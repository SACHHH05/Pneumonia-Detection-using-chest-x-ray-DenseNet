import os
import random
from glob import glob
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

DATA_DIR = r"put your dataset"
CHECKPOINT_DIR = r"put your checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------
# DATASET
# ----------------------
class ChestXrayDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.filepaths[idx]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

def make_file_list(root_dir, split="train"):
    filepaths, labels = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        pattern = os.path.join(root_dir, split, cls, "*.jpeg")
        files = glob(pattern)
        filepaths += files
        labels += [cls_idx] * len(files)
    return filepaths, labels

# ----------------------
# TRANSFORMS
# ----------------------
def build_transforms(img_size, strong_aug=False):
    train_tf = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]
    if strong_aug:
        train_tf += [A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)]
    train_tf += [A.Normalize(), ToTensorV2()]

    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    return A.Compose(train_tf), val_tf

# ----------------------
# MODELS
# ----------------------
def build_model(model_name: str, num_classes=2, pretrained=True):
    name = model_name.lower()
    if name in ["resnet18", "rn18"]:
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name in ["densenet121", "dense121", "dn121"]:
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    elif name in ["efficientnet_b0", "effb0", "efficientnet"]:
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ----------------------
# TRAIN / VAL
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss, preds, targets = 0.0, [], []
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds += torch.argmax(logits, dim=1).detach().cpu().tolist()
        targets += labels.detach().cpu().tolist()
    return running_loss / len(loader.dataset), preds, targets

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, preds, targets, probs_all = 0.0, [], [], []
    for imgs, labels in tqdm(loader, desc="Val"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        preds += torch.argmax(logits, dim=1).detach().cpu().tolist()
        targets += labels.detach().cpu().tolist()
        probs_all += probs
    return running_loss / len(loader.dataset), preds, targets, probs_all

# ----------------------
# UTILS
# ----------------------
def build_weighted_sampler(labels):
    counts = Counter(labels)
    # inverse frequency per class → sample weights per item
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True), class_weights

def compute_class_weights_for_ce(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ----------------------
# MAIN
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet121", help="densenet121 | resnet18 | efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--use_sampler", action="store_true", help="Use class-balanced WeightedRandomSampler")
    args = parser.parse_args() if hasattr(__import__('__main__'), '__file__') else parser.parse_args([])

    set_seed()

    train_files, train_labels = make_file_list(DATA_DIR, "train")
    val_files, val_labels = make_file_list(DATA_DIR, "val")

    # If no explicit val split exists, make one from train
    if len(val_files) == 0:
        combined = list(zip(train_files, train_labels))
        random.shuffle(combined)
        split = int(0.8 * len(combined))
        train_files, train_labels = zip(*combined[:split])
        val_files, val_labels = zip(*combined[split:])
        train_files, train_labels = list(train_files), list(train_labels)
        val_files, val_labels = list(val_files), list(val_labels)

    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")
    print(f"Class balance (train):", Counter(train_labels))

    train_tf, val_tf = build_transforms(args.img_size, strong_aug=args.strong_aug)
    train_ds = ChestXrayDataset(train_files, train_labels, transform=train_tf)
    val_ds = ChestXrayDataset(val_files, val_labels, transform=val_tf)

    if args.use_sampler:
        sampler, _ = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

    ce_weights = compute_class_weights_for_ce(train_labels)
    criterion = nn.CrossEntropyLoss(weight=ce_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auc = 0.0
    ckpt_name = f"best_{args.model}_img{args.img_size}_bs{args.batch_size}.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_preds, train_targets = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_preds, val_targets, val_probs = validate(model, val_loader, criterion)

        try:
            auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            auc = 0.0

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val AUC: {auc:.4f}")

        report = classification_report(val_targets, val_preds, target_names=CLASS_NAMES, digits=4)
        print("Classification Report:\n", report)
        cm = confusion_matrix(val_targets, val_preds)
        print("Confusion Matrix:\n", cm)

        scheduler.step(val_loss)

        if auc > best_auc:
            best_auc = auc
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "auc": best_auc,
                "args": vars(args),
                "ce_weights": ce_weights.detach().cpu().numpy().tolist(),
            }, os.path.join(CHECKPOINT_DIR, ckpt_name))
            print(f"Saved best model → {ckpt_name}")

    print("Training complete. Best Val AUC:", best_auc)

if __name__ == "__main__":
    main()
