import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
import torch
import torch.nn as nn
import torchvision.models as tv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = r"dataset"
CHECKPOINT_PATH = r"checkpoint path"
THRESHOLD_JSON = "best_threshold.json"  # if threshold sweep done
IMG_SIZE = 320
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = r"replace with your path"  # where to save the figures
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOAD THRESHOLD IF EXISTS
# ---------------------------
if os.path.exists(THRESHOLD_JSON):
    with open(THRESHOLD_JSON, "r") as f:
        thr_data = json.load(f)
    THRESHOLD = float(thr_data.get("threshold", 0.5))
    print(f"✅ Loaded threshold: {THRESHOLD:.4f}")
else:
    THRESHOLD = 0.5
    print("⚠️ No threshold file found, defaulting to 0.50")

# ---------------------------
# TRANSFORMS
# ---------------------------
test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

# ---------------------------
# DATA LOADING
# ---------------------------
def load_test_files(root):
    files, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(root, "test", cls)
        # Support common image extensions
        patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]
        cls_files = []
        for p in patterns:
            cls_files += glob(os.path.join(cls_dir, p))
        files += cls_files
        labels += [idx] * len(cls_files)
    return files, labels

# ---------------------------
# MODEL BUILDER
# ---------------------------
def load_model(checkpoint_path):
    model = tv.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # supports {"model_state": ...} or direct state_dict / "state_dict"
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip DataParallel prefixes if any
    new_sd = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        print("⚠️ Unexpected keys in checkpoint:", unexpected)
    if missing:
        print("⚠️ Missing keys when loading checkpoint:", missing)

    model.to(DEVICE)
    model.eval()
    print("✅ Model Loaded")
    return model

# ---------------------------
# INFERENCE
# ---------------------------
def infer(model, files):
    probs, preds = [], []
    with torch.no_grad():
        for f in tqdm(files, desc="Testing"):
            img = np.array(Image.open(f).convert("RGB"))
            img_t = test_transform(image=img)["image"].unsqueeze(0).to(DEVICE)

            logits = model(img_t)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)

            pred = 1 if p >= THRESHOLD else 0
            preds.append(pred)

    return np.array(probs, dtype=np.float32), np.array(preds, dtype=np.int64)

# ---------------------------
# PLOTTING
# ---------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"💾 Saved confusion matrix: {outpath}")

def plot_roc(y_true, y_prob, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"💾 Saved ROC curve: {outpath}")

def plot_per_class_prf(y_true, y_pred, class_names, outpath):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    values_by_metric = {m: [report[cls][m] for cls in class_names] for m in metrics}

    x = np.arange(len(class_names))
    width = 0.22
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x - width, values_by_metric["precision"], width, label="Precision")
    ax.bar(x,          values_by_metric["recall"],    width, label="Recall")
    ax.bar(x + width,  values_by_metric["f1-score"],  width, label="F1-score")
    ax.set_title("Per-class Precision, Recall, and F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"💾 Saved PRF bars: {outpath}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print(f"✅ Device: {DEVICE}")

    test_files, test_labels = load_test_files(DATA_DIR)
    print(f"📂 Test Images: {len(test_files)}")

    model = load_model(CHECKPOINT_PATH)

    probs, preds = infer(model, test_files)
    y_true = np.array(test_labels, dtype=np.int64)

    # Metrics
    acc = accuracy_score(y_true, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, preds, labels=[0, 1], zero_division=0)
    auc_val = roc_auc_score(y_true, probs)

    print("\n🔎 RESULTS on TEST SET")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc_val:.4f}")
    print("Per-class Metrics [NORMAL, PNEUMONIA]:")
    print("Precision:", pr)
    print("Recall   :", rc)
    print("F1-score :", f1)

    print("\n📊 Classification Report:\n",
          classification_report(y_true, preds, target_names=CLASS_NAMES, digits=4))
    print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_true, preds))

    # Plots
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    prf_path = os.path.join(OUTPUT_DIR, "per_class_prf.png")

    plot_confusion_matrix(y_true, preds, CLASS_NAMES, cm_path)
    plot_roc(y_true, probs, roc_path)
    plot_per_class_prf(y_true, preds, CLASS_NAMES, prf_path)
    