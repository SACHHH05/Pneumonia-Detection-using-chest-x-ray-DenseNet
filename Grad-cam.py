import os, sys
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F

# ---------- CONFIG ----------
CHECKPOINT_PATH = r"Path"
OUT_DIR = r"Grad-cam path to save"
IMG_SIZE = 320
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
TARGET_LAYER = "features.denseblock4"  # DenseNet-121 last conv block
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

print(">> Device:", DEVICE)
if DEVICE.type == "cuda":
    print(">> GPU:", torch.cuda.get_device_name(0))

# ---------- Grad-CAM core ----------
class GradCAM:
    def __init__(self, model, target_layer: str):
        self.model = model.eval()
        self.target_layer_name = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _get_target_layer(self):
        layer = self.model
        for attr in self.target_layer_name.split("."):
            layer = getattr(layer, attr)
        return layer

    def _register_hooks(self):
        tl = self._get_target_layer()

        def fwd_hook(module, inp, out):
            self.activations = out  # keep with grad

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        tl.register_forward_hook(fwd_hook)
        tl.register_full_backward_hook(bwd_hook)

    def __call__(self, inputs, class_idx=None):
        # Make sure grads are enabled for CAM
        with torch.enable_grad():
            logits = self.model(inputs)  # forward with grad
            if class_idx is None:
                idx = torch.argmax(logits, dim=1)
            else:
                if isinstance(class_idx, int):
                    idx = torch.tensor([class_idx]*inputs.size(0), device=inputs.device)
                else:
                    idx = torch.tensor(class_idx, device=inputs.device)

            selected = logits.gather(1, idx.view(-1,1)).sum()
            self.model.zero_grad(set_to_none=True)
            selected.backward(retain_graph=True)

            # activations: (B,C,h,w), gradients: (B,C,h,w)
            weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)  # (B,C,1,1)
            cam = torch.sum(weights * self.activations, dim=1)             # (B,h,w)
            cam = F.relu(cam)
            # normalize per-sample to [0,1]
            cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy(), idx.detach().cpu().numpy()

def overlay_cam_on_rgb(img_rgb, cam_resized, alpha=0.40):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    over = np.uint8(img_rgb * (1 - alpha) + heatmap * alpha)
    return over

def mask_and_circle_from_cam(cam_resized, method="percentile", thr=0.85):
    if method == "otsu":
        cm = (cam_resized * 255).astype(np.uint8)
        _, thr_val = cv2.threshold(cm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_mask = (cm >= thr_val).astype(np.uint8) * 255
    else:
        t = np.quantile(cam_resized, thr)  # keep top (1 - thr) mass
        bin_mask = (cam_resized >= t).astype(np.uint8) * 255

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_mask, None
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return bin_mask, (int(x), int(y), int(r))

def draw_circle(image_rgb, circle, color=(255,255,255), thickness=2):
    out = image_rgb.copy()
    if circle is not None:
        x, y, r = circle
        cv2.circle(out, (x, y), r, color, thickness)
    return out

# ---------- Model / transforms ----------
def build_model():
    m = tv.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state"], strict=True)
    m.to(DEVICE).eval()
    return m

def aug(img_size):
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])

# ---------- Helpers ----------
def is_image(path):
    return os.path.splitext(path)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]

def collect_images(path):
    if os.path.isfile(path) and is_image(path):
        return [path]
    if os.path.isdir(path):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            imgs.extend(glob(os.path.join(path, ext)))
        return sorted(imgs)
    return []

# ---------- Inference ----------
def run_one(model, gcam, img_path, out_dir, percentile=0.85):
    arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    t = aug(IMG_SIZE)(image=arr)["image"].unsqueeze(0).to(DEVICE)

    # inference (no_grad ok for probs)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0]
        prob_pneu = float(probs[1].item())
        pred_idx = int(torch.argmax(probs).item())
        pred_name = CLASS_NAMES[pred_idx]

    # Grad-CAM needs grads enabled
    cams, _ = gcam(t, class_idx=pred_idx)
    cam_small = cams[0]
    cam_resized = cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_CUBIC)

    overlay = overlay_cam_on_rgb(arr, cam_resized, alpha=0.40)
    mask, circle = mask_and_circle_from_cam(cam_resized, method="percentile", thr=percentile)
    circled = draw_circle(overlay, circle, color=(255,255,255), thickness=2)

    # triptych
    heatmap_rgb = overlay_cam_on_rgb(arr, cam_resized, alpha=0.70)
    trip = np.concatenate([arr, heatmap_rgb, circled], axis=1)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base}_pred-{pred_name}_p{prob_pneu:.3f}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))

    return out_path, pred_name, prob_pneu

def prompt_path():
    print("\n📥 Paste an X-ray image path OR a folder of images (drag & drop works).")
    p = input("Path: ").strip().strip('"').strip("'")
    return p

def main():
    model = build_model()
    gcam = GradCAM(model, TARGET_LAYER)

    user_path = prompt_path()
    images = collect_images(user_path)
    if not images:
        print("⚠️ No images found. Exiting.")
        sys.exit(1)

    print(f"🔎 Found {len(images)} image(s). Processing...")
    for i, img_path in enumerate(images, 1):
        try:
            out_path, pred, p = run_one(model, gcam, img_path, OUT_DIR, percentile=0.85)
            print(f"[{i}/{len(images)}] saved: {out_path} | pred: {pred} | prob_pneumonia: {p:.3f}")
        except Exception as e:
            print(f"[{i}/{len(images)}] ERROR on {img_path}: {e}")

    print("\n✅ Done. Outputs in:", OUT_DIR)
    print("🩺 Research prototype only. Not a medical device.")

if __name__ == "__main__":
    main()
