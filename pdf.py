import os, sys, re, datetime
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

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

CHECKPOINT_PATH = r"put your checkpoint"
OUT_IMG_DIR     = r"gradcam path to save"
OUT_PDF_DIR     = r"pdf directory"
IMG_SIZE        = 320
CLASS_NAMES     = ["NORMAL", "PNEUMONIA"]
TARGET_LAYER    = "features.denseblock4"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_PDF_DIR, exist_ok=True)

print(">> Device:", DEVICE)
if DEVICE.type == "cuda":
    print(">> GPU:", torch.cuda.get_device_name(0))
class GradCAM:
    def __init__(self, model, target_layer: str):
        self.model = model.eval()
        self.tname = target_layer
        self.activations = None
        self.gradients = None
        self._register()

    def _layer(self):
        l = self.model
        for a in self.tname.split("."):
            l = getattr(l, a)
        return l

    def _register(self):
        tl = self._layer()
        def fwd_hook(m, i, o): self.activations = o
        def bwd_hook(m, gi, go): self.gradients = go[0]
        tl.register_forward_hook(fwd_hook)
        tl.register_full_backward_hook(bwd_hook)

    def __call__(self, x, class_idx=None):
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                idx = torch.argmax(logits, dim=1)
            else:
                if isinstance(class_idx, int):
                    idx = torch.tensor([class_idx]*x.size(0), device=x.device)
                else:
                    idx = torch.tensor(class_idx, device=x.device)
            sel = logits.gather(1, idx.view(-1,1)).sum()
            self.model.zero_grad(set_to_none=True)
            sel.backward(retain_graph=True)

            w = torch.mean(self.gradients, dim=(2,3), keepdim=True)
            cam = torch.sum(w * self.activations, dim=1)
            cam = F.relu(cam)
            cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            return cam.detach().cpu().numpy(), idx.detach().cpu().numpy()
def overlay_cam_on_rgb(img_rgb, cam_resized, alpha=0.40):
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(img_rgb*(1-alpha) + heatmap*alpha)

def mask_and_circle_from_cam(cam_resized, method="percentile", thr=0.85):
    if method == "otsu":
        cm = (cam_resized*255).astype(np.uint8)
        _, thr_val = cv2.threshold(cm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_mask = (cm >= thr_val).astype(np.uint8)*255
    else:
        q = np.quantile(cam_resized, thr)
        bin_mask = (cam_resized >= q).astype(np.uint8)*255
    contours,_ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_mask, None
    c = max(contours, key=cv2.contourArea)
    (x,y), r = cv2.minEnclosingCircle(c)
    return bin_mask, (int(x), int(y), int(r))

def draw_circle(image_rgb, circle, color=(255,255,255), thickness=2):
    out = image_rgb.copy()
    if circle is not None:
        x,y,r = circle
        cv2.circle(out, (x,y), r, color, thickness)
    return out

def build_model():
    m = tv.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state"], strict=True)
    m.to(DEVICE).eval()
    return m

def aug(img_size):
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])

def risk_level(prob):
    p = prob*100
    if p >= 85: return "High", colors.red
    if p >= 60: return "Medium", colors.orange
    return "Low", colors.green

def severity_from_mask(mask):
    frac = float(mask.sum() / 255) / (mask.shape[0]*mask.shape[1] + 1e-8)
    if frac > 0.30: return "Severe", frac
    if frac > 0.10: return "Moderate", frac
    return "Mild", frac

def is_image(path):
    return os.path.splitext(path)[1].lower() in [".jpg",".jpeg",".png",".bmp"]

def collect_images(path):
    if os.path.isfile(path) and is_image(path): return [path]
    if os.path.isdir(path):
        imgs=[]
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            imgs.extend(glob(os.path.join(path, ext)))
        return sorted(imgs)
    return []

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip() or "N_A"
def run_case(model, gcam, img_path, patient_name, patient_age, patient_sex, percentile=0.85):
    arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    t = aug(IMG_SIZE)(image=arr)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0]
        prob_pneu = float(probs[1].item())
        pred_idx = int(torch.argmax(probs).item())
        pred_name = CLASS_NAMES[pred_idx]

    cams,_ = gcam(t, class_idx=pred_idx)
    cam_small = cams[0]
    cam_resized = cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_CUBIC)

    overlay = overlay_cam_on_rgb(arr, cam_resized, alpha=0.40)
    mask, circle = mask_and_circle_from_cam(cam_resized, method="percentile", thr=percentile)
    circled = draw_circle(overlay, circle, color=(255,255,255), thickness=2)
    heatmap_rgb = overlay_cam_on_rgb(arr, cam_resized, alpha=0.70)

    base = os.path.splitext(os.path.basename(img_path))[0]
    img_orig_path = os.path.join(OUT_IMG_DIR, f"{base}_orig.png")
    img_heat_path = os.path.join(OUT_IMG_DIR, f"{base}_heat.png")
    img_circ_path = os.path.join(OUT_IMG_DIR, f"{base}_circ.png")
    cv2.imwrite(img_orig_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_heat_path, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_circ_path, cv2.cvtColor(circled, cv2.COLOR_RGB2BGR))

    severity, frac = severity_from_mask(mask)
    return {
        "img_path": img_path,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "pred_name": pred_name,
        "prob_pneumonia": prob_pneu,
        "risk": risk_level(prob_pneu),     
        "severity": (severity, frac),       
        "paths": (img_orig_path, img_heat_path, img_circ_path)
    }

def mmx(x): return x*mm

def draw_header(c, title):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(mmx(20), mmx(285), title)
    c.setFont("Helvetica", 9)
    c.drawString(mmx(20), mmx(279), f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(mmx(20), mmx(277), mmx(190), mmx(277))

def draw_meta(c, case):
    c.setFont("Helvetica", 10)
    c.drawString(mmx(20), mmx(270), f"Patient Name: {case['patient_name']}")
    c.drawString(mmx(20), mmx(265), f"Age: {case['patient_age']}")
    c.drawString(mmx(20), mmx(260), f"Sex: {case['patient_sex']}")
    c.drawString(mmx(20), mmx(254), f"Case File: {os.path.basename(case['img_path'])}")
    c.drawString(mmx(20), mmx(248), "Model: DenseNet-121 + Grad-CAM Explainability")

    pred = case["pred_name"]; pp = case["prob_pneumonia"]
    risk_label, risk_color = case["risk"]
    c.drawString(mmx(20), mmx(242), f"Prediction: {pred}  |  P(pneumonia) = {pp:.3f}")
    c.setFillColor(risk_color); c.rect(mmx(20), mmx(237), mmx(15), mmx(5), fill=1, stroke=0)
    c.setFillColor(colors.black); c.drawString(mmx(37), mmx(237), f"Risk Level: {risk_label}")

    sev_label, frac = case["severity"]
    c.drawString(mmx(20), mmx(231), f"Severity (CAM-area proxy): {sev_label}  |  area ≈ {frac*100:.1f}%")

def draw_images_row(c, case):
    orig, heat, circ = case["paths"]
    w_img = mmx(55); h_img = mmx(80); y = mmx(150)
    c.drawImage(orig, mmx(20),  y, width=w_img, height=h_img, preserveAspectRatio=True, mask='auto')
    c.drawImage(heat, mmx(80),  y, width=w_img, height=h_img, preserveAspectRatio=True, mask='auto')
    c.drawImage(circ, mmx(140), y, width=w_img, height=h_img, preserveAspectRatio=True, mask='auto')
    c.setFont("Helvetica", 9)
    c.drawCentredString(mmx(20)+w_img/2,  y - mmx(4), "Original")
    c.drawCentredString(mmx(80)+w_img/2,  y - mmx(4), "Grad-CAM Heatmap")
    c.drawCentredString(mmx(140)+w_img/2, y - mmx(4), "Circled Lesion")

def draw_disclaimer(c):
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(mmx(20), mmx(20),
        "Research prototype — not a medical device. Predictions are assistive only and must be verified by a qualified clinician.")
    c.setFillColor(colors.black)

def save_pdf(case):
    base = os.path.splitext(os.path.basename(case["img_path"]))[0]
    safe_name = sanitize(case["patient_name"])
    pdf_path = os.path.join(OUT_PDF_DIR, f"{safe_name}_{base}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    draw_header(c, "Chest X-ray Pneumonia — Case Report")
    draw_meta(c, case)
    draw_images_row(c, case)
    draw_disclaimer(c)
    c.showPage()
    c.save()
    return pdf_path

def prompt_inputs():
    print("\nEnter Patient Name:")
    name = input("Name: ").strip() or "N/A"
    print("Enter Patient Age:")
    age = input("Age: ").strip() or "N/A"
    print("Enter Sex (M/F):")
    sex = input("Sex: ").strip().upper()
    if sex not in ["M", "F"]: sex = "N/A"
    print("\nPaste an X-ray image path OR a folder of images (drag & drop works).")
    in_path = input("Path: ").strip().strip('"').strip("'")
    return name, age, sex, in_path

def main():
    model = build_model()
    gcam = GradCAM(model, TARGET_LAYER)

    patient_name, patient_age, patient_sex, in_path = prompt_inputs()

    imgs = []
    if os.path.isfile(in_path) and is_image(in_path):
        imgs = [in_path]
    elif os.path.isdir(in_path):
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            imgs.extend(glob(os.path.join(in_path, ext)))
        imgs = sorted(imgs)
    else:
        print("No valid image(s) found. Exiting."); sys.exit(1)

    print(f"Found {len(imgs)} image(s). Generating reports...")
    for i, p in enumerate(imgs, 1):
        try:
            case = run_case(model, gcam, p, patient_name, patient_age, patient_sex, percentile=0.85)
            pdf_path = save_pdf(case)
            print(f"[{i}/{len(imgs)}] Saved report → {pdf_path}")
        except Exception as e:
            print(f"[{i}/{len(imgs)}] ERROR on {p}: {e}")

    print("\nDone. PDF reports in:", OUT_PDF_DIR)
    print("Reminder: Research prototype — not a medical device.")

if __name__ == "__main__":
    main()