import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms

# ================== CONFIG ==================
st.set_page_config(
    page_title="Bottle Anomaly Detection",
    layout="centered"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== MODELS ==================
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")

    backbone = models.resnet18(weights="IMAGENET1K_V1")
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.to(device).eval()

    return yolo, backbone

yolo_model, backbone = load_models()

# ================== TRANSFORM ==================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================== UTILS ==================
def resize_for_display(image, max_width=450):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)))

# ================== FUNCTIONS ==================
def extract_features(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone(x)
    return feat.squeeze(0)

def compute_anomaly_map(test_feat, memory_bank):
    dists = []
    for ref in memory_bank:
        dists.append(torch.norm(test_feat - ref, dim=0))
    return torch.stack(dists).min(dim=0)[0].cpu().numpy()

def compute_confidence(score, normal_scores):
    return float(np.mean(normal_scores < score) * 100)

def get_severity(score, threshold):
    r = score / threshold
    if r < 1.2:
        return "LOW"
    elif r < 1.8:
        return "MEDIUM"
    else:
        return "HIGH"

def defect_coverage(anomaly_map, ratio=0.6):
    mask = anomaly_map > (ratio * anomaly_map.max())
    return float(100 * np.sum(mask) / mask.size)

def draw_anomaly_mask(image, anomaly_map, alpha=0.4):
    h, w, _ = image.shape
    amap = cv2.resize(anomaly_map, (w, h))
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)

    mask = (amap > 0.6).astype(np.uint8) * 255
    red = np.zeros_like(image)
    red[:, :, 0] = mask

    return cv2.addWeighted(red, alpha, image, 1 - alpha, 0)

# ================== LOAD NORMAL BANK ==================
@st.cache_resource
def load_normal_bank():
    normal_features = np.load("normal_features.npy", allow_pickle=True)
    normal_scores = np.load("normal_scores.npy")
    threshold = np.percentile(normal_scores, 95)
    return list(normal_features), normal_scores, threshold

normal_feature_bank, normal_scores, AUTO_THRESHOLD = load_normal_bank()

# ================== UI ==================
st.title("🍾 Bottle Anomaly Detection")
st.caption("Visual anomaly detection with severity & coverage - By Arka")

uploaded = st.file_uploader(
    "Upload bottle image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model(img)[0]
    detected = False

    for box in results.boxes:
        label = yolo_model.names[int(box.cls[0])]
        if label not in ["bottle", "vase"]:
            continue

        detected = True

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        feat = extract_features(crop)
        amap = compute_anomaly_map(feat, normal_feature_bank)

        score = float(amap.max())
        status = "ANOMALOUS" if score > AUTO_THRESHOLD else "NORMAL"
        confidence = compute_confidence(score, normal_scores)
        severity = get_severity(score, AUTO_THRESHOLD)
        coverage = defect_coverage(amap)

        heatmap = cv2.applyColorMap(
            cv2.resize(
                (amap / amap.max() * 255).astype(np.uint8),
                (crop.shape[1], crop.shape[0])
            ),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)
        overlay = draw_anomaly_mask(overlay, amap)

        display_img = resize_for_display(overlay, max_width=450)

        # ===== DASHBOARD LAYOUT =====
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"{status} | {severity}")
            st.write(f"**Score:** {score:.2f}")
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write(f"**Affected Area:** {coverage:.1f}%")

        with col2:
            st.image(
                display_img,
                caption="Anomaly Visualization"
            )

    if not detected:
        st.warning("No bottle detected in the image.")
