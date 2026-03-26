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

def compute_confidence(score, threshold):
    # k controls the steepness of the confidence curve.
    k = 5.0 / (threshold + 1e-6) 
    prob_anomaly = 1 / (1 + np.exp(-k * (score - threshold)))
    
    if score > threshold:
        return float(prob_anomaly * 100)
    else:
        return float((1 - prob_anomaly) * 100)

def get_severity(score, threshold):
    r = score / threshold
    if r < 1.0:
        return "NONE"
    elif r < 1.4:
        return "LOW"
    elif r < 1.8:
        return "MEDIUM"
    else:
        return "HIGH"

# ================== LOAD NORMAL BANK ==================
@st.cache_resource
def load_normal_bank():
    try:
        normal_features = np.load("normal_features.npy", allow_pickle=True)
        normal_scores = np.load("normal_scores.npy")
        
        # Avoid empty array errors causing the Limit: 0.00 bug
        if len(normal_scores) > 0:
            threshold = np.percentile(normal_scores, 95)
        else:
            threshold = 0.0
            
        return list(normal_features), normal_scores, threshold
    except FileNotFoundError:
        st.error("Missing normal_features.npy or normal_scores.npy. Please ensure they are in the same folder.")
        return [], np.array([]), 0.0

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

        # Extract features and calculate anomaly map
        feat = extract_features(crop)
        
        # Prevent crash if memory bank is empty
        if not normal_feature_bank:
            st.error("Cannot compute anomaly. Memory bank is empty.")
            break
            
        amap = compute_anomaly_map(feat, normal_feature_bank)

        # 🚨 TEMPORARY OVERRIDE FOR TESTING 🚨
        # If your threshold is 0.00 (from empty/bad npy files), we force it to 45.0 for testing.
        if AUTO_THRESHOLD <= 0.01:
            AUTO_THRESHOLD = 45.0  

        score = float(amap.max())
        is_anomalous = score > AUTO_THRESHOLD
        
        status = "ANOMALOUS" if is_anomalous else "NORMAL"
        confidence = compute_confidence(score, AUTO_THRESHOLD)
        severity = get_severity(score, AUTO_THRESHOLD)
        
        # Resize amap to match the crop exactly
        amap_resized = cv2.resize(amap, (crop.shape[1], crop.shape[0]))
        
        # Create a boolean mask of ONLY the pixels that exceed the threshold
        mask = amap_resized > AUTO_THRESHOLD
        coverage = float(100 * np.sum(mask) / mask.size) if is_anomalous else 0.0

        # ===== VISUALIZATION LOGIC =====
        # Scale the colors based on the threshold so 'hot' is always an anomaly
        max_score = max(AUTO_THRESHOLD * 1.5, score + 1e-5)
        normalized_amap = np.clip(amap_resized / max_score, 0, 1)

        # Generate the heatmap
        heatmap = cv2.applyColorMap(
            (normalized_amap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Copy original crop for the overlay
        overlay = crop.copy()

        if is_anomalous:
            # Blend the original crop and the heatmap
            blended = cv2.addWeighted(crop, 0.4, heatmap, 0.6, 0)
            
            # Apply the blended heatmap ONLY to the defective pixels
            overlay[mask] = blended[mask]

        display_img = resize_for_display(overlay, max_width=450)

        # ===== DASHBOARD LAYOUT =====
        col1, col2 = st.columns([1, 2])

        with col1:
            if is_anomalous:
                st.error(f"**{status}** | {severity}")
            else:
                st.success(f"**{status}**")
                
            st.write(f"**Score:** {score:.2f} *(Limit: {AUTO_THRESHOLD:.2f})*")
            st.write(f"**Confidence:** {confidence:.1f}%")
            if is_anomalous:
                st.write(f"**Affected Area:** {coverage:.1f}%")

        with col2:
            st.image(
                display_img,
                caption="Anomaly Visualization"
            )

    if not detected:
        st.warning("No bottle detected in the image.")