import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from pathlib import Path

# -------------------------------
# 1. Helper Functions
# -------------------------------

def lbp_image(img):
    img = img.astype('uint8')
    rows, cols = img.shape
    lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            c = img[i, j]
            b = 0
            b |= (1 << 0) if img[i - 1, j - 1] >= c else 0
            b |= (1 << 1) if img[i - 1, j] >= c else 0
            b |= (1 << 2) if img[i - 1, j + 1] >= c else 0
            b |= (1 << 3) if img[i, j + 1] >= c else 0
            b |= (1 << 4) if img[i + 1, j + 1] >= c else 0
            b |= (1 << 5) if img[i + 1, j] >= c else 0
            b |= (1 << 6) if img[i + 1, j - 1] >= c else 0
            b |= (1 << 7) if img[i, j - 1] >= c else 0
            lbp[i - 1, j - 1] = b
    return lbp

def extract_features(img):
    edges = cv2.Canny(img, 100, 200)
    edge_feat = cv2.resize(edges, (8, 8)).flatten() / 255.0
    hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-9)
    lbp = lbp_image(img)
    lbp_hist, _ = np.histogram(lbp.flatten(), bins=16, range=(0, 256))
    lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-9)
    return np.concatenate([edge_feat, hist, lbp_hist])

def detect_and_crop_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return gray
    faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    crop = gray[y1:y2, x1:x2]
    return crop

# -------------------------------
# 2. Load Model
# -------------------------------

MODEL_PATH = "deepfake_model.npz"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    data = np.load(MODEL_PATH)
    return data["W"], float(data["b"])

model = load_model()

# -------------------------------
# 3. Prediction & Visualization
# -------------------------------

def predict_from_gray(img_gray):
    img_r = cv2.resize(img_gray, (64, 64))
    feat = extract_features(img_r)
    if model is None:
        return None
    W, b = model
    linear = np.dot(feat, W) + b
    prob = 1 / (1 + np.exp(-linear))
    label = "Fake" if prob >= 0.5 else "Real"
    confidence = prob if prob >= 0.5 else 1 - prob
    return {"label": label, "prob": float(prob), "confidence": float(confidence), "img": img_r}

def make_visuals(img):
    edges = cv2.Canny(img, 100, 200)
    lbp = lbp_image(img)
    hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-9)
    return edges, lbp, hist

def generate_html_report(image_path, label, prob, edges, lbp, hist, out_path):
    fig1 = plt.figure(figsize=(4, 3))
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    fig2 = plt.figure(figsize=(4, 3))
    plt.imshow(lbp, cmap="gray")
    plt.axis("off")
    fig3 = plt.figure(figsize=(6, 3))
    plt.plot(hist)
    plt.title("Intensity Histogram")

    buf1, buf2, buf3 = BytesIO(), BytesIO(), BytesIO()
    fig1.savefig(buf1, format="png")
    fig2.savefig(buf2, format="png")
    fig3.savefig(buf3, format="png")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    b64_edges = base64.b64encode(buf1.getvalue()).decode("utf-8")
    b64_lbp = base64.b64encode(buf2.getvalue()).decode("utf-8")
    b64_hist = base64.b64encode(buf3.getvalue()).decode("utf-8")

    html = f"""
    <html><body>
    <h2>Deepfake Authentication Report</h2>
    <p><b>File:</b> {os.path.basename(image_path)}</p>
    <p><b>Prediction:</b> {label}</p>
    <p><b>Probability:</b> {prob:.4f}</p>
    <img src="data:image/png;base64,{b64_edges}" width="300">
    <img src="data:image/png;base64,{b64_lbp}" width="300">
    <br><img src="data:image/png;base64,{b64_hist}" width="600">
    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

# -------------------------------
# 4. Streamlit App UI
# -------------------------------

st.set_page_config(page_title="Deepfake Auth XAI", layout="wide")
st.title("🧠 Deepfake Authentication & Explainability System")

st.sidebar.header("Options")
crop_face = st.sidebar.checkbox("Crop detected face before analysis", value=True)
save_html = st.sidebar.checkbox("Save HTML report", value=True)

if model is None:
    st.sidebar.warning("⚠️ Model not found! Please place deepfake_model.npz in project folder.")
else:
    st.sidebar.success("✅ Model loaded successfully!")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    arr = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("❌ Could not read image.")
    else:
        if crop_face:
            img_gray = detect_and_crop_face(img_bgr)
        else:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        result = predict_from_gray(img_gray)
        if result is None:
            st.error("⚠️ Model not loaded.")
        else:
            edges, lbp, hist = make_visuals(result["img"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
            with col2:
                st.image(edges, caption="Edges (Canny)", use_column_width=True)
            with col3:
                st.image(lbp, caption="Local Binary Pattern", use_column_width=True)

            st.subheader(f"Prediction: {result['label']}")
            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            st.line_chart(hist)

            if save_html:
                report_dir = Path("reports")
                report_dir.mkdir(exist_ok=True)
                out_html = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                generate_html_report(uploaded.name, result["label"], result["prob"], edges, lbp, hist, str(out_html))
                st.success(f"Report saved: {out_html}")

# -------------------------------
# 5. Batch Mode
# -------------------------------

st.markdown("---")
st.header("🧩 Batch Report Mode")
batch_folder = st.text_input("Enter folder path containing images", "batch_images")

if st.button("Generate Batch Reports"):
    if model is None:
        st.warning("⚠️ Model not loaded!")
    else:
        in_dir = Path(batch_folder)
        if not in_dir.exists():
            st.error("Folder not found!")
        else:
            out_dir = Path("batch_reports")
            out_dir.mkdir(exist_ok=True)
            files = [f for f in in_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            st.info(f"Found {len(files)} images. Generating reports...")
            progress = st.progress(0)
            for i, img_path in enumerate(files):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                gray = detect_and_crop_face(bgr) if crop_face else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                result = predict_from_gray(gray)
                if result is None:
                    continue
                edges, lbp, hist = make_visuals(result["img"])
                report_file = out_dir / f"{img_path.stem}_report.html"
                generate_html_report(str(img_path), result["label"], result["prob"], edges, lbp, hist, str(report_file))
                progress.progress((i + 1) / len(files))
            st.success(f"Batch reports saved in {out_dir.resolve()}")
