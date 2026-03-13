import os
import streamlit as st
import numpy as np
import time
import tempfile
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime

# ── CONFIGURATION ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS STYLE ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap](https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap)');
    
    html, body, .stApp {
        background: #060810;
        color: #C9D1E0;
        font-family: 'Syne', sans-serif;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 30px;
    }
    .stat-item {
        background: rgba(10, 14, 22, 0.8);
        border: 1px solid rgba(99, 179, 237, 0.1);
        padding: 20px;
        border-radius: 4px;
        text-align: center;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 900;
        color: #63B3ED;
    }
    .scan-result {
        background: rgba(10, 14, 22, 0.8);
        border: 1px solid rgba(99, 179, 237, 0.1);
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 10px;
        position: relative;
    }
    .scan-result.invalid {
        border-left: 5px solid #ecc94b;
    }
    .finding-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #63B3ED;
        padding: 15px;
        margin-top: 10px;
        font-size: 13px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

FINDINGS = {
    "Glioma": "Diffuse infiltrative tumor detected in parenchymal tissue. Typically requires contrast-enhanced follow-up.",
    "Meningioma": "Extra-axial mass likely originating from meninges. Often well-circumscribed.",
    "Pituitary": "Sellar region expansion detected. May impact endocrine function or optic chiasm.",
    "No Tumor": "No abnormal intracranial mass detected. Physiological structures appear within normal limits.",
    "INVALID": "Image rejected: Structural analysis indicates this is not a valid Brain MRI sequence."
}

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def is_valid_mri(image):
    img_gray = np.array(image.convert("L"))
    if np.mean(img_gray) > 200 or np.mean(img_gray) < 5:
        return False
    if np.std(img_gray) < 18:
        return False
    return True

def is_genuine_mri(image):
    img_gray = np.array(image.convert("L"))
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    peak_black = hist[0:10].sum() / hist.sum()
    # Logic: MRI has specific texture (Laplacian) and high background black ratio
    if laplacian_var < 10 or laplacian_var > 1000:
        return False
    if peak_black < 0.15:
        return False
    return True

def apply_clahe(image):
    img = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    return Image.fromarray(enhanced).convert("RGB")

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "NEUROSCAN AI CLINICAL REPORT", ln=True, align="C")
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    for i, r in enumerate(results):
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"CASE {i+1} - {r['filename']}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"Prediction: {r['label'].upper()}", ln=True)
        pdf.cell(0, 6, f"Confidence: {r['confidence']:.1f}%", ln=True)
        pdf.cell(0, 6, f"Inference Latency: {r['time']:.3f}s", ln=True)
        pdf.ln(4)
    
    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = "resnet_model.onnx"
        if os.path.exists(path):
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess(image):
    img = image.resize((224, 224))
    arr = np.array(img).astype("float32")[:, :, ::-1] # RGB to BGR
    # ResNet Mean Subtraction
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, 0)

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.title("🧠 NeuroScan AI")
st.caption("Precision Brain Tumor MRI Classification Sequence")

session = load_model()

if session is None:
    st.warning("⚠️ Model File Error: 'resnet_model.onnx' not found in root directory.")
    st.stop()

files = st.file_uploader(
    "Upload MRI Scan Sequences",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ── ANALYSIS ENGINE ──────────────────────────────────────────────────────────
if files:
    if st.button(f"RUN DIAGNOSTIC / {len(files)} SCANS"):
        results = []
        total_start = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, f in enumerate(files):
            start_time = time.time()
            status_text.text(f"Processing: {f.name}")
            
            img = Image.open(f).convert("RGB")

            # Validation Pipeline
            if not is_valid_mri(img) or not is_genuine_mri(img):
                results.append({
                    "image": img,
                    "filename": f.name,
                    "label": "INVALID",
                    "confidence": 0,
                    "probs": [0] * 4,
                    "time": 0
                })
            else:
                # Inference
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: preprocess(img)})[0]
                probs = output[0]
                
                # Confidence Logic
                idx = int(np.argmax(probs))
                conf = float(np.max(probs)) * 100
                final_label = CLASS_NAMES[idx] if conf > 92 else "INVALID"
                
                results.append({
                    "image": img,
                    "filename": f.name,
                    "label": final_label,
                    "confidence": conf,
                    "probs": probs.tolist(),
                    "time": time.time() - start_time
                })

            progress_bar.progress((i + 1) / len(files))

        st.session_state["results"] = results
        st.session_state["latency"] = time.time() - total_start
        status_text.success("Analysis Complete")

# ── RESULTS DISPLAY ──────────────────────────────────────────────────────────
if "results" in st.session_state:
    res = st.session_state["results"]

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item">Processed<div class="stat-value">{len(res)}</div></div>
        <div class="stat-item">Tumor Detected<div class="stat-value">{sum(1 for r in res if r["label"] not in ["No Tumor", "INVALID"])}</div></div>
        <div class="stat-item">Anomalies/Invalid<div class="stat-value">{sum(1 for r in res if r["label"] == "INVALID")}</div></div>
        <div class="stat-item">Total Latency<div class="stat-value">{st.session_state.get('latency', 0):.2f}s</div></div>
    </div>
    """, unsafe_allow_html=True)

    for i, r in enumerate(res):
        is_inv = r["label"] == "INVALID"
        
        st.markdown(f"""
        <div class="scan-result {'invalid' if is_inv else ''}">
            <b style="color: {'#ecc94b' if is_inv else '#63B3ED'}">{r["label"].upper()}</b> | {r["filename"]}<br>
            <small>Confidence: {r["confidence"]:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

        t_img, t_prob, t_findings = st.tabs(["VIEWPORT", "PROBABILITY MAP", "CLINICAL DATA"])

        with t_img:
            col1, col2 = st.columns(2)
            with col1:
                st.image(r["image"], caption="Original Input", use_container_width=True)
            with col2:
                # Interactive Enhancement
                br = st.slider("Brightness Adjust", 0.5, 2.0, 1.0, key=f"br_{i}_{r['filename']}")
                enhanced = ImageEnhance.Brightness(r["image"]).enhance(br)
                if st.toggle("Enable CLAHE Enhancement", key=f"clahe_{i}"):
                    enhanced = apply_clahe(enhanced)
                st.image(enhanced, caption="Enhanced Preview", use_container_width=True)

        with t_prob:
            if not is_inv:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(CLASS_NAMES.values(), r["probs"], color=['#63B3ED', '#4FD1C5', '#F6AD55', '#BEE3F8'])
                ax.set_facecolor('#060810')
                fig.patch.set_facecolor('#060810')
                ax.tick_params(colors='white')
                ax.set_ylabel("Probability Score", color='white')
                st.pyplot(fig)
            else:
                st.warning("Probability data unavailable for non-MRI inputs.")

        with t_findings:
            ranking = sorted(zip(CLASS_NAMES.values(), r["probs"]), key=lambda x: x[1], reverse=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                for name, val in ranking[:3]:
                    st.metric(name, f"{val*100:.1f}%")
            with c2:
                st.markdown(f'<div class="finding-box"><b>Clinical Observation:</b><br>{FINDINGS.get(r["label"])}</div>', unsafe_allow_html=True)

    st.divider()
    st.download_button(
        label="📥 GENERATE CLINICAL PDF REPORT",
        data=create_pdf(res),
        file_name=f"NeuroScan_Report_{datetime.now().strftime('%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.markdown("<br><center style='opacity:0.2; font-size:10px;'>NEUROSCAN AI · RESEARCH ARCHITECTURE v2.9 · 2026</center>", unsafe_allow_html=True)
