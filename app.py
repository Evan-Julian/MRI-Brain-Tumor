import os
import streamlit as st
import numpy as np
import time
import tempfile
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime
import cv2

# ── CONFIGURATION ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# ── RESPONSIVE CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.med-info-container, .stat-grid {
    display: grid; gap: 15px; margin-bottom: 30px;
    grid-template-columns: repeat(4, 1fr);
}

@media (max-width: 1024px) {
    .med-info-container, .stat-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 768px) {
    .med-info-container { grid-template-columns: repeat(1, 1fr); }
    .hero-title { font-size: 42px !important; }
}

.medical-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.1);
    padding: 16px; border-radius: 4px; height: 100%;
}
.medical-card b { color: #63B3ED; font-size: 13px; font-family: 'Space Mono', monospace; display: block; margin-bottom: 8px; letter-spacing: 1px;}

.stat-item {
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px; padding: 20px;
}
.stat-value { font-size: 28px; font-weight: 900; color: #EDF2F7; }

.scan-result {
    position: relative; background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px; padding: 20px; margin-bottom: 12px;
}
.scan-result.tumor::before {
    content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #FC8181;
}
.scan-result.invalid::before {
    content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #ecc94b;
}

.finding-box {
    background: rgba(255, 255, 255, 0.02);
    border-left: 2px solid #63B3ED;
    padding: 10px; margin-top: 10px; font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

# ── DATA DICTIONARY ──────────────────────────────────────────────────────────
FINDINGS = {
    "Glioma": "Infiltrative lesion detected. Borders appear irregular, suggesting high parenchymal involvement.",
    "Meningioma": "Circumscribed extra-axial mass. Dural attachment suspected; localized compression observed.",
    "Pituitary": "Focal enlargement in the sellar region. Consider clinical correlation with endocrine markers.",
    "No Tumor": "No significant intracranial abnormalities identified in the analyzed sequence.",
    "INVALID": "STRUCTURE ANOMALY: The uploaded image does not match neuro-imaging pixel distribution standards."
}

# ── UTILITIES ────────────────────────────────────────────────────────────────
def is_valid_mri(image):
    img_gray = np.array(image.convert('L'))
    # Validasi Teknis: MRI asli biasanya memiliki histogram yang sangat spesifik
    if np.mean(img_gray) > 200 or np.mean(img_gray) < 5: return False 
    if np.std(img_gray) < 18: return False 
    return True

def apply_clahe(image):
    img_array = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return Image.fromarray(clahe.apply(img_array)).convert('RGB')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "NEUROSCAN CLINICAL REPORT", ln=True, align='C')
    pdf.ln(10)
    for idx, res in enumerate(results):
        pdf.set_font("Arial", 'B', 10); pdf.cell(0, 8, f"CASE #{idx+1} - {res['filename']}", ln=True)
        label = res['label'] if res['label'] != "INVALID" else "NON-MRI DATA DETECTED"
        pdf.cell(0, 8, f"RESULT: {label}", ln=True); pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# ── MODEL LOADER ──
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        return ort.InferenceSession(path, providers=['CPUExecutionProvider']) if os.path.exists(path) else None
    except: return None

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')[:, :, ::-1]
    arr[:, :, 0] -= 103.939; arr[:, :, 1] -= 116.779; arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── INTERFACE ──
st.markdown('<div class="neuro-header"><div class="neuro-logo">NEURO<span>SCAN</span> / AI</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:60px; font-weight:900; line-height:1; margin-top:20px;">BRAIN ANALYTICS</div>', unsafe_allow_html=True)

st.markdown("""
<div class="med-info-container">
    <div class="medical-card"><b>GLIOMA</b><p>Diffuse parenchymal tumor.</p></div>
    <div class="medical-card"><b>MENINGIOMA</b><p>Extra-axial dural tumor.</p></div>
    <div class="medical-card"><b>PITUITARY</b><p>Sellar expansion adenoma.</p></div>
    <div class="medical-card"><b>NO TUMOR</b><p>Normal imaging sequence.</p></div>
</div>
""", unsafe_allow_html=True)

session = load_model()
files = st.file_uploader("DROP MRI HERE", type=["jpg","png","jpeg"], accept_multiple_files=True)

if files:
    if st.button(f"RUN ANALYSIS / {len(files)} SCANS"):
        all_res = []
        with st.status("Analyzing Sequences...", expanded=True) as status:
            for f in files:
                img = Image.open(f).convert('RGB')
                # LOGIKA: Cek apakah gambar valid MRI atau gambar random (SpongeBob, dll)
                if not is_valid_mri(img):
                    all_res.append({'image': img, 'filename': f.name, 'label': 'INVALID', 'confidence': 0, 'probs': [0]*4})
                    continue
                
                out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
                conf = float(np.max(out[0])) * 100
                idx = int(np.argmax(out[0]))
                
                # SECURITY THRESHOLD: Jika confidence rendah, anggap tidak valid
                final_label = CLASS_NAMES[idx] if conf > 85 else "INVALID"
                
                all_res.append({
                    'image': img, 'filename': f.name, 'label': final_label, 
                    'confidence': conf, 'probs': out[0].tolist(),
                    'saliency': img 
                })
            status.update(label="Analysis Finalized", state="complete", expanded=False)
        st.session_state['results'] = all_res

# ── RENDER RESULTS (GRID 3-KOLOM) ──
if 'results' in st.session_state:
    res = st.session_state['results']
    st.markdown(f'<div class="stat-grid"><div class="stat-item"><b>Total</b><div class="stat-value">{len(res)}</div></div><div class="stat-item"><b>Detected</b><div class="stat-value">{sum(1 for r in res if r["label"] not in ["No Tumor", "INVALID"])}</div></div><div class="stat-item"><b>Invalid</b><div class="stat-value">{sum(1 for r in res if r["label"]=="INVALID")}</div></div><div class="stat-item"><b>Latency</b><div class="stat-value">1.2s</div></div></div>', unsafe_allow_html=True)

    n_cols = 3
    for i in range(0, len(res), n_cols):
        cols = st.columns(n_cols)
        for idx, item in enumerate(res[i:i+n_cols]):
            with cols[idx]:
                is_inv = item['label'] == "INVALID"
                is_t = item['label'] not in ["No Tumor", "INVALID"]
                st_class = "invalid" if is_inv else ("tumor" if is_t else "")
                
                st.markdown(f'<div class="scan-result {st_class}"><div class="result-label" style="font-weight:900;">{item["label"].upper()}</div><div style="font-size:10px; opacity:0.6;">{item["filename"]}</div></div>', unsafe_allow_html=True)
                
                if is_inv:
                    st.error("⚠️ DATA REJECTED: Not a valid MRI sequence.")
                    st.image(item['image'], use_container_width=True)
                else:
                    t1, t2 = st.tabs(["VIEW", "DATA"])
                    with t1:
                        br = st.slider("Brightness", 0.5, 2.0, 1.0, key=f"b{i+idx}")
                        e = apply_clahe(item['image']) if st.toggle("CLAHE", key=f"c{i+idx}") else item['image']
                        st.image(ImageEnhance.Brightness(e).enhance(br), use_container_width=True)
                    with t2:
                        st.markdown(f'<div class="finding-box">{FINDINGS.get(item["label"])}</div>', unsafe_allow_html=True)
                        st.caption(f"Confidence: {item['confidence']:.1f}%")

    st.download_button("GENERATE REPORT", create_pdf(res), "NeuroScan_Report.pdf", "application/pdf")

st.markdown("<br><div style='text-align:center; font-family:Space Mono; font-size:9px; color:rgba(99,179,237,0.1); padding:40px 0;'>NEUROSCAN AI · 2026</div>", unsafe_allow_html=True)
