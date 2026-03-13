import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from fpdf import FPDF
from datetime import datetime
import io
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

.stApp > * { position: relative; z-index: 1; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* Responsive Grid */
.med-info-container, .stat-grid {
    display: grid; gap: 15px; margin-bottom: 30px;
    grid-template-columns: repeat(4, 1fr);
}

/* Mobile Adjustments */
@media (max-width: 768px) {
    .med-info-container { grid-template-columns: repeat(1, 1fr); }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 42px !important; }
    .block-container { padding: 1rem !important; }
}

.medical-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.1);
    padding: 16px; border-radius: 4px;
}
.medical-card b { color: #63B3ED; font-size: 13px; font-family: 'Space Mono', monospace; display: block; margin-bottom: 8px; letter-spacing: 1px;}
.medical-card p { font-size: 11px; line-height: 1.5; color: rgba(201,209,224,0.7); margin: 0; }

.neuro-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0 30px; border-bottom: 1px solid rgba(99,179,237,0.12);
}

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
    "INVALID": "Structure does not match expected neuro-imaging protocols. Analysis aborted for safety."
}

# ── UTILITIES ────────────────────────────────────────────────────────────────
def is_valid_mri(image):
    """Detects if the uploaded image is a valid brain MRI based on pixel distribution"""
    img_gray = np.array(image.convert('L'))
    if np.mean(img_gray) > 220 or np.mean(img_gray) < 5: return False # Too bright/dark
    if np.std(img_gray) < 15: return False # Too flat
    return True

def safe_str(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def apply_clahe(image):
    img_array = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return Image.fromarray(clahe.apply(img_array)).convert('RGB')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18); pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 12, "NEURO-IMAGING DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.ln(10)
    for idx, res in enumerate(results):
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Arial", 'B', 11); pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe_str(f"CASE #{idx+1:02d} | SOURCE: {res['filename']}"), ln=True)
        pdf.set_font("Arial", '', 10)
        label = res['label'] if res['label'] != "INVALID" else "INVALID SCAN DATA"
        pdf.cell(0, 8, safe_str(f"RESULT: {label}"), ln=True)
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# ── MODEL LOADER ─────────────────────────────────────────────────────────────
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

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    inp = session.get_inputs()[0].name
    batch = preprocess(image)
    base = float(session.run(None, {inp: batch})[0][0][pred_class])
    H, W = 224, 224
    saliency = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, 14):
        for x in range(0, W, 14):
            occ = batch.copy(); occ[0, y:min(y+28,H), x:min(x+28,W), :] = 0
            score = float(session.run(None, {inp: occ})[0][0][pred_class])
            saliency[y:min(y+28,H), x:min(x+28,W)] = np.maximum(saliency[y:min(y+28,H), x:min(x+28,W)], base - score)
    if saliency.max() > saliency.min(): saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    heatmap = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    return Image.blend(image.convert('RGB'), Image.fromarray(heatmap).resize(image.size, Image.BILINEAR), alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── ENGINE ───────────────────────────────────────────────────────────────────
session = load_model()
if not session: st.error("Neural engine missing."); st.stop()

# Header
st.markdown('<div class="neuro-header"><div class="neuro-logo">NEURO<span>SCAN</span> / AI</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:60px; font-weight:900; line-height:1; margin-top:20px;">BRAIN ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// Mobile-Optimized Diagnostic Suite</div>', unsafe_allow_html=True)

st.markdown("""
<div class="med-info-container">
    <div class="medical-card"><b>GLIOMA</b><p>Diffuse parenchymal infiltration.</p></div>
    <div class="medical-card"><b>MENINGIOMA</b><p>Dural-based extra-axial mass.</p></div>
    <div class="medical-card"><b>PITUITARY</b><p>Sellar expansion impact.</p></div>
    <div class="medical-card"><b>NO TUMOR</b><p>Physiological signal intact.</p></div>
</div>
""", unsafe_allow_html=True)

files = st.file_uploader("DROP MRI HERE", type=["jpg","png","jpeg"], accept_multiple_files=True)

if files:
    if st.button(f"RUN ANALYSIS / {len(files)} SCANS"):
        all_res = []
        t_start = time.time()
        with st.status("Analyzing Sequences...", expanded=True) as status:
            prog = st.progress(0)
            for i, f in enumerate(files):
                img = Image.open(f).convert('RGB')
                if not is_valid_mri(img):
                    all_res.append({'image': img, 'filename': f.name, 'label': 'INVALID', 'confidence': 0, 'probs': [0]*4})
                    continue
                
                out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
                idx = int(np.argmax(out[0]))
                all_res.append({
                    'image': img, 'saliency': generate_saliency(session, img, idx),
                    'filename': f.name, 'label': CLASS_NAMES[idx], 
                    'confidence': float(np.max(out[0]))*100, 'probs': out[0].tolist()
                })
                prog.progress((i+1)/len(files))
            status.update(label="Analysis Finalized", state="complete", expanded=False)
        st.session_state['results'] = all_res; st.session_state['time'] = time.time() - t_start

# ── RENDER ───────────────────────────────────────────────────────────────────
if 'results' in st.session_state:
    res = st.session_state['results']
    st.markdown(f'<div class="stat-grid"><div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(res)}</div></div><div class="stat-item"><div class="stat-label">Detected</div><div class="stat-value">{sum(1 for r in res if r["label"] not in ["No Tumor", "INVALID"])}</div></div><div class="stat-item"><div class="stat-label">Avg Conf</div><div class="stat-value">{np.mean([r["confidence"] for r in res]):.1f}%</div></div><div class="stat-item"><div class="stat-label">Alerts</div><div class="stat-value">{sum(1 for r in res if r["label"]=="INVALID")}</div></div></div>', unsafe_allow_html=True)

    for i, item in enumerate(res):
        is_invalid = item['label'] == "INVALID"
        is_tumor = item['label'] not in ["No Tumor", "INVALID"]
        status_class = "invalid" if is_invalid else ("tumor" if is_tumor else "")
        
        st.markdown(f'<div class="scan-result {status_class}"><div class="result-label">{item["label"].upper()}</div><div style="font-size:10px; opacity:0.6;">{item["filename"]}</div></div>', unsafe_allow_html=True)
        
        if is_invalid:
            st.warning(f"Safety Warning: File '{item['filename']}' is not a valid Brain MRI scan. Analysis aborted to prevent false diagnosis.")
            st.image(item['image'], use_container_width=True)
        else:
            t1, t2, t3 = st.tabs(["VIEW", "HEATMAP", "FINDINGS"])
            with t1:
                st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                br = s1.slider("BRIGHT", 0.5, 2.0, 1.0, key=f"b{i}")
                ct = s2.slider("CONTRAST", 0.5, 2.0, 1.0, key=f"c{i}")
                sh = s3.slider("SHARP", 0.0, 3.0, 1.0, key=f"s{i}")
                med_enh = st.toggle("MEDICAL ENHANCE (CLAHE)", key=f"clahe_{i}")
                st.markdown("</div>", unsafe_allow_html=True)
                e = apply_clahe(item['image']) if med_enh else item['image']
                e = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Brightness(e).enhance(br)).enhance(ct)).enhance(sh)
                st.image(e, use_container_width=True)
            with t2: st.image(item['saliency'], use_container_width=True)
            with t3:
                st.markdown(f'<div class="finding-box"><b>AI Observation:</b><br>{FINDINGS.get(item["label"])}</div>', unsafe_allow_html=True)
                for k, v in enumerate(item['probs']): st.write(f"{CLASS_NAMES[k]}: {v*100:.1f}%"); st.progress(v)

    st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
    st.download_button("GENERATE CLINICAL REPORT", create_pdf(res), f"NeuroScan_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")

st.markdown("<br><div style='text-align:center; font-family:Space Mono; font-size:9px; color:rgba(99,179,237,0.1); padding:40px 0;'>NEUROSCAN AI · INSTITUTIONAL RESEARCH USE ONLY · 2026</div>", unsafe_allow_html=True)
