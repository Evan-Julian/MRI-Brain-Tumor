import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime

st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
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

/* ── RESPONSIVE CONTAINER ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem !important; }
}
@media (max-width: 480px) {
    .block-container { padding: 0.75rem 0.5rem !important; }
}

/* ── SIDEBAR INFO ── */
[data-testid="stSidebar"] {
    background-color: #030508 !important;
    border-right: 1px solid rgba(99,179,237,0.1) !important;
}
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 14px; color: #63B3ED;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 20px; border-bottom: 1px solid rgba(99,179,237,0.2);
    padding-bottom: 10px;
}
.medical-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.1);
    padding: 12px; border-radius: 4px;
    margin-bottom: 15px;
}
.medical-card b { color: #63B3ED; font-size: 12px; }
.medical-card p { font-size: 11px; line-height: 1.4; color: rgba(201,209,224,0.7); margin-top: 5px; }

/* ── HEADER ── */
.neuro-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0 32px;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 36px;
    flex-wrap: wrap;
    gap: 12px;
}
.neuro-logo {
    font-family: 'Space Mono', monospace;
    font-size: 13px; letter-spacing: 4px;
    color: rgba(99,179,237,0.6);
    text-transform: uppercase;
}
.neuro-logo span { color: #63B3ED; font-weight: 700; }
.neuro-badge {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 3px;
    color: rgba(99,179,237,0.4);
    border: 1px solid rgba(99,179,237,0.15);
    padding: 6px 14px; border-radius: 2px;
    text-transform: uppercase;
}
@media (max-width: 480px) {
    .neuro-logo { font-size: 11px; letter-spacing: 2px; }
    .neuro-badge { font-size: 8px; letter-spacing: 1px; padding: 4px 8px; }
    .neuro-header { padding: 14px 0 20px; margin-bottom: 20px; }
}

/* ── HERO TITLE ── */
.hero-title {
    font-size: clamp(28px, 5vw, 70px);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -3px;
    color: #EDF2F7;
    margin-bottom: 16px;
}
.hero-title .accent {
    color: transparent;
    -webkit-text-stroke: 1px rgba(99,179,237,0.5);
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px; letter-spacing: 2px;
    color: rgba(99,179,237,0.5);
    text-transform: uppercase;
    margin-bottom: 32px;
}
@media (max-width: 768px) {
    .hero-title { letter-spacing: -1px; margin-bottom: 10px; }
    .hero-sub { font-size: 10px; margin-bottom: 20px; }
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] > div {
    background: rgba(99,179,237,0.02) !important;
    border: 1px dashed rgba(99,179,237,0.2) !important;
    border-radius: 4px !important;
    padding: 40px !important;
}
@media (max-width: 768px) {
    [data-testid="stFileUploader"] > div { padding: 20px !important; }
}

/* ── STAT GRID ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 32px 0;
}
@media (max-width: 900px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
}
@media (max-width: 480px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 16px 0; }
}
.stat-item {
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px; padding: 20px 24px;
}
@media (max-width: 480px) {
    .stat-item { padding: 14px 16px; }
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px; letter-spacing: 3px;
    color: rgba(99,179,237,0.35);
    text-transform: uppercase; margin-bottom: 8px;
}
.stat-value {
    font-size: 32px; font-weight: 900;
    letter-spacing: -2px; color: #EDF2F7;
}
@media (max-width: 480px) {
    .stat-value { font-size: 24px; }
}
.stat-value span { color: #63B3ED; font-size: 16px; font-weight: 400; }

/* ── RESULT CARDS ── */
.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 12px;
    overflow: hidden;
}
.scan-result::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #63B3ED, transparent);
}
.scan-result.tumor::before {
    background: linear-gradient(180deg, #FC8181, transparent);
}
.result-label {
    font-size: clamp(16px, 3vw, 22px);
    font-weight: 900;
    letter-spacing: -1px;
    color: #EDF2F7;
    margin-bottom: 4px;
}
.result-label.tumor { color: #FC8181; }
.result-meta {
    font-family: 'Space Mono', monospace;
    font-size: 9px; letter-spacing: 2px;
    color: rgba(99,179,237,0.4);
    text-transform: uppercase;
    word-break: break-all;
}
.result-confidence {
    font-family: 'Space Mono', monospace;
    font-size: clamp(18px, 3vw, 28px);
    font-weight: 700;
    color: #63B3ED;
    text-align: right;
    white-space: nowrap;
}
.result-confidence.tumor { color: #FC8181; }

.neo-divider { border: none; border-top: 1px solid rgba(99,179,237,0.08); margin: 24px 0; }

.idle-state {
    text-align: center; padding: 60px 24px;
    border: 1px solid rgba(99,179,237,0.06);
    border-radius: 4px;
    background: rgba(99,179,237,0.01);
}
@media (max-width: 480px) {
    .idle-state { padding: 40px 16px; }
}
.idle-icon { font-size: 48px; margin-bottom: 20px; opacity: 0.3; }
.idle-text {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 3px;
    color: rgba(99,179,237,0.25);
    text-transform: uppercase;
}

/* ── CONTROL PANEL ── */
.control-panel {
    background: rgba(99,179,237,0.03);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 15px;
    margin-top: 10px;
}

/* ── KNOWLEDGE SECTION ── */
.knowledge-section {
    background: rgba(6, 8, 16, 0.9);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 4px;
    padding: 24px 28px;
    margin-bottom: 36px;
}
@media (max-width: 768px) {
    .knowledge-section { padding: 16px; margin-bottom: 24px; }
}
.knowledge-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 4px;
    color: rgba(99,179,237,0.5);
    text-transform: uppercase;
    margin-bottom: 20px;
}
.knowledge-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
@media (max-width: 900px) {
    .knowledge-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }
}
@media (max-width: 480px) {
    .knowledge-grid { grid-template-columns: 1fr; gap: 10px; }
}
.knowledge-card {
    background: rgba(99,179,237,0.03);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px;
    padding: 16px;
    transition: border-color 0.2s;
}
.knowledge-card:hover { border-color: rgba(99,179,237,0.25); }
.knowledge-card .k-icon { font-size: 20px; margin-bottom: 8px; }
.knowledge-card .k-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2px; color: #63B3ED;
    text-transform: uppercase; margin-bottom: 6px;
}
.knowledge-card .k-type {
    font-size: 9px; font-family: 'Space Mono', monospace;
    letter-spacing: 1px; color: rgba(99,179,237,0.35);
    text-transform: uppercase; margin-bottom: 8px;
    padding: 2px 6px; border: 1px solid rgba(99,179,237,0.15);
    border-radius: 2px; display: inline-block;
}
.knowledge-card .k-desc {
    font-size: 11px; color: rgba(201,209,224,0.65); line-height: 1.6;
}
.knowledge-card.no-tumor { border-color: rgba(72,187,120,0.12); }
.knowledge-card.no-tumor .k-title { color: #68D391; }
.knowledge-card.no-tumor .k-type {
    color: rgba(72,187,120,0.5); border-color: rgba(72,187,120,0.2);
}

/* ── INVALID WARNING CARD ── */
.invalid-card {
    background: rgba(254,178,178,0.05);
    border: 1px solid rgba(252,129,129,0.3);
    border-radius: 4px; padding: 16px 20px;
    margin-bottom: 12px; position: relative; overflow: hidden;
}
.invalid-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #FC8181, transparent);
}
.invalid-card .inv-title {
    font-family: 'Space Mono', monospace;
    font-size: 12px; font-weight: 700; letter-spacing: 2px;
    color: #FC8181; margin-bottom: 4px;
    word-break: break-word;
}
.invalid-card .inv-meta {
    font-family: 'Space Mono', monospace;
    font-size: 9px; letter-spacing: 2px;
    color: rgba(252,129,129,0.5); text-transform: uppercase;
}
.invalid-card .inv-reason {
    font-size: 11px; color: rgba(252,129,129,0.7); margin-top: 6px;
}

/* ── RESPONSIVE: Streamlit columns stack on mobile ── */
@media (max-width: 768px) {
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    [data-testid="stTabs"] button {
        font-size: 10px !important;
        padding: 6px 8px !important;
    }
}

/* ── BUTTONS ── */
.stButton > button {
    width: 100% !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 2px !important;
    font-size: 11px !important;
}
@media (max-width: 480px) {
    .stButton > button { font-size: 9px !important; letter-spacing: 1px !important; }
}
.stDownloadButton > button {
    width: 100% !important;
    font-family: 'Space Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR KNOWLEDGE CENTER ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Medical Knowledge Center</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="medical-card">
        <b>🧠 GLIOMA</b>
        <p>Tumor arising from glial cells, the brain's support network for neurons. Tends to be invasive and may spread into surrounding brain tissue.</p>
    </div>
    <div class="medical-card">
        <b>🛡️ MENINGIOMA</b>
        <p>Tumor originating from the meninges, the protective membranes covering the brain. Usually benign but can compress vital nerves.</p>
    </div>
    <div class="medical-card">
        <b>💧 PITUITARY</b>
        <p>Tumor located at the pituitary gland at the base of the brain. May cause hormonal imbalance and disrupt vision.</p>
    </div>
    <div class="medical-card">
        <b>✅ NO TUMOR</b>
        <p>No abnormal mass, pathological lesion, or irregular tissue growth detected within the analyzed scan region.</p>
    </div>
    <p style="font-size: 9px; color: rgba(201,209,224,0.4); text-align: center; margin-top:20px;">
        Press <b>></b> in the top-left corner if this panel is collapsed.
    </p>
    """, unsafe_allow_html=True)

# ── PDF GENERATOR ────────────────────────────────────────────────────────────
def safe(text):
    return text.encode('latin-1', errors='replace').decode('latin-1')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 12, "NEUROSCAN AI - CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"), ln=True, align='C')
    pdf.ln(8)

    for idx, res in enumerate(results):
        if idx > 0 and idx % 2 == 0:
            pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe(f"#{idx+1:02d}  {res['filename']}"), ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mri:
            res['image'].save(tmp_mri.name)
            tmp_mri_path = tmp_mri.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            res['saliency'].save(tmp_heat.name)
            tmp_heat_path = tmp_heat.name

        y = pdf.get_y()

        pdf.image(tmp_mri_path, x=10, y=y, w=55)
        pdf.image(tmp_heat_path, x=68, y=y, w=55)

        pdf.set_xy(130, y + 2)
        pdf.set_font("Arial", 'B', 13)
        is_tumor = res['label'].lower() != 'no tumor'
        pdf.set_text_color(220, 50, 50) if is_tumor else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe(res['label'].upper()), ln=True)

        pdf.set_x(130)
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 6, safe(f"Confidence: {res['confidence']:.2f}%"), ln=True)

        pdf.set_x(130)
        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, safe(f"Image Size: {res['size']} px"), ln=True)

        pdf.set_x(130)
        pdf.cell(0, 5, safe(f"Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)

        pdf.set_font("Arial", 'I', 7)
        pdf.set_text_color(150, 150, 150)
        pdf.set_xy(10, y + 57)
        pdf.cell(55, 4, "[ MRI SCAN ]", align='C')
        pdf.set_xy(68, y + 57)
        pdf.cell(55, 4, "[ SALIENCY HEATMAP ]", align='C')

        pdf.set_y(y + 65)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        if os.path.exists(tmp_mri_path):
            os.remove(tmp_mri_path)
        if os.path.exists(tmp_heat_path):
            os.remove(tmp_heat_path)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, safe("DISCLAIMER: Research use only. Final diagnosis must be confirmed by a licensed medical professional."))
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)

# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        if not os.path.exists(path): return None
        return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except: return None

# ── IMAGE VALIDATION / SECURITY FILTER ───────────────────────────────────────
def is_valid_mri(image: Image.Image) -> tuple[bool, str]:
    """
    Validates whether an uploaded image is likely an MRI scan.
    Uses 3 heuristics:
      1. Grayscale histogram — MRI scans are near-grayscale (low color saturation)
      2. Laplacian texture — MRI scans have distinctive texture (not too smooth, not too noisy)
      3. Black background ratio — MRI scans typically have significant dark/black regions
    Returns: (is_valid: bool, reason: str)
    """
    img_rgb = image.convert("RGB")
    img_arr = np.array(img_rgb, dtype=np.float32)

    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))
    gb_diff = np.mean(np.abs(g - b))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3.0
    if avg_color_diff > 30.0:
        return False, f"Image is too colorful (color diff={avg_color_diff:.1f}). MRI scans must be near-grayscale."

    gray = np.array(image.convert("L"), dtype=np.float32)
    lap_x = np.abs(gray[:, 1:] - gray[:, :-1])
    lap_y = np.abs(gray[1:, :] - gray[:-1, :])
    texture_score = (np.mean(lap_x) + np.mean(lap_y)) / 2.0
    if texture_score < 1.5:
        return False, f"Image is too smooth/flat (texture={texture_score:.2f}). MRI scans have characteristic medical texture."
    if texture_score > 80.0:
        return False, f"Image is too noisy (texture={texture_score:.2f}). Not consistent with MRI scan characteristics."

    dark_pixel_ratio = np.mean(gray < 20)
    if dark_pixel_ratio < 0.08:
        return False, f"Insufficient dark background ({dark_pixel_ratio*100:.1f}%). MRI scans typically have a dominant black background."

    return True, "OK"

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1]
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])
    H, W = 224, 224
    stride, patch = 14, 28
    saliency = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = base_batch.copy()
            y1, y2 = y, min(y + patch, H)
            x1, x2 = x, min(x + patch, W)
            occ[0, y1:y2, x1:x2, :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            drop = base_score - score
            saliency[y1:y2, x1:x2] = np.maximum(saliency[y1:y2, x1:x2], drop)

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)

    r = np.clip(saliency * 3 - 1, 0, 1)
    g = np.clip(saliency * 3 - 0.5, 0, 1) * np.clip(2 - saliency * 3, 0, 1)
    b = np.clip(1 - saliency * 2, 0, 1)
    heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    orig = image.convert('RGB')
    return Image.blend(orig, heatmap_img, alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div class="neuro-badge">ResNet50 · ONNX Runtime · v2.4</div>
</div>
""", unsafe_allow_html=True)

# ── KNOWLEDGE SECTION ─────────────────────────────────────────────────────────
st.markdown("""
<div class="knowledge-section">
    <div class="knowledge-title">// Brain Tumor Knowledge Base</div>
    <div class="knowledge-grid">
        <div class="knowledge-card">
            <div class="k-icon">🧠</div>
            <div class="k-title">Glioma</div>
            <div class="k-type">Invasive · Grade I–IV</div>
            <div class="k-desc">
                Tumor originating from glial cells, the brain's support and protection system for neurons.
                Gliomas are invasive and can spread into surrounding brain tissue.
                High-grade forms (GBM) are aggressive and require immediate medical intervention.
            </div>
        </div>
        <div class="knowledge-card">
            <div class="k-icon">🛡️</div>
            <div class="k-title">Meningioma</div>
            <div class="k-type">Usually Benign · Slow-Growing</div>
            <div class="k-desc">
                Arises from the meninges, the protective membranes surrounding the brain and spinal cord.
                Generally benign and slow-growing, but certain locations may compress vital nerves
                and cause serious neurological complications.
            </div>
        </div>
        <div class="knowledge-card">
            <div class="k-icon">💧</div>
            <div class="k-title">Pituitary</div>
            <div class="k-type">Hormonal · Pituitary Gland</div>
            <div class="k-desc">
                Tumor located in the pituitary gland at the base of the brain, which regulates the body's hormonal system.
                Can trigger overproduction or deficiency of hormones. Common symptoms include
                vision disturbances and hormonal imbalance.
            </div>
        </div>
        <div class="knowledge-card no-tumor">
            <div class="k-icon">✅</div>
            <div class="k-title">No Tumor</div>
            <div class="k-type">Normal · Clear Scan</div>
            <div class="k-desc">
                No abnormal mass, pathological lesion, or irregular tissue growth detected
                within the analyzed scan area. A clear result indicates brain structure
                is within normal limits.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_hero, col_upload = st.columns([1, 1], gap="large")

with col_hero:
    st.markdown("""
    <div class="hero-title">
        BRAIN<br>
        <span class="accent">TUMOR</span><br>
        SCAN
    </div>
    <div class="hero-sub">// Precision Neuro-Imaging Analytics</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(99,179,237,0.05); padding: 20px; border-radius: 4px; border: 1px solid rgba(99,179,237,0.1);">
        <p style="font-family:'Space Mono', monospace; font-size:11px; color:#63B3ED; margin-bottom:10px;">[ SYSTEM_CAPABILITIES ]</p>
        <ul style="font-size:12px; color:rgba(201,209,224,0.7); line-height:1.6; list-style-type: '→ '; padding-left:15px;">
            <li>Saliency Mapping (Occlusion Sensitivity)</li>
            <li>Real-time Image Augmentation Engine</li>
            <li>Medical Knowledge Center Support</li>
            <li>Clinical PDF Export (MRI + Heatmap)</li>
            <li>MRI Image Validation &amp; Security Filter</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_upload:
    session = load_model()
    if session is None:
        st.error("⚠ resnet_model.onnx not found.")
        st.stop()

    uploaded_files = st.file_uploader(
        "DROP MRI SCANS HERE",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:10px;
    letter-spacing:2px; color:rgba(99,179,237,0.25); text-align:center;
    margin-top:8px; text-transform:uppercase;'>
    JPG / PNG format · Batch analysis active
    </div>""", unsafe_allow_html=True)

# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if uploaded_files:
    n = len(uploaded_files)
    if st.button(f"RUN ANALYSIS · {n} SEQUENCE{'S' if n > 1 else ''}"):
        all_results = []
        invalid_results = []
        t_start = time.time()
        progress_bar = st.progress(0, text="Initializing...")

        for fi, f in enumerate(uploaded_files):
            progress_bar.progress((fi) / n, text=f"Analyzing {f.name}...")
            img = Image.open(f).convert('RGB')

            valid, reason = is_valid_mri(img)
            if not valid:
                invalid_results.append({
                    'filename': f.name,
                    'reason': reason,
                    'size': f"{img.size[0]}x{img.size[1]}"
                })
                continue

            batch = preprocess(img)
            inp = session.get_inputs()[0].name
            out = session.run(None, {inp: batch})[0]
            pred_idx = int(np.argmax(out[0]))
            conf = float(np.max(out[0])) * 100
            saliency_img = generate_saliency(session, img, pred_idx)

            all_results.append({
                'image': img,
                'saliency': saliency_img,
                'filename': f.name,
                'label': CLASS_NAMES.get(pred_idx, f"Class {pred_idx}"),
                'confidence': conf,
                'probs': out[0].tolist(),
                'size': f"{img.size[0]}x{img.size[1]}"
            })

        progress_bar.empty()
        st.session_state['analysis_results'] = all_results
        st.session_state['invalid_results'] = invalid_results
        st.session_state['total_time'] = time.time() - t_start

if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    invalid_list = st.session_state.get('invalid_results', [])
    t_total = st.session_state['total_time']

    # ── INVALID FILES WARNING ──
    if invalid_list:
        st.markdown(f"""
        <div style="margin-bottom: 8px; font-family:'Space Mono',monospace; font-size:9px;
        letter-spacing:3px; color:rgba(252,129,129,0.5); text-transform:uppercase;">
        ⚠ {len(invalid_list)} FILE{'S' if len(invalid_list) > 1 else ''} REJECTED — NOT A VALID MRI SCAN
        </div>
        """, unsafe_allow_html=True)
        for inv in invalid_list:
            st.markdown(f"""
            <div class="invalid-card">
                <div class="inv-title">⛔ INVALID — {inv['filename']}</div>
                <div class="inv-meta">{inv['size']} px · Rejected by MRI Security Filter</div>
                <div class="inv-reason">{inv['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    if results:
        avg_conf = np.mean([r['confidence'] for r in results])
        tumor_found = sum(1 for r in results if r['label'].lower() != 'no tumor')

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(results)}<span> seq</span></div></div>
            <div class="stat-item"><div class="stat-label">Abnormalities</div><div class="stat-value">{tumor_found}<span> detect</span></div></div>
            <div class="stat-item"><div class="stat-label">Avg Conf</div><div class="stat-value">{avg_conf:.1f}<span>%</span></div></div>
            <div class="stat-item"><div class="stat-label">Time</div><div class="stat-value">{t_total:.2f}<span>s</span></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)

        # ── RESULTS GRID ──
        ncols = 3
        for row_start in range(0, len(results), ncols):
            row_items = results[row_start:row_start + ncols]
            cols = st.columns(ncols, gap="medium")
            for col_idx, res in enumerate(row_items):
                is_tumor = res["label"].lower() != "no tumor"
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="scan-result {'tumor' if is_tumor else ''}">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
                            <div style="min-width:0;">
                                <div class="result-label {'tumor' if is_tumor else ''}">{res["label"].upper()}</div>
                                <div class="result-meta">{res["filename"][:18]}… | {res['size']} PX</div>
                            </div>
                            <div class="result-confidence {'tumor' if is_tumor else ''}">{res["confidence"]:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["📷 VIEWPORT", "🔥 HEATMAP", "📊 PROBS"])

                    with tab1:
                        st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
                        c_col1, c_col2, c_col3 = st.columns(3)
                        with c_col1: br = st.slider("BRIGHT", 0.5, 2.0, 1.0, 0.1, key=f"br_{res['filename']}")
                        with c_col2: ct = st.slider("CONTRAST", 0.5, 2.0, 1.0, 0.1, key=f"ct_{res['filename']}")
                        with c_col3: sh = st.slider("SHARP", 0.0, 3.0, 1.0, 0.1, key=f"sh_{res['filename']}")
                        st.markdown("</div>", unsafe_allow_html=True)

                        enhanced = ImageEnhance.Brightness(res["image"]).enhance(br)
                        enhanced = ImageEnhance.Contrast(enhanced).enhance(ct)
                        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sh)
                        st.image(enhanced, use_container_width=True)

                    with tab2:
                        st.image(res["saliency"], use_container_width=True)

                    with tab3:
                        for ci, prob in enumerate(res["probs"]):
                            name = CLASS_NAMES.get(ci, f"C{ci}")
                            st.markdown(f"<p style='font-size:10px; margin-bottom:2px; color:rgba(255,255,255,0.6)'>{name} ({prob*100:.1f}%)</p>", unsafe_allow_html=True)
                            st.progress(min(prob, 1.0))

        st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)

        # ── PDF DOWNLOAD ──
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            pdf_bytes = create_pdf(results)
            st.download_button(
                label="↓ DOWNLOAD CLINICAL REPORT",
                data=pdf_bytes,
                file_name=f"NeuroScan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        with col_info:
            st.info("ℹ️ Tumor types: Glioma (Invasive), Meningioma (Protective Membrane), Pituitary (Hormonal), No Tumor (Normal). PDF report includes MRI + Saliency Heatmap.")

    elif not invalid_list:
        st.markdown("""
        <div class="idle-state">
            <div class="idle-icon">◎</div>
            <div class="idle-text">No valid results // Upload MRI sequences to analyze</div>
        </div>
        """, unsafe_allow_html=True)

elif not uploaded_files:
    st.markdown("""
    <div class="idle-state">
        <div class="idle-icon">◎</div>
        <div class="idle-text">System offline // Awaiting MRI sequence upload</div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<br><div style='text-align:center; font-family:"Space Mono",monospace; font-size:9px; letter-spacing:4px; text-transform:uppercase; color:rgba(99,179,237,0.12); padding: 24px 0;'>
NEUROSCAN AI &nbsp;·&nbsp; INSTITUTIONAL RESEARCH USE &nbsp;·&nbsp; v2.4
</div>
""", unsafe_allow_html=True)
