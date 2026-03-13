import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw
from fpdf import FPDF
from datetime import datetime

st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded", 
)

# ── CSS (FULL & RESPONSIVE) ──────────────────────────────────────────────────
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
    padding: 28px 0 40px;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 48px;
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

/* ── HERO TITLE ── */
.hero-title {
    font-size: clamp(32px, 5vw, 70px);
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
    margin-bottom: 48px;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] > div {
    background: rgba(99,179,237,0.02) !important;
    border: 1px dashed rgba(99,179,237,0.2) !important;
    border-radius: 4px !important;
    padding: 40px !important;
}

/* ── STAT GRID ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 32px 0;
}
@media (max-width: 768px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
}

.stat-item {
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px; padding: 20px 24px;
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
.stat-value span { color: #63B3ED; font-size: 16px; font-weight: 400; }

/* ── RESULT CARDS ── */
.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 20px;
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
    font-size: 22px; font-weight: 900;
    letter-spacing: -1px;
    color: #EDF2F7;
    margin-bottom: 4px;
}
.result-label.tumor { color: #FC8181; }
.result-meta {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    color: rgba(99,179,237,0.4);
    text-transform: uppercase;
}
.result-confidence {
    font-family: 'Space Mono', monospace;
    font-size: 28px; font-weight: 700;
    color: #63B3ED;
    text-align: right;
}
.result-confidence.tumor { color: #FC8181; }

.neo-divider { border: none; border-top: 1px solid rgba(99,179,237,0.08); margin: 32px 0; }

/* Control Panel for Image Augmentation */
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
    padding: 28px 32px;
    margin-top: 40px;
    margin-bottom: 40px;
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
@media (max-width: 1024px) {
    .knowledge-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .knowledge-grid { grid-template-columns: 1fr; }
}

.knowledge-card {
    background: rgba(99,179,237,0.03);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px;
    padding: 16px;
    transition: border-color 0.2s;
}
.knowledge-card:hover {
    border-color: rgba(99,179,237,0.25);
}
.knowledge-card .k-icon {
    font-size: 20px;
    margin-bottom: 8px;
}
.knowledge-card .k-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2px;
    color: #63B3ED;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.knowledge-card .k-type {
    font-size: 9px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    color: rgba(99,179,237,0.35);
    text-transform: uppercase;
    margin-bottom: 8px;
    padding: 2px 6px;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 2px;
    display: inline-block;
}
.knowledge-card .k-desc {
    font-size: 11px;
    color: rgba(201,209,224,0.65);
    line-height: 1.6;
}
.knowledge-card.no-tumor {
    border-color: rgba(72,187,120,0.12);
}
.knowledge-card.no-tumor .k-title {
    color: #68D391;
}
.knowledge-card.no-tumor .k-type {
    color: rgba(72,187,120,0.5);
    border-color: rgba(72,187,120,0.2);
}

/* ── INVALID WARNING CARD ── */
.invalid-card {
    background: rgba(254,178,178,0.05);
    border: 1px solid rgba(252,129,129,0.3);
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.invalid-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #FC8181, transparent);
}
.invalid-card .inv-title {
    font-family: 'Space Mono', monospace;
    font-size: 13px; font-weight: 700;
    letter-spacing: 2px;
    color: #FC8181;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Medical Knowledge Center</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="medical-card">
        <b>GLIOMA</b>
        <p>Tumors originating from glial cells. Invasive and spreads through surrounding brain matter.</p>
    </div>
    <div class="medical-card">
        <b>MENINGIOMA</b>
        <p>Tumors from the protective meninges. Usually benign but can compress vital nerves.</p>
    </div>
    <div class="medical-card">
        <b>PITUITARY</b>
        <p>Tumors on the base gland. Affects hormonal balance and visual fields.</p>
    </div>
    <div class="medical-card">
        <b>NO TUMOR</b>
        <p>No indication of abnormal mass found within the analyzed scan area.</p>
    </div>
    """, unsafe_allow_html=True)

# ── PDF GENERATOR (FIXED LAYOUT & COMPATIBILITY) ─────────────────────────────
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
        
        y_pos = pdf.get_y()
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe(f"#{idx+1:02d} | {res['filename']}"), ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mri:
            res['image'].save(tmp_mri.name)
            tmp_mri_path = tmp_mri.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            res['saliency'].save(tmp_heat.name)
            tmp_heat_path = tmp_heat.name

        img_y_pos = pdf.get_y()
        pdf.image(tmp_mri_path, x=10, y=img_y_pos, w=58)
        pdf.image(tmp_heat_path, x=72, y=img_y_pos, w=58)

        # Labels
        pdf.set_font("Arial", 'I', 7)
        pdf.set_text_color(150, 150, 150)
        pdf.set_xy(10, img_y_pos + 60)
        pdf.cell(58, 4, "[ MRI RAW SCAN ]", align='C')
        pdf.set_xy(72, img_y_pos + 60)
        pdf.cell(58, 4, "[ SALIENCY HEATMAP ]", align='C')

        # Info
        pdf.set_xy(135, img_y_pos + 2)
        pdf.set_font("Arial", 'B', 14)
        is_tumor = res['label'].lower() != 'no tumor'
        pdf.set_text_color(220, 50, 50) if is_tumor else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe(res['label'].upper()), ln=True)

        pdf.set_x(135)
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 6, safe(f"Confidence: {res['confidence']:.2f}%"), ln=True)

        pdf.set_y(img_y_pos + 70)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        if os.path.exists(tmp_mri_path): os.remove(tmp_mri_path)
        if os.path.exists(tmp_heat_path): os.remove(tmp_heat_path)

    return pdf.output(dest='S').encode('latin-1')

# ── LOGIC ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        if not os.path.exists(path): return None
        return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except: return None

def is_valid_mri(image: Image.Image) -> tuple[bool, str]:
    img_rgb = image.convert("RGB")
    img_arr = np.array(img_rgb, dtype=np.float32)
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    avg_color_diff = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3.0
    if avg_color_diff > 30.0: return False, "Grayscale MRI profile required."
    gray = np.array(image.convert("L"), dtype=np.float32)
    texture_score = (np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) + np.mean(np.abs(gray[1:, :] - gray[:-1, :]))) / 2.0
    if texture_score < 1.5 or texture_score > 80.0: return False, "Image lacks MRI characteristics."
    return True, "OK"

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1]
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_saliency_and_box(session, image: Image.Image, pred_class: int, label: str):
    """Generates heatmap and auto-calculated bounding box for tumor classes."""
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])
    
    H, W = 224, 224
    stride, patch = 14, 28
    saliency = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = base_batch.copy()
            y1, y2, x1, x2 = y, min(y+patch, H), x, min(x+patch, W)
            occ[0, y1:y2, x1:x2, :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            saliency[y1:y2, x1:x2] = np.maximum(saliency[y1:y2, x1:x2], base_score - score)
            
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)
    
    # ── HEATMAP GENERATION ──
    heatmap_raw = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_raw).resize(image.size, Image.BILINEAR)
    blended = Image.blend(image.convert('RGB'), heatmap_img, alpha=0.55)
    
    # ── BOUNDING BOX GENERATION (If Tumor detected) ──
    box_image = image.convert('RGB')
    if label.lower() != "no tumor":
        # Threshold saliency to find the most 'important' region
        threshold = 0.6
        coords = np.argwhere(saliency > threshold)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Rescale coords to original image size
            orig_w, orig_h = image.size
            scale_x, scale_y = orig_w / W, orig_h / H
            
            left, top = x_min * scale_x, y_min * scale_y
            right, bottom = x_max * scale_x, y_max * scale_y
            
            # Draw box on original image
            draw = ImageDraw.Draw(box_image)
            draw.rectangle([left, top, right, bottom], outline="#FC8181", width=3)
            # Small label tag
            draw.rectangle([left, top-12, left+60, top], fill="#FC8181")
            draw.text((left+5, top-11), "POS_MASS", fill="white")

    return blended, box_image

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("""<div class="neuro-header"><div class="neuro-logo">NEURO<span>SCAN</span> AI</div><div class="neuro-badge">ResNet50 · v2.4</div></div>""", unsafe_allow_html=True)

col_hero, col_upload = st.columns([1, 1], gap="large")
with col_hero:
    st.markdown("""<div class="hero-title">BRAIN<br><span class="accent">TUMOR</span><br>SCAN</div><div class="hero-sub">// Precision Neuro-Imaging Analytics</div>""", unsafe_allow_html=True)
with col_upload:
    session = load_model()
    uploaded_files = st.file_uploader("UPLOAD MRI", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

# ── KNOWLEDGE BASE ──
st.markdown("""<div class="knowledge-section"><div class="knowledge-grid">
    <div class="knowledge-card"><b>🧠 Glioma</b><p>Invasive glial tumor.</p></div>
    <div class="knowledge-card"><b>🛡️ Meningioma</b><p>Protective layer tumor.</p></div>
    <div class="knowledge-card"><b>💧 Pituitary</b><p>Glandular base tumor.</p></div>
    <div class="knowledge-card"><b>✅ No Tumor</b><p>Normal brain scan.</p></div>
</div></div>""", unsafe_allow_html=True)

# ── ANALYSIS ──
if uploaded_files and session:
    if st.button("EXECUTE ANALYSIS"):
        all_results = []
        t_start = time.time()
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            valid, reason = is_valid_mri(img)
            if not valid: continue
            
            batch = preprocess(img)
            out = session.run(None, {session.get_inputs()[0].name: batch})[0]
            pred_idx = int(np.argmax(out[0]))
            label = CLASS_NAMES.get(pred_idx, "Unknown")
            
            heat, boxed = generate_saliency_and_box(session, img, pred_idx, label)
            
            all_results.append({
                'image': img, 'boxed': boxed, 'saliency': heat,
                'filename': f.name, 'label': label,
                'confidence': float(np.max(out[0])) * 100, 'probs': out[0].tolist(),
                'size': f"{img.size[0]}x{img.size[1]}"
            })
        st.session_state['analysis_results'] = all_results
        st.session_state['total_time'] = time.time() - t_start

if 'analysis_results' in st.session_state:
    results = st.session_state['analysis_results']
    st.markdown(f'<div class="stat-grid"><div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(results)}</div></div></div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, res in enumerate(results):
        with cols[idx % 3]:
            is_tumor = res["label"] != "No Tumor"
            st.markdown(f'<div class="scan-result {"tumor" if is_tumor else ""}"><b>{res["label"].upper()}</b> | {res["confidence"]:.1f}%</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["VIEWPORT", "HEATMAP", "PROBABILITY"])
            
            with t1:
                st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
                ca, cb, cc = st.columns(3)
                with ca: br = st.slider("BRIGHT", 0.5, 2.0, 1.0, 0.1, key=f"br_{res['filename']}")
                with cb: ct = st.slider("CONTRAST", 0.5, 2.0, 1.0, 0.1, key=f"ct_{res['filename']}")
                with cc: sh = st.slider("SHARP", 0.0, 3.0, 1.0, 0.1, key=f"sh_{res['filename']}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Apply enhancements to the BOXED image
                enh = ImageEnhance.Brightness(res["boxed"]).enhance(br)
                enh = ImageEnhance.Contrast(enh).enhance(ct)
                enh = ImageEnhance.Sharpness(enh).enhance(sh)
                st.image(enh, use_container_width=True)
                
            with t2: st.image(res["saliency"], use_container_width=True)
            with t3:
                for ci, p in enumerate(res["probs"]):
                    st.markdown(f"<div style='font-size:12px; color:rgba(255,255,255,0.7)'>{CLASS_NAMES[ci]} ({p*100:.1f}%)</div>", unsafe_allow_html=True)
                    st.progress(p)

    st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
    col_dl, col_info = st.columns([1, 2])
    with col_dl: 
        pdf_bytes = create_pdf(results)
        st.download_button("DOWNLOAD CLINICAL REPORT", pdf_bytes, f"NeuroScan_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
    with col_info: st.info("Institutional research use only.")

st.markdown("<br><div style='text-align:center; font-size:9px; color:rgba(99,179,237,0.12); padding: 24px 0;'>NEUROSCAN AI · v2.4</div>", unsafe_allow_html=True)
