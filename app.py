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

# ── ENHANCED RESPONSIVE CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Syne', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}

/* Responsive Grid for Stat Grid */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 24px 0;
}

/* Responsive Knowledge Grid */
.knowledge-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
}

/* Mobile Optimizations */
@media (max-width: 768px) {
    .hero-title {
        font-size: 40px !important;
        line-height: 1.0 !important;
    }
    .block-container {
        padding: 1rem !important;
    }
    .stat-value {
        font-size: 24px !important;
    }
    .knowledge-card {
        padding: 12px !important;
    }
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

/* ── COMPONENTS ── */
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

.neuro-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 30px;
}

.hero-title {
    font-size: clamp(32px, 5vw, 70px);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -3px;
    color: #EDF2F7;
}

.hero-title .accent {
    color: transparent;
    -webkit-text-stroke: 1px rgba(99,179,237,0.5);
}

.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px; padding: 15px;
    margin-bottom: 10px;
}

.invalid-card {
    background: rgba(254,178,178,0.05);
    border: 1px solid rgba(252,129,129,0.3);
    padding: 12px; border-radius: 4px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR KNOWLEDGE CENTER (English) ───────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Medical Knowledge Center</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="medical-card">
        <b>🧠 GLIOMA</b>
        <p>Tumors originating from glial cells. Often invasive and spread into surrounding brain tissue.</p>
    </div>
    <div class="medical-card">
        <b>🛡️ MENINGIOMA</b>
        <p>Arises from the meninges (protective layers). Mostly benign but can compress vital nerves.</p>
    </div>
    <div class="medical-card">
        <b>💧 PITUITARY</b>
        <p>Tumors on the pituitary gland. Can affect hormone production and vision.</p>
    </div>
    <div class="medical-card">
        <b>✅ NO TUMOR</b>
        <p>No indication of abnormal mass or pathological lesions in the analyzed scan area.</p>
    </div>
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
        
        # MRI and Heatmap Logic
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
        pdf.ln(40)

        # Cleanup
        os.remove(tmp_mri_path)
        os.remove(tmp_heat_path)

    return bytes(pdf.output(dest="S"))

# ── UTILITIES (Validation & Model) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        if not os.path.exists(path): return None
        return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except: return None

def is_valid_mri(image: Image.Image) -> tuple[bool, str]:
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    avg_color_diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3.0
    
    if avg_color_diff > 30.0:
        return False, "Image too colorful. MRI scans should be grayscale."
    
    gray = np.array(image.convert("L"), dtype=np.float32)
    dark_pixel_ratio = np.mean(gray < 20)
    if dark_pixel_ratio < 0.08:
        return False, "Insufficient dark background. Not characteristic of an MRI."
    
    return True, "OK"

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1] # BGR
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
            occ[0, y:min(y+patch, H), x:min(x+patch, W), :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            saliency[y:min(y+patch, H), x:min(x+patch, W)] = np.maximum(saliency[y:min(y+patch, H), x:min(x+patch, W)], base_score - score)

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)
    
    heatmap = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert('RGB'), heatmap_img, alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── MAIN UI (English) ────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div class="neuro-badge">ResNet50 · v2.4</div>
</div>
""", unsafe_allow_html=True)

# Knowledge Section with Responsive Grid
st.markdown("""
<div class="knowledge-grid">
    <div class="medical-card"><b>🧠 Glioma</b><p>Invasive tumor from glial cells.</p></div>
    <div class="medical-card"><b>🛡️ Meningioma</b><p>Protective layer tumor, mostly benign.</p></div>
    <div class="medical-card"><b>💧 Pituitary</b><p>Hormonal gland tumor at brain base.</p></div>
    <div class="medical-card"><b>✅ No Tumor</b><p>Clean scan, normal brain structure.</p></div>
</div>
""", unsafe_allow_html=True)

col_hero, col_upload = st.columns([1, 1])

with col_hero:
    st.markdown("""
    <div class="hero-title">BRAIN<br><span class="accent">TUMOR</span><br>SCAN</div>
    <div style="font-family:'Space Mono'; font-size:12px; color:#63B3ED; margin-top:10px;">// PRECISION NEURO-IMAGING</div>
    """, unsafe_allow_html=True)

with col_upload:
    session = load_model()
    uploaded_files = st.file_uploader("UPLOAD MRI SCANS", type=["jpg","png","jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files and st.button(f"ANALYZE {len(uploaded_files)} FILES"):
        results, invalids = [], []
        t_start = time.time()
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            valid, reason = is_valid_mri(img)
            if not valid:
                invalids.append({'name': f.name, 'reason': reason})
                continue
            
            batch = preprocess(img)
            out = session.run(None, {session.get_inputs()[0].name: batch})[0]
            idx = int(np.argmax(out[0]))
            results.append({
                'image': img, 'saliency': generate_saliency(session, img, idx),
                'filename': f.name, 'label': CLASS_NAMES[idx],
                'confidence': float(np.max(out[0]))*100, 'probs': out[0].tolist(),
                'size': f"{img.size[0]}x{img.size[1]}"
            })
        st.session_state.update({'results': results, 'invalids': invalids, 'time': time.time()-t_start})

# ── RESULTS DISPLAY ──────────────────────────────────────────────────────────
if 'results' in st.session_state:
    res, inv = st.session_state['results'], st.session_state['invalids']
    
    if inv:
        for i in inv: st.error(f"Rejected: {i['name']} - {i['reason']}")

    if res:
        st.markdown(f'<div class="stat-grid">'
                    f'<div class="medical-card"><b>PROCESSED</b><br>{len(res)}</div>'
                    f'<div class="medical-card"><b>TIME</b><br>{st.session_state["time"]:.2f}s</div>'
                    f'</div>', unsafe_allow_html=True)

        # Responsive Grid for Results
        cols = st.columns(min(len(res), 3))
        for i, r in enumerate(res):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="scan-result">
                    <div style="color:#63B3ED; font-weight:bold;">{r['label'].upper()}</div>
                    <div style="font-size:10px; opacity:0.6;">{r['confidence']:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(r['image'], use_container_width=True)

        pdf_bytes = create_pdf(res)
        st.download_button("↓ DOWNLOAD REPORT", pdf_bytes, f"Report_{datetime.now().strftime('%M%S')}.pdf", "application/pdf")
