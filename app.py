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
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.med-info-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 40px;
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

.hero-title {
    font-size: clamp(32px, 5vw, 70px);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -3px;
    color: #EDF2F7;
    margin-bottom: 16px;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px; letter-spacing: 2px;
    color: rgba(99,179,237,0.5);
    text-transform: uppercase;
    margin-bottom: 32px;
}

.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 32px 0;
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

.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 12px;
}
.scan-result.tumor::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: #FC8181;
}

.control-panel {
    background: rgba(99,179,237,0.03);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 15px;
    margin-top: 10px;
}

.neo-divider { border: none; border-top: 1px solid rgba(99,179,237,0.08); margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> / AI DIAGNOSTIC</div>
    <div class="neuro-badge">ResNet50 · ONNX · v2.8 PRO</div>
</div>
""", unsafe_allow_html=True)

# ── HERO & MEDICAL KNOWLEDGE ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">BRAIN ANALYTICS</div>
<div class="hero-sub">// Deep Learning MRI Diagnostic Suite</div>
<div class="med-info-container">
    <div class="medical-card">
        <b>GLIOMA</b>
        <p>Invasive primary tumors. Characterized by parenchymal infiltration and ill-defined margins.</p>
    </div>
    <div class="medical-card">
        <b>MENINGIOMA</b>
        <p>Extra-axial tumors arising from meninges. Often show dural tail sign and clear margins.</p>
    </div>
    <div class="medical-card">
        <b>PITUITARY</b>
        <p>Sellar region adenomas. Can result in bitemporal hemianopsia and hormonal dysregulation.</p>
    </div>
    <div class="medical-card">
        <b>NO TUMOR</b>
        <p>Negative study. No evidence of space-occupying lesions or pathological signal intensity.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UTILITIES ────────────────────────────────────────────────────────────────
def safe_str(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def apply_clahe(image):
    # Convert PIL to CV2
    img_array = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_array)
    return Image.fromarray(cl1).convert('RGB')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header & Meta
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 12, "CLINICAL NEURO-IMAGING REPORT", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 5, safe_str(f"REF ID: {datetime.now().strftime('%Y%m%d%H%M%S')}"), ln=True, align='R')
    pdf.cell(0, 5, safe_str(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"), ln=True, align='R')
    pdf.ln(5)
    
    pdf.set_draw_color(26, 115, 232)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    for idx, res in enumerate(results):
        if pdf.get_y() > 200:
            pdf.add_page()
            
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe_str(f"CASE #{idx+1:02d} | SOURCE: {res['filename']}"), ln=True)
        
        y_img = pdf.get_y()
        # Original Image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_o:
            res['image'].save(tmp_o.name)
            pdf.image(tmp_o.name, x=10, y=y_img, w=55)
        # Heatmap Image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_h:
            res['saliency'].save(tmp_h.name)
            pdf.image(tmp_h.name, x=70, y=y_img, w=55)
            
        pdf.set_xy(130, y_img + 2)
        pdf.set_font("Arial", 'B', 10); pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, "AI CLASSIFICATION:", ln=True)
        
        pdf.set_x(130)
        pdf.set_font("Arial", 'B', 14)
        is_t = res['label'].lower() != 'no tumor'
        pdf.set_text_color(200, 50, 50) if is_t else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe_str(res['label'].upper()), ln=True)
        
        pdf.set_x(130)
        pdf.set_font("Arial", '', 11); pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 7, safe_str(f"Confidence: {res['confidence']:.2f}%"), ln=True)
        
        os.remove(tmp_o.name); os.remove(tmp_h.name)
        pdf.set_y(y_img + 60)
        pdf.set_draw_color(230, 230, 230); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(10)

    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, "NOTICE: This automated report is for institutional research purposes. Findings must be correlated with clinical symptoms and confirmed by a board-certified neuroradiologist.")
    
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
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)
    heatmap = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    return Image.blend(image.convert('RGB'), Image.fromarray(heatmap).resize(image.size, Image.BILINEAR), alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── ENGINE ───────────────────────────────────────────────────────────────────
session = load_model()
if not session: st.error("Neural engine core missing."); st.stop()

uploaded_files = st.file_uploader("DROP MRI HERE", type=["jpg","png","jpeg"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    if st.button(f"RUN ANALYSIS / {len(uploaded_files)} SCANS"):
        all_res = []
        t_start = time.time()
        with st.status("Initializing AI Diagnostic Layer...", expanded=True) as status:
            prog = st.progress(0)
            for i, f in enumerate(uploaded_files):
                status.write(f"Analyzing sequence: {f.name}")
                img = Image.open(f).convert('RGB')
                out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
                idx = int(np.argmax(out[0]))
                all_res.append({
                    'image': img, 'saliency': generate_saliency(session, img, idx),
                    'filename': f.name, 'label': CLASS_NAMES[idx], 
                    'confidence': float(np.max(out[0]))*100, 'probs': out[0].tolist(), 'size': f"{img.size[0]}x{img.size[1]}"
                })
                prog.progress((i+1)/len(uploaded_files))
            status.update(label="Sequence Analysis Finalized", state="complete", expanded=False)
        st.session_state['results'] = all_res; st.session_state['time'] = time.time() - t_start

# ── RENDER ───────────────────────────────────────────────────────────────────
if 'results' in st.session_state:
    res = st.session_state['results']
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(res)}<span> scans</span></div></div>
        <div class="stat-item"><div class="stat-label">Abnormalities</div><div class="stat-value">{sum(1 for r in res if r['label']!='No Tumor')}<span> detect</span></div></div>
        <div class="stat-item"><div class="stat-label">Avg Conf</div><div class="stat-value">{np.mean([r['confidence'] for r in res]):.1f}<span>%</span></div></div>
        <div class="stat-item"><div class="stat-label">Latency</div><div class="stat-value">{st.session_state['time']:.2f}<span>s</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    for i in range(0, len(res), 3):
        cols = st.columns(3)
        for idx, item in enumerate(res[i:i+3]):
            with cols[idx]:
                is_t = item['label'] != 'No Tumor'
                st.markdown(f'<div class="scan-result {"tumor" if is_t else ""}"><div class="result-label {"tumor" if is_t else ""}">{item["label"].upper()}</div><div style="font-size:10px; opacity:0.6;">{item["filename"]}</div></div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["VIEWPORT", "SALIENCY", "PROBABILITY"])
                with t1:
                    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
                    s1, s2, s3 = st.columns(3)
                    br = s1.slider("BRIGHT", 0.5, 2.0, 1.0, key=f"b{i+idx}")
                    ct = s2.slider("CONTRAST", 0.5, 2.0, 1.0, key=f"c{i+idx}")
                    sh = s3.slider("SHARP", 0.0, 3.0, 1.0, key=f"s{i+idx}")
                    med_enh = st.toggle("MEDICAL ENHANCE (CLAHE)", key=f"clahe_{i+idx}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    e = item['image']
                    if med_enh: e = apply_clahe(e)
                    e = ImageEnhance.Brightness(e).enhance(br)
                    e = ImageEnhance.Contrast(e).enhance(ct)
                    e = ImageEnhance.Sharpness(e).enhance(sh)
                    st.image(e, use_container_width=True)
                with t2: st.image(item['saliency'], use_container_width=True)
                with t3:
                    for k, v in enumerate(item['probs']):
                        st.write(f"{CLASS_NAMES[k]}: {v*100:.1f}%"); st.progress(v)
    
    st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1: 
        report_data = create_pdf(res)
        st.download_button("GENERATE CLINICAL REPORT", report_data, f"NeuroScan_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")
    with c2: st.info("Diagnostic Notice: This tool is intended for institutional research use only.")

elif not uploaded_files:
    st.markdown('<div style="text-align:center; padding:100px; opacity:0.2; font-family:Space Mono; letter-spacing:2px;">AWAITING MRI INPUT SEQUENCE</div>', unsafe_allow_html=True)

st.markdown("<br><div style='text-align:center; font-family:Space Mono; font-size:9px; color:rgba(99,179,237,0.1); padding:40px 0;'>NEUROSCAN AI · INSTITUTIONAL RESEARCH USE ONLY · 2026</div>", unsafe_allow_html=True)
