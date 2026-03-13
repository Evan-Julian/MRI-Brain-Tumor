import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── FULL RESPONSIVE CSS (MODERN & SPACE-SAVING) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Syne', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}

/* Responsive Grid System */
.main-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

/* Stat Cards Grid */
.stat-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

/* Mobile Optimizations */
@media (max-width: 768px) {
    .hero-title { font-size: 45px !important; line-height: 0.9 !important; }
    .neuro-header { flex-direction: column; align-items: flex-start; gap: 10px; }
    .stImage { width: 100% !important; }
    .block-container { padding: 1rem !important; }
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

/* Components Styling */
.sidebar-title {
    font-family: 'Space Mono', monospace; font-size: 14px; color: #63B3ED;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 20px;
    border-bottom: 1px solid rgba(99,179,237,0.2); padding-bottom: 10px;
}

.medical-card {
    background: rgba(99,179,237,0.04); border: 1px solid rgba(99,179,237,0.1);
    padding: 15px; border-radius: 8px; margin-bottom: 15px;
    transition: all 0.3s ease;
}

.medical-card:hover { border-color: rgba(99,179,237,0.4); background: rgba(99,179,237,0.07); }

.neuro-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0; border-bottom: 1px solid rgba(99,179,237,0.12); margin-bottom: 30px;
}

.neuro-logo { font-family: 'Space Mono'; font-weight: 700; font-size: 18px; letter-spacing: 2px; }
.neuro-logo span { color: #63B3ED; }

.hero-title {
    font-size: clamp(40px, 8vw, 90px); font-weight: 900; line-height: 0.85;
    letter-spacing: -4px; color: #EDF2F7;
}

.hero-title .accent {
    color: transparent; -webkit-text-stroke: 1px rgba(99,179,237,0.5);
}

.scan-result-card {
    background: rgba(10,14,22,0.8); border: 1px solid rgba(99,179,237,0.1);
    padding: 20px; border-radius: 12px; height: 100%;
}

.divider {
    height: 1px; background: linear-gradient(90deg, rgba(99,179,237,0.3), transparent);
    margin: 40px 0;
}
</style>
""", unsafe_allow_html=True)

# ── LOGIC FUNCTIONS (KEEPING ALL YOUR ORIGINAL LOGIC) ────────────────────────

def safe_latin1(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

@st.cache_resource
def load_session():
    try:
        import onnxruntime as ort
        model_path = 'resnet_model.onnx' # Pastikan nama file sesuai
        if not os.path.exists(model_path): return None
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def is_valid_mri(image: Image.Image) -> tuple[bool, str]:
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    # 1. Color check (MRI should be grayscale-ish)
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    avg_diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3.0
    if avg_diff > 35.0:
        return False, "Image contains too much color. Genuine MRI scans are grayscale."
    
    # 2. Texture/Contrast check
    gray = np.array(image.convert("L"), dtype=np.float32)
    dark_ratio = np.mean(gray < 25)
    if dark_ratio < 0.05:
        return False, "Missing dark background characteristic of MRI scans."
    
    return True, "Valid"

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    img_data = np.array(img).astype('float32')
    # BGR Conversion & Mean Subtraction (Standard for ResNet)
    img_data = img_data[:, :, ::-1] 
    img_data[:, :, 0] -= 103.939
    img_data[:, :, 1] -= 116.779
    img_data[:, :, 2] -= 123.68
    return np.expand_dims(img_data, axis=0)

def generate_saliency(session, image: Image.Image, pred_idx: int) -> Image.Image:
    input_name = session.get_inputs()[0].name
    original_batch = preprocess_image(image)
    base_probs = session.run(None, {input_name: original_batch})[0]
    base_score = float(base_probs[0][pred_idx])
    
    # Occlusion sensitivity (Keeping your specific logic)
    H, W = 224, 224
    stride, size = 14, 28
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ_input = original_batch.copy()
            y_end, x_end = min(y + size, H), min(x + size, W)
            occ_input[0, y:y_end, x:x_end, :] = 0
            new_score = float(session.run(None, {input_name: occ_input})[0][0][pred_idx])
            heatmap[y:y_end, x:x_end] = np.maximum(heatmap[y:y_end, x:x_end], base_score - new_score)
    
    # Normalize & Colorize
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min: heatmap = (heatmap - h_min) / (h_max - h_min)
    
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    color_map[:,:,0] = (np.clip(heatmap * 3 - 1, 0, 1) * 255).astype(np.uint8) # R
    color_map[:,:,1] = (np.clip(heatmap * 3 - 0.5, 0, 1) * np.clip(2 - heatmap * 3, 0, 1) * 255).astype(np.uint8) # G
    color_map[:,:,2] = (np.clip(1 - heatmap * 2, 0, 1) * 255).astype(np.uint8) # B
    
    heatmap_img = Image.fromarray(color_map).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_img, alpha=0.5)

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 15, "NEUROSCAN AI - CLINICAL REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, safe_latin1(f"Diagnostic Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True, align='C')
    pdf.ln(10)

    for i, r in enumerate(results):
        if i > 0 and i % 2 == 0: pdf.add_page()
        
        # Save temp images for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
            r['image'].save(f1.name); r['heatmap'].save(f2.name)
            
            y_pos = pdf.get_y()
            pdf.image(f1.name, x=10, y=y_pos, w=60)
            pdf.image(f2.name, x=75, y=y_pos, w=60)
            
            pdf.set_xy(140, y_pos + 5)
            pdf.set_font("Arial", 'B', 14)
            color = (200, 0, 0) if r['label'] != "No Tumor" else (0, 150, 0)
            pdf.set_text_color(*color)
            pdf.cell(0, 10, safe_latin1(r['label'].upper()), ln=True)
            
            pdf.set_x(140)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 7, safe_latin1(f"Confidence: {r['confidence']:.2f}%"), ln=True)
            pdf.ln(50)
            
            os.remove(f1.name); os.remove(f2.name)
            
    return pdf.output(dest="S")

# ── SIDEBAR (English Content) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Clinical Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="medical-card">
        <b>🧠 Glioma</b><br><small>Tumors that start in the glial cells of the brain or the spine. Highly invasive.</small>
    </div>
    <div class="medical-card">
        <b>🛡️ Meningioma</b><br><small>A tumor that arises from the meninges — the membranes that surround your brain.</small>
    </div>
    <div class="medical-card">
        <b>💧 Pituitary</b><br><small>Abnormal growths that develop in your pituitary gland (master gland).</small>
    </div>
    <div class="medical-card">
        <b>✅ No Tumor</b><br><small>Clear scan. No malignant or benign masses detected in the sequence.</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("NeuroScan AI v2.4 | ResNet50 Engine")

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div style="font-family:'Space Mono'; font-size:12px; color:rgba(99,179,237,0.5);">STATUS: SYSTEM_READY</div>
</div>
""", unsafe_allow_html=True)

# Hero Section
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.markdown("""
    <div class="hero-title">ADVANCED<br><span class="accent">NEURO</span><br>IMAGING</div>
    <div style="font-family:'Space Mono'; font-size:14px; color:#63B3ED; margin-top:15px; letter-spacing:1px;">
        // AUTOMATED DETECTION OF PATHOLOGICAL LESIONS
    </div>
    """, unsafe_allow_html=True)

with col_right:
    session = load_session()
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("DROP MRI SEQUENCES HERE", type=['jpg','png','jpeg'], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files ready for batch analysis.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ANALYSIS PROCESS ─────────────────────────────────────────────────────────
if uploaded_files and st.button("🚀 INITIATE CLINICAL ANALYSIS", use_container_width=True):
    if not session:
        st.error("Model file 'resnet_model.onnx' not found.")
    else:
        results_list = []
        invalid_list = []
        start_time = time.time()
        
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB")
            is_valid, reason = is_valid_mri(img)
            
            if not is_valid:
                invalid_list.append({"file": file.name, "reason": reason})
            else:
                # Prediction Logic
                input_arr = preprocess_image(img)
                outputs = session.run(None, {session.get_inputs()[0].name: input_arr})[0][0]
                pred_idx = int(np.argmax(outputs))
                labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
                
                # Saliency Mapping
                heatmap = generate_saliency(session, img, pred_idx)
                
                results_list.append({
                    "image": img, "heatmap": heatmap, "label": labels[pred_idx],
                    "confidence": float(outputs[pred_idx]) * 100, "filename": file.name
                })
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        total_time = time.time() - start_time
        
        # Display Stats Grid (Responsive)
        st.markdown(f"""
        <div class="stat-container">
            <div class="medical-card"><b>SUCCESSFUL</b><br><span style="font-size:24px; color:#63B3ED;">{len(results_list)}</span></div>
            <div class="medical-card"><b>REJECTED</b><br><span style="font-size:24px; color:#FC8181;">{len(invalid_list)}</span></div>
            <div class="medical-card"><b>LATENCY</b><br><span style="font-size:24px;">{total_time:.2f}s</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Handle Rejections
        for inv in invalid_list:
            st.warning(f"⚠️ **{inv['file']}** rejected: {inv['reason']}")

        # Display Results in Responsive Grid
        if results_list:
            st.markdown('<div class="main-grid">', unsafe_allow_html=True)
            # Karena Streamlit tidak mendukung penutupan tag HTML custom di tengah st.columns, 
            # kita gunakan layout kolom standar Streamlit yang sudah responsif.
            cols = st.columns(min(len(results_list), 3))
            for idx, res in enumerate(results_list):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class="scan-result-card">
                            <div style="font-family:'Space Mono'; font-size:10px; color:#63B3ED;">{res['filename']}</div>
                            <div style="font-size:22px; font-weight:800; margin:5px 0;">{res['label'].upper()}</div>
                            <div style="font-size:12px; opacity:0.7;">Confidence: {res['confidence']:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        # Tabs for original vs heatmap
                        t1, t2 = st.tabs(["Scan", "Saliency"])
                        t1.image(res['image'], use_container_width=True)
                        t2.image(res['heatmap'], use_container_width=True)
            
            # PDF Download
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            pdf_data = create_pdf(results_list)
            st.download_button("↓ DOWNLOAD FULL CLINICAL REPORT", data=pdf_data, file_name="Diagnostic_Report.pdf", mime="application/pdf", use_container_width=True)

elif not uploaded_files:
    # Idle State
    st.markdown("""
    <div style="text-align:center; padding:100px 20px; border:1px dashed rgba(99,179,237,0.2); border-radius:20px;">
        <div style="font-size:40px; margin-bottom:20px; opacity:0.3;">◎</div>
        <div style="font-family:'Space Mono'; letter-spacing:2px; opacity:0.5;">AWAITING MRI SEQUENCE UPLOAD</div>
    </div>
    """, unsafe_allow_html=True)
