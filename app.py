import os
import streamlit as st
import numpy as np
import time
import tempfile
import cv2
from PIL import Image
from fpdf import FPDF
from datetime import datetime

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (Enterprise Dark Mode) ───────────────────────────────────────────────
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

.hero-title {
    font-size: clamp(42px, 6vw, 80px);
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

.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px; padding: 20px;
    margin-bottom: 12px; overflow: hidden;
}
.scan-result::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #63B3ED, transparent);
}
.scan-result.tumor::before { background: linear-gradient(180deg, #FC8181, transparent); }

.stButton > button {
    background: transparent !important;
    color: #63B3ED !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
}

.stat-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin: 32px 0;
}
.stat-item {
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px; padding: 20px 24px;
}
</style>
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
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            res['image'].save(tmp.name)
            y = pdf.get_y()
            pdf.image(tmp.name, x=10, y=y, w=55)
        
        pdf.set_xy(75, y + 6)
        pdf.set_font("Arial", 'B', 13)
        is_tumor = res['label'].lower() != 'no tumor'
        pdf.set_text_color(220, 50, 50) if is_tumor else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe(res['label'].upper()), ln=True)
        
        pdf.set_x(75)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, safe(f"Confidence: {res['confidence']:.2f}%"), ln=True)
        
        pdf.set_y(y + 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)
        if os.path.exists(tmp.name): os.remove(tmp.name)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, safe("DISCLAIMER: This report is generated by AI for research purposes. Final diagnosis must be by a professional."))
    return bytes(pdf.output(dest="S"))

# ── GRAD-CAM & SALIENCY ENGINES ──────────────────────────────────────────────
@st.cache_resource
def load_onnx_session():
    import onnxruntime as ort
    return ort.InferenceSession('resnet_model.onnx', providers=['CPUExecutionProvider'])

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1] # RGB to BGR
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_gradcam(image: Image.Image, model_path: str) -> Image.Image:
    """True Gradient-weighted Class Activation Mapping via TensorFlow."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Preprocessing khusus TF
    img_array = preprocess(image)
    
    # Cari layer konvolusi terakhir
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(last_conv_layer_output, (list, tuple)): last_conv_layer_output = last_conv_layer_output[0]
        class_idx = np.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    # Resizing and Coloring
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    original = np.array(image.convert('RGB'))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    """Occlusion-based saliency (ONNX compatible)."""
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])

    H, W = 224, 224
    stride, patch = 16, 32
    saliency = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = base_batch.copy()
            y1, y2, x1, x2 = y, min(y+patch, H), x, min(x+patch, W)
            occ[0, y1:y2, x1:x2, :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            saliency[y1:y2, x1:x2] = np.maximum(saliency[y1:y2, x1:x2], base_score - score)

    s_max = saliency.max()
    if s_max > 0: saliency /= s_max
    
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert('RGB'), heatmap_img, alpha=0.5)

# ── UI RENDER ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
MODEL_H5 = 'best_resnet_20260307-162330.h5'

st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div class="neuro-badge">Hybrid Engine: ONNX + TF</div>
</div>
""", unsafe_allow_html=True)

col_hero, col_upload = st.columns([1, 1], gap="large")
with col_hero:
    st.markdown("""<div class="hero-title">BRAIN<br><span class="accent">TUMOR</span><br>SCAN</div>
    <div class="hero-sub">// Dual-Engine Visualization (Grad-CAM & Saliency)</div>""", unsafe_allow_html=True)

with col_upload:
    onnx_sess = load_model() # Dari cache_resource Anda
    uploaded_files = st.file_uploader("DROP MRI SCANS", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    if st.button(f"EXECUTE NEURAL ANALYSIS ({len(uploaded_files)} SCANS)"):
        all_results = []
        t_start = time.time()
        
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            batch = preprocess(img)
            
            # 1. Fast Prediction via ONNX
            out_onnx = onnx_sess.run(None, {onnx_sess.get_inputs()[0].name: batch})[0]
            pred_idx = int(np.argmax(out_onnx[0]))
            conf = float(np.max(out_onnx[0])) * 100
            
            # 2. XAI Visualizations
            with st.spinner(f"Generating XAI for {f.name}..."):
                saliency_img = generate_saliency(onnx_sess, img, pred_idx)
                
                # Grad-CAM menggunakan file .h5
                gradcam_img = None
                if os.path.exists(MODEL_H5):
                    try:
                        gradcam_img = generate_gradcam(img, MODEL_H5)
                    except: gradcam_img = None

            all_results.append({
                'image': img, 'saliency': saliency_img, 'gradcam': gradcam_img,
                'filename': f.name, 'label': CLASS_NAMES.get(pred_idx),
                'confidence': conf, 'probs': out_onnx[0].tolist()
            })

        # ── STATS RENDER ──
        t_total = time.time() - t_start
        avg_conf = np.mean([r['confidence'] for r in all_results])
        st.markdown(f"""<div class="stat-grid">
            <div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(all_results)}<span> seq</span></div></div>
            <div class="stat-item"><div class="stat-label">Avg Confidence</div><div class="stat-value">{avg_conf:.0f}<span>%</span></div></div>
            <div class="stat-item"><div class="stat-label">Total Time</div><div class="stat-value">{t_total:.1f}<span>s</span></div></div>
        </div>""", unsafe_allow_html=True)

        # ── RESULTS GRID ──
        for res in all_results:
            is_tumor = res['label'].lower() != 'no tumor'
            st.markdown(f"""<div class="scan-result {'tumor' if is_tumor else ''}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><div class="result-label {'tumor' if is_tumor else ''}">{res['label'].upper()}</div><div class="result-meta">{res['filename']}</div></div>
                    <div class="result-confidence {'tumor' if is_tumor else ''}">{res['confidence']:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.image(res['image'], caption="Original", use_container_width=True)
            with c2: st.image(res['gradcam'], caption="Grad-CAM (Gradient)", use_container_width=True) if res['gradcam'] else st.warning("H5 Missing")
            with c3: st.image(res['saliency'], caption="Saliency (Occlusion)", use_container_width=True)
            
            # Prob bars
            cols = st.columns(4)
            for i, p in enumerate(res['probs']):
                with cols[i]:
                    st.caption(f"{CLASS_NAMES[i]}")
                    st.progress(p)

        st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
        st.download_button("↓ DOWNLOAD CLINICAL REPORT (PDF)", create_pdf(all_results), 
                         file_name=f"NeuroScan_{datetime.now().strftime('%m%d_%H%M')}.pdf")

else:
    st.markdown('<div class="idle-state"><div class="idle-icon">◎</div><div class="idle-text">Awaiting scan sequence input</div></div>', unsafe_allow_html=True)

st.markdown("<br><div style='text-align:center; font-family:\"Space Mono\",monospace; font-size:9px; letter-spacing:4px; color:rgba(99,179,237,0.12); padding: 24px 0;'>NEUROSCAN AI · RESEARCH USE ONLY · v2.5</div>", unsafe_allow_html=True)
