import os
import streamlit as st
import numpy as np
import time
import tempfile
import cv2
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CSS (Enterprise Dark Mode) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@400;700&family=Poppins:wght@400;700;900&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}
.stApp::before {
    content: ''; position: fixed; inset: 0; z-index: 0;
    background-image: linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 48px 48px; pointer-events: none;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
.neuro-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 0; border-bottom: 1px solid rgba(99,179,237,0.12); margin-bottom: 40px; }
.neuro-logo { font-family: 'Space Mono', monospace; font-size: 14px; letter-spacing: 4px; color: #63B3ED; font-weight: 700; }
.hero-title { font-size: clamp(40px, 5vw, 70px); font-weight: 900; line-height: 1; color: #EDF2F7; margin-bottom: 10px; }
.scan-result { background: rgba(10,14,22,0.8); border-left: 4px solid #63B3ED; padding: 20px; margin-bottom: 20px; border-radius: 4px; }
.scan-result.tumor { border-left-color: #FC8181; }
.stButton > button { background: transparent !important; color: #63B3ED !important; border: 1px solid #63B3ED !important; letter-spacing: 2px; text-transform: uppercase; width: 100%; padding: 15px; font-family: 'Space Mono', monospace; }
.stat-box { background: rgba(10,14,22,0.8); border: 1px solid rgba(99,179,237,0.1); padding: 20px; text-align: center; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- PDF GENERATOR ---
def safe(text):
    return text.encode('latin-1', errors='replace').decode('latin-1')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "NEUROSCAN AI CLINICAL REPORT", ln=True, align='C')
    pdf.ln(10)
    for idx, res in enumerate(results):
        if idx > 0 and idx % 2 == 0: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, safe(f"SCAN #{idx+1}: {res['filename']}"), ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            res['image'].save(tmp.name)
            y = pdf.get_y()
            pdf.image(tmp.name, x=10, y=y, w=50)
        pdf.set_xy(70, y + 5)
        pdf.cell(0, 10, safe(f"RESULT: {res['label'].upper()}"), ln=True)
        pdf.set_x(70)
        pdf.cell(0, 10, safe(f"CONFIDENCE: {res['confidence']:.2f}%"), ln=True)
        pdf.set_y(y + 55)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        if os.path.exists(tmp.name): os.remove(tmp.name)
    return pdf.output(dest='S').encode('latin-1', errors='replace')

# --- ENGINES ---
@st.cache_resource
def load_onnx_session():
    import onnxruntime as ort
    return ort.InferenceSession('resnet_model.onnx', providers=['CPUExecutionProvider'])

def preprocess(image):
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1] # RGB to BGR
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

# --- GRAD-CAM ENGINE v3.7 (ADAPTIVE) ---
def generate_gradcam(image, model_path):
    import tensorflow as tf

    def build_and_load():
        # Membangun dengan API Functional (lebih stabil daripada Sequential untuk loading bobot)
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        
        # 1. Coba load utuh (Keras 3 standar)
        try:
            full_model = tf.keras.models.load_model(model_path, compile=False)
            # Cari resnet part di dalam model yang baru dimuat
            resnet_part = None
            for layer in full_model.layers:
                if 'resnet50' in layer.name.lower():
                    resnet_part = layer
                    break
            if resnet_part is None: resnet_part = full_model
            return full_model, resnet_part
        except:
            # 2. Fallback: Load bobot secara parsial (Keras 2 compat)
            # by_name=True sangat penting karena struktur wrapper Sequential sering berbeda
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            return model, base

    try:
        model, resnet_part = build_and_load()
    except Exception as e:
        return None, f"Structure Error: {e}"

    img_array = preprocess(image)
    
    try:
        # Cari layer konvolusi terakhir secara dinamis dengan pengecekan mendalam
        target_layer = None
        # Jika resnet_part adalah model functional (biasanya iya), kita cari di .layers-nya
        layers_to_search = resnet_part.layers if hasattr(resnet_part, 'layers') else model.layers
        
        for layer in reversed(layers_to_search):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv5_block3_out' in layer.name:
                target_layer = layer
                break
        
        if not target_layer: return None, "Target layer not detected."

        # Model Grad-CAM
        grad_model = tf.keras.models.Model(
            [resnet_part.inputs], 
            [target_layer.output, resnet_part.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tf.cast(img_array, tf.float32))
            # Unpacking tensor (Keras 3 safety)
            if isinstance(conv_outputs, (list, tuple)): conv_outputs = conv_outputs[0]
            if isinstance(predictions, (list, tuple)): predictions = predictions[0]
            
            loss = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Kalkulasi Heatmap dengan broadcast multiplication
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        heatmap = cam[0]
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        # Color mapping
        heatmap_np = heatmap.numpy()
        heatmap_res = cv2.resize(heatmap_np, (image.size[0], image.size[1]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(cv2.addWeighted(np.array(image.convert('RGB')), 0.6, heatmap_rgb, 0.4, 0)), None

    except Exception as e:
        return None, f"XAI Logic Error: {e}"

def generate_saliency(session, image, pred_class):
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])
    saliency = np.zeros((224, 224), dtype=np.float32)
    step = 16
    for y in range(0, 224, step):
        for x in range(0, 224, step):
            occ = base_batch.copy()
            occ[0, y:y+step, x:x+step, :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            saliency[y:y+step, x:x+step] = base_score - score
    saliency = np.maximum(saliency, 0) / (saliency.max() + 1e-10)
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.blend(image.convert('RGB'), Image.fromarray(heatmap).resize(image.size), alpha=0.5)

# --- UI MAIN ---
st.markdown('<div class="neuro-header"><div class="neuro-logo">NEUROSCAN AI / DIAGNOSTIC</div></div>', unsafe_allow_html=True)
col_h, col_u = st.columns([1, 1])
with col_h:
    st.markdown('<div class="hero-title">BRAIN <span style="color:#63B3ED">TUMOR</span> ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("v3.7 Stable XAI: Functional Mapping & Dimension Safety")
with col_u:
    onnx_sess = load_onnx_session()
    uploaded_files = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    if st.button(f"EXECUTE ANALYSIS ({len(uploaded_files)} SCANS)"):
        all_results = []
        t_start = time.time()
        MODEL_PATH = 'best_resnet_20260307-162330.h5'
        
        for f in uploaded_files:
            img = Image.open(f).convert('RGB')
            batch = preprocess(img)
            out = onnx_sess.run(None, {onnx_sess.get_inputs()[0].name: batch})[0]
            pred_idx = int(np.argmax(out[0]))
            
            with st.spinner(f"Analyzing {f.name}..."):
                saliency_img = generate_saliency(onnx_sess, img, pred_idx)
                
                gradcam_img = None
                gc_error = None
                if os.path.exists(MODEL_PATH):
                    gradcam_img, gc_error = generate_gradcam(img, MODEL_PATH)
                else:
                    gc_error = "Model .h5 not found."
            
            all_results.append({
                'image': img, 'saliency': saliency_img, 'gradcam': gradcam_img, 'error': gc_error,
                'filename': f.name, 'label': {0:"Glioma", 1:"Meningioma", 2:"No Tumor", 3:"Pituitary"}.get(pred_idx, "Unknown"),
                'confidence': float(np.max(out[0])) * 100
            })
        
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"<div class='stat-box'>SCANS<br><h2>{len(all_results)}</h2></div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='stat-box'>AVG ACC<br><h2>{np.mean([r['confidence'] for r in all_results]):.1f}%</h2></div>", unsafe_allow_html=True)
        s3.markdown(f"<div class='stat-box'>TIME<br><h2>{time.time()-t_start:.1f}s</h2></div>", unsafe_allow_html=True)

        for res in all_results:
            is_tumor = res['label'] != "No Tumor"
            st.markdown(f'<div class="scan-result {"tumor" if is_tumor else ""}">{res["label"].upper()} - {res["filename"]} ({res["confidence"]:.1f}%)</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.image(res['image'], caption="Original MRI", use_container_width=True)
            with c2: 
                if res['gradcam']: 
                    st.image(res['gradcam'], caption="Grad-CAM Focus", use_container_width=True)
                else: 
                    st.error(f"XAI Engine Error: {res['error']}")
            with c3: st.image(res['saliency'], caption="Saliency Map", use_container_width=True)

        st.download_button("DOWNLOAD PDF REPORT", create_pdf(all_results), file_name=f"NeuroScan_Report_{datetime.now().strftime('%H%M')}.pdf")
else:
    st.markdown("<center><br><br><div style='opacity:0.3'>Awaiting MRI scan input...</div></center>", unsafe_allow_html=True)
