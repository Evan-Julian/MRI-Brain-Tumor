import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INJEKSI CSS UNTUK UI PROFESIONAL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background and Cards */
    .main {
        background-color: #f8f9fa;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }

    /* Card Style */
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    /* Analysis Result Header */
    .result-header {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .confidence-value {
        font-size: 16px;
        color: #495057;
        font-weight: 500;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 12px 24px;
        background-color: #1a73e8;
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #1557b0;
        box-shadow: 0 4px 12px rgba(26,115,232,0.3);
    }

    /* Sidebar info box */
    .sidebar-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1a73e8;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_onnx_model():
    try:
        import onnxruntime as ort
        model_path = 'resnet_model.onnx'
        if not os.path.exists(model_path): return None
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except:
        return None

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("<h2 style='color:#1a73e8;'>NeuroScan AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:#6c757d;'>Medical Imaging Intelligence</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="sidebar-box">
            <small style="color:#1a73e8; font-weight:bold;">SYSTEM STATUS</small><br>
            <span style="font-size:14px;">ResNet50 Engine Active</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("**Technical Specs**")
    st.caption("Architecture: ResNet50")
    st.caption("Accelerator: ONNX Runtime")
    st.caption("Dataset: Multi-source MRI")
    
    if st.button("System Reset"):
        st.cache_resource.clear()
        st.rerun()

# --- HEADER ---
st.markdown("<h1 style='font-weight:700;'>MRI Diagnostic Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6c757d; font-size:18px;'>Analisis klasifikasi citra MRI otomatis berbasis Deep Learning.</p>", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("SYSTEM ERROR: Model file 'resnet_model.onnx' not found.")
else:
    # Grid Layout
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("<h4 style='font-weight:600; margin-bottom:20px;'>Upload Patient Records</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            analyze_btn = st.button('Run Diagnostic Analysis')

    with col_output:
        st.markdown("<h4 style='font-weight:600; margin-bottom:20px;'>Diagnostic Report</h4>", unsafe_allow_html=True)
        
        if uploaded_file and analyze_btn:
            with st.spinner('Processing neural network...'):
                try:
                    # Preprocessing
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # BGR
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_batch = np.expand_dims(img_array, axis=0)

                    # Prediction
                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]
                    
                    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, "Unknown")
                    
                    # Warna indikator
                    res_color = "#28a745" if "No Tumor" in label else "#d93025"

                    # Card Result
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="result-header">Diagnosis Result</div>
                            <div class="result-value" style="color: {res_color};">{label.upper()}</div>
                            <div class="confidence-value">Confidence Level: {confidence:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<p style='font-weight:600; font-size:14px; margin-top:30px;'>PROBABILITY DISTRIBUTION</p>", unsafe_allow_html=True)
                    
                    # Progress bars
                    for i, prob in enumerate(output[0]):
                        name = CLASS_NAMES.get(i, f"Class {i}")
                        val = max(0.0, min(float(prob), 1.0))
                        
                        cols = st.columns([3, 7])
                        cols[0].caption(name)
                        cols[1].progress(val)

                except Exception as e:
                    st.error(f"Inference Error: {e}")
        else:
            st.markdown("""
                <div style='text-align:center; padding: 100px 20px; border: 2px dashed #e9ecef; border-radius:12px;'>
                    <p style='color:#adb5bd;'>Waiting for patient data upload...</p>
                </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='border-top: 1px solid #e9ecef; padding-top: 20px; color: #adb5bd; font-size: 12px; text-align: center;'>
        NeuroScan AI v1.0 | IEEE Journal Documentation Reference | For Research Use Only
    </div>
""", unsafe_allow_html=True)
