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

# --- INJEKSI CSS UNTUK UI KONTRAST TINGGI & PROFESIONAL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Memaksa font Inter ke seluruh aplikasi */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Memastikan teks terbaca di mode apapun */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1A1D23 !important;
    }

    /* Container Background */
    [data-testid="stAppViewContainer"] {
        background-color: #F4F7FA;
    }

    /* Card Style - Putih Bersih dengan Shadow Lembut */
    .metric-card {
        background-color: #FFFFFF !important;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        color: #1A1D23 !important;
    }

    .result-header {
        font-size: 12px;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .result-value {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .confidence-value {
        font-size: 16px;
        color: #475569 !important;
        background: #F1F5F9;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }

    /* Tombol Biru Industrial */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 15px;
        background-color: #2563EB;
        color: white !important;
        border: none;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .sidebar-status {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2563EB;
        margin-bottom: 20px;
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
    st.markdown("<h2 style='color:#2563EB; font-weight:800; margin-bottom:0;'>NEUROSCAN</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:12px; margin-top:0;'>PRECISION DIAGNOSTIC v1.0</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="sidebar-status">
            <span style="color:#64748B; font-size:10px; font-weight:700; text-transform:uppercase;">System Status</span><br>
            <span style="color:#1E293B; font-size:14px; font-weight:600;">Engine: ResNet50_ONNX</span><br>
            <span style="color:#22C55E; font-size:12px;">● Online & Optimized</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:14px; font-weight:600; color:#1E293B;'>Clinical Specs</p>", unsafe_allow_html=True)
    st.caption("Training Base: ResNet50")
    st.caption("Target Classes: 4 (Multi-class)")
    st.caption("Input Matrix: 224x224x3")
    
    if st.button("Refresh System"):
        st.cache_resource.clear()
        st.rerun()

# --- HEADER SECTION ---
st.markdown("<h1 style='font-weight:800; color:#1E293B; margin-bottom:5px;'>Automated MRI Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:16px;'>Deep Learning system for Brain Tumor Classification based on MRI Scans.</p>", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("FATAL ERROR: Inference engine (resnet_model.onnx) failed to initialize.")
else:
    # Grid Layout
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("<h4 style='font-weight:700; color:#1E293B;'>Digital Imaging Input</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True, channels="RGB")
            analyze_btn = st.button('GENERATE DIAGNOSTIC REPORT')

    with col_output:
        st.markdown("<h4 style='font-weight:700; color:#1E293B;'>Diagnostic Findings</h4>", unsafe_allow_html=True)
        
        if uploaded_file and analyze_btn:
            with st.spinner('Neural processing...'):
                try:
                    # Preprocessing
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # Konversi RGB ke BGR (Standard ResNet)
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
                    
                    # Warna indikator status (Hijau jika aman, Merah jika terdeteksi)
                    res_color = "#16A34A" if "No Tumor" in label else "#DC2626"

                    # Result Card UI
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="result-header">Diagnosis Classification</div>
                            <div class="result-value" style="color: {res_color};">{label.upper()}</div>
                            <div class="confidence-value">Statistical Confidence: <b>{confidence:.2f}%</b></div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<p style='font-weight:700; font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:1px;'>Class Probabilities</p>", unsafe_allow_html=True)
                    
                    # Progress bars minimalis
                    for i, prob in enumerate(output[0]):
                        name = CLASS_NAMES.get(i, f"Class {i}")
                        val = max(0.0, min(float(prob), 1.0))
                        
                        cols = st.columns([4, 6])
                        cols[0].markdown(f"<span style='font-size:13px; color:#1E293B;'>{name}</span>", unsafe_allow_html=True)
                        cols[1].progress(val)

                except Exception as e:
                    st.error(f"Computation Error: {e}")
        else:
            # Placeholder State
            st.markdown("""
                <div style='text-align:center; padding: 120px 20px; border: 2px dashed #CBD5E1; border-radius:16px; background-color: #F8FAFC;'>
                    <p style='color:#94A3B8; font-size:14px;'>Awaiting patient scan data to initiate diagnostic sequence...</p>
                </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='border-top: 1px solid #E2E8F0; padding-top: 30px; color: #94A3B8; font-size: 11px; text-align: center; letter-spacing: 0.5px;'>
        <b>SYSTEM INFORMATION:</b> NeuroScan v1.0-Release | IEEE Journal Ref: 2026.B.011 | Deep Learning Analysis | Research Purposes Only
    </div>
""", unsafe_allow_html=True)
