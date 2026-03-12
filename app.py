import os
import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="NeuroScan AI | Enterprise Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUNGSI GENERATE PDF (Batch Report) ---
def create_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 15, "NEUROSCAN AI - BATCH DIAGNOSTIC SUMMARY", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Summary Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(80, 10, "Filename", 1, 0, 'C', True)
    pdf.cell(60, 10, "Diagnosis", 1, 0, 'C', True)
    pdf.cell(50, 10, "Confidence", 1, 1, 'C', True)

    pdf.set_font("Arial", size=10)
    for res in results:
        pdf.cell(80, 10, str(res['filename']), 1)
        pdf.cell(60, 10, str(res['label']), 1)
        pdf.cell(50, 10, f"{res['confidence']:.2f}%", 1, 1)

    pdf.ln(15)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "CONFIDENTIAL: This document is generated for medical research documentation. ResNet50 Architecture Inference.")
    return pdf.output(dest='S').encode('latin-1')

# --- INJEKSI CSS CUSTOM (CYBER DARK UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #05070A !important;
        color: #E6EDF3 !important;
    }

    /* Result Card Glassmorphism */
    .result-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(88, 166, 255, 0.2);
        margin-bottom: 20px;
        transition: 0.4s ease;
    }
    .result-card:hover {
        border-color: #58A6FF;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
        transform: translateY(-5px);
    }

    /* Stat Box Header */
    .stat-box {
        background: linear-gradient(145deg, #0D1117, #161B22);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        text-align: center;
    }

    /* Primary Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(90deg, #1A73E8 0%, #0052D1 100%) !important;
        color: white !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: none;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363D;
    }

    /* Progress Bar Neon */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #58A6FF, #00D4FF) !important;
    }

    /* Titles */
    h1, h2, h3 {
        letter-spacing: -1px;
        font-weight: 800 !important;
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
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#58A6FF;'>NEUROSCAN</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B949E; font-size:12px; margin-top:-15px;'>Enterprise Medical AI v1.6</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🛠️ Hardware Info")
    st.caption("Engine: ONNX Runtime")
    st.caption("Backend: ResNet50 CPU-Inference")
    
    st.markdown("---")
    if st.button("🔄 REFRESH ENGINE"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN HEADER ---
st.markdown("<h1>Batch Neural Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8B949E; font-size:16px;'>Clinical-grade automated tumor classification for multiple MRI sequences.</p>", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("FATAL: resnet_model.onnx not detected.")
else:
    # Uploader Section
    uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f'⚡ EXECUTE DIAGNOSTIC ON {len(uploaded_files)} FILES'):
            all_results = []
            
            # --- START BATCH PROCESS ---
            container = st.container()
            with container:
                st.markdown("### 📊 Inference Statistics")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                placeholder_grid = st.empty()
                results_grid = st.container()
                
                total_start = time.time()
                
                with results_grid:
                    ui_cols = st.columns(3) # Grid 3 Kolom agar lebih compact
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # Preprocessing
                        img_224 = image.resize((224, 224))
                        img_array = np.array(img_224).astype('float32')
                        img_array = img_array[:, :, ::-1] # BGR
                        img_array[:, :, 0] -= 103.939
                        img_array[:, :, 1] -= 116.779
                        img_array[:, :, 2] -= 123.68
                        img_batch = np.expand_dims(img_array, axis=0)

                        # Run AI
                        input_name = session.get_inputs()[0].name
                        output = session.run(None, {input_name: img_batch})[0]
                        
                        CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
                        pred_idx = int(np.argmax(output[0]))
                        confidence = float(np.max(output[0])) * 100
                        label = CLASS_NAMES.get(pred_idx, "Unknown")
                        
                        all_results.append({
                            'filename': uploaded_file.name,
                            'label': label,
                            'confidence': confidence
                        })

                        # Grid UI
                        with ui_cols[idx % 3]:
                            st.markdown(f"""
                                <div class="result-card">
                                    <small style="color:#8B949E;">SCAN #{idx+1}</small>
                                    <h3 style="color:#58A6FF; margin:5px 0; font-size:20px;">{label.upper()}</h3>
                                    <p style="font-size:13px; margin:0;">Accuracy: <b>{confidence:.1f}%</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.image(image, use_container_width=True)

                total_time = time.time() - total_start
                
                # Update Stats
                stat_col1.markdown(f"<div class='stat-box'><small>TOTAL FILES</small><h2>{len(all_results)}</h2></div>", unsafe_allow_html=True)
                stat_col2.markdown(f"<div class='stat-box'><small>AVG CONFIDENCE</small><h2>{np.mean([r['confidence'] for r in all_results]):.1f}%</h2></div>", unsafe_allow_html=True)
                stat_col3.markdown(f"<div class='stat-box'><small>TOTAL TIME</small><h2>{total_time:.2f}s</h2></div>", unsafe_allow_html=True)

            # --- REPORT SECTION ---
            st.markdown("---")
            pdf_data = create_pdf(all_results)
            st.download_button(
                label="📥 DOWNLOAD CLINICAL SUMMARY REPORT",
                data=pdf_data,
                file_name=f"Batch_Report_{datetime.now().strftime('%H%M')}.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("""
            <div style='text-align:center; padding: 120px 20px; border: 2px dashed #30363D; border-radius:30px; background-color: #0D1117; margin-top:50px;'>
                <h2 style='color:#58A6FF; margin-bottom:10px;'>Ready for Multi-Inference</h2>
                <p style='color:#8B949E;'>Upload patient MRI scans to initiate the neural classification sequence.</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><br><div style='text-align:center; opacity:0.3; font-size:10px; letter-spacing:2px;'>NEUROSCAN ENTERPRISE EDITION v1.6 | IEEE-754 COMPLIANT | BATCH CORE</div>", unsafe_allow_html=True)
