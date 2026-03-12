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

# --- INJEKSI CSS CUSTOM (MINIMALIST CYBER DARK) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@400;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #05070A !important;
        color: #E6EDF3 !important;
    }

    /* Sidepanel Font Refinement */
    [data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363D;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 13px;
        line-height: 1.6;
        color: #8B949E !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3 {
        letter-spacing: 0.5px;
    }

    /* Result Card Glassmorphism */
    .result-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(88, 166, 255, 0.15);
        margin-bottom: 20px;
        transition: 0.3s ease;
    }
    .result-card:hover {
        border-color: #58A6FF;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.1);
    }

    /* Numeric Stat Boxes */
    .stat-box {
        background: #0D1117;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .stat-box h2 {
        font-family: 'Inter', sans-serif;
        color: #58A6FF !important;
        font-weight: 700 !important;
    }
    .stat-box small {
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8B949E;
        font-weight: 600;
    }

    /* CSS Icons Style */
    .icon-status {
        height: 10px;
        width: 10px;
        background-color: #238636;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px #238636;
    }

    /* Primary Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 14px;
        background: #1F6FEB !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border: none;
        transition: 0.2s;
        font-size: 14px;
    }
    .stButton>button:hover {
        background: #388BFD !important;
        box-shadow: 0 0 20px rgba(31, 111, 235, 0.3);
    }

    /* File Uploader Custom Text */
    [data-testid="stFileUploadDropzone"] div div span {
        font-size: 14px;
        color: #8B949E !important;
    }

    h1, h2, h3 { letter-spacing: -1px; font-weight: 800 !important; }
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
    st.markdown("<h1 style='color:#58A6FF; font-size: 24px;'>NEUROSCAN</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-15px; font-weight: 600;'>System Diagnostic v1.7</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### SYSTEM LOGS")
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <span class="icon-status"></span> <span style="font-size:13px; color:#E6EDF3;">Inference Engine: Active</span>
        </div>
        <div style="margin-bottom: 20px;">
            <span class="icon-status" style="background-color:#58A6FF; box-shadow: 0 0 8px #58A6FF;"></span> <span style="font-size:13px; color:#E6EDF3;">Backend: ONNX Optimized</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #8B949E !important; font-weight: 700;'>Technical Parameters</p>", unsafe_allow_html=True)
    st.markdown("- Arch: ResNet50<br>- Input: 224x224 RGB<br>- Threads: Multi-Core", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RESET ENGINE"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN HEADER ---
st.markdown("<h1>Batch Neural Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8B949E; font-size:15px;'>Automated multi-sequence MRI classification system for neuro-pathological research.</p>", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("SYSTEM ERROR: resnet_model.onnx not found.")
else:
    uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f'RUN ANALYSIS ON {len(uploaded_files)} SCANS'):
            all_results = []
            container = st.container()
            with container:
                st.markdown("### ANALYSIS METRICS")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                results_grid = st.container()
                total_start = time.time()
                
                with results_grid:
                    ui_cols = st.columns(3)
                    
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

                        output = session.run(None, {session.get_inputs()[0].name: img_batch})[0]
                        
                        CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
                        pred_idx = int(np.argmax(output[0]))
                        confidence = float(np.max(output[0])) * 100
                        label = CLASS_NAMES.get(pred_idx, "Unknown")
                        
                        all_results.append({
                            'filename': uploaded_file.name,
                            'label': label,
                            'confidence': confidence
                        })

                        with ui_cols[idx % 3]:
                            st.markdown(f"""
                                <div class="result-card">
                                    <p style="font-size:10px; color:#8B949E; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Scan Sequence {idx+1}</p>
                                    <h3 style="color:#58A6FF; margin:0; font-size:18px;">{label.upper()}</h3>
                                    <p style="font-family:'Inter'; font-size:14px; margin-top:4px;">Accuracy: {confidence:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.image(image, use_container_width=True)

                total_time = time.time() - total_start
                
                # Update Stats with Inter Font
                stat_col1.markdown(f"<div class='stat-box'><small>SCANS</small><h2>{len(all_results)}</h2></div>", unsafe_allow_html=True)
                stat_col2.markdown(f"<div class='stat-box'><small>AVG CONF</small><h2>{np.mean([r['confidence'] for r in all_results]):.1f}%</h2></div>", unsafe_allow_html=True)
                stat_col3.markdown(f"<div class='stat-box'><small>LATENCY</small><h2>{total_time:.2f}s</h2></div>", unsafe_allow_html=True)

            st.markdown("---")
            pdf_data = create_pdf(all_results)
            st.download_button(
                label="EXPORT DIAGNOSTIC REPORT (PDF)",
                data=pdf_data,
                file_name=f"NeuroScan_Report_{datetime.now().strftime('%H%M')}.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("""
            <div style='text-align:center; padding: 100px 20px; border: 1px solid #30363D; border-radius:20px; background-color: #0D1117; margin-top:40px;'>
                <p style='color:#8B949E; font-size:14px; font-weight:400;'>SYSTEM IDLE: Waiting for image input...</p>
                <p style='color:#484F58; font-size:11px; text-transform:uppercase; letter-spacing:1px;'>Select MRI files to initiate batch sequence</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><br><div style='text-align:center; opacity:0.3; font-size:10px; font-weight:700; letter-spacing:2px; color:#8B949E;'>NEUROSCAN ENTERPRISE v1.7 | RESEARCH CORE</div>", unsafe_allow_html=True)
