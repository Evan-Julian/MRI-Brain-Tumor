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
    page_title="NeuroScan AI | Multi-Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUNGSI GENERATE PDF (Updated for Multi-Report) ---
def create_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 15, "NEUROSCAN AI - MULTI-SCAN REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    for i, res in enumerate(results):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"SCAN #{i+1}: {res['filename']}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Diagnosis: {res['label'].upper()}", ln=True)
        pdf.cell(0, 8, f"Confidence: {res['confidence']:.2f}%", ln=True)
        pdf.ln(5)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "Disclaimer: All results are AI-generated for research purposes. Final decisions must be made by qualified medical staff.")
    return pdf.output(dest='S').encode('latin-1')

# --- INJEKSI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; background-color: #0B0E14 !important; color: #FFFFFF !important; }
    .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
    .result-card {
        background-color: #161B22 !important;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        margin-bottom: 15px;
        transition: 0.3s;
    }
    .result-card:hover { border-color: #58A6FF; }
    .stButton>button {
        width: 100%; border-radius: 12px; padding: 12px;
        background: linear-gradient(135deg, #238636 0%, #2EA043 100%) !important;
        color: white !important; font-weight: 700; text-transform: uppercase;
    }
    .info-pill {
        background-color: #1F2937; padding: 5px 12px; border-radius: 50px;
        font-size: 11px; border: 1px solid #374151; display: inline-block; margin-right: 5px;
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
    st.markdown("<h1 style='font-size:26px; font-weight:800; color:#58A6FF;'>NEUROSCAN AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B949E; font-size:12px;'>Multi-Scan Diagnostic v1.5</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.success("● Engine: ResNet50 Active")
    st.info("● Batch Processing Enabled")
    if st.button("REBOOT SYSTEM"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN CONTENT ---
st.markdown("<h1 style='font-weight:800; font-size:40px; letter-spacing:-1.5px;'>Neuro-Batch Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""<div><span class='info-pill'>v1.5 Enterprise</span><span class='info-pill'>ResNet-50</span><span class='info-pill'>Multi-Upload</span></div>""", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("FATAL ERROR: 'resnet_model.onnx' not found.")
else:
    # Upload multiple images
    uploaded_files = st.file_uploader("Upload patient MRI scans (Multiple files supported)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        analyze_btn = st.button(f'🚀 ANALYZE {len(uploaded_files)} SCANS NOW')
        
        if analyze_btn:
            all_results = []
            cols = st.columns(2) # Tampilan grid 2 kolom untuk hasil
            
            for idx, uploaded_file in enumerate(uploaded_files):
                start_time = time.time()
                image = Image.open(uploaded_file).convert('RGB')
                
                # Preprocessing
                img_224 = image.resize((224, 224))
                img_array = np.array(img_224).astype('float32')
                img_array = img_array[:, :, ::-1] # BGR
                img_array[:, :, 0] -= 103.939
                img_array[:, :, 1] -= 116.779
                img_array[:, :, 2] -= 123.68
                img_batch = np.expand_dims(img_array, axis=0)

                # Inference
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: img_batch})[0]
                
                inference_speed = (time.time() - start_time) * 1000
                CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
                pred_idx = int(np.argmax(output[0]))
                confidence = float(np.max(output[0])) * 100
                label = CLASS_NAMES.get(pred_idx, "Unknown")
                
                # Simpan hasil
                res_data = {
                    'filename': uploaded_file.name,
                    'label': label,
                    'confidence': confidence,
                    'speed': inference_speed
                }
                all_results.append(res_data)

                # Tampilkan di UI secara Grid
                with cols[idx % 2]:
                    st.markdown(f"""
                        <div class="result-card">
                            <p style="font-size:11px; color:#8B949E; margin:0;">FILE: {uploaded_file.name}</p>
                            <h3 style="color:#58A6FF; margin:5px 0;">{label.upper()}</h3>
                            <p style="font-size:14px; margin:0;">Confidence: <b>{confidence:.2f}%</b></p>
                            <small style="color:#484F58;">Processed in {inference_speed:.1f}ms</small>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
            
            # Global Report Download
            st.markdown("---")
            pdf_data = create_pdf(all_results)
            st.download_button(
                label="📄 DOWNLOAD BATCH DIAGNOSTIC REPORT (PDF)",
                data=pdf_data,
                file_name=f"NeuroScan_Batch_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("""
            <div style='text-align:center; padding: 80px 20px; border: 2px dashed #30363D; border-radius:20px; background-color: #161B22; margin-top:30px;'>
                <p style='color:#8B949E; font-size:16px;'>Awaiting Multi-Scan Sequence...</p>
                <p style='color:#484F58; font-size:12px;'>Select one or more MRI files to begin batch diagnostic</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><br><div style='text-align:center; border-top:1px solid #30363D; padding-top:20px;'><p style='color:#484F58; font-size:10px;'>NEUROSCAN v1.5 | BATCH ENGINE | IEEE COMPLIANT</p></div>", unsafe_allow_html=True)
