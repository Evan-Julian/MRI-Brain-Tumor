import os
import streamlit as st
import numpy as np
import cv2
import time
import tempfile
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

# --- FUNGSI GENERATE PDF (Dengan Integrasi Gambar) ---
def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header Dokumen
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(26, 115, 232) # Blue Accent
    pdf.cell(0, 15, "NEUROSCAN AI - CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Analysis Performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Iterasi Hasil untuk dicetak ke PDF
    for idx, res in enumerate(results):
        # Tambahkan halaman baru setiap 2 hasil agar tidak terlalu padat
        if idx > 0 and idx % 2 == 0: 
            pdf.add_page()

        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"SEQUENCE #{idx+1}: {res['filename']}", ln=True)
        
        # Logika Penyisipan Gambar
        # Kita perlu simpan sementara gambar PIL ke file agar fpdf bisa membacanya
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            res['image'].save(tmpfile.name)
            curr_y = pdf.get_y()
            # Letakkan gambar di sebelah kiri (lebar 60mm)
            pdf.image(tmpfile.name, x=10, y=curr_y, w=60)
            
        # Letakkan detail teks di sebelah kanan gambar
        pdf.set_xy(80, curr_y + 10)
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 10, f"RESULT: {res['label'].upper()}", ln=True)
        
        pdf.set_x(80)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Confidence Score: {res['confidence']:.2f}%", ln=True)
        
        # Reset Y ke bawah gambar untuk sequence berikutnya
        pdf.set_y(curr_y + 65)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Garis pemisah horizontal
        pdf.ln(10)
        
        # Cleanup file temporary
        if os.path.exists(tmpfile.name):
            os.remove(tmpfile.name)

    # Disclaimer di akhir dokumen
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, "DISCLAIMER: This automated report is generated using a Deep Learning ResNet50 model. The results provided are for research and documentation purposes only. A final diagnosis should always be conducted by a qualified radiologist or medical professional.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- INJEKSI CSS CUSTOM (MINIMALIST ENTERPRISE DARK) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #05070A !important;
        color: #E6EDF3 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #010409 !important;
        border-right: 1px solid #30363D;
    }
    
    .result-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(88, 166, 255, 0.15);
        margin-bottom: 15px;
    }

    .stat-box {
        background: #0D1117;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .stat-box h2 {
        font-family: 'Inter', sans-serif;
        color: #58A6FF !important;
        font-weight: 700 !important;
    }

    .status-dot {
        height: 8px; width: 8px; background-color: #238636;
        border-radius: 50%; display: inline-block; margin-right: 5px;
    }

    .stButton>button {
        width: 100%; border-radius: 8px; padding: 12px;
        background: #1F6FEB !important; color: white !important;
        font-weight: 600; text-transform: uppercase; border: none; letter-spacing: 1px;
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
    st.markdown("<h1 style='color:#58A6FF; font-size: 22px;'>NEUROSCAN</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-15px; font-size:12px; color:#8B949E;'>Diagnostic Core v1.8</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size:11px; font-weight:700; color:#8B949E; text-transform:uppercase;'>System Status</p>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:13px;'><span class="status-dot"></span> Neural Engine Active</div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RELOAD SYSTEM"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN CONTENT ---
st.title("Batch Neural Analysis")
st.markdown("<p style='color:#8B949E;'>Automated multi-sequence MRI classification with integrated PDF visual reporting.</p>", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.error("SYSTEM ERROR: resnet_model.onnx was not found in the root directory.")
else:
    uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f'RUN DIAGNOSTIC ON {len(uploaded_files)} SCANS'):
            all_results = []
            total_start = time.time()
            
            # Tampilan Grid UI
            ui_cols = st.columns(3)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert('RGB')
                
                # Preprocessing & Inference
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized).astype('float32')
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
                    'image': image, # Simpan original PIL image untuk PDF
                    'filename': uploaded_file.name,
                    'label': label,
                    'confidence': confidence
                })

                with ui_cols[idx % 3]:
                    st.markdown(f"""
                        <div class="result-card">
                            <h3 style="color:#58A6FF; margin:0; font-size:18px;">{label.upper()}</h3>
                            <p style="font-family:'Inter'; font-size:14px; margin-top:4px; color:#8B949E;">{confidence:.1f}% Accuracy</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(image, use_container_width=True)

            total_time = time.time() - total_start
            
            # Dashboard Metrik
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.markdown(f"<div class='stat-box'><small style='color:#8B949E;'>TOTAL SCANS</small><h2>{len(all_results)}</h2></div>", unsafe_allow_html=True)
            m_col2.markdown(f"<div class='stat-box'><small style='color:#8B949E;'>AVG CONFIDENCE</small><h2>{np.mean([r['confidence'] for r in all_results]):.1f}%</h2></div>", unsafe_allow_html=True)
            m_col3.markdown(f"<div class='stat-box'><small style='color:#8B949E;'>TOTAL LATENCY</small><h2>{total_time:.2f}s</h2></div>", unsafe_allow_html=True)

            st.markdown("---")
            # PDF Generation
            with st.spinner("Compiling full medical report with images..."):
                pdf_data = create_pdf(all_results)
                st.download_button(
                    label="📥 DOWNLOAD FULL CLINICAL REPORT (PDF)",
                    data=pdf_data,
                    file_name=f"NeuroScan_FullReport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.markdown("<div style='text-align:center; padding: 100px 20px; border: 1px solid #30363D; border-radius:20px; background-color: #0D1117; margin-top:40px;'><p style='color:#8B949E;'>Awaiting patient scan input sequence...</p></div>", unsafe_allow_html=True)

# Footer Minimalis
st.markdown("<br><div style='text-align:center; opacity:0.3; font-size:10px; letter-spacing:2px; font-weight:700;'>NEUROSCAN ENTERPRISE v1.8 | IMAGING DIAGNOSTIC UNIT</div>", unsafe_allow_html=True)
