import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Brain Tumor Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS UNTUK TAMPILAN MODERN ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-tag {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (ONNX ONLY) ---
@st.cache_resource
def load_onnx_model():
    try:
        import onnxruntime as ort
        model_path = 'resnet_model.onnx'
        if not os.path.exists(model_path):
            return None
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491214.png", width=100)
    st.title("Navigation")
    st.info("Aplikasi ini menggunakan model **ResNet50** yang dioptimasi dengan ONNX untuk klasifikasi tumor otak melalui citra MRI.")
    
    st.markdown("---")
    st.subheader("Model Info")
    st.write("✅ Format: ONNX Runtime")
    st.write("✅ Arsitektur: ResNet50")
    st.write("✅ Dataset: Figshare, SARTAJ, Br35H")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# --- MAIN CONTENT ---
st.title("🧠 AI Brain Tumor Diagnostic System")
st.write("Upload hasil scan MRI pasien untuk melakukan analisis deteksi dini.")

session = load_onnx_model()

if session is None:
    st.error("❌ Sistem gagal memuat model. Pastikan 'resnet_model.onnx' tersedia di direktori.")
else:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📂 Input Citra MRI")
        uploaded_file = st.file_uploader("Drop image here or browse", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Gambar terpilih', use_container_width=True)
            
            analyze_btn = st.button('🚀 Mulai Analisis Otomatis')

    with col2:
        st.subheader("📊 Hasil Diagnosis")
        
        if uploaded_file and analyze_btn:
            with st.spinner('AI sedang menganalisis pola citra...'):
                try:
                    # Preprocessing
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # BGR
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_batch = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]
                    
                    # Logika Label
                    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, "Unknown")

                    # Tampilan Hasil Utama
                    st.markdown(f"""
                        <div class="result-card">
                            <p style="color: #666; margin-bottom: 5px;">Klasifikasi Terdeteksi:</p>
                            <h2 style="color: {'#28a745' if 'No Tumor' in label else '#dc3545'};">{label}</h2>
                            <p style="font-size: 1.1em;">Tingkat Keyakinan: <b>{confidence:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    if "No Tumor" in label:
                        st.balloons()
                    
                    st.markdown("---")
                    st.write("**Probabilitas Detail:**")
                    
                    # Visualisasi Bar untuk semua kelas
                    for i, prob in enumerate(output[0]):
                        name = CLASS_NAMES.get(i, f"Kelas {i}")
                        val = max(0.0, min(float(prob), 1.0))
                        st.write(f"{name} ({val*100:.1f}%)")
                        st.progress(val)

                except Exception as e:
                    st.error(f"Terjadi kesalahan teknis: {e}")
        else:
            st.info("Silakan unggah gambar dan klik tombol analisis untuk melihat hasil.")

# --- FOOTER ---
st.markdown("---")
st.caption("Aplikasi ini merupakan bagian dari Proyek Skripsi. Hasil prediksi bersifat informatif dan harus dikonfirmasi oleh tenaga medis profesional.")
