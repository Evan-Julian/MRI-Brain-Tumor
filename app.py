import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Set page config
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="centered")

@st.cache_resource
def load_onnx_model():
    try:
        import onnxruntime as ort
        model_path = 'resnet_model.onnx'
        if not os.path.exists(model_path):
            st.error(f"❌ File {model_path} tidak ditemukan")
            return None
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"⚠️ Error ONNX: {e}")
        return None

def generate_gradcam(img_batch, model_h5_path):
    try:
        import tensorflow as tf
        # 1. Load Model H5
        model = tf.keras.models.load_model(model_h5_path)
        
        # 2. Cari layer konvolusi terakhir
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            return None, "Layer konvolusi tidak ditemukan."

        # 3. Buat Grad-Model
        # Kita pastikan mengambil .output secara eksplisit
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            # Jalankan model
            conv_outputs, predictions = grad_model(img_batch)
            
            # --- FIX RADIKAL UNTUK TUPLE ERROR ---
            # Jika output masih berbentuk list/tuple bersarang, bongkar sampai ketemu tensornya
            while isinstance(conv_outputs, (list, tuple)):
                conv_outputs = conv_outputs[0]
            while isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            
            # Ambil skor untuk kelas tertinggi
            class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # 4. Hitung Gradien
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return None, "Gagal menghitung gradien. Pastikan model tidak dalam mode inference-only."

        # Global Average Pooling pada gradien
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 5. Kalkulasi Heatmap
        # conv_outputs[0] untuk menghilangkan dimensi batch
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 6. Normalisasi Heatmap 0 ke 1
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy(), None
        
    except Exception as e:
        return None, f"Detail Error: {str(e)}"

# Konfigurasi Label
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

session = load_onnx_model()

if session:
    st.title("🧠 Klasifikasi Tumor Otak MRI")
    st.markdown("Dashboard Skripsi: **ResNet50 + Grad-CAM Visualizer**")
    st.markdown("---")

    uploaded_file = st.file_uploader("Pilih gambar MRI...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption='Gambar Asli', use_container_width=True)

        if st.button('Mulai Analisis & Visualisasi', use_container_width=True):
            with st.spinner('Sedang menganalisis area tumor...'):
                try:
                    # PREPROCESSING (Sesuai Colab: BGR & Mean Subtraction)
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # RGB -> BGR
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_batch = np.expand_dims(img_array, axis=0)

                    # 1. PREDIKSI UTAMA (Menggunakan ONNX untuk Kecepatan)
                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]
                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, "Unknown")

                    # 2. GRAD-CAM (Visualisasi area fokus AI)
                    model_h5 = 'best_resnet_20260307-162330.h5' 
                    heatmap = None
                    
                    if os.path.exists(model_h5):
                        heatmap, err = generate_gradcam(img_batch, model_h5)
                        if err: 
                            st.error(f"Grad-CAM Error: {err}")
                    else:
                        st.warning(f"⚠️ File model h5 tidak ditemukan.")

                    # 3. TAMPILKAN HASIL
                    st.success(f"### Prediksi: **{label}** ({confidence:.2f}%)")
                    
                    if heatmap is not None:
                        # Resizing dan coloring heatmap
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        
                        # Gabungkan dengan gambar asli (Overlay)
                        original_img = np.array(img_224)
                        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
                        
                        with col2:
                            st.image(superimposed_img, caption='Hasil Grad-CAM', use_container_width=True)
                            st.info("💡 Area merah/kuning menunjukkan bagian yang paling diperhatikan AI.")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
else:
    st.error("Gagal memuat sistem.")
