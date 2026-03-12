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
        
        # 1. Load Model
        model = tf.keras.models.load_model(model_h5_path, compile=False)
        
        # 2. Identifikasi Layer Konvolusi Terakhir
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            return None, "Layer Conv2D tidak ditemukan."

        # 3. Bangun Grad-Model dengan penanganan output eksplisit
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # 4. Hitung Gradient dengan perlakuan khusus Tensor
        @tf.function
        def get_grads(inputs):
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(inputs)
                # Bongkar tuple jika ada
                if isinstance(conv_out, (list, tuple)): conv_out = conv_out[0]
                if isinstance(preds, (list, tuple)): preds = preds[0]
                
                class_idx = tf.argmax(preds[0])
                loss = preds[:, class_idx]
            return tape.gradient(loss, conv_out), conv_out

        grads, conv_outputs = get_grads(img_batch)

        if grads is None:
            return None, "Gradient bernilai None. Pastikan layer tidak 'frozen'."

        # 5. Global Average Pooling pada gradien
        # Menangani dimensi secara dinamis untuk menghindari error 'None' shape
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 6. Kalkulasi Heatmap
        # conv_outputs[0] adalah (7, 7, 2048)
        # weights adalah (2048,)
        heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 7. Normalisasi Heatmap 0 ke 1
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
                    # PREPROCESSING (Sesuai dataset Anda: BGR & Mean Subtraction)
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # Konversi ke BGR
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_batch = np.expand_dims(img_array, axis=0)

                    # 1. PREDIKSI (ONNX)
                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]
                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, "Unknown")

                    # 2. GRAD-CAM
                    model_h5 = 'best_resnet_20260307-162330.h5' 
                    heatmap = None
                    
                    if os.path.exists(model_h5):
                        heatmap, err = generate_gradcam(img_batch, model_h5)
                        if err: 
                            st.error(f"Grad-CAM Error: {err}")
                    else:
                        st.warning(f"⚠️ File {model_h5} tidak ditemukan.")

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
                            st.info("💡 Area merah menunjukkan bagian yang dideteksi AI sebagai ciri tumor.")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
else:
    st.error("Gagal memuat sistem.")
