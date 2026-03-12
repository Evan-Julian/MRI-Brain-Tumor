import os
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

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

# Fungsi untuk Grad-CAM (Menggunakan TF karena butuh akses ke layer gradien)
def generate_gradcam(img_batch, model_h5_path='best_resnet50_model.h5'):
    try:
        # Load model h5 untuk akses layer
        model = tf.keras.models.load_model(model_h5_path)
        last_conv_layer_name = "conv5_block3_out" # Layer terakhir ResNet50
        
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_batch)
            class_channel = preds[:, np.argmax(preds[0])]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        return None

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
                    # Preprocessing
                    img_224 = image.resize((224, 224))
                    img_array = np.array(img_224).astype('float32')
                    img_array = img_array[:, :, ::-1] # BGR
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_batch = np.expand_dims(img_array, axis=0)

                    # 1. Prediksi (ONNX)
                    input_name = session.get_inputs()[0].name
                    output = session.run(None, {input_name: img_batch})[0]
                    pred_idx = int(np.argmax(output[0]))
                    confidence = float(np.max(output[0])) * 100
                    label = CLASS_NAMES.get(pred_idx, "Unknown")

                    # 2. Grad-CAM (Hanya jika file .h5 tersedia di repo)
                    heatmap = None
                    if os.path.exists('best_resnet50_model.h5'):
                        heatmap = generate_gradcam(img_batch, 'best_resnet50_model.h5')

                    # 3. Tampilkan Hasil
                    st.success(f"### Prediksi: **{label}** ({confidence:.2f}%)")
                    
                    if heatmap is not None:
                        # Overlay Heatmap
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        
                        original_img = np.array(img_224)
                        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
                        
                        with col2:
                            st.image(superimposed_img, caption='Grad-CAM (Area Fokus AI)', use_container_width=True)
                            st.info("💡 Area merah menunjukkan bagian yang paling mempengaruhi keputusan AI.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
