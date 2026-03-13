```python
import os
import streamlit as st
import numpy as np
import time
import tempfile
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    body {
        background:#060810;
        color:#C9D1E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# MODEL INFO PANEL
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Model Information")
    st.write("Architecture: **ResNet50**")
    st.write("Framework: **ONNX Runtime**")
    st.write("Input Size: **224x224 MRI**")

    st.write("Classes:")
    st.write("- Glioma")
    st.write("- Meningioma")
    st.write("- No Tumor")
    st.write("- Pituitary")

# ─────────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────────

def safe(text):
    return text.encode("latin-1", errors="replace").decode("latin-1")


def create_pdf(results):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "NEUROSCAN AI - CLINICAL REPORT", ln=True, align="C")

    pdf.ln(10)

    for idx, res in enumerate(results):

        if idx > 0 and idx % 2 == 0:
            pdf.add_page()

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, safe(f"{idx+1}. {res['filename']}"), ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

            res["image"].save(tmp.name)

            y = pdf.get_y()
            pdf.image(tmp.name, x=10, y=y, w=55)

        pdf.set_xy(75, y + 6)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, safe(res["label"]), ln=True)

        pdf.set_x(75)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 7, safe(f"Confidence: {res['confidence']:.2f}%"), ln=True)

        pdf.set_y(y + 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        os.remove(tmp.name)

    return pdf.output(dest="S").encode("latin-1")


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():

    import onnxruntime as ort

    path = "resnet_model.onnx"

    if not os.path.exists(path):
        return None

    return ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )


# ─────────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────────

def preprocess(image):

    img = image.resize((224, 224))

    arr = np.array(img).astype("float32")

    arr = arr[:, :, ::-1]

    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68

    return np.expand_dims(arr, 0)


# ─────────────────────────────────────────────
# SALIENCY MAP
# ─────────────────────────────────────────────

def generate_saliency(session, image, pred_class):

    inp_name = session.get_inputs()[0].name

    base_batch = preprocess(image)

    base_score = float(
        session.run(None, {inp_name: base_batch})[0][0][pred_class]
    )

    H, W = 224, 224
    step = 16

    saliency = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, step):
        for x in range(0, W, step):

            occ = base_batch.copy()

            occ[0, y:y+step, x:x+step, :] = 0

            score = float(
                session.run(None, {inp_name: occ})[0][0][pred_class]
            )

            drop = base_score - score

            saliency[y:y+step, x:x+step] = drop

    saliency = np.maximum(saliency, 0)

    if saliency.max() > 0:
        saliency /= saliency.max()

    heatmap = (saliency * 255).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap).resize(image.size)

    heatmap_rgb = Image.merge(
        "RGB",
        (heatmap_img, heatmap_img, heatmap_img)
    )

    return Image.blend(
        image.convert("RGB"),
        heatmap_rgb,
        alpha=0.5
    )


# ─────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🧠 NeuroScan AI")

st.caption("Deep Learning Brain Tumor Detection System")


# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────

session = load_model()

if session is None:
    st.error("resnet_model.onnx not found.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload MRI images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if uploaded_files:

    if st.button("Run Analysis"):

        results = []

        start = time.time()

        for f in uploaded_files:

            img = Image.open(f).convert("RGB")

            batch = preprocess(img)

            inp = session.get_inputs()[0].name

            out = session.run(None, {inp: batch})[0]

            probs = out[0]

            pred = int(np.argmax(probs))

            conf = float(np.max(probs)) * 100

            saliency_img = generate_saliency(session, img, pred)

            focus_score = float(
                np.mean(np.array(saliency_img)) / 255 * 100
            )

            results.append(
                {
                    "image": img,
                    "saliency": saliency_img,
                    "filename": f.name,
                    "label": CLASS_NAMES[pred],
                    "confidence": conf,
                    "probs": probs.tolist(),
                    "focus": focus_score,
                }
            )

        latency = time.time() - start


        # ─────────────────────────
        # SYSTEM STATS
        # ─────────────────────────

        st.subheader("System Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric("Scans", len(results))

        c2.metric(
            "Average Confidence",
            f"{np.mean([r['confidence'] for r in results]):.1f}%"
        )

        c3.metric("Latency", f"{latency:.2f}s")

        st.divider()


        # ─────────────────────────
        # RESULTS
        # ─────────────────────────

        for r in results:

            st.subheader(r["filename"])

            col1, col2 = st.columns(2)

            with col1:
                st.image(r["image"], caption="Original MRI")

            with col2:
                st.image(r["saliency"], caption="Saliency Map")

            st.markdown(f"### Prediction: **{r['label']}**")

            st.write(f"Confidence: **{r['confidence']:.2f}%**")


            # CONFIDENCE INTERPRETATION

            if r["confidence"] > 90:
                st.success("High certainty prediction")

            elif r["confidence"] > 70:
                st.warning("Moderate certainty prediction")

            else:
                st.error("Low certainty prediction")


            # TUMOR WARNING

            if r["label"] != "No Tumor":
                st.error("⚠ Possible tumor detected. Clinical review recommended.")
            else:
                st.success("No tumor pattern detected.")


            # CLASS RANKING

            st.markdown("#### Class Ranking")

            ranking = sorted(
                zip(CLASS_NAMES.values(), r["probs"]),
                key=lambda x: x[1],
                reverse=True,
            )

            for name, prob in ranking:
                st.write(f"{name}: {prob*100:.2f}%")


            # PROBABILITY CHART

            fig, ax = plt.subplots()

            ax.bar(CLASS_NAMES.values(), r["probs"])

            ax.set_ylabel("Probability")

            ax.set_title("Class Probability Distribution")

            st.pyplot(fig)


            # XAI FOCUS SCORE

            st.write(f"XAI Focus Score: **{r['focus']:.1f}%**")


            # AI EXPLANATION

            explanation = f"""
            The AI model identified patterns in the MRI scan
            consistent with **{r['label']}**.

            The highlighted saliency regions indicate areas
            that strongly influenced the neural network decision.
            """

            st.info(explanation)

            st.divider()


        # ─────────────────────────
        # PDF DOWNLOAD
        # ─────────────────────────

        pdf_bytes = create_pdf(results)

        st.download_button(
            "Download Clinical Report",
            pdf_bytes,
            file_name="neuroscan_report.pdf"
        )

else:

    st.info("Upload MRI scans to begin analysis.")
```
