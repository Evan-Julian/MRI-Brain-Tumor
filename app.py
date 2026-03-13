```python
import os
import streamlit as st
import numpy as np
import time
import tempfile
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime
import cv2
import matplotlib.pyplot as plt

# ── CONFIGURATION ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp {
    background:#060810;
    color:#C9D1E0;
    font-family:'Syne',sans-serif;
}

.stat-grid{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:15px;
margin-bottom:30px;
}

.stat-item{
background:rgba(10,14,22,0.8);
border:1px solid rgba(99,179,237,0.1);
padding:20px;
border-radius:4px;
}

.stat-value{
font-size:28px;
font-weight:900;
}

.scan-result{
background:rgba(10,14,22,0.8);
border:1px solid rgba(99,179,237,0.1);
border-radius:4px;
padding:15px;
margin-bottom:10px;
}

.scan-result.tumor::before{
content:'';
position:absolute;
left:0;
top:0;
width:4px;
height:100%;
background:#FC8181;
}

.scan-result.invalid::before{
content:'';
position:absolute;
left:0;
top:0;
width:4px;
height:100%;
background:#ecc94b;
}

.finding-box{
background:rgba(255,255,255,0.03);
border-left:3px solid #63B3ED;
padding:10px;
margin-top:10px;
font-size:12px;
}
</style>
""", unsafe_allow_html=True)

# ── FINDINGS DICTIONARY ──────────────────────────────────────────────────────
FINDINGS = {
    "Glioma":"Diffuse infiltrative tumor detected in parenchymal tissue.",
    "Meningioma":"Extra-axial mass likely originating from meninges.",
    "Pituitary":"Sellar region expansion detected.",
    "No Tumor":"No abnormal intracranial mass detected.",
    "INVALID":"Image rejected: not valid MRI structure."
}

# ── VALIDATION 1 : BASIC MRI CHECK ───────────────────────────────────────────
def is_valid_mri(image):

    img_gray = np.array(image.convert("L"))

    if np.mean(img_gray) > 200 or np.mean(img_gray) < 5:
        return False

    if np.std(img_gray) < 18:
        return False

    return True

# ── VALIDATION 2 : ADVANCED TEXTURE CHECK ─────────────────────────────────────
def is_genuine_mri(image):

    img_gray = np.array(image.convert("L"))

    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])

    peak_black = hist[0:10].sum() / hist.sum()

    if laplacian_var < 10 or laplacian_var > 1000:
        return False

    if peak_black < 0.2:
        return False

    return True

# ── IMAGE ENHANCEMENT ────────────────────────────────────────────────────────
def apply_clahe(image):

    img = np.array(image.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    enhanced = clahe.apply(img)

    return Image.fromarray(enhanced).convert("RGB")

# ── PDF REPORT ───────────────────────────────────────────────────────────────
def create_pdf(results):

    pdf = FPDF()

    pdf.set_auto_page_break(auto=True,margin=15)

    pdf.add_page()

    pdf.set_font("Arial","B",16)

    pdf.cell(0,10,"NEUROSCAN AI CLINICAL REPORT",ln=True,align="C")

    pdf.ln(10)

    for i,r in enumerate(results):

        pdf.set_font("Arial","B",11)

        pdf.cell(0,8,f"CASE {i+1} - {r['filename']}",ln=True)

        pdf.set_font("Arial","",10)

        pdf.cell(0,6,f"Prediction: {r['label']}",ln=True)

        pdf.cell(0,6,f"Confidence: {r['confidence']:.1f}%",ln=True)

        pdf.cell(0,6,f"Inference Time: {r['time']:.3f}s",ln=True)

        pdf.ln(4)

    return pdf.output(dest="S").encode("latin-1")

# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():

    try:
        import onnxruntime as ort

        path = "resnet_model.onnx"

        if os.path.exists(path):

            return ort.InferenceSession(path,providers=["CPUExecutionProvider"])

        return None

    except:
        return None

# ── PREPROCESS ───────────────────────────────────────────────────────────────
def preprocess(image):

    img = image.resize((224,224))

    arr = np.array(img).astype("float32")[:,:,::-1]

    arr[:,:,0] -= 103.939
    arr[:,:,1] -= 116.779
    arr[:,:,2] -= 123.68

    return np.expand_dims(arr,0)

CLASS_NAMES = {
    0:"Glioma",
    1:"Meningioma",
    2:"No Tumor",
    3:"Pituitary"
}

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🧠 NeuroScan AI")

st.caption("Advanced Brain Tumor MRI Classification System")

# ── FILE UPLOADER ────────────────────────────────────────────────────────────
session = load_model()

if session is None:

    st.error("resnet_model.onnx not found")

    st.stop()

files = st.file_uploader(
"Upload MRI images",
type=["jpg","png","jpeg"],
accept_multiple_files=True
)

# ── ANALYSIS ENGINE ──────────────────────────────────────────────────────────
if files:

    if st.button(f"RUN ANALYSIS / {len(files)} SCANS"):

        results = []

        total_start = time.time()

        progress = st.progress(0)

        for i,f in enumerate(files):

            start = time.time()

            img = Image.open(f).convert("RGB")

            if not is_valid_mri(img) or not is_genuine_mri(img):

                results.append({
                "image":img,
                "filename":f.name,
                "label":"INVALID",
                "confidence":0,
                "probs":[0]*4,
                "time":0
                })

                continue

            output = session.run(
                None,
                {session.get_inputs()[0].name:preprocess(img)}
            )[0]

            probs = output[0]

            idx = int(np.argmax(probs))

            conf = float(np.max(probs))*100

            final_label = CLASS_NAMES[idx] if conf > 92 else "INVALID"

            elapsed = time.time() - start

            results.append({
                "image":img,
                "filename":f.name,
                "label":final_label,
                "confidence":conf,
                "probs":probs.tolist(),
                "time":elapsed
            })

            progress.progress((i+1)/len(files))

        st.session_state["results"] = results

        st.session_state["latency"] = time.time()-total_start

# ── DISPLAY RESULTS ──────────────────────────────────────────────────────────
if "results" in st.session_state:

    res = st.session_state["results"]

    st.markdown(f"""
    <div class="stat-grid">
    <div class="stat-item"><b>Total</b><div class="stat-value">{len(res)}</div></div>
    <div class="stat-item"><b>Tumor</b><div class="stat-value">{sum(1 for r in res if r["label"] not in ["No Tumor","INVALID"])}</div></div>
    <div class="stat-item"><b>Invalid</b><div class="stat-value">{sum(1 for r in res if r["label"]=="INVALID")}</div></div>
    <div class="stat-item"><b>Total Time</b><div class="stat-value">{st.session_state["latency"]:.2f}s</div></div>
    </div>
    """,unsafe_allow_html=True)

    for r in res:

        is_inv = r["label"] == "INVALID"

        st.markdown(f"""
        <div class="scan-result {'invalid' if is_inv else ''}">
        <b>{r["label"]}</b><br>
        {r["filename"]}<br>
        Confidence: {r["confidence"]:.1f}%
        </div>
        """,unsafe_allow_html=True)

        tabs = st.tabs(["IMAGE","PROBABILITY","ANALYSIS"])

        with tabs[0]:

            st.image(r["image"],use_container_width=True)

            br = st.slider("Brightness",0.5,2.0,1.0,key=r["filename"])

            enhanced = ImageEnhance.Brightness(r["image"]).enhance(br)

            if st.toggle("CLAHE Enhancement",key="clahe"+r["filename"]):
                enhanced = apply_clahe(enhanced)

            st.image(enhanced,use_container_width=True)

        with tabs[1]:

            fig,ax = plt.subplots()

            ax.bar(CLASS_NAMES.values(),r["probs"])

            ax.set_ylabel("Probability")

            st.pyplot(fig)

        with tabs[2]:

            ranking = sorted(
                zip(CLASS_NAMES.values(),r["probs"]),
                key=lambda x:x[1],
                reverse=True
            )

            for name,val in ranking[:3]:

                st.write(f"{name}: {val*100:.2f}%")

            st.markdown(
                f'<div class="finding-box">{FINDINGS.get(r["label"])}</div>',
                unsafe_allow_html=True
            )

    st.download_button(
        "GENERATE REPORT",
        create_pdf(res),
        "NeuroScan_Report.pdf",
        "application/pdf"
    )

st.markdown(
"<br><center style='opacity:0.2'>NEUROSCAN AI · 2026</center>",
unsafe_allow_html=True
)
```
