import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image
from fpdf import FPDF
from datetime import datetime

st.set_page_config(
    page_title="NeuroScan AI | Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (Dengan Tambahan untuk Chart & Chat) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}

.stApp > * { position: relative; z-index: 1; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* ── HEADER ── */
.neuro-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 28px 0 40px;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 48px;
}
.neuro-logo {
    font-family: 'Space Mono', monospace;
    font-size: 13px; letter-spacing: 4px;
    color: rgba(99,179,237,0.6);
    text-transform: uppercase;
}
.neuro-logo span {
    color: #63B3ED; font-weight: 700;
}
.neuro-badge {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 3px;
    color: rgba(99,179,237,0.4);
    border: 1px solid rgba(99,179,237,0.15);
    padding: 6px 14px; border-radius: 2px;
    text-transform: uppercase;
}

/* ── HERO TITLE ── */
.hero-title {
    font-size: clamp(42px, 6vw, 80px);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -3px;
    color: #EDF2F7;
    margin-bottom: 16px;
}
.hero-title .accent {
    color: transparent;
    -webkit-text-stroke: 1px rgba(99,179,237,0.5);
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px; letter-spacing: 2px;
    color: rgba(99,179,237,0.5);
    text-transform: uppercase;
    margin-bottom: 48px;
}

/* ── UPLOAD ZONE ── */
.upload-wrapper [data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(99,179,237,0.02) !important;
    border: 1px dashed rgba(99,179,237,0.2) !important;
    border-radius: 4px !important;
    padding: 40px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] > div:hover {
    background: rgba(99,179,237,0.05) !important;
    border-color: rgba(99,179,237,0.4) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: transparent !important;
    color: #63B3ED !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.08) !important;
    border-color: #63B3ED !important;
    color: #EDF2F7 !important;
}

/* ── RESULT CARDS ── */
.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 12px;
    overflow: hidden;
    transition: border-color 0.3s;
}
.scan-result:hover { border-color: rgba(99,179,237,0.3); }
.scan-result::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #63B3ED, transparent);
}
.scan-result.tumor::before {
    background: linear-gradient(180deg, #FC8181, transparent);
}
.result-label {
    font-size: 22px; font-weight: 900;
    letter-spacing: -1px;
    color: #EDF2F7;
    margin-bottom: 4px;
}
.result-label.tumor { color: #FC8181; }
.result-meta {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    color: rgba(99,179,237,0.4);
    text-transform: uppercase;
}
.result-confidence {
    font-family: 'Space Mono', monospace;
    font-size: 28px; font-weight: 700;
    color: #63B3ED;
    text-align: right;
}
.result-confidence.tumor { color: #FC8181; }

/* ── STAT GRID ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin: 32px 0;
}
.stat-item {
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.08);
    border-radius: 4px; padding: 20px 24px;
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px; letter-spacing: 3px;
    color: rgba(99,179,237,0.35);
    text-transform: uppercase; margin-bottom: 8px;
}
.stat-value {
    font-size: 32px; font-weight: 900;
    letter-spacing: -2px; color: #EDF2F7;
}
.stat-value span { color: #63B3ED; font-size: 16px; font-weight: 400; }

/* ── CHATBOX MOCKUP ── */
.chat-container {
    background: rgba(10,14,22,0.9);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 8px;
    padding: 15px;
    height: 300px;
    overflow-y: auto;
}
.chat-msg {
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
}
.chat-ai { color: #63B3ED; }
.chat-user { color: #EDF2F7; opacity: 0.8; }

.neo-divider {
    border: none; border-top: 1px solid rgba(99,179,237,0.08);
    margin: 32px 0;
}

.idle-state {
    text-align: center; padding: 80px 40px;
    border: 1px solid rgba(99,179,237,0.06);
    border-radius: 4px;
    background: rgba(99,179,237,0.01);
}
.idle-icon {
    font-size: 48px; margin-bottom: 20px;
    opacity: 0.3;
}
.idle-text {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 3px;
    color: rgba(99,179,237,0.25);
    text-transform: uppercase;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    color: rgba(99,179,237,0.5) !important;
    padding: 8px 16px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
}
.stTabs [aria-selected="true"] {
    border-color: #63B3ED !important;
    color: #63B3ED !important;
}
</style>
""", unsafe_allow_html=True)

# ── PDF GENERATOR ────────────────────────────────────────────────────────────
def safe(text):
    return text.encode('latin-1', errors='replace').decode('latin-1')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 12, "NEUROSCAN AI - CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"), ln=True, align='C')
    pdf.ln(8)

    for idx, res in enumerate(results):
        if idx > 0 and idx % 2 == 0:
            pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe(f"#{idx+1:02d}  {res['filename']}"), ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            res['image'].save(tmp.name)
            y = pdf.get_y()
            pdf.image(tmp.name, x=10, y=y, w=55)
        pdf.set_xy(75, y + 6)
        pdf.set_font("Arial", 'B', 13)
        is_tumor = res['label'].lower() != 'no tumor'
        pdf.set_text_color(220, 50, 50) if is_tumor else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe(res['label'].upper()), ln=True)
        pdf.set_x(75)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, safe(f"Confidence: {res['confidence']:.2f}%"), ln=True)
        pdf.set_y(y + 60)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, safe("DISCLAIMER: This report is generated by a Deep Learning model for research purposes only. Final diagnosis must be conducted by a qualified medical professional."))
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)

# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        if not os.path.exists(path): return None
        return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    except: return None

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')
    arr = arr[:, :, ::-1] # BGR conversion
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])

    H, W = 224, 224
    stride = 14 # Reduced for better resolution
    patch  = 28 
    saliency = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = base_batch.copy()
            y1, y2 = y, min(y + patch, H)
            x1, x2 = x, min(x + patch, W)
            occ[0, y1:y2, x1:x2, :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            drop = base_score - score
            saliency[y1:y2, x1:x2] = np.maximum(saliency[y1:y2, x1:x2], drop)

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)

    r = np.clip(saliency * 3 - 1, 0, 1)
    g = np.clip(saliency * 3 - 0.5, 0, 1) * np.clip(2 - saliency * 3, 0, 1)
    b = np.clip(1 - saliency * 2, 0, 1)
    heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    orig = image.convert('RGB')
    return Image.blend(orig, heatmap_img, alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div class="neuro-badge">ResNet50 · ONNX Runtime · v2.1</div>
</div>
""", unsafe_allow_html=True)

col_hero, col_upload = st.columns([1, 1], gap="large")

with col_hero:
    st.markdown("""
    <div class="hero-title">
        BRAIN<br>
        <span class="accent">TUMOR</span><br>
        SCAN
    </div>
    <div class="hero-sub">// Precision Neuro-Imaging Analytics</div>
    """, unsafe_allow_html=True)
    
    # ── ADDITIONAL FEATURE: AI ASSISTANT CHAT ──
    with st.expander("💬 CLINICAL AI ASSISTANT", expanded=False):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-msg chat-ai"><b>AI:</b> System ready. How can I help you interpret the scan results?</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.text_input("Query model (e.g., 'What is Glioma?')", key="chat_input", label_visibility="collapsed")

with col_upload:
    session = load_model()
    if session is None:
        st.error("⚠ resnet_model.onnx not found.")
        st.stop()

    uploaded_files = st.file_uploader(
        "DROP MRI SCANS HERE",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:10px;
    letter-spacing:2px; color:rgba(99,179,237,0.25); text-align:center;
    margin-top:8px; text-transform:uppercase;'>
    Batch support enabled · DICOM to PNG suggested
    </div>""", unsafe_allow_html=True)

# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if uploaded_files:
    n = len(uploaded_files)
    if st.button(f"RUN ANALYSIS · {n} SEQUENCE{'S' if n > 1 else ''}"):
        all_results = []
        t_start = time.time()
        progress_bar = st.progress(0, text="Initializing...")
        
        for fi, f in enumerate(uploaded_files):
            progress_bar.progress((fi) / n, text=f"Analyzing {f.name}...")
            img = Image.open(f).convert('RGB')
            batch = preprocess(img)
            inp = session.get_inputs()[0].name
            out = session.run(None, {inp: batch})[0]
            pred_idx = int(np.argmax(out[0]))
            conf = float(np.max(out[0])) * 100
            saliency_img = generate_saliency(session, img, pred_idx)
            
            all_results.append({
                'image': img,
                'saliency': saliency_img,
                'filename': f.name,
                'label': CLASS_NAMES.get(pred_idx, f"Class {pred_idx}"),
                'confidence': conf,
                'probs': out[0].tolist(),
                'size': f"{img.size[0]}x{img.size[1]}"
            })
            
        progress_bar.empty()
        t_total = time.time() - t_start

        # ── EXTENDED STATS ──
        avg_conf = np.mean([r['confidence'] for r in all_results])
        tumor_found = sum(1 for r in all_results if r['label'].lower() != 'no tumor')
        
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-label">Scans Processed</div>
                <div class="stat-value">{len(all_results)}<span> units</span></div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Abnormalities</div>
                <div class="stat-value">{tumor_found}<span> detected</span></div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{avg_conf:.1f}<span>%</span></div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Processing Time</div>
                <div class="stat-value">{t_total:.2f}<span>s</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── BATCH INSIGHTS (NEW FEATURE) ──
        if n > 1:
            with st.expander("📊 BATCH DISTRIBUTION ANALYSIS", expanded=True):
                df = pd.DataFrame([{'Label': r['label'], 'Conf': r['confidence']} for r in all_results])
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.caption("Detection Frequency")
                    st.bar_chart(df['Label'].value_counts())
                with chart_col2:
                    st.caption("Confidence Levels")
                    st.line_chart(df['Conf'])

        st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)

        # ── RESULTS GRID ──
        ncols = 3
        for row_start in range(0, len(all_results), ncols):
            row_items = all_results[row_start:row_start + ncols]
            cols = st.columns(ncols, gap="medium")
            for col_idx, res in enumerate(row_items):
                is_tumor = res["label"].lower() != "no tumor"
                accent = "#FC8181" if is_tumor else "#63B3ED"
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="scan-result {'tumor' if is_tumor else ''}">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                                <div class="result-label {'tumor' if is_tumor else ''}">{res["label"].upper()}</div>
                                <div class="result-meta">{res["filename"][:25]}...<br>{res['size']} PX</div>
                            </div>
                            <div class="result-confidence {'tumor' if is_tumor else ''}">{res["confidence"]:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["📷 RAW", "🔥 HEATMAP", "📊 PROBS"])
                    with tab1:
                        st.image(res["image"], use_container_width=True)
                    with tab2:
                        st.image(res["saliency"], use_container_width=True)
                    with tab3:
                        for ci, prob in enumerate(res["probs"]):
                            name = CLASS_NAMES.get(ci, f"C{ci}")
                            pct = prob * 100
                            is_top = ci == int(np.argmax(res["probs"]))
                            st.markdown(f"<p style='font-size:10px; margin-bottom:2px; color:rgba(255,255,255,0.6)'>{name}</p>", unsafe_allow_html=True)
                            st.progress(min(pct / 100, 1.0))

        st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)

        # ── PDF DOWNLOAD ──
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            with st.spinner("Finalizing PDF..."):
                pdf_bytes = create_pdf(all_results)
            st.download_button(
                label="↓ DOWNLOAD CLINICAL REPORT",
                data=pdf_bytes,
                file_name=f"NeuroScan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        with col_info:
            st.info("💡 Pro-Tip: The Saliency Map shows where the AI is looking. Red areas indicate high influence on the final diagnosis.")

else:
    st.markdown("""
    <div class="idle-state">
        <div class="idle-icon">◎</div>
        <div class="idle-text">System offline // Awaiting MRI sequence upload</div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style='text-align:center; font-family:"Space Mono",monospace;
font-size:9px; letter-spacing:4px; text-transform:uppercase;
color:rgba(99,179,237,0.12); padding: 24px 0;'>
NEUROSCAN AI &nbsp;·&nbsp; INSTITUTIONAL RESEARCH USE &nbsp;·&nbsp; v2.1
</div>
""", unsafe_allow_html=True)
