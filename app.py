import os
import streamlit as st
import numpy as np
import time
import tempfile
import pandas as pd
from PIL import Image, ImageEnhance
from fpdf import FPDF
from datetime import datetime

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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #060810 !important;
    color: #C9D1E0 !important;
}

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

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* ── MEDICAL INFO GRID ── */
.med-info-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 40px;
}
.medical-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.1);
    padding: 16px; border-radius: 4px;
    transition: all 0.3s ease;
}
.medical-card:hover {
    background: rgba(99,179,237,0.08);
    border-color: rgba(99,179,237,0.3);
}
.medical-card b { color: #63B3ED; font-size: 13px; font-family: 'Space Mono', monospace; display: block; margin-bottom: 8px; letter-spacing: 1px;}
.medical-card p { font-size: 11px; line-height: 1.5; color: rgba(201,209,224,0.7); margin: 0; }

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
.neuro-logo span { color: #63B3ED; font-weight: 700; }
.neuro-badge {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 3px;
    color: rgba(99,179,237,0.4);
    border: 1px solid rgba(99,179,237,0.15);
    padding: 6px 14px; border-radius: 2px;
    text-transform: uppercase;
}

/* ── HERO ── */
.hero-title {
    font-size: clamp(32px, 5vw, 70px);
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
    margin-bottom: 32px;
}

/* ── UPLOAD ── */
[data-testid="stFileUploader"] > div {
    background: rgba(99,179,237,0.02) !important;
    border: 1px dashed rgba(99,179,237,0.2) !important;
    border-radius: 4px !important;
    padding: 40px !important;
}

/* ── STATS ── */
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

/* ── RESULTS ── */
.scan-result {
    position: relative;
    background: rgba(10,14,22,0.8);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 12px;
    overflow: hidden;
}
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

.control-panel {
    background: rgba(99,179,237,0.03);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 4px;
    padding: 15px;
    margin-top: 10px;
}

.neo-divider { border: none; border-top: 1px solid rgba(99,179,237,0.08); margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="neuro-header">
    <div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div>
    <div class="neuro-badge">ResNet50 · ONNX · v2.5 STABLE</div>
</div>
""", unsafe_allow_html=True)

# ── HERO & MEDICAL INFO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    BRAIN<br>
    <span class="accent">TUMOR</span><br>
    SCAN
</div>
<div class="hero-sub">// Precision Neuro-Imaging Knowledge Base</div>
""", unsafe_allow_html=True)

# PINDAHAN DARI SIDEBAR KE ATAS
st.markdown("""
<div class="med-info-container">
    <div class="medical-card">
        <b>🧠 GLIOMA</b>
        <p>Tumor invasif dari sel glial. Sering menyebar di jaringan otak sekitarnya secara difus.</p>
    </div>
    <div class="medical-card">
        <b>🛡️ MENINGIOMA</b>
        <p>Tumor dari selaput otak. Biasanya tumbuh lambat dan bersifat kompresif pada jaringan saraf.</p>
    </div>
    <div class="medical-card">
        <b>💧 PITUITARY</b>
        <p>Tumor kelenjar hormon dasar otak. Mempengaruhi sistem endokrin dan traktus optikus.</p>
    </div>
    <div class="medical-card">
        <b>✅ NO TUMOR</b>
        <p>Scan normal. Tidak terdeteksi massa abnormal, lesi patologis, atau tanda neoplasma.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PDF ENGINE (BUG FIX) ─────────────────────────────────────────────────────
def safe_str(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(26, 115, 232)
    pdf.cell(0, 12, "NEUROSCAN AI - CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, safe_str(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"), ln=True, align='C')
    pdf.ln(8)
    for idx, res in enumerate(results):
        if idx > 0 and idx % 2 == 0: pdf.add_page()
        pdf.set_font("Arial", 'B', 11); pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe_str(f"#{idx+1:02d}  {res['filename']}"), ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            res['image'].save(tmp.name)
            y_pos = pdf.get_y(); pdf.image(tmp.name, x=10, y=y_pos, w=55)
        pdf.set_xy(75, y_pos + 6); pdf.set_font("Arial", 'B', 13)
        is_t = res['label'].lower() != 'no tumor'
        pdf.set_text_color(220, 50, 50) if is_t else pdf.set_text_color(26, 115, 232)
        pdf.cell(0, 8, safe_str(res['label'].upper()), ln=True)
        pdf.set_x(75); pdf.set_font("Arial", '', 11); pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, safe_str(f"Confidence: {res['confidence']:.2f}%"), ln=True)
        pdf.set_y(y_pos + 60); pdf.set_draw_color(220, 220, 220); pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        if os.path.exists(tmp.name): os.remove(tmp.name)
    pdf.ln(8); pdf.set_font("Arial", 'I', 7)
    pdf.multi_cell(0, 4, safe_str("DISCLAIMER: Research use only. Final diagnosis must be by a professional."))
    return pdf.output()

# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'resnet_model.onnx'
        return ort.InferenceSession(path, providers=['CPUExecutionProvider']) if os.path.exists(path) else None
    except: return None

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    arr = np.array(img).astype('float32')[:, :, ::-1]
    arr[:, :, 0] -= 103.939; arr[:, :, 1] -= 116.779; arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    inp = session.get_inputs()[0].name
    batch = preprocess(image)
    base = float(session.run(None, {inp: batch})[0][0][pred_class])
    H, W = 224, 224
    saliency = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, 14):
        for x in range(0, W, 14):
            occ = batch.copy(); occ[0, y:min(y+28,H), x:min(x+28,W), :] = 0
            score = float(session.run(None, {inp: occ})[0][0][pred_class])
            saliency[y:min(y+28,H), x:min(x+28,W)] = np.maximum(saliency[y:min(y+28,H), x:min(x+28,W)], base - score)
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)
    heatmap = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    return Image.blend(image.convert('RGB'), Image.fromarray(heatmap).resize(image.size, Image.BILINEAR), alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── UPLOAD SECTION ───────────────────────────────────────────────────────────
session = load_model()
if not session: st.error("resnet_model.onnx not found."); st.stop()

uploaded_files = st.file_uploader("DROP MRI HERE", type=["jpg","png","jpeg"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    if st.button(f"RUN ANALYSIS · {len(uploaded_files)} SCANS"):
        all_res = []
        t_start = time.time()
        prog = st.progress(0)
        for i, f in enumerate(uploaded_files):
            img = Image.open(f).convert('RGB')
            out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
            idx = int(np.argmax(out[0]))
            all_res.append({
                'image': img, 'saliency': generate_saliency(session, img, idx),
                'filename': f.name, 'label': CLASS_NAMES[idx], 
                'confidence': float(np.max(out[0]))*100, 'probs': out[0].tolist(), 'size': f"{img.size[0]}x{img.size[1]}"
            })
            prog.progress((i+1)/len(uploaded_files))
        st.session_state['results'] = all_res; st.session_state['time'] = time.time() - t_start

# ── DISPLAY RESULTS ──────────────────────────────────────────────────────────
if 'results' in st.session_state:
    res = st.session_state['results']
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">Processed</div><div class="stat-value">{len(res)}<span> scans</span></div></div>
        <div class="stat-item"><div class="stat-label">Detected</div><div class="stat-value">{sum(1 for r in res if r['label']!='No Tumor')}<span> cases</span></div></div>
        <div class="stat-item"><div class="stat-label">Avg Conf</div><div class="stat-value">{np.mean([r['confidence'] for r in res]):.1f}<span>%</span></div></div>
        <div class="stat-item"><div class="stat-label">Latency</div><div class="stat-value">{st.session_state['time']:.2f}<span>s</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    for i in range(0, len(res), 3):
        row_items = res[i:i+3]
        cols = st.columns(3)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                is_t = item['label'] != 'No Tumor'
                st.markdown(f'<div class="scan-result {"tumor" if is_t else ""}"><div class="result-label {"tumor" if is_t else ""}">{item["label"].upper()}</div><div style="font-size:10px; opacity:0.6;">{item["filename"]} | {item["confidence"]:.1f}%</div></div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["📷 VIEW", "🔥 HEAT", "📊 PROB"])
                with t1:
                    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
                    sl1, sl2 = st.columns(2)
                    br = sl1.slider("BRIGHT", 0.5, 2.0, 1.0, key=f"b{i+idx}")
                    ct = sl2.slider("CONTRAST", 0.5, 2.0, 1.0, key=f"c{i+idx}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.image(ImageEnhance.Contrast(ImageEnhance.Brightness(item['image']).enhance(br)).enhance(ct), use_container_width=True)
                with t2: st.image(item['saliency'], use_container_width=True)
                with t3:
                    for k, v in enumerate(item['probs']):
                        st.write(f"{CLASS_NAMES[k]}: {v*100:.1f}%"); st.progress(v)
    
    st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
    c_dl, c_inf = st.columns([1, 2])
    with c_dl: st.download_button("↓ DOWNLOAD REPORT", create_pdf(res), f"NeuroReport_{datetime.now().strftime('%M%S')}.pdf", "application/pdf")
    with c_inf: st.info("ℹ️ Diagnosis klinis final harus dilakukan oleh ahli radiologi profesional.")

elif not uploaded_files:
    st.markdown('<div style="text-align:center; padding:100px; opacity:0.2;">◎ Awaiting MRI Scan Input</div>', unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<br><div style='text-align:center; font-family:Space Mono; font-size:9px; letter-spacing:4px; color:rgba(99,179,237,0.1); padding:40px 0;'>
NEUROSCAN AI &nbsp;·&nbsp; RESEARCH USE ONLY &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
