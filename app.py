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
    initial_sidebar_state="expanded", 
)

# ── CSS (BUG FIX: CLEANED) ───────────────────────────────────────────────────
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

/* SIDEBAR STYLING */
[data-testid="stSidebar"] {
    background-color: #030508 !important;
    border-right: 1px solid rgba(99,179,237,0.1) !important;
}

.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 14px; color: #63B3ED;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 20px; border-bottom: 1px solid rgba(99,179,237,0.2);
    padding-bottom: 10px;
}

.medical-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.1);
    padding: 12px; border-radius: 4px;
    margin-bottom: 15px;
}
.medical-card b { color: #63B3ED; font-size: 12px; }
.medical-card p { font-size: 11px; line-height: 1.4; color: rgba(201,209,224,0.7); margin-top: 5px; }

/* HEADER */
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

/* STATS */
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

/* RESULTS */
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

# ── SIDEBAR (FIXED: KNOWLEDGE CENTER) ────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Medical Knowledge Center</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="medical-card">
        <b>🧠 GLIOMA</b>
        <p>Tumor invasif yang tumbuh dari sel glial penyokong otak.</p>
    </div>
    <div class="medical-card">
        <b>🛡️ MENINGIOMA</b>
        <p>Tumor pada selaput pelindung otak (meninges). Seringkali jinak.</p>
    </div>
    <div class="medical-card">
        <b>💧 PITUITARY</b>
        <p>Tumor pada kelenjar hormon dasar otak. Bisa mengganggu hormon & visi.</p>
    </div>
    <div class="medical-card">
        <b>✅ NO TUMOR</b>
        <p>Kondisi normal; tidak ditemukan massa abnormal dalam scan ini.</p>
    </div>
    <hr style="opacity:0.1">
    <p style="font-size:10px; color:rgba(99,179,237,0.4); text-align:center;">
        Gunakan tombol sidebar di kiri atas untuk menutup/membuka panel ini.
    </p>
    """, unsafe_allow_html=True)

# ── PDF UTILITY ──────────────────────────────────────────────────────────────
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
        if idx > 0 and idx % 2 == 0: pdf.add_page()
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
        if os.path.exists(tmp.name): os.remove(tmp.name)

    pdf.ln(8)
    pdf.set_font("Arial", 'I', 7)
    pdf.multi_cell(0, 4, safe("DISCLAIMER: Research use only. Final diagnosis must be by a medical professional."))
    return bytes(pdf.output(dest="S"))

# ── MODEL ENGINE ─────────────────────────────────────────────────────────────
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
    arr = arr[:, :, ::-1] # BGR
    arr[:, :, 0] -= 103.939
    arr[:, :, 1] -= 116.779
    arr[:, :, 2] -= 123.68
    return np.expand_dims(arr, axis=0)

def generate_saliency(session, image: Image.Image, pred_class: int) -> Image.Image:
    inp_name = session.get_inputs()[0].name
    base_batch = preprocess(image)
    base_score = float(session.run(None, {inp_name: base_batch})[0][0][pred_class])
    H, W = 224, 224
    stride, patch = 14, 28
    saliency = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = base_batch.copy()
            occ[0, y:min(y+patch,H), x:min(x+patch,W), :] = 0
            score = float(session.run(None, {inp_name: occ})[0][0][pred_class])
            saliency[y:min(y+patch,H), x:min(x+patch,W)] = np.maximum(saliency[y:min(y+patch,H), x:min(x+patch,W)], base_score - score)
    
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min: saliency = (saliency - s_min) / (s_max - s_min)
    heatmap = (np.stack([np.clip(saliency*3-1,0,1), np.clip(saliency*3-0.5,0,1)*np.clip(2-saliency*3,0,1), np.clip(1-saliency*2,0,1)], axis=-1)*255).astype(np.uint8)
    return Image.blend(image.convert('RGB'), Image.fromarray(heatmap).resize(image.size, Image.BILINEAR), alpha=0.55)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ── MAIN INTERFACE ───────────────────────────────────────────────────────────
st.markdown('<div class="neuro-header"><div class="neuro-logo">NEURO<span>SCAN</span> &nbsp;/&nbsp; AI DIAGNOSTIC</div><div class="neuro-badge">v2.5 FINAL</div></div>', unsafe_allow_html=True)

c_hero, c_up = st.columns([1, 1], gap="large")
with c_hero:
    st.markdown('<div style="font-size:60px; font-weight:900; line-height:1; color:#EDF2F7;">BRAIN<br><span style="color:transparent;-webkit-text-stroke:1px rgba(99,179,237,0.5);">TUMOR</span><br>SCAN</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Space Mono; color:rgba(99,179,237,0.5); font-size:12px;">// Deep Learning MRI Classification</p>', unsafe_allow_html=True)

with c_up:
    session = load_model()
    if not session: st.error("Model not found."); st.stop()
    files = st.file_uploader("DROP MRI HERE", type=["jpg","png","jpeg"], accept_multiple_files=True, label_visibility="collapsed")

if files:
    if st.button(f"RUN ANALYSIS · {len(files)} SCANS"):
        all_res = []
        t_start = time.time()
        prog = st.progress(0)
        for i, f in enumerate(files):
            img = Image.open(f).convert('RGB')
            out = session.run(None, {session.get_inputs()[0].name: preprocess(img)})[0]
            idx = int(np.argmax(out[0]))
            all_res.append({
                'image': img, 'saliency': generate_saliency(session, img, idx),
                'filename': f.name, 'label': CLASS_NAMES[idx], 
                'confidence': float(np.max(out[0]))*100, 'probs': out[0].tolist(), 'size': f"{img.size[0]}x{img.size[1]}"
            })
            prog.progress((i+1)/len(files))
        st.session_state['results'] = all_res
        st.session_state['time'] = time.time() - t_start

# ── RESULTS DISPLAY ──────────────────────────────────────────────────────────
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

    # LOOP RESULTS (BUG FIX: CLEAN COLUMN HANDLING)
    for i in range(0, len(res), 3):
        row_items = res[i:i+3]
        cols = st.columns(3)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                is_t = item['label'] != 'No Tumor'
                st.markdown(f"""
                <div class="scan-result {'tumor' if is_t else ''}">
                    <div class="result-label {'tumor' if is_t else ''}">{item['label'].upper()}</div>
                    <div style="font-size:10px; opacity:0.6; font-family:Space Mono;">{item['filename']} | {item['confidence']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
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
                        st.write(f"{CLASS_NAMES[k]}: {v*100:.1f}%")
                        st.progress(v)
    
    st.markdown("<hr class='neo-divider'>", unsafe_allow_html=True)
    
    # PDF SECTION (CLEANED)
    pdf_col1, pdf_col2 = st.columns([1, 2])
    with pdf_col1:
        st.download_button("↓ DOWNLOAD CLINICAL REPORT", create_pdf(res), f"NeuroReport_{datetime.now().strftime('%M%S')}.pdf", "application/pdf")
    with pdf_col2:
        st.info("ℹ️ INFO: Gunakan tab VIEW untuk pengaturan visual manual.")

elif not files:
    st.markdown('<div style="text-align:center; padding:100px; opacity:0.2;">◎ Awaiting MRI Scan Input</div>', unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<br><div style='text-align:center; font-family:Space Mono; font-size:9px; letter-spacing:4px; color:rgba(99,179,237,0.1); padding:40px 0;'>
NEUROSCAN AI &nbsp;·&nbsp; INSTITUTIONAL RESEARCH USE ONLY &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
