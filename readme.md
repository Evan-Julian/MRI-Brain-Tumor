# 🧠 NeuroScan AI | Precision Neuro-Imaging Diagnostic

**NeuroScan AI** is a specialized medical imaging tool designed for the detection and classification of brain tumors from MRI scans. Leveraging deep learning architectures and advanced saliency mapping, this system provides clinical-grade analytics to assist in institutional research and preliminary neurological assessments.

![Platform Version](https://img.shields.io/badge/Version-2.4-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Model](https://img.shields.io/badge/Model-ResNet50-orange)

---

# 🔬 System Overview

The application utilizes a **ResNet50** model optimized via **ONNX Runtime** for high-speed, CPU-efficient inference. It classifies scans into four distinct categories:

* **Glioma** – Invasive tumors originating from glial cells
* **Meningioma** – Typically benign tumors arising from the protective layers (meninges)
* **Pituitary** – Tumors located at the brain base affecting hormonal regulation
* **No Tumor** – MRI scans indicating no abnormal masses or structures

---

# ✨ Key Features

* **🎯 Automated Tumor Localization**
  Automatically identifies and generates bounding boxes around suspected tumorous regions using saliency thresholding.

* **🗺️ Saliency Mapping**
  Visualizes the AI's decision-making process through heatmaps, highlighting areas of high diagnostic importance.

* **🖼️ Real-time Image Augmentation**
  Provides manual controls for **Brightness, Contrast, and Sharpness** to enhance visual inspection.

* **📄 Clinical PDF Export**
  Generates diagnostic reports containing original MRI scans, heatmaps, model confidence scores, and resolution data.

* **🛡️ MRI Validation Filter**
  Ensures uploaded files follow MRI-like characteristics (grayscale validation and texture inspection).

---

# 🛠️ Technical Stack

| Component               | Technology                   |
| :---------------------- | :--------------------------- |
| **Frontend**            | Streamlit (Responsive UI/UX) |
| **Deep Learning Model** | ResNet50 (ONNX Format)       |
| **Inference Engine**    | ONNX Runtime                 |
| **Image Processing**    | Pillow (PIL), NumPy          |
| **Report Generation**   | FPDF                         |

---

# 🚀 Installation & Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Evan-Julian/MRI-Brain-Tumor.git
cd MRI-Brain-Tumor
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Model Setup

Ensure the pretrained model file is placed in the project root:

```
resnet_model.onnx
```

## 4. Run the Application

```bash
streamlit run app.py
```

After running, the application will launch automatically in your browser.

---

# 📂 Project Structure

```
MRI-Brain-Tumor
│
├── app.py
├── resnet_model.onnx
├── requirements.txt
└── .devcontainer/
```

**Description**

* **app.py** — Main Streamlit application containing UI logic, preprocessing, inference, and visualization pipeline
* **resnet_model.onnx** — Pretrained ResNet50 deep learning model
* **requirements.txt** — Python dependencies required to run the system
* **.devcontainer/** — Development container configuration for reproducible environments

---

# 📄 Disclaimer

**Institutional Research Use Only**

This tool is intended for research assistance and preliminary diagnostic exploration.
Final clinical decisions must always be performed by licensed medical professionals.

---

**Author:** Evan-Julian
**Status:** Active Development
**Last Updated:** March 2026
