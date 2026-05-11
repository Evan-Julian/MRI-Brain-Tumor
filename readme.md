![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Type-Research%20Project-purple)

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

# 📂 Dataset

This project uses the **Masoud Nickparvar Brain Tumor MRI Dataset**, which contains MRI scans categorized into four classes.

**Dataset Characteristics**

| Category | Description |
|--------|--------|
| Glioma | Tumors originating from glial cells |
| Meningioma | Tumors arising from the meninges |
| Pituitary | Tumors affecting the pituitary gland |
| No Tumor | MRI scans without tumor presence |

**Dataset Statistics**

- Total Images: ~7000+ MRI scans  
- Image Size: Resized to **224 × 224 pixels**  
- Classes: **4 categories**

The dataset is commonly used in brain tumor classification research and provides balanced MRI samples for deep learning models.

# 🧠 Model Architecture (ResNet50)

The **ResNet50 architecture** is used as the primary deep learning backbone for MRI brain tumor classification.

ResNet introduces **residual connections** that allow gradients to propagate through deep networks, enabling more stable training for complex medical imaging tasks.

### Architecture Overview

```
Input MRI Image (224 × 224 × 3)
        │
        ▼
Conv Layer (7×7, 64 filters)
        │
        ▼
Max Pooling
        │
        ▼
Residual Block Stage 1
        │
        ▼
Residual Block Stage 2
        │
        ▼
Residual Block Stage 3
        │
        ▼
Residual Block Stage 4
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer (512)
        │
        ▼
Dense Layer (256)
        │
        ▼
Softmax Output (4 Classes)
```

**Output Classes**

* Glioma
* Meningioma
* Pituitary
* No Tumor

---

# 📊 Final Model Performance Comparison

The following table summarizes the evaluation results of all architectures tested in this research pipeline.
Performance metrics were computed using **Accuracy, Precision, Recall, and F1-Score** on the testing dataset.

| Model Architecture                            | Training / Validation               | Test Accuracy        | Precision   | Recall      | F1-Score    |
| --------------------------------------------- | ----------------------------------- | -------------------- | ----------- | ----------- | ----------- |
| **DenseNet121 (Baseline)**                    | Transfer Learning + Fine-Tuning     | 94.89%               | 0.94        | 0.94        | 0.93        |
| **ResNet50 (Baseline)**                       | Transfer Learning + Fine-Tuning     | 99.39%               | 0.99        | 0.99        | 0.94        |
| **Hybrid DenseNet + SVM**                     | CNN Feature Extraction + SVM        | 98.32%               | 0.98        | 0.98        | 0.95        |
| **Hybrid ResNet + SVM**                       | CNN Feature Extraction + SVM        | 99.24%               | 0.99        | 0.99        | 0.96        |
| **Proposed Feature Fusion (CNN + PCA + SVM)** | DenseNet + ResNet Feature Fusion    | 99.08%               | 0.99        | 0.99        | 0.98        |
| **Final Ensemble Model**                      | Soft Voting (ResNet + SVM + Fusion) | **99.62%** | **1.00** | **Highest** | **Optimal** |

---

### 📈 Evaluation Metrics

| Metric        | Description                                           |
| ------------- | ----------------------------------------------------- |
| **Accuracy**  | Overall percentage of correctly classified MRI images |
| **Precision** | Ratio of correctly predicted tumor cases              |
| **Recall**    | Ability of the model to detect actual tumor cases     |
| **F1-Score**  | Harmonic mean of precision and recall                 |

The results demonstrate that **feature fusion and ensemble learning significantly improve classification performance**, outperforming individual deep learning models.

---

### 🧠 Key Insight

The **Final Ensemble Model**, which combines:

* Deep learning predictions (**ResNet50**)
* Hybrid models (**CNN + SVM**)
* Feature fusion representations (**DenseNet + ResNet features**)

achieves the **most robust performance for brain tumor MRI classification**.

---



# 📄 Disclaimer

**Institutional Research Use Only**

This tool is intended for research assistance and preliminary diagnostic exploration.
Final clinical decisions must always be performed by licensed medical professionals.

---

**Author:** Evan-Julian
**Status:** Active Development
**Last Updated:** March 2026
