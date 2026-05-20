# 🧠 Multi-Modal Brain Tumor Analysis System

An advanced AI-based medical imaging system that performs **brain tumor segmentation, classification, and prediction** using multi-modal MRI scans (T1, T2, FLAIR).

---

## 🚀 Features

- 🧬 3D Brain Tumor Segmentation (U-Net / Attention U-Net)
- 🧠 Tumor Type Classification (Classifier Model)
- 📊 MRI-based Tumor Analysis Pipeline
- 📁 Multi-modal MRI Support (T1, T2, FLAIR)
- 🔍 Inference on real MRI scans (.nii / .nii.gz)
- 📉 Training + Prediction pipeline
- 🧪 Research-grade modular architecture

---

## 📂 Project Structure
Multi-Modal Brain Tumor Analysis System/
│
├── brain_tumor_ai/ # Core launcher (main.py)
├── data/ # Dataset (BraTS format)
├── training/ # Model training scripts
├── inference/ # Prediction scripts
├── models/ # Neural network architectures
├── utils/ # Loss functions & metrics
├── venv/ # Virtual environment (ignored in GitHub)

---

## 🧠 Dataset Used

This project uses the **BraTS (Brain Tumor Segmentation) Dataset**:

- T1-weighted MRI
- T2-weighted MRI
- FLAIR MRI
- Ground truth segmentation masks

📌 Download dataset:
https://www.synapse.org/#!Synapse:syn51156910

---

## ⚙️ Installation Guide

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/brain-tumor-analysis.git
cd brain-tumor-analysis