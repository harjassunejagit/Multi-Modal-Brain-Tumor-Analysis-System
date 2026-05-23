# 🧠 Multi-Modal Brain Tumor Analysis System

An advanced AI-powered medical imaging system for **brain tumor segmentation and analysis** using **multi-modal MRI scans** and **3D deep learning**.

This project uses a custom **3D U-Net architecture** trained on the **BraTS (Brain Tumor Segmentation) Dataset** to automatically identify and segment tumor regions from MRI scans.

---

# 🚀 Features

✅ Multi-modal MRI processing  
✅ 3D Brain Tumor Segmentation  
✅ Deep Learning using PyTorch  
✅ MRI Visualization & Overlay Generation  
✅ Automated Inference Pipeline  
✅ Dice Loss Optimization  
✅ NIfTI (.nii.gz) MRI Support  
✅ Professional Segmentation Overlay Outputs  

---

# 🧠 MRI Modalities Used

| Modality | Purpose |
|---|---|
| T1 | Brain anatomy |
| T2 | Fluid & edema |
| FLAIR | Tumor visibility |

The system combines all three modalities to improve segmentation performance.

---

# 🏗️ Project Architecture

```text
MRI Scans (T1 + T2 + FLAIR)
            ↓
      Preprocessing
            ↓
      3D U-Net Model
            ↓
    Tumor Segmentation
            ↓
   Visualization & Overlay
```

---

# 📂 Project Structure

```text
Multi-Modal Brain Tumor Analysis System/
│
├── checkpoints/              # Training checkpoints
│
├── data/
│   ├── dataset.py            # Dataset loader
│   └── BraTS2024/            # BraTS dataset
│
├── inference/
│   ├── __init__.py
│   └── predict.py            # Inference pipeline
│
├── models/
│   ├── __init__.py
│   ├── unet3d.py             # 3D U-Net model
│   ├── attention_unet.py
│   └── classifier.py
│
├── outputs/                  # Prediction outputs
│
├── test_sample/              # Sample MRI scans
│
├── training/
│   ├── __init__.py
│   ├── train_segmentation.py
│   └── train_classifier.py
│
├── utils/
│   ├── __init__.py
│   ├── loss.py
│   └── metrices.py
│
├── weights/
│   ├── unet3d.pth
│   └── classifier.pth
│
├── README.md
└── requirements.txt
```

---

# 📦 Dataset Used

## BraTS Dataset

This project uses the **BraTS (Brain Tumor Segmentation Challenge)** dataset.

Dataset contains:
- Multi-modal MRI scans
- Ground truth tumor masks
- Glioma cases

---

# 📥 Download Dataset

Register and download:

- BraTS Dataset:  
  https://www.synapse.org/

After downloading:

```text
data/
└── BraTS2024/
    └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/brain-tumor-analysis-system.git
cd brain-tumor-analysis-system
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training the Model

Run segmentation training:

```bash
python -m training.train_segmentation
```

Trained weights will be saved inside:

```text
checkpoints/
```

Copy final trained models to:

```text
weights/
```

Example:

```text
weights/
├── unet3d.pth
└── classifier.pth
```

---

# 🧪 Running Inference

## Add Test MRI Scans

Place sample patient folders inside:

```text
test_sample/
```

Example:

```text
test_sample/
├── BraTS-GLI-00000-000/
├── BraTS-GLI-00001-000/
└── BraTS-GLI-00002-000/
```

---

## Run Prediction

```bash
python -m inference.predict
```

The system will:
- load trained weights
- preprocess MRI scans
- generate tumor masks
- create visualization overlays
- save outputs

---

# 📊 Output Example

The output includes:

✅ Original MRI  
✅ Predicted Tumor Mask  
✅ Tumor Overlay Visualization  

Saved in:

```text
outputs/
```

---

# 🧠 Deep Learning Details

## Model Used

### 3D U-Net

The architecture consists of:
- Encoder
- Bottleneck
- Decoder
- Skip Connections

---

## Loss Function

Dice Loss is used for segmentation optimization.

```text
Dice = 2TP / (2TP + FP + FN)
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| PyTorch | Deep learning |
| NumPy | Numerical computation |
| Nibabel | MRI loading |
| Matplotlib | Visualization |
| tqdm | Training progress |

---

# ⚠️ Challenges Faced

- MRI preprocessing
- Large 3D data handling
- Tensor dimension mismatches
- CPU memory limitations
- Segmentation quality optimization

---

# 🚀 Future Improvements

- Attention U-Net
- Vision Transformer (ViT)
- Multi-class tumor segmentation
- FastAPI backend
- React frontend
- PDF medical report generation
- Grad-CAM explainability
- MONAI framework integration

---

# 📈 Results

The model successfully:
- loads multi-modal MRI scans
- performs tumor segmentation
- generates overlay visualizations
- predicts tumor regions automatically

---

# 🎯 Learning Outcomes

Through this project I learned:
- Medical image processing
- 3D deep learning
- MRI preprocessing
- Segmentation architectures
- PyTorch model training
- Inference pipeline development

---

# 👨‍💻 Author

Your Name

---

# 📜 License

This project is for educational and research purposes.