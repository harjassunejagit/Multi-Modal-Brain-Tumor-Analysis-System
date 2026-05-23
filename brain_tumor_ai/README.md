🧠 Multi-Modal Brain Tumor Analysis System

An advanced deep learning project for analyzing brain MRI scans using multiple imaging modalities (T1, T2, and FLAIR). The system performs tumor segmentation, tumor classification, growth risk estimation, and AI-assisted clinical explanation.

🚀 Features
🧠 3D Tumor Segmentation using 3D U-Net and Attention U-Net
🏷️ Tumor Type Classification using a Vision Transformer (ViT)
📈 Growth Risk Prediction based on extracted radiomic features
🤖 AI Clinical Explanation in natural language
🖼️ Multi-Modal MRI Support
T1
T2
FLAIR
💾 Support for real BraTS dataset (.nii.gz files)
🏗️ System Pipeline
T1 + T2 + FLAIR MRI
          │
          ▼m 
   Preprocessing
(normalization + resizing)
          │
          ▼
   3D U-Net / Attention U-Net
          │
          ▼
   Tumor Segmentation Mask
          │
          ▼
   Feature Extraction
          │
          ▼
 Vision Transformer (ViT)
          │
          ▼
 Tumor Classification
          │
          ▼
 Risk Prediction
          │
          ▼
 AI Clinical Report
🧪 Supported Tumor Types
Glioblastoma (GBM)
Astrocytoma
Meningioma
Other / Benign
📊 Output

The system produces:

Tumor segmentation mask
Tumor type prediction
Risk score (0–100)
AI-generated clinical summary
📁 Project Structure
Multi-Modal Brain Tumor Analysis System/
├── brain_tumor_ai/
│   ├── main.py
│   ├── requirements.txt
│   └── venv/
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── BraTS2024/
│       └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│           ├── BraTS-GLI-00000-000/
│           ├── BraTS-GLI-00001-000/
│           └── ...
│
├── models/
│   ├── __init__.py
│   ├── unet3d.py
│   ├── attention_unet.py
│   └── classifier.py
│
├── training/
│   ├── __init__.py
│   ├── train_segmentation.py
│   └── train_classifier.py
│
├── inference/
│   ├── __init__.py
│   └── predict.py
│
├── utils/
│   ├── __init__.py
│   ├── loss.py
│   └── metrices.py
│
├── checkpoints/
├── outputs/
├── weights/
├── .gitignore
└── README.md
🗂️ Dataset

This project uses the Brain Tumor Segmentation Challenge dataset.

Download

BraTS Challenge Website

Expected Dataset Structure
data/
└── BraTS2024/
    └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
        ├── BraTS-GLI-00000-000/
        │   ├── BraTS-GLI-00000-000-t1n.nii.gz
        │   ├── BraTS-GLI-00000-000-t2w.nii.gz
        │   ├── BraTS-GLI-00000-000-t2f.nii.gz
        │   └── BraTS-GLI-00000-000-seg.nii.gz
        └── ...
⚙️ Installation
1. Clone the Repository
git clone <your-repository-url>
cd "Multi-Modal Brain Tumor Analysis System"
2. Create Virtual Environment
python -m venv brain_tumor_ai/venv
3. Activate Virtual Environment
Windows PowerShell
brain_tumor_ai\venv\Scripts\activate
Linux / macOS
source brain_tumor_ai/venv/bin/activate
4. Install Dependencies
pip install -r brain_tumor_ai/requirements.txt
📦 Required Packages
torch
torchvision
nibabel
monai
numpy
scikit-learn
matplotlib
opencv-python
tqdm
🏋️ Training
Train Segmentation Model
python -m training.train_segmentation
Train Classification Model
python -m training.train_classifier
💾 Saved Models

During training:

checkpoints/unet3d_epoch_1.pth
checkpoints/unet3d_epoch_2.pth
...

Final model:

unet3d.pth

Recommended location for deployment:

weights/unet3d.pth
weights/classifier.pth
🔍 Inference

Run prediction on trained models:

python -m inference.predict

Outputs will be saved in:

outputs/
📈 Evaluation Metrics
Segmentation Metrics
Dice Score
Hausdorff Distance
Sensitivity
Specificity
Classification Metrics
Accuracy
Precision
Recall
F1 Score
🧠 Models Used
3D U-Net

Primary model for volumetric tumor segmentation.

Attention U-Net

Improves segmentation by focusing on relevant regions.

Vision Transformer (ViT)

Classifies tumor type from extracted features.

Risk Predictor

Estimates tumor aggressiveness using radiomic features.

🛠️ Technologies Used
PyTorch
MONAI
NumPy
Nibabel
OpenCV
Visual Studio Code
🎯 Future Enhancements
FastAPI backend for REST API deployment
React frontend for interactive visualization
DICOM support
PDF clinical report generation
Docker deployment
Cloud integration with AWS
🧑‍💻 Author

Your Name

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgements
MICCAI
ASNR
Brain Tumor Segmentation Challenge
PyTorch Foundation
📌 Quick Start
git clone <your-repository-url>
cd "Multi-Modal Brain Tumor Analysis System"
python -m venv brain_tumor_ai/venv
brain_tumor_ai\venv\Scripts\activate
pip install -r brain_tumor_ai/requirements.txt
python -m training.train_segmentation
python -m training.train_classifier
python -m inference.predict