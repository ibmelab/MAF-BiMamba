# MAF-BiMamba: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **"MAF-BiMamba: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification"**.

## 📌 Overview
MAF-BiMamba is a novel multimodal architecture designed to bridge the gap between high-performance computing and clinical workflows in dermatological diagnosis. It effectively integrates dermoscopic imagery with patient clinical metadata (age, sex, location) to resolve the "texture-context trade-off."

### Key Contributions:
1. **Resolving the Texture-Context Trade-off:** A hybrid architecture combining a DenseNet Stem (to capture high-frequency visual cues like texture and borders) with Bidirectional Mamba blocks (to assess global lesion asymmetry).
2. **Optimized Metadata Encoder with Adaptive FiLM:** A multimodal modulation mechanism allowing clinical metadata to dynamically impact the image feature extraction process with $O(N)$ complexity.
3. **Anatomically-Aware Regularization:** A novel Joint-Training V-JEPA framework with Semantic Block Masking to prevent overfitting, particularly for underrepresented classes.
4. **Adaptive Clinical Workflow Control:** A dual-track inference strategy providing a lightweight Single Model (178 FPS) for real-time screening and a high-precision Ensemble track (35 FPS) for in-depth reference diagnosis.

## 🏆 Main Results
MAF-BiMamba achieves state-of-the-art performance on the HAM10000 dataset and demonstrates strong cross-dataset generalization on PAD-UFES-20 and ISIC 2019.

| Dataset | Model Mode | Accuracy (%) | AUC | Weighted F1 (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HAM10000** | Single Model | $95.67 \pm 0.28$ | $0.969 \pm 0.003$ | $96.01 \pm 0.31$ | **178** |
| **HAM10000** | Ensemble (5-Fold + TTA) | **$96.43 \pm 0.21$** | **$0.971 \pm 0.002$** | **$96.47 \pm 0.24$** | 35 |
| **PAD-UFES-20** | Zero-shot Generalization | $89.45 \pm 0.52$ | $0.931 \pm 0.005$ | $88.67 \pm 0.55$ | - |
| **ISIC 2019** (Disjoint)| Zero-shot Generalization | $87.85 \pm 0.48$ | $0.928 \pm 0.004$ | $87.15 \pm 0.50$ | - |

## 📁 Project Structure

```text
MAF-BiMamba/
├── results/
│   ├── final_confusion_matrix.png
│   ├── final_result.png
│   └── final_roc_curves.png
├── scripts/
│   ├── run_ensemble_TTA.py     # Inference using 5-fold ensemble with Weighted TTA
│   └── train.py                # Main script for model training (supports V-JEPA)
└── src/
    ├── __init__.py
    ├── augmentations.py        # Semantic Block Masking and standard augmentations
    ├── config.py               # Hyperparameters and path configurations
    ├── dataset.py              # Multimodal Dataset class for HAM10000/ISIC/PAD-UFES-20
    ├── engine.py               # Training and validation loops with EMA updates
    ├── model.py                # MAF-BiMamba architecture (DenseNet Stem, Bi-Mamba, Adaptive FiLM)
    └── utils.py                # Helper functions
```
## ⚙️ Environment Setup

```bash
# 1. Create and activate a Conda environment
conda create -n maf-bimamba python=3.9 -y
conda activate maf-bimamba

# 2. Install PyTorch (Adjust CUDA version to match your hardware)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install State Space Model dependencies
pip install mamba-ssm==1.1.0

# 4. Install other requirements
pip install -r requirements.txt 
```
## 🚀 Usage
```bash
# 1. Training
# Train the MAF-BiMamba model from scratch using the Joint-Training V-JEPA strategy
python scripts/train.py

# 2. Inference & Evaluation
# Reproduce the best results using the 5-fold ensemble and weighted TTA strategy.
# This script will evaluate the model and automatically generate all final reports
# (final_result.png, final_confusion_matrix.png, final_roc_curves.png)
python scripts/run_ensemble_TTA.py
```
## 🤝 Contact
Email: ibme.lab@gmail.com
