# MAF-BiMamba: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **"MAF-BiMamba: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification"**.

## ðŸ“Œ Overview
MAF-BiMamba is a novel multimodal architecture designed to bridge the gap between high-performance computing and clinical workflows in dermatological diagnosis. It effectively integrates dermoscopic imagery with patient clinical metadata (age, sex, location) to resolve the "texture-context trade-off."

### Key Contributions:
1. **Resolving the Texture-Context Trade-off:** A hybrid architecture combining a DenseNet Stem (to capture high-frequency visual cues like texture and borders) with Bidirectional Mamba blocks (to assess global lesion asymmetry).
2. **Optimized Metadata Encoder with Adaptive FiLM:** A multimodal modulation mechanism allowing clinical metadata to dynamically impact the image feature extraction process with $O(N)$ complexity.
3. **Anatomically-Aware Regularization:** A novel Joint-Training I-JEPA framework with Semantic Block Masking to prevent overfitting, particularly for underrepresented classes.
4. **Adaptive Clinical Workflow Control:** A dual-track inference strategy providing a lightweight Single Model (178 FPS) for real-time screening and a high-precision Ensemble track (35 FPS) for in-depth reference diagnosis.

## Ã°Å¸Ââ€  Main Results
MAF-BiMamba achieves state-of-the-art performance on the HAM10000 dataset and demonstrates strong cross-dataset generalization on PAD-UFES-20 and ISIC 2019.

| Dataset | Model Mode | Params | MACs | Accuracy (%) | AUC | Weighted F1 (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HAM10000** | Single Model | 24.3M | ~4.2G | $95.67 \pm 0.28$ | $0.969 \pm 0.003$ | $96.01 \pm 0.31$ | **178** |
| **HAM10000** | Ensemble (5-Fold + TTA) | 121.5M | ~21.0G | **$96.43 \pm 0.21$** | **$0.971 \pm 0.002$** | **$96.47 \pm 0.24$** | 35 |
| **PAD-UFES-20** | Zero-shot Generalization | - | - | $89.45 \pm 0.52$ | $0.931 \pm 0.005$ | $88.67 \pm 0.55$ | - |
| **ISIC 2019** (Disjoint)| Zero-shot Generalization | - | - | $87.85 \pm 0.48$ | $0.928 \pm 0.004$ | $87.15 \pm 0.50$ | - |

## ðŸ”¬ Ablation Studies
Our codebase is designed to easily reproduce the ablation studies via flags in `src/config.py` (`USE_FILM`, `USE_IJEPA`, etc).

| Config | Acc (%) | BAcc (%) | Kappa (%) | AUC | Macro F1 (%) | W-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (DenseNet121) | $91.50 \pm 0.55$ | $85.12 \pm 0.61$ | $87.45 \pm 0.58$ | $0.932 \pm 0.005$ | $82.33 \pm 0.65$ | $91.33 \pm 0.52$ |
| + Unidirectional Mamba | $93.23 \pm 0.48$ | $87.45 \pm 0.52$ | $90.12 \pm 0.49$ | $0.952 \pm 0.004$ | $85.12 \pm 0.55$ | $93.12 \pm 0.45$ |
| + Bidirectional Mamba | $94.01 \pm 0.41$ | $88.23 \pm 0.45$ | $91.34 \pm 0.42$ | $0.960 \pm 0.003$ | $87.45 \pm 0.48$ | $94.45 \pm 0.38$ |
| + Metadata | $94.28 \pm 0.39$ | $88.56 \pm 0.42$ | $91.78 \pm 0.40$ | $0.962 \pm 0.003$ | $88.01 \pm 0.44$ | $94.89 \pm 0.35$ |
| + Adaptive FiLM | $94.55 \pm 0.35$ | $88.78 \pm 0.38$ | $92.12 \pm 0.36$ | $0.964 \pm 0.003$ | $88.89 \pm 0.41$ | $95.23 \pm 0.32$ |
| + Focal Loss ($\gamma=2.0$) | $95.12 \pm 0.30$ | $89.34 \pm 0.33$ | $92.67 \pm 0.31$ | $0.967 \pm 0.002$ | $89.67 \pm 0.35$ | $95.67 \pm 0.28$ |
| + I-JEPA (block masking) | $95.67 \pm 0.28$ | $89.78 \pm 0.30$ | $93.23 \pm 0.29$ | $0.969 \pm 0.002$ | $90.45 \pm 0.32$ | $96.01 \pm 0.26$ |
| + 5-Fold Ens (no TTA) | $96.12 \pm 0.24$ | $89.89 \pm 0.27$ | $93.78 \pm 0.25$ | $0.970 \pm 0.002$ | $91.12 \pm 0.29$ | $96.34 \pm 0.23$ |
| **+ Weighted TTA (Full)** | **$96.43 \pm 0.21$** | **$90.03 \pm 0.25$** | **$93.18 \pm 0.22$** | **$0.971 \pm 0.002$** | **$88.30 \pm 0.28$** | **$96.47 \pm 0.24$** |


## ðŸ’¾ Dataset Setup
Please organize your dataset as follows before running any scripts. Ensure `CSV_FILE` and `IMG_ROOTS` in `src/config.py` point to these locations.

```text
data/
â””â”€â”€ HAM10000/
    â”œâ”€â”€ HAM10000_metadata.csv
    â””â”€â”€ images/
        â”œâ”€â”€ ISIC_0024306.jpg
        â”œâ”€â”€ ISIC_0024307.jpg
        â””â”€â”€ ...
```

## Ã°Å¸â€œÂ Project Structure

```text
MAF-BiMamba/
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ final_confusion_matrix.png
â”‚   â”œâ”€â”€ final_result.png
â”‚   â””â”€â”€ final_roc_curves.png
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ run_ensemble_TTA.py     # Inference using 5-fold ensemble with Weighted TTA
â”‚   â””â”€â”€ train.py                # Main script for model training (supports I-JEPA)
â””â”€â”€ src/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ augmentations.py        # Semantic Block Masking and standard augmentations
    â”œâ”€â”€ config.py               # Hyperparameters and path configurations
    â”œâ”€â”€ dataset.py              # Multimodal Dataset class for HAM10000/ISIC/PAD-UFES-20
    â”œâ”€â”€ engine.py               # Training and validation loops with EMA updates
    â”œâ”€â”€ model.py                # MAF-BiMamba architecture (DenseNet Stem, Bi-Mamba, Adaptive FiLM)
    â””â”€â”€ utils.py                # Helper functions
```
## Ã¢Å¡â„¢Ã¯Â¸Â Environment Setup

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
## ðŸš€ Usage
```bash
# 1. Training
# Train the MAF-BiMamba model from scratch using the Joint-Training I-JEPA strategy
python scripts/train.py

# 2. Inference & Evaluation
# Reproduce the best results using the 5-fold ensemble and weighted TTA strategy.
# This script will evaluate the model and automatically generate all final reports
# (final_result.png, final_confusion_matrix.png, final_roc_curves.png)
python scripts/run_ensemble_TTA.py
```
## Ã°Å¸Â¤Â Contact
Email: ibme.lab@gmail.com
