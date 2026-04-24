import sys
import os
import gc
import warnings
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.optimize import minimize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, cohen_kappa_score, f1_score, 
    classification_report, roc_auc_score, roc_curve, auc, 
    balanced_accuracy_score, recall_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast

# --- FIX PATH: To run from scripts directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- CONFIG ---
warnings.filterwarnings("ignore")
# Import your module (ensure src is in parent_dir)
from src.config import cfg
from src.dataset import HAM10000Dataset, preprocess_metadata_for_transformer
from src.model import MAF_BiMamba 
from src.augmentations import valid_tf 

# --- SETTINGS ---
CHECKPOINT_DIR = cfg.OUTPUT_DIR
DEVICE = cfg.DEVICE
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
N_FOLDS = 5


# =============================================================================
# 1. IMPROVED TTA FUNCTION (WEIGHTED)
# =============================================================================
def advanced_tta_inference(model, images, metas):
    # View 1: Original (Highest weight - original confidence)
    p1 = F.softmax(model(images, metas), dim=1)
    
    # View 2-4: Geometric transforms
    p2 = F.softmax(model(TF.hflip(images), metas), dim=1)
    p3 = F.softmax(model(TF.vflip(images), metas), dim=1)
    p4 = F.softmax(model(torch.rot90(images, 1, [2, 3]), metas), dim=1)
    
    # View 5: Zoom (Center Crop -> Resize) - Good for small lesions
    _, _, h, w = images.shape
    crop_h, crop_w = int(h * 0.85), int(w * 0.85)
    img_zoom = TF.center_crop(images, [crop_h, crop_w])
    img_zoom = TF.resize(img_zoom, [h, w], antialias=True)
    p5 = F.softmax(model(img_zoom, metas), dim=1)
    
    # Weighted Average: Prioritize View 1 and View 5
    final_p = (p1 * 0.35) + (p5 * 0.20) + (p2 * 0.15) + (p3 * 0.15) + (p4 * 0.15)
    return final_p

# =============================================================================
# 2. LOAD DATA & INFERENCE
# =============================================================================
print("\n" + "█"*70)
print("STARTING: FULL EVALUATION PIPELINE")
print("█"*70)

# Load Data
df_full = pd.read_csv(cfg.CSV_FILE)
LABEL_MAP = {name: idx for idx, name in enumerate(sorted(df_full['dx'].unique()))}
meta_processed, cat_dims, num_continuous = preprocess_metadata_for_transformer(df_full, df_full, df_full)

# Fix tensor/dataframe index error
if isinstance(meta_processed[0], torch.Tensor):
    meta_df = meta_processed[0].cpu()
else:
    meta_df = meta_processed[0].reset_index(drop=True)

ds = HAM10000Dataset(df_full, meta_df, cfg.IMG_ROOTS, LABEL_MAP, valid_tf)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

evaluation_labels = df_full['dx'].map(LABEL_MAP).values
ensemble_probs = np.zeros((len(df_full), len(CLASSES)))

print(f"Running Inference for {N_FOLDS} Folds with Weighted TTA...")

# --- INFERENCE LOOP ---
for fold_id in range(1, N_FOLDS + 1):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{fold_id}.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Skip Fold {fold_id} (Not found)")
        continue
        
    print(f"  -> Fold {fold_id} processing...", end=" ")
    model = MAF_BiMamba(num_classes=len(LABEL_MAP), cat_dims=cat_dims, num_continuous=num_continuous, use_cross_scale=cfg.USE_CROSS_SCALE)
    model.to(DEVICE)
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("OK")
    except Exception as e:
        print(f"ERR: {e}")
        continue
    
    fold_probs_list = []
    with torch.no_grad(), autocast('cuda'):
        for imgs, metas, _ in tqdm(loader, leave=False):
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            probs = advanced_tta_inference(model, imgs, metas)
            fold_probs_list.append(probs.cpu().numpy())
    
    ensemble_probs += np.concatenate(fold_probs_list)
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Average across folds
ensemble_probs /= N_FOLDS
# Handle NaN just in case
ensemble_probs = np.nan_to_num(ensemble_probs, nan=1.0/len(CLASSES))
# =============================================================================
# 3. EVALUATION
# =============================================================================
final_probs = ensemble_probs
final_preds = np.argmax(final_probs, axis=1)

# =============================================================================
# 4. FINAL REPORT
# =============================================================================
acc = accuracy_score(evaluation_labels, final_preds)
macro_f1 = f1_score(evaluation_labels, final_preds, average='macro')
try:
    macro_auc = roc_auc_score(label_binarize(evaluation_labels, classes=range(len(CLASSES))), final_probs, multi_class='ovr', average='macro')
except: macro_auc = 0.0

lines = []
lines.append("\n" + "="*60)
lines.append("FINAL EVALUATION RESULTS")
lines.append("="*60)
lines.append(f"1. Overall Accuracy      : {acc*100:.2f}%")
lines.append(f"2. Macro F1-Score        : {macro_f1*100:.2f}%")
lines.append(f"3. Macro AUC             : {macro_auc:.4f}")
lines.append("-" * 60)
report = classification_report(evaluation_labels, final_preds, target_names=CLASSES, output_dict=True)
lines.append(f"{'CLASS':<8} {'RECALL':<10} {'PRECISION':<10} {'F1-SCORE':<10} {'SUPPORT':<8}")
for cls in CLASSES:
    r = report[cls]
    lines.append(f"{cls.upper():<8} {r['recall']*100:>8.2f}% {r['precision']*100:>8.2f}% {r['f1-score']*100:>8.2f}% {r['support']:>8}")
lines.append("="*60)

report_text = "\n".join(lines)
print(report_text)

def text_to_image(text, filename):
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.text(0.05, 0.95, text, 
             fontsize=12, 
             family='monospace', 
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=1))
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[IMAGE] Saved report image: {filename}")

text_to_image(report_text, 'final_result.png')

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# A. CONFUSION MATRIX
try:
    cm = confusion_matrix(evaluation_labels, final_preds)
    cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cmn, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=[c.upper() for c in CLASSES], yticklabels=[c.upper() for c in CLASSES],
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f'Confusion Matrix (Acc={acc*100:.2f}%)', fontsize=14)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=300)
    print("\n[IMAGE] Saved CM.")
except: pass

# B. ROC CURVES
try:
    plt.figure(figsize=(11, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    print("Drawing ROC Curves...")

    y_true_bin = label_binarize(evaluation_labels, classes=range(len(CLASSES)))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], final_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        label_txt = f'{cls.upper()} (AUC = {roc_auc:.3f})'
        lw = 2.5 if cls in ['vasc', 'df', 'akiec'] else 1.5
        alp = 0.85 if cls in ['vasc', 'df', 'akiec'] else 0.7
        
        plt.plot(fpr, tpr, color=colors[i], lw=lw, alpha=alp, label=label_txt)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    plt.xlim([-0.01, 1.0]); plt.ylim([0.0, 1.02])
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10, shadow=True)
    plt.tight_layout()
    plt.savefig('final_roc_curves.png', dpi=300)
    print("[IMAGE] Saved Final ROC.")

except Exception as e: print(f"Error: {e}")