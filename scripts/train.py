import sys
import os

# --- FIX SRC IMPORT ERROR (MOST IMPORTANT) ---
# This line helps Python find the 'src' directory from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__)) # Get scripts/ directory path
project_root = os.path.dirname(current_dir)              # Get parent directory (project root)
sys.path.append(project_root)
# -----------------------------------------------

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import StratifiedKFold

# Import from .py modules in 'src'
from src.config import cfg
from src.utils import seed_everything, FocalLoss
from src.dataset import preprocess_metadata_for_transformer, HAM10000Dataset
from src.augmentations import train_tf, valid_tf
from src.model import MAF_BiMamba
from src.engine import train_one_epoch, valid_one_epoch

# --- 0. SEEDING ---
seed_everything(cfg.SEED)

def main():
    print("="*70)
    print(f"MAIN TRAINING - V36 (Safe Mode + V-JEPA + Fixed Import)")
    print(f"Dataset: HAM10000 | Folds: {cfg.FOLDS_TO_RUN}")
    print(f"RUNNING ON DEVICE: {cfg.DEVICE}")
    print("="*70)

    # 1. DATA SETUP (HAM10000)
    df_full = pd.read_csv(cfg.CSV_FILE)
    nunique_labels = sorted(df_full['dx'].unique()) 
    LABEL_MAP = {name: idx for idx, name in enumerate(nunique_labels)}
    
    if cfg.NUM_CLASSES != len(LABEL_MAP):
        print(f"Update NUM_CLASSES: {cfg.NUM_CLASSES} -> {len(LABEL_MAP)}")
        cfg.NUM_CLASSES = len(LABEL_MAP)

    print(f"Loaded {len(df_full)} samples. Classes: {LABEL_MAP}\n")

    labels = df_full['dx'] 
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    splits = list(skf.split(df_full, labels))

    print(f"Created {len(splits)} folds (StratifiedKFold)")

    # --- 2. Start iterating over Folds --
    all_fold_metrics = []

    for fold_id in cfg.FOLDS_TO_RUN:
        train_idx, val_idx = splits[fold_id]
        
        print(f"\n{'='*80}")
        print(f"STARTING FOLD {fold_id + 1}/{cfg.N_SPLITS}")
        print(f"{'='*80}")
        
        train_df = df_full.iloc[train_idx].copy()
        val_df = df_full.iloc[val_idx].copy()
        
        print(f"  Train: {len(train_df)}")
        print(f"  Val:   {len(val_df)}")

        # --- 3. Preprocessing (Updated function call) --
        # New function in src/dataset.py returns: (train_meta, val_meta, test_meta), cat_dims, num_continuous
        # We pass None to the 3rd parameter because we don't use test set here
        (train_meta, val_meta, _), cat_dims, num_continuous = \
            preprocess_metadata_for_transformer(train_df, val_df, None) 
            
        print(f"  Meta-features: Num={num_continuous}, Cat={len(cat_dims)} {cat_dims}")

        # --- 4. Datasets & Dataloaders --
        train_ds = HAM10000Dataset(train_df, train_meta, cfg.IMG_ROOTS, LABEL_MAP, train_tf)
        val_ds = HAM10000Dataset(val_df, val_meta, cfg.IMG_ROOTS, LABEL_MAP, valid_tf)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
        print("  Dataloaders created.")

        # --- 5. Loss --
        if cfg.USE_FOCAL_LOSS:
            # IMPORTANT: Must pass cfg.LABEL_SMOOTHING here
            # If not provided, it defaults to 0.0
            criterion = FocalLoss(
                gamma=cfg.FOCAL_LOSS_GAMMA, 
                label_smoothing=cfg.LABEL_SMOOTHING  # <--- This will override the 0.0
            ).to(cfg.DEVICE)
            
            print(f"  Using FocalLoss (gamma={cfg.FOCAL_LOSS_GAMMA}, LS={cfg.LABEL_SMOOTHING}).")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING).to(cfg.DEVICE)
            print(f"  Using CrossEntropyLoss (LS={cfg.LABEL_SMOOTHING}).")
        # --- 6. Model, Optimizer, Scheduler --
        model = MAF_BiMamba(num_classes=cfg.NUM_CLASSES, cat_dims=cat_dims, num_continuous=num_continuous, use_cross_scale=cfg.USE_CROSS_SCALE).to(cfg.DEVICE)
        
        # Optimizer 4-part (Keep your logic)
        backbone_decay = []; backbone_no_decay = []; head_decay = []; head_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if name.startswith("stem.backbone.") or name.startswith("backbone."): # Update backbone name to match
                if param.ndim <= 1 or name.endswith(".bias"): backbone_no_decay.append(param)
                else: backbone_decay.append(param)
            else:
                if param.ndim <= 1 or name.endswith(".bias"): head_no_decay.append(param)
                else: head_decay.append(param)
        
        optimizer_grouped_parameters = [
            {'params': backbone_decay, 'lr': cfg.BACKBONE_LR, 'weight_decay': cfg.WEIGHT_DECAY},       
            {'params': backbone_no_decay, 'lr': cfg.BACKBONE_LR, 'weight_decay': 0.0},   
            {'params': head_decay, 'lr': cfg.HEAD_LR, 'weight_decay': cfg.WEIGHT_DECAY},               
            {'params': head_no_decay, 'lr': cfg.HEAD_LR, 'weight_decay': 0.0}              
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.HEAD_LR, eps=cfg.EPS, betas=cfg.BETAS)
        print(f"Optimizer: AdamW (4-Part Smart Weight Decay)")
        
        # Scheduler (Keep your logic)
        steps_per_epoch = len(train_loader)
        
        if cfg.SCHEDULER_TYPE == 'cosine':
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=cfg.WARMUP_EPOCHS * steps_per_epoch
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=(cfg.EPOCHS - cfg.WARMUP_EPOCHS) * steps_per_epoch, 
                eta_min=1e-7 
            )
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[cfg.WARMUP_EPOCHS * steps_per_epoch]
            )
            print(f"Scheduler: CosineAnnealingLR (Max LR: {cfg.HEAD_LR})\n")
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[cfg.BACKBONE_LR, cfg.BACKBONE_LR, cfg.HEAD_LR, cfg.HEAD_LR], 
                epochs=cfg.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                pct_start=cfg.WARMUP_EPOCHS / cfg.EPOCHS
            )
            print(f"Scheduler: OneCycleLR (Max LR: {cfg.HEAD_LR})\n")

        # Reset TTA log 
        if hasattr(valid_one_epoch, 'logged_tta'): del valid_one_epoch.logged_tta
        if hasattr(valid_one_epoch, 'logged_no_tta'): del valid_one_epoch.logged_no_tta
            
        # --- 7. Training Loop ---
        best_kappa = 0 
        best_acc = 0
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        total_start_time = time.time()

        for epoch in range(1, cfg.EPOCHS + 1):
            # train_one_epoch function in engine.py now has V-JEPA
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, epoch, scheduler)
            val_loss, metrics = valid_one_epoch(model, val_loader, criterion, cfg.DEVICE)
            
            lr_now = optimizer.param_groups[2]['lr'] 
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"ERROR: Loss is NaN at Epoch {epoch}. Stopping fold.")
                break

            print(f"Ep {epoch:3d} | T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | LR: {lr_now:.1e} | "
                  f"Kappa: {metrics['Kappa']*100:.2f}% | F1-Score: {metrics['F1-Score']*100:.2f}% | "
                  f"Precision: {metrics['Precision']*100:.2f}% | Recall: {metrics['Recall']*100:.2f}% | "
                  f"Accuracy: {metrics['Accuracy']*100:.2f}%")
            
            # Save Best
            if metrics['Kappa'] > best_kappa:
                best_kappa = metrics['Kappa']
                best_acc = metrics['Accuracy']
                best_f1 = metrics['F1-Score']
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_kappa': best_kappa
                }, os.path.join(cfg.OUTPUT_DIR, f"best_fold{fold_id+1}.pth"))
                
                print(f"  NEW BEST (Fold {fold_id+1})! K: {best_kappa*100:.2f}% | F1: {metrics['F1-Score']*100:.2f}% | A: {metrics['Accuracy']*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    print(f"\n  Early stop at epoch {epoch}")
                    break

        # --- 8. End of Fold ---
        total_time = (time.time() - total_start_time) / 60
        
        print(f"\n{'='*80}")
        print(f"FINISHED FOLD {fold_id+1} / {cfg.N_SPLITS} (After {total_time:.2f} minutes)")
        print(f"{'-'*80}")
        print(f"  Best Model Metrics of Fold {fold_id+1}:")
        print(f"  > Best Epoch:    {best_epoch}")
        print(f"  > Best Kappa:    {best_kappa*100:.2f}%")
        print(f"  > Best F1-Score: {best_f1*100:.2f}%")
        print(f"  > Best Accuracy: {best_acc*100:.2f}%")
        print(f"  > (Model saved at: {os.path.join(cfg.OUTPUT_DIR, f'best_fold{fold_id+1}.pth')})")
        print(f"{'='*80}\n") 

        all_fold_metrics.append({
            'fold': fold_id + 1,
            'kappa': best_kappa,
            'f1': best_f1,
            'acc': best_acc,
            'recall_per_class': metrics.get('recall_per_class', []) # Use get for safety
        })
        
        del model, train_loader, val_loader, optimizer, scheduler, criterion
        gc.collect()
        torch.cuda.empty_cache()

    # --- 9. Print final results ---
    print(f"\n{'='*80}")
    print(f"COMPLETED {len(cfg.FOLDS_TO_RUN)} FOLDS (Fold ID: {cfg.FOLDS_TO_RUN})")
    print(f"Using Focal Loss: {cfg.USE_FOCAL_LOSS}")
    print(f"{'='*80}")

    metrics_df = pd.DataFrame(all_fold_metrics)
    
    print("--- Detailed Results (Folds just ran) ---")
    columns_to_print = ['fold', 'kappa', 'f1', 'acc']
    print(metrics_df[columns_to_print].to_string())
    print(f"{'-'*80}")

    # --- SUMMARY OF AVERAGE RESULTS (Mean ± Std) ---
    if len(metrics_df) > 1:
        mean_kappa = metrics_df['kappa'].mean() * 100
        std_kappa = metrics_df['kappa'].std() * 100
        
        mean_f1 = metrics_df['f1'].mean() * 100
        std_f1 = metrics_df['f1'].std() * 100
        
        mean_acc = metrics_df['acc'].mean() * 100
        std_acc = metrics_df['acc'].std() * 100
        
        print("--- Overall Average Summary (Mean ± Std) ---")
        print(f"  > Average Kappa:    {mean_kappa:.2f}% ± {std_kappa:.2f}%")
        print(f"  > Average F1-Score: {mean_f1:.2f}% ± {std_f1:.2f}%")
        print(f"  > Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

        print("\n  --- Average Class Recall (Mean ± Std) ---")
        try:
            # Convert list of arrays into one large numpy array to calculate average
            all_recalls = np.stack(metrics_df['recall_per_class'].to_numpy()) 
            mean_recalls = np.mean(all_recalls, axis=0) * 100
            std_recalls = np.std(all_recalls, axis=0) * 100
            
            inv_label_map = {v: k for k, v in LABEL_MAP.items()} 
            
            for i in range(len(mean_recalls)):
                class_name = inv_label_map.get(i, f"Class {i}").upper()
                print(f"    > {class_name.ljust(6)}: {mean_recalls[i]:.2f}% ± {std_recalls[i]:.2f}%")
        except Exception as e:
            print(f"    (Error calculating average recall: {e})")
        
        print(f"{'='*80}")
    
    elif len(metrics_df) == 1:
        print("--- Summary Results (Only 1 fold ran) ---")
        print(f"  > Kappa:    {metrics_df['kappa'].iloc[0]*100:.2f}%")
        print(f"  > F1-Score: {metrics_df['f1'].iloc[0]*100:.2f}%")
        print(f"  > Accuracy: {metrics_df['acc'].iloc[0]*100:.2f}%")
        
        print("\n  --- Class Recall (Fold 1) ---")
        try:
            all_recalls = metrics_df['recall_per_class'].iloc[0] * 100
            inv_label_map = {v: k for k, v in LABEL_MAP.items()}
            for i in range(len(all_recalls)):
                class_name = inv_label_map.get(i, f"Class {i}").upper()
                print(f"    > {class_name.ljust(6)}: {all_recalls[i]:.2f}%")
        except Exception as e:
            print(f"    (Error printing recall: {e})")
            
        print(f"{'='*80}")

    print("\n(Finished.)")

if __name__ == "__main__":
    main()