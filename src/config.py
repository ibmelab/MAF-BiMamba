
# ===============================================================
# CONFIG - V36 (Single-Stream Bidirectional Mamba)
# ===============================================================
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Enable if needed
import torch
import random
import numpy as np

class Config:
    # ============ DATA & PATHS (HAM10000) ===========
    CSV_FILE = "./data/HAM10000_metadata.csv"
    IMG_ROOTS = [
        "./data/HAM10000_images_part_1", 
        "./data/HAM10000_images_part_2",
    ]
    OUTPUT_DIR = "./checkpoints"
    
    # DO NOT USE RESUME - Train from scratch
    RESUME_CHECKPOINT = None 
    
    # SPECIFY GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    SEED = 42
    N_SPLITS = 5
    
    # ============ MODEL ARCHITECTURE (V36 - Single Stream Lite) ===========
    NUM_CLASSES = 7
    IMG_SIZE = 224            # Keep 224x224 for safety and speed
    META_DIM = 256
    

    USE_CROSS_SCALE = False   # V36 is Single Stream, Cross Scale not needed
    
    # ============ REGULARIZATION (V36 - Stable) ===========
    META_DROPOUT = 0.15
    FUSION_DROPOUT = 0.15
    META_FEATURE_DROPOUT_RATE = 0.1
    MODALITY_DROPOUT_RATE = 0.15
    STOCHASTIC_DEPTH_RATE = 0.05 # Very light degree
    
    # ============ TRAINING (Optimized for speed) ===========
    BATCH_SIZE = 16       # Increased to 16 (Since V36 is lighter than V35)
    EPOCHS = 200 
    PATIENCE = 200
    FOLDS_TO_RUN = [0,1,2,3,4]
    
    # ============ LOSS & STRATEGY ===========
    USE_HYBRID_LOSS = False    
    LABEL_SMOOTHING = 0.05
    USE_FOCAL_LOSS = True 
    FOCAL_LOSS_GAMMA = 2.0 
    
    # ============ OPTIMIZER ===========
    WEIGHT_DECAY = 0.05       
    BETAS = (0.9, 0.999)    
    EPS = 1e-6
    
    # ============ LEARNING RATE (Safe) ===========
    HEAD_LR = 5e-5            # Kept around 1e-4 to avoid NaN
    BACKBONE_LR = 5e-6        
    
    # ============ SCHEDULER ===========
    SCHEDULER_TYPE = 'cosine' 
    WARMUP_EPOCHS = 15        
    
    # ============ AUGMENTATION (Disabled Mixup to use Masking) ===========
    USE_TTA = True
    USE_MIXUP = False 
    MIXUP_PROB = 0.5 
    MIXUP_ALPHA = 0.4
    
    # ============ CONSISTENCY LOSS (V36 - Masking) ===========
    USE_AMP = False           # Disable AMP (FP32) to strictly avoid NaN
    GRAD_CLIP = 0.5           
    USE_MASKING_LOSS = True   # Enable Masking Consistency
    MASKING_RATIO = 0.3  
    MASKING_LOSS_WEIGHT = 1.0 
    
cfg = Config()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"\nConfig V36 (Single-Stream Bidirectional + Masking) Ready:")
print(f"   GPU: {cfg.DEVICE}")
print(f"   Model: DenseNet121 -> Single Stream Mamba (512D)")
print(f"   Strategy: Masking Loss (15%) | FP32 | Batch={cfg.BATCH_SIZE}")