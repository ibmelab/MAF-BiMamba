import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import cfg # Import cfg

print("="*70)
print(f"AUGS: V9 - Fixed p=0.5 (Added A.Resize before)")
print("="*70)

# ============= TRAINING: Added A.Resize(p=1.0) =============
train_tf = A.Compose([
    
    # 1. Standard Resize (Now 224)
    A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE, p=1.0),
    
    # 2. Light Crop
    A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.85, 1.0), p=0.5),
    
    # 3. Basic geometric transformations 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7), 
    
    # 4. Light image distortion
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2), 
    
    # 5. Color adjustments
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    
    # REMOVED 'CoarseDropout' to avoid conflict with Masking Consistency Loss
    
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
# ============= VALIDATION: KEPT UNCHANGED =============
valid_tf = A.Compose([
    A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("Augmentation: train_tf (Resize p=1.0 + RandomResizedCrop p=0.5 FIXED)")
print("Augmentation: valid_tf (Resize)")
print("="*70 + "\n")