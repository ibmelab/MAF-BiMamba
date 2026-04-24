import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import cfg

print("="*70)
print("DATA PREPROCESSING (HAM10000 - SAFE MODE)")
print("="*70)

def preprocess_metadata_for_transformer(df_train, df_val, df_test=None):
    """
    ABSOLUTELY SAFE VERSION: 
    1. Combine all data to fit LabelEncoder (prevents unseen labels in Val/Test).
    2. Handle NaN thoroughly for both numerical and categorical columns.
    """
    print("Starting Metadata processing...")
    
    # Copy to avoid modifying original data
    df_train = df_train.copy()
    df_val = df_val.copy()
    if df_test is not None:
        df_test = df_test.copy()
    
    # Define columns
    cat_cols = ["dx_type", "sex", "localization"]
    num_cols = ["age"]
    
    # 1. COMBINE DATA (IMPORTANT TO AVOID INDEX ERROR)
    # Combine all so LabelEncoder learns every possible value
    all_dfs = [df_train, df_val]
    if df_test is not None:
        all_dfs.append(df_test)
    
    full_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # 2. PROCESS NUMERICAL
    for c in num_cols:
        # Convert to numeric, errors to NaN
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
        
        # Fill missing age with mean (Safer than median if data is scarce)
        mean_val = full_df[c].mean()
        if pd.isna(mean_val): mean_val = 50.0 # Fallback
        
        full_df[c] = full_df[c].fillna(mean_val)
        
        # Normalize (StandardScaler is better than MinMaxScaler for Transformers)
        scaler = StandardScaler()
        full_df[[c]] = scaler.fit_transform(full_df[[c]])

    # 3. PROCESS CATEGORICAL
    cat_dims = []
    encoders = {}
    
    for c in cat_cols:
        # Convert to string and handle NaN
        full_df[c] = full_df[c].fillna("unknown").astype(str)
        full_df[c] = full_df[c].replace(['nan', 'NaN', 'UNK'], "unknown")
        
        # Fit LabelEncoder on ALL data
        le = LabelEncoder()
        le.fit(full_df[c])
        
        # Transform
        full_df[c] = le.transform(full_df[c])
        
        # Save number of classes (Add 1 as a fallback for Embedding)
        n_classes = len(le.classes_)
        cat_dims.append(n_classes)
        encoders[c] = le
        print(f"   > Column '{c}': {n_classes} classes -> {le.classes_}")

    # 4. RETURN DATA IN SPLITS
    # Split full_df back into train, val, test
    len_train = len(df_train)
    len_val = len(df_val)
    
    train_meta_df = full_df.iloc[:len_train].copy()
    val_meta_df = full_df.iloc[len_train : len_train + len_val].copy()
    
    test_meta_df = None
    if df_test is not None:
        test_meta_df = full_df.iloc[len_train + len_val :].copy()
    
    # Select columns to return
    meta_cols = num_cols + cat_cols
    
    # Helper: Convert DataFrame to Tensor
    def to_tensor(df):
        return torch.tensor(df[meta_cols].values, dtype=torch.float32)

    train_tensor = to_tensor(train_meta_df)
    val_tensor = to_tensor(val_meta_df)
    test_tensor = to_tensor(test_meta_df) if test_meta_df is not None else None
    
    print(f"Metadata processing completed! (Num: {len(num_cols)}, Cat: {len(cat_cols)})")
    return (train_tensor, val_tensor, test_tensor), cat_dims, len(num_cols)


class HAM10000Dataset(Dataset):
    def __init__(self, df, meta_data, img_root, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_data = meta_data # This is a Tensor
        self.img_root = img_root   # List of image paths
        self.label_map = label_map
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. Get basic info
        row = self.df.iloc[idx]
        img_id = str(row['image_id']).strip()
        
        # 2. Find image (Supports multiple folders & extensions)
        img_path = None
        extensions = [".jpg", ".jpeg", ".png"]
        
        if isinstance(self.img_root, str):
            self.img_root = [self.img_root]
            
        for root in self.img_root:
            for ext in extensions:
                temp_path = os.path.join(root, img_id + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if img_path: break
            
        # 3. Load image (With Fallback if error occurs)
        try:
            if img_path:
                img = np.array(Image.open(img_path).convert("RGB"))
            else:
                # Create black image if not found (prevent crash)
                img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        except Exception:
            img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        
        # 4. Augmentation
        if self.transform:
            img = self.transform(image=img)['image']
            
        # 5. Get Metadata & Label
        meta = self.meta_data[idx] # Pre-processed Tensor
        label = torch.tensor(self.label_map[row['dx']], dtype=torch.long)
        
        return img, meta, label