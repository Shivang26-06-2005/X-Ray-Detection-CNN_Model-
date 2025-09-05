import os
import shutil
import re

# Paths
original_dataset = r"D:\XRAY DETECTION\COVID-19_Radiography_Dataset"
train_folder = os.path.join(original_dataset, "train")
val_folder = os.path.join(original_dataset, "val")
classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# Helper function to extract numeric part from filename
def get_number(fname, cls):
    match = re.search(rf"{cls}-(\d+)", fname, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

# ----------------------------
# 1. Create training and validation folders
# ----------------------------
for cls in classes:
    img_src = os.path.join(original_dataset, cls, "images")
    mask_src = os.path.join(original_dataset, cls, "masks")

    # Create train folders
    train_img_dest = os.path.join(train_folder, cls, "images")
    train_mask_dest = os.path.join(train_folder, cls, "masks")
    os.makedirs(train_img_dest, exist_ok=True)
    os.makedirs(train_mask_dest, exist_ok=True)

    # Create validation folders
    val_img_dest = os.path.join(val_folder, cls, "images")
    val_mask_dest = os.path.join(val_folder, cls, "masks")
    os.makedirs(val_img_dest, exist_ok=True)
    os.makedirs(val_mask_dest, exist_ok=True)

    # Sort files by numeric index
    all_files = os.listdir(img_src)
    all_files.sort(key=lambda x: get_number(x, cls))

    # ----------------------------
   
    # ----------------------------
    train_files = all_files[:1200]
    for fname in train_files:
        shutil.copy(os.path.join(img_src, fname), os.path.join(train_img_dest, fname))
        mask_path = os.path.join(mask_src, fname)
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(train_mask_dest, fname))

    # ----------------------------
    
    # ----------------------------
    val_files = all_files[1200:1300]
    for fname in val_files:
        shutil.copy(os.path.join(img_src, fname), os.path.join(val_img_dest, fname))
        mask_path = os.path.join(mask_src, fname)
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(val_mask_dest, fname))

print("Training and validation folders created successfully!")
