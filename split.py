import os
import shutil
import random
from tqdm import tqdm

# === Configuration ===
SOURCE_DIR = 'dataset'           # Original folder with class subfolders
DEST_DIR = 'dataset_split'       # New folder for train/val split
VAL_RATIO = 0.2                  # 20% for validation

# Create target directories
for split in ['train', 'val']:
    for cls in ['correct', 'incorrect']:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# Split and copy files
for cls in ['correct', 'incorrect']:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)
    
    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]
    train_images = images[val_count:]

    for img in tqdm(train_images, desc=f"Copying {cls} train"):
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, 'train', cls, img)
        shutil.copyfile(src, dst)

    for img in tqdm(val_images, desc=f"Copying {cls} val"):
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, 'val', cls, img)
        shutil.copyfile(src, dst)

print("✅ Split complete. Check the 'dataset_split' folder.")
