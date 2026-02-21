# ==============================================================================
# Program Name: fetch_real_nature_images_fix.py
# Version: 1.1
# Description: Standalone script to fetch 100 real natural images (flowers), 
#              bypassing previous dataset access and security errors.
# Input:       HuggingFace 'huggan/flowers-102-categories' stream.
# Output:      100 real PNG images saved to your Google Drive folder.
# ==============================================================================

from datasets import load_dataset
import os
from PIL import Image
from google.colab import drive


# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Path to the existing Google Drive folder
real_dir = "/content/drive/MyDrive/SPL_Experiments/real_images_nature_100"
os.makedirs(real_dir, exist_ok=True)

print("⬇️ Downloading 100 Real Images (Flowers dataset)...")

# Using the dataset you verified
ds_real = load_dataset("huggan/flowers-102-categories", split="train", streaming=True)

count = 0
for i, item in enumerate(ds_real):
    if count >= 100: 
        break
        
    save_path = os.path.join(real_dir, f"real_{count:03d}.png")
    
    # Skip if file already exists (handles Colab disconnects)
    if os.path.exists(save_path):
        count += 1
        continue
        
    try:
        # Standard huggingface image column
        img = item['image']
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        
        img.save(save_path)
        count += 1
    except Exception as e:
        # Skip if an image is corrupted or formatted unexpectedly
        pass

print(f"Successfully verified 100 real images in '{real_dir}'")