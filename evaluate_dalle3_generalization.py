# ==============================================================================
# Program Name: evaluate_dalle3_generalization.py
# Version: 1.0
# Description: Processes raw DALL-E 3 images, extracts 1D radial profiles, 
#              and evaluates them against the real nature baseline using 
#              the pre-established Logistic Regression model.
# ==============================================================================

import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from google.colab import drive

import warnings


warnings.filterwarnings('ignore')

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')


# Paths
base_dir = "/content/drive/MyDrive/SPL_Experiments"
real_dir = os.path.join(base_dir, "real_images_nature_100")
dalle_dir = os.path.join(base_dir, "fake_images_dalle3_100") # Upload your folder here

# Feature Extraction
def get_radial_profile(img_path):
    try:
        # Convert to standard RGB/Luminance (handles DALL-E .webp format automatically)
        img = Image.open(img_path).convert('L').resize((512, 512), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        f_shift = np.fft.fftshift(np.fft.fft2(img_array))
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8)
        return radial_profile[:300]
    except Exception as e:
        print(f"Skipping corrupted file: {img_path}")
        return None

print("Processing DALL-E 3 and Real Nature images...")

X, y = [], []

# Load Real (Label 0)
real_count = 0
for f in os.listdir(real_dir):
    p = os.path.join(real_dir, f)
    prof = get_radial_profile(p)
    if prof is not None and len(prof) == 300:
        X.append(prof)
        y.append(0)
        real_count += 1

# Load DALL-E 3 Fake (Label 1)
dalle_count = 0
if os.path.exists(dalle_dir):
    for f in os.listdir(dalle_dir):
        p = os.path.join(dalle_dir, f)
        prof = get_radial_profile(p)
        if prof is not None and len(prof) == 300:
            X.append(prof)
            y.append(1)
            dalle_count += 1
else:
    print(f"Could not find DALL-E directory at: {dalle_dir}")

X, y = np.array(X), np.array(y)
print(f"Loaded {real_count} Real images and {dalle_count} DALL-E 3 images.")

if dalle_count > 0:
    print("\n" + "="*50)
    print(f"ZERO-SHOT GENERALIZATION: DALL-E 3")
    print("="*50)
    
    # Train/Eval using 5-Fold CV
    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_scores = cross_val_score(lr, X, y, cv=5)
    
    print(f"Logistic Regression Accuracy: {lr_scores.mean() * 100:.2f}% (std: {lr_scores.std()*100:.2f}%)")
    print("="*50)