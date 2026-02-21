# ==============================================================================
# Program Name: run_ablation_and_generalization.py
# Version: 1.0
# Description: Evaluates the cross-architecture generalization (SDXL vs Nature) 
#              and performs classifier and feature ablation studies using 
#              5-fold cross-validation to address reviewer comments.
# Input:       Images from Google Drive folders (real_images_nature_100 and 
#              fake_images_sdxl_100).
# Output:      Accuracy metrics for LR, SVM, and RF on 1D radial profiles, 
#              and baseline LR accuracy on raw 2D flattened spectra.
# ==============================================================================

import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
from google.colab import drive


# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set paths to your newly generated datasets
base_dir = "/content/drive/MyDrive/SPL_Experiments"
real_dir = os.path.join(base_dir, "real_images_nature_100")
fake_dir = os.path.join(base_dir, "fake_images_sdxl_100")

# ==============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ==============================================================================
def get_radial_profile(img_path):
    """Proposed Method: 1D Azimuthal Radial Profile (300 features)"""
    try:
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
        return None

def get_2d_profile_flattened(img_path):
    """Ablation Baseline: Raw 2D Spectrum Flattened (Resized to 128x128 for memory)"""
    try:
        img = Image.open(img_path).convert('L').resize((128, 128), Image.Resampling.LANCZOS)
        f_shift = np.fft.fftshift(np.fft.fft2(np.array(img)))
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        return magnitude_spectrum.flatten() # Creates 16,384 features
    except Exception as e:
        return None

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Extracting spectral features from Google Drive... (This takes a minute)")

X_1D, X_2D, y = [], [], []

# Load Real (Label 0)
for f in os.listdir(real_dir):
    p = os.path.join(real_dir, f)
    prof_1d = get_radial_profile(p)
    prof_2d = get_2d_profile_flattened(p)
    if prof_1d is not None and len(prof_1d) == 300 and prof_2d is not None:
        X_1D.append(prof_1d)
        X_2D.append(prof_2d)
        y.append(0)

# Load Fake (Label 1)
for f in os.listdir(fake_dir):
    p = os.path.join(fake_dir, f)
    prof_1d = get_radial_profile(p)
    prof_2d = get_2d_profile_flattened(p)
    if prof_1d is not None and len(prof_1d) == 300 and prof_2d is not None:
        X_1D.append(prof_1d)
        X_2D.append(prof_2d)
        y.append(1)

X_1D, X_2D, y = np.array(X_1D), np.array(X_2D), np.array(y)
print(f"Loaded {len(y)} images successfully.")

# ==============================================================================
# RUN EXPERIMENTS
# ==============================================================================
if len(y) > 0:
    print("\n" + "="*50)
    print("CLASSIFIER ABLATION (Proposed 1D Features)")
    print("   Dataset: SDXL vs Flowers (N=" + str(len(y)) + ")")
    print("   Validation: 5-Fold Cross Validation")
    print("="*50)
    
    # 1. Logistic Regression (Your paper's proposed model)
    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_scores = cross_val_score(lr, X_1D, y, cv=5)
    print(f"Logistic Regression (Proposed): {lr_scores.mean() * 100:.2f}% (std: {lr_scores.std()*100:.2f}%)")

    # 2. SVM
    svm = SVC(kernel='rbf', C=1.0)
    svm_scores = cross_val_score(svm, X_1D, y, cv=5)
    print(f"SVM (RBF Kernel):             {svm_scores.mean() * 100:.2f}% (std: {svm_scores.std()*100:.2f}%)")

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X_1D, y, cv=5)
    print(f"Random Forest:                {rf_scores.mean() * 100:.2f}% (std: {rf_scores.std()*100:.2f}%)")

    print("\n" + "="*50)
    print("FEATURE ABLATION (1D Azimuthal vs Raw 2D)")
    print("="*50)
    
    # Baseline: Logistic Regression on raw 2D features
    lr_2d = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
    lr_2d_scores = cross_val_score(lr_2d, X_2D, y, cv=5)
    
    print(f"Proposed (1D Azimuthal - 300 features): {lr_scores.mean() * 100:.2f}%")
    print(f"Baseline (Raw 2D - 16,384 features):    {lr_2d_scores.mean() * 100:.2f}%")
    print("="*50)
else:
    print("Error: Dataset empty. Please check paths.")