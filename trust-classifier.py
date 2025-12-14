#Program lightweight linear classifier that provides interpretable decisions based on frequency-domain signatures.

decisions based on frequency-domain signatures
import numpy as np
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. FEATURE EXTRACTION (Fixed with Resizing)
def get_radial_profile(img_path):
    try:
        img = Image.open(img_path).convert('L')
        
        # Resize to 512x512 so all feature vectors are the same length
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8)
        
        # Return exactly 300 bins as features
        return radial_profile[:300]
    except Exception as e:
        return None

# 2. PREPARE DATASET
def prepare_dataset(real_dir, fake_dir):
    print("Extracting features from images...")
    
    features = []
    labels = []
    
    # Load Real (Label 0)
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg'))]
    for p in real_files:
        prof = get_radial_profile(p)
        if prof is not None and len(prof) == 300:
            features.append(prof)
            labels.append(0)

    # Load Fake (Label 1)
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.png', '.jpg'))]
    for p in fake_files:
        prof = get_radial_profile(p)
        if prof is not None and len(prof) == 300:
            features.append(prof)
            labels.append(1)

    return np.array(features), np.array(labels)

# 3. RUN TRAINING
# Load data from our 100-image folders
X, y = prepare_dataset('real_images_100', 'fake_images_100')

print(f"Dataset Shape: {X.shape} (Images, Features)")

if len(X) == 0:
    print("⚠️ Error: No features extracted. Check folder paths!")
else:
    # 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"🧠 Training Logistic Regression on {len(X_train)} images...")
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    
    # Test
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {acc * 100:.2f}%")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))