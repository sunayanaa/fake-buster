#program to compare spectral images

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy import stats

# 1. UPDATED CORE FUNCTION (With Resizing) ---

def get_radial_profile(img_path):
    """
    Computes the 1D Azimuthal Radial Profile.
    NOW INCLUDES: Resizing to 512x512 to ensure consistent array lengths.
    """
    try:
        # Load and convert to Grayscale
        img = Image.open(img_path).convert('L')
        
        # [FIX] Resize to fixed 512x512 so all spectra align mathematically
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)

        # Compute DFT
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

        # Azimuthal Integration
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

        # Average energy per radius
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8)
        
        # Return exactly 300 bins to be safe
        return radial_profile[:300], magnitude_spectrum
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

# 2. EXECUTION & VISUALIZATION ---

def compare_spectra(real_dir, fake_dir):
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort files to ensure deterministic order
    real_files.sort()
    fake_files.sort()

    print(f"Processing {len(real_files)} Real and {len(fake_files)} Fake images...")

    plt.figure(figsize=(12, 6))
    
    all_real_profiles = []
    all_fake_profiles = []

    # -- Process Real Images --
    for p in real_files:
        profile, _ = get_radial_profile(p)
        if profile is not None and len(profile) == 300:
            all_real_profiles.append(profile)
            plt.plot(profile, color='blue', alpha=0.05) # Lower alpha for 100 images

    # -- Process Fake Images --
    for p in fake_files:
        profile, _ = get_radial_profile(p)
        if profile is not None and len(profile) == 300:
            all_fake_profiles.append(profile)
            plt.plot(profile, color='red', alpha=0.05)

    # -- Plot Averages --
    if all_real_profiles:
        avg_real = np.mean(all_real_profiles, axis=0)
        plt.plot(avg_real, color='navy', linewidth=3, label='Real (Average)')

    if all_fake_profiles:
        avg_fake = np.mean(all_fake_profiles, axis=0)
        plt.plot(avg_fake, color='darkred', linewidth=3, label='Fake (Diffusion)')

    plt.title("Spectral Analysis: Real vs. Diffusion Generated (N=100)")
    plt.xlabel("Spatial Frequency (Radius)")
    plt.ylabel("Log Power Spectrum Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 3. RUN AND COMPARE ---
compare_spectra('real_images_100', 'fake_images_100')