#Program to remove fake (black/corrupted) images
import os
import numpy as np
from PIL import Image

def remove_black_images(folder):
    removed = 0
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert('L')
            # If the average pixel value is near 0 (black)
            if np.mean(img) < 5: 
                print(f"Removing black image: {filename}")
                os.remove(path)
                removed += 1
        except:
            pass
    return removed

print("🧹 Cleaning dataset...")
r_removed = remove_black_images('real_images_100')
f_removed = remove_black_images('fake_images_100')

print(f"\nDone! Removed {r_removed} real and {f_removed} fake (black/corrupted) images.")
