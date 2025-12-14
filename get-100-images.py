#program to get 100 real images and 100 fake images

# STEP 1: INSTALL LIBRARIES
!pip install -q diffusers transformers accelerate datasets scipy scikit-learn

import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm

# Setup Folders
os.makedirs("real_images_100", exist_ok=True)
os.makedirs("fake_images_100", exist_ok=True)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Processing on device: {device.upper()}")

if device == "cpu":
    print("⚠️ WARNING: You are using CPU. Generating 100 images will take ~2 hours.")
    print("👉 Please switch to GPU: Runtime -> Change runtime type -> T4 GPU")

# STEP 2: GET 100 REAL IMAGES (Food101)
print("\n⬇️  Downloading 100 Real Images (Food101)...")

# We use 'food101' which has high-res natural texture photos
# trust_remote_code=True fixes the 'script not supported' error
ds_real = load_dataset("food101", split="train", streaming=True, trust_remote_code=True)

count = 0
for i, item in enumerate(ds_real):
    if count >= 100: break
    try:
        img = item['image']
        if img.mode != 'RGB': img = img.convert('RGB')
        
        # Save high quality
        save_path = os.path.join("real_images_100", f"real_{count:03d}.png")
        img.save(save_path)
        count += 1
    except Exception as e:
        pass

print(f"Saved {count} Real images to 'real_images_100/'")

# STEP 3: GENERATE 100 FAKE IMAGES (Stable Diffusion)
print("\n🎨 Loading Stable Diffusion Model...")

# Handle FP16 vs FP32 based on device
dtype = torch.float16 if device == "cuda" else torch.float32

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True) 

print("Generating 100 Diffusion Images...")

prompts = [
    "A professional photograph of a gourmet burger",
    "A high resolution photo of sushi on a plate",
    "A delicious pizza with cheese and basil",
    "A bowl of fresh pasta with tomato sauce",
    "A stack of pancakes with syrup",
    "A fresh salad with vegetables",
    "A chocolate cake slice on a white plate",
    "A cup of coffee and a croissant",
    "A taco with meat and salsa",
    "A bowl of ramen with egg and pork"
]

for i in tqdm(range(100)):
    prompt = prompts[i % len(prompts)] 
    
    # Generate
    if device == "cuda":
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
    else:
        image = pipe(prompt).images[0]
    
    save_path = os.path.join("fake_images_100", f"fake_{i:03d}.png")
    image.save(save_path)

print(f"Generated 100 Fake images to 'fake_images_100/'")
print("\nDATASET COMPLETE! You are ready to run the Classifier.")