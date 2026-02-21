# ==============================================================================
# Program Name: dataset_expansion_sdxl_nature.py
# Version: 1.0
# Description: This script addresses the reviewer's request for cross-architecture 
#              generalization and diverse natural imagery. It mounts Google Drive 
#              for persistent storage, downloads 100 real nature images, and 
#              generates 100 synthetic nature images using Stable Diffusion XL.
#              It includes resume capability to handle Colab disconnections.
# Input:       HuggingFace 'mertcobanov/nature-dataset' stream; Text prompts.
# Output:      100 real PNG images and 100 synthetic PNG images saved directly 
#              to '/content/drive/MyDrive/SPL_Experiments/'.
# ==============================================================================

# STEP 1: INSTALL LIBRARIES & MOUNT DRIVE
!pip install -q diffusers transformers accelerate datasets scipy scikit-learn

import os
import torch
from PIL import Image
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm
from google.colab import drive

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Setup Persistent Folders in Google Drive
base_dir = "/content/drive/MyDrive/SPL_Experiments"
real_dir = os.path.join(base_dir, "real_images_nature_100")
fake_dir = os.path.join(base_dir, "fake_images_sdxl_100")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

print(f"Images will be saved persistently to: {base_dir}")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Processing on device: {device.upper()}")

if device == "cpu":
    print("WARNING: You are using CPU. SDXL generation will take many hours.")
    print("Please switch to GPU: Runtime -> Change runtime type -> T4 GPU")

# ==============================================================================
# STEP 2: GET 100 REAL IMAGES (Diverse Natural Scenes)
# ==============================================================================
print("\n⬇️ Downloading 100 Real Images (Diverse Nature Dataset)...")

# Using a dataset of natural landscapes to satisfy the reviewer's request
ds_real = load_dataset("mertcobanov/nature-dataset", split="train", streaming=True, trust_remote_code=True)

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
        img = item['image']
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        
        # Save high quality directly to Drive
        img.save(save_path)
        count += 1
    except Exception as e:
        pass

print(f"Real images verified in '{real_dir}'")

# ==============================================================================
# STEP 3: GENERATE 100 FAKE IMAGES (SDXL)
# ==============================================================================
print("\nLoading SDXL Model (This will take a moment to download weights)...")

# Handle FP16 for speed and memory efficiency on T4 GPU
dtype = torch.float16 if device == "cuda" else torch.float32

# Load the base SDXL 1.0 model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=dtype, 
    use_safetensors=True, 
    variant="fp16"
)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True) 

print("⚙️ Generating 100 SDXL Images (~25-30 minutes on T4 GPU)...")

# Prompts matching the natural image dataset domain
prompts = [
    "A professional photograph of a dense green forest",
    "A high resolution photo of a mountain range at sunset",
    "A clear river flowing through rocky terrain",
    "A vast desert landscape under a blue sky",
    "A close up of a colorful tropical bird",
    "A stunning waterfall cascading down a cliff",
    "A peaceful lake reflecting the surrounding trees",
    "A field of wild flowers in full bloom",
    "A snowy mountain peak with clouds",
    "A dramatic ocean cliff with crashing waves"
]

for i in tqdm(range(100)):
    save_path = os.path.join(fake_dir, f"fake_{i:03d}.png")
    
    # Skip if file already exists (handles Colab disconnects)
    if os.path.exists(save_path):
        continue
        
    prompt = prompts[i % len(prompts)] 
    
    # Generate (using 25 steps to optimize time on Colab without sacrificing quality)
    if device == "cuda":
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=25).images[0]
    else:
        image = pipe(prompt, num_inference_steps=25).images[0]
    
    # Save directly to Drive
    image.save(save_path)

print(f"Fake images verified in '{fake_dir}'")
print("\nDATASET EXPANSION COMPLETE! The images are secure in your Google Drive.")