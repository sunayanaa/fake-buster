# Spectral Forensics: Detecting Diffusion-Generated Imagery via $1/f^\\beta$ Power Law Violations



This repository contains the official implementation code for the paper \*\*"Spectral Forensics: Detecting Diffusion-Generated Imagery via $1/f^\\beta$ Power Law Violations"\*\* (IEEE Signal Processing Letters submission `SPL-45372-2025.R1`).



## Overview



As generative AI democratizes the creation of hyper-realistic media, distinguishing authentic photography from synthetic imagery has become a critical information security priority. Current deep-learning-based forensic detectors function as opaque "black boxes" and struggle to generalize across evolving diffusion architectures. 



This project proposes a robust, explainable, physics-based forensic framework. We hypothesize that the iterative upsampling operations in Latent Diffusion Models (LDMs) introduce systematic violations of the natural $1/f^\\beta$ power-law decay in the frequency domain. By analyzing the azimuthally averaged radial power spectrum, our pipeline extracts a 1D spectral profile to identify a distinct high-frequency "energy surplus" unique to diffusion synthesis. A lightweight linear classifier (Logistic Regression) is then used to achieve highly accurate and verifiable detection.



\## Repository Structure



The codebase is organized into two phases: the initial baseline experiments (focusing on Stable Diffusion v1.5) and the expanded revision experiments (focusing on SDXL, DALL-E 3, and ablation studies).



\### Phase 1: Baseline Experiments (Initial Submission)

These scripts establish the core methodology, using the `Food-101` dataset and Stable Diffusion v1.5.



\* `get-100-images.py`: Downloads 100 high-resolution real images from the Hugging Face `food101` dataset and generates 100 synthetic counterparts using the `runwayml/stable-diffusion-v1-5` pipeline.

\* `remove-black-images.py`: A utility script to clean the dataset by automatically detecting and removing corrupted or solid black images (often triggered by diffusion safety checkers).

\* `spectral-analysis.py`: Computes the 2D Discrete Fourier Transform (DFT), extracts the 1D azimuthally averaged radial profile, and generates the comparative visualization plot showing the high-frequency energy surplus.

\* `trust-classifier.py`: Trains a lightweight Logistic Regression model on the extracted 300-dimensional spectral features and evaluates accuracy using an 80/20 train/test split.

\* `unzipscript.py`: A simple utility to extract pre-downloaded dataset zip archives if running in a fresh environment.



\### Phase 2: Expanded Generalization \& Ablations (Revision)

These scripts were introduced to rigorously validate cross-architecture generalization and justify architectural choices via ablation studies.



\* `dataset\_expansion\_sdxl\_nature.py`: Generates 100 high-resolution synthetic images using the `stabilityai/stable-diffusion-xl-base-1.0` (SDXL) model, utilizing nature-based prompts. Designed to run on a T4 GPU and save directly to Google Drive.

\* `fetch\_real\_nature\_images\_fix.py`: Downloads 100 real, complex natural images from the `huggan/flowers-102-categories` dataset to serve as the authentic baseline for the SDXL/DALL-E 3 evaluations.

\* `run\_ablation\_and\_generalization.py`: The master evaluation script for the revision. It performs feature extraction and runs a 5-fold cross-validation suite. It includes:


&nbsp;   \* \*\*Classifier Ablation:\*\* Compares Logistic Regression, SVM (RBF kernel), and Random Forest to prove linear separability.

&nbsp;   \* \*\*Feature Ablation:\*\* Compares the proposed 300-dimensional 1D azimuthal profile against a baseline using the raw flattened 2D power spectrum (16,384 features).

&nbsp;   \* \*\*Cross-Architecture Evaluation:\*\* Tests the pipeline on the SDXL vs. Flowers-102 dataset.

\* `evaluate\_dalle3\_generalization.py`: Processes raw DALL-E 3 images, extracts 1D radial profiles, and evaluates them against the real nature baseline using the pre-established Logistic Regression model


\*(Note: The DALL-E 3 images utilized in the final paper evaluation were sourced from the public Kaggle repository `sunayanaa/dalle3-100`.)\*



\## Prerequisites \& Installation



To run the code, you will need Python 3.8+ and a CUDA-enabled GPU (highly recommended for generating diffusion images). The required libraries can be installed via pip:



```bash

pip install torch diffusers transformers accelerate datasets scipy scikit-learn matplotlib Pillow numpy tqdm



