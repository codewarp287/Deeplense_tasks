# DeepLense GSoC 2025 Evaluation – Solution Repository

This repository contains my solutions for the **DeepLense GSoC 2025 Evaluation Tests**, including the **Common Test (I)** and **Specific Tests II–V**. Each task has been implemented in **PyTorch** and presented via Jupyter Notebooks, as per the official guidelines.

---

## Contents

| Task | Notebook | Title | Model Used |
|------|----------|-------|------------|
| I | `01_multiclass_classification.ipynb` | Multi-Class Classification | EfficientNet-B0 |
| II | `02_lens_finding.ipynb` | Binary Lens Finding | ResNet-34 |
| III.A | `03a_super_resolution_synthetic.ipynb` | Super-Resolution (Synthetic) | RCAN |
| III.B | `03b_super_resolution_real.ipynb` | Super-Resolution (Real) | Fine-tuned RCAN |
| IV | `04_diffusion_model_generation.ipynb` | Diffusion Models | DDPM + Super-Resolution |
| V | `05_physics_guided_classification.ipynb` | Physics-Guided Classification | EfficientNet + Lensiformer |

---

## Task I – Multi-Class Classification

- **Objective:** Classify lensing images into three categories: `no_sub`, `cdm`, and `axion`.
- **Architecture:** EfficientNet-b0
- **Dataset:** Pre-processed, min-max normalized lensing images.
- **Evaluation:** ROC Curve, AUC Score (per class)

---

## Task II – Lens Finding

- **Objective:** Binary classification to detect presence of strong gravitational lensing.
- **Challenge:** Severe class imbalance (non-lenses >> lenses).
- **Architecture:** ResNet-34
- **Evaluation:** ROC Curve, AUC Score

---

## Task III – Super-Resolution

### Task III.A – Synthetic Dataset

- **Objective:** Upscale low-resolution simulated lensing images using high-resolution counterparts.
- **Model Used:** RCAN (Residual Channel Attention Network)
- **Loss Functions:** MSE, SSIM
- **Evaluation:** PSNR, SSIM, MSE

---

### Task III.B – Real HR/LR Pairs

- **Objective:** Super-resolve real lensing images using limited HR/LR data.
- **Method:** Fine-tuned the RCAN model from Task III.A with additional augmentations.
- **Techniques Used:** Transfer learning, few-shot learning strategies, data augmentation
- **Evaluation:** Same as Task III.A

---

## Task IV – Diffusion Models

- **Objective:** Generate realistic strong lensing images using generative modeling.
- **Approach:**
  - Trained a DDPM model to generate low-resolution `(64, 64)` lensing images (Due to lack of Compute).
  - Then upscaled the generated image using a super-resolution model to reach target resolution of `(150, 150)`.
- **Evaluation:** PSNR, SSIM, sample visualizations

---

## Task V – Physics-Guided ML

- **Objective:** Improve classification using a physics-informed model.
- **Integration:** Combined **EfficientNet-B0** with an **inverse lens law module** from **Lensiformer**.
- **Benefit:** Enforced physically consistent learning in the architecture.
- **Evaluation:** AUC

---

## Evaluation Summary

| Task | Metrics | Best Scores |
|------|---------|------------------------|
| I | AUC (3 classes) | 0.9913 |
| II | AUC | 0.9990 |
| III.A | PSNR / SSIM | 42.32/0.9745 |
| III.B | PSNR / SSIM | 29.95/0.8217 |
| IV | PSNR / SSIM | 22.0495/0.9306 |
| V | AUC | 0.9933 |


---

## Running the Notebooks

- All notebooks are self-contained and runnable
- Make sure to adjust dataset paths as per your local setup
- Python dependencies: `torch`, `torchvision`, `efficientnet_pytorch`, `scikit-image`, etc.

---
