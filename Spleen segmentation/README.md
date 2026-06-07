# Spleen Segmentation

3D spleen segmentation from CT scans using MONAI and a 3D UNet trained on the Task09_Spleen dataset (Medical Segmentation Decathlon).

## Notebook Overview

1. **Setup** — Installs MONAI, tqdm, matplotlib, and pins numpy 1.26.4 for compatibility.
2. **Data Download** — Downloads and extracts the Task09_Spleen NIfTI dataset.
3. **Data Preparation** — MONAI transforms: resampling, intensity scaling, cropping, random cropping with positive/negative sampling, and augmentation.
4. **Model** — 3D UNet with 5 encoder levels and Dice Loss.
5. **Training** — Uses `ReduceLROnPlateau` scheduler and early stopping with patience 30.
6. **Inference** — Sliding-window inference on full volumes.
7. **Post-Processing** — `KeepLargestConnectedComponent` and `FillHoles` to clean predictions.

## Model

The best checkpoint is saved as `best_metric_model.pth`.

## Requirements

- MONAI (with all extras)
- PyTorch
- matplotlib, tqdm, numpy
