# Spleen Segmentation

3D spleen segmentation from CT scans using MONAI and a 3D UNet trained on the Task09_Spleen dataset (Medical Segmentation Decathlon).

## Notebook Steps

1. **Setup** — Installs MONAI, tqdm, matplotlib, and pins numpy 1.26.4 for compatibility. Imports modules and verifies NumPy compatibility.

2. **Data Download** — Downloads and extracts the Task09_Spleen NIfTI dataset (~1.5 GB) from MONAI's public S3 resource.

3. **NIfTI Visualization** — Loads a NIfTI file and displays axial, coronal, and sagittal views using matplotlib.

4. **Dataset Preparation** — Applies MONAI transforms: `LoadImaged`, `EnsureChannelFirstd`, `Orientationd` (RAS), `Spacingd` (1.5×1.5×2.0 mm), `ScaleIntensityRanged` (HU window −57 to 164), `CropForegroundd`, `RandCropByPosNegLabeld` (128³ patches, pos:neg = 3:1, 4 samples), `DivisiblePadd`, `RandFlipd`, `RandRotate90d`, and `RandGaussianNoised`. Uses `CacheDataset` and `pad_list_data_collate` for batching.

5. **Image & Mask Visualization** — Verifies the segmentation mask aligns with the spleen anatomy on a central slice.

6. **Model Definition** — Defines a 3D UNet (`in_channels=1`, `out_channels=2`, channels=(16, 32, 64, 128, 256), `strides=(2, 2, 2, 2)`, `num_res_units=2`) with `DiceLoss` (softmax, one-hot) and Adam optimizer (1e-4).

7. **Training (Commented Out)** — Placeholder for a training loop with `ReduceLROnPlateau` scheduler and early stopping (patience 30, max 300 epochs).

8. **Model Predictions** — Loads the best checkpoint (`best_metric_model.pth`), runs inference on a validation batch, and compares the prediction, ground truth, and original image slice.

9. **Full-Volume Inference** — Applies the same preprocessing (without random cropping) and uses `sliding_window_inference` (ROI size 128³, batch size 4) to segment entire volumes.

10. **Post-Processing** — Cleans predictions with `KeepLargestConnectedComponent` (label 1) and `FillHoles` (label 1) to remove small false positives and fill gaps.

11. **Quantitative Evaluation** — Computes the mean Dice score on a 10-sample validation subset using `DiceMetric` (excluding background). Achieved **0.9066**.

12. **Model Explainability (Grad-CAM)** — Uses MONAI's `GradCAM` on layers `model.1` (intermediate), `model.0` (shallow), and a multi-layer comparison (`model.0`–`model.2`) to visualize which regions drive the spleen prediction. Heatmaps are normalized, smoothed (Gaussian σ=2), and inverted for better contrast.

## Model

The best checkpoint is saved as `best_metric_model.pth` and loaded for inference and evaluation.

## Requirements

- MONAI (with all extras)
- PyTorch
- matplotlib, tqdm, numpy 1.26.4
- scipy (for Gaussian blur in Grad-CAM)
