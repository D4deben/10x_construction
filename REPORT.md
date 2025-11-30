## Drywall Prompted Segmentation – Report

- Task: Text‑conditioned segmentation with prompts: “segment taping area” (drywall seams) and “segment crack”.
- Dataset: Roboflow Drywall‑Join‑Detect (YOLOv8). Cracks dataset pending published version for API download.
- Model: CLIP text encoder (ViT-B/32) + lightweight UNet‑style decoder with text conditioning; focal loss; data augmentations.

### Data Splits
- Train: populated from YOLOv8 `train/` split
- Validation: populated from `valid/` split
- Test: populated from `test/` split (if available)
- Exact counts are printed in the notebook during entries collection.

### Training & Runtime
- Training: 6 epochs, AdamW (lr=3e-4), focal loss (α=0.75, γ=2.0), batch size 8, CPU in this environment.
- Augmentations: resize to 352, color jitter, horizontal flip.
- Runtime: epoch losses and total time printed in notebook logs.

### Validation Metrics (threshold sweep)
- Thresholds evaluated: 0.3, 0.4, 0.5, 0.6
- Reported metrics: mean IoU and Dice across validation batches.
- Best operating point: printed in the notebook as “Best: (thr, IoU, Dice)”.

### Outputs
- Improved test masks saved to: `pred_masks_improved/` with filenames `imageid__prompt.png` (prompt spaces replaced by `_`).
- Summary overlays rendered in the notebook (gallery cell) to visually verify alignment.

### Notes & Next Steps
- Visual QA added to confirm mask rasterization from YOLOv8 polygons (normalized coords correctly denormalized to image size; masks resized with nearest interpolation).
- If validation metrics remain low, consider: stronger image encoder, additional augmentations, class‑balanced sampling, and integrating the cracks dataset when available.
- For full assignment scope, integrate Cracks and re‑run training/evaluation with both prompts.

### Reproducibility
- Seeds: fixed via `random`, `numpy`, and `torch` in notebook.
- Environment: see `drywall_qa/requirements.txt` and the venv packages.
