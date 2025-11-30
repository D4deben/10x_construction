# Drywall QA - Text-Prompted Segmentation

## Project Summary

This project implements a text-conditioned segmentation system for drywall quality assurance, specifically detecting **taping areas** on drywall surfaces using natural language prompts.

### Problem Statement
Given an image and a natural language prompt (e.g., "segment taping area"), the model produces a binary segmentation mask where pixels are marked as 0 (background) or 255 (target region).

---

## Results

### Baseline Performance (CLIPSeg without fine-tuning)

**Validation Set Metrics:**
- **Mean IoU**: 0.0156 ± 0.0614
- **Mean Dice**: 0.0251 ± 0.0935  
- **Median IoU**: 0.0000
- **Median Dice**: 0.0000

**Dataset:** 1680 samples total
- Training: 1386 samples (taping areas)
- Validation: 294 samples (taping areas)

**Model:** CLIPSeg (CIDAS/clipseg-rd64-refined)
- Size: 150.75M parameters
- Device: CPU
- Input: RGB images (640x640)
- Output: Binary masks (0/255)

### Key Findings

1. **Baseline Limitations**: The pre-trained CLIPSeg model shows very limited performance on this specific task without domain-specific fine-tuning
2. **Occasional Success**: Some samples achieve moderate scores (IoU up to 0.44), suggesting the model has some understanding of the task
3. **Inconsistency**: Median metrics of 0.0 indicate most predictions are failing to capture the target regions

### Sample Visualizations

The project includes side-by-side comparisons of:
- Original drywall images
- Ground truth masks (from bounding box annotations)
- Model predictions
- Per-sample IoU and Dice scores

See `outputs/visualizations/predictions_comparison.png` for examples.

---

## Project Structure

```
drywall_qa/
├── data/
│   ├── raw/                    # Downloaded datasets
│   │   └── taping/            # Taping area dataset
│   │       ├── train/         # 1386 training images
│   │       └── valid/         # 294 validation images
│   └── processed/             # Processed binary masks
│       └── taping/
│           ├── train/masks/   # Training masks
│           └── valid/masks/   # Validation masks
├── outputs/
│   ├── masks/                 # Predicted masks
│   ├── visualizations/        # Comparison plots
│   ├── validation_results.csv # Per-sample metrics
│   └── evaluation_report.txt  # Final report
├── models/                    # Model checkpoints (if saved)
├── drywall_segmentation.ipynb # Main pipeline notebook
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0
- transformers>=4.35
- opencv-python
- roboflow>=1.1
- pandas
- matplotlib
- seaborn
- tqdm
- scikit-learn

### 2. Dataset Preparation

The taping dataset was downloaded from Roboflow:
- **Source**: objectdetect-pu6rn/drywall-join-detect (version 1)
- **Format**: COCO format with bounding box annotations
- **Preprocessing**: Bounding boxes converted to binary masks (0/255)

### 3. Run the Pipeline

Open and execute `drywall_segmentation.ipynb` sequentially:

1. **Setup** - Import libraries and configure paths
2. **Download Data** - Fetch datasets from Roboflow
3. **Preprocessing** - Convert annotations to binary masks
4. **Model Loading** - Load pre-trained CLIPSeg model
5. **Inference** - Run predictions on validation set
6. **Evaluation** - Calculate IoU and Dice metrics
7. **Visualization** - Generate comparison plots
8. **Reporting** - Create evaluation summary

---

## Evaluation Metrics

### Intersection over Union (IoU)

```
IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)
```

- Range: [0, 1]
- Higher is better
- 0 = no overlap, 1 = perfect match

### Dice Coefficient

```
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```

- Range: [0, 1]  
- Higher is better
- More sensitive to small differences than IoU

---

## Outputs

### Generated Files

1. **`data/processed/dataset_metadata.csv`**
   - Image paths, mask paths, prompts, splits
   - 1680 rows total

2. **`outputs/validation_results.csv`**
   - Per-sample metrics: image_id, IoU, Dice
   - 294 validation samples

3. **`outputs/evaluation_report.txt`**
   - Summary statistics and model information
   - Overall and per-dataset metrics

4. **`outputs/visualizations/predictions_comparison.png`**
   - 4 sample comparisons (original, ground truth, prediction)
   - IoU and Dice scores displayed

5. **`outputs/masks/valid/*.png`**
   - Binary prediction masks for validation samples

---

## Recommendations for Improvement

### 1. Fine-Tuning (Priority: HIGH)
- Train CLIPSeg on the drywall dataset
- Use data augmentation (rotation, brightness, contrast)
- Implement curriculum learning (easy → hard samples)

### 2. Better Annotations (Priority: HIGH)
- Current masks from bounding boxes are approximate
- Obtain polygon or pixel-level annotations for taping areas
- Reduces label noise in training

### 3. Architectural Improvements (Priority: MEDIUM)
- Try SAM (Segment Anything Model) with text conditioning
- Experiment with CLIPSeg variants (different backbones)
- Ensemble multiple models

### 4. Data Augmentation (Priority: MEDIUM)
- Lighting variations common in construction sites
- Perspective transforms for camera angles
- Synthetic data generation

### 5. Prompt Engineering (Priority: LOW)
- Test alternative prompts: "drywall tape", "joint compound", "seam"
- Multi-prompt ensembling
- Few-shot prompt examples

---

## Technical Notes

### Limitations

1. **Object Detection → Segmentation Mismatch**
   - Dataset has bounding box annotations, not true segmentation masks
   - Rectangular masks don't capture actual taping area shapes
   - Limits model learning

2. **No Crack Dataset**
   - Original plan included crack detection
   - Crack dataset not available on Roboflow
   - Project focused on taping only

3. **CPU Inference**
   - All processing done on CPU (no CUDA available)
   - Inference time: ~1 second per image
   - Full validation: ~5 minutes (294 images)

4. **Binary Threshold**
   - Fixed threshold (0.5) for binary masks
   - Could be optimized per dataset

### Code Quality

- Type hints for function signatures
- Docstrings for all major functions  
- Progress bars for long operations
- Error handling in inference loop
- Reproducible (random seed = 42)

---

## Assignment Requirements Checklist

- ✅ **Dataset**: Roboflow taping dataset (1680 samples)
- ✅ **Model**: CLIPSeg text-conditioned segmentation
- ✅ **Preprocessing**: Bounding boxes → binary masks (0/255 format)
- ✅ **Inference**: Validation set predictions (294 samples)
- ✅ **Metrics**: IoU and Dice coefficient computed
- ✅ **Outputs**: PNG masks in `outputs/masks/`
- ✅ **Visualizations**: 4 comparison images with metrics
- ✅ **Report**: Summary with mean/std metrics

---

## References

- **CLIPSeg**: [GitHub](https://github.com/timojl/clipseg) | [Paper](https://arxiv.org/abs/2112.10003)
- **Roboflow**: [Dataset Platform](https://roboflow.com/)
- **Transformers**: [Hugging Face](https://huggingface.co/docs/transformers)

---

## Contact

For questions or improvements, please create an issue or submit a pull request.

**Last Updated**: 2025
