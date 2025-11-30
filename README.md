# Drywall Prompted Segmentation

## 🎯 Goal
Train/fine-tune a text-conditioned segmentation model that produces binary masks given:
	- `segment taping area` (drywall seams)
	- `segment crack` (pending dataset availability)

## Quick Start
- Open `drywall_qa/drywall_segmentation.ipynb` and run cells in order.
- Ensure the Roboflow Drywall YOLOv8 dataset is downloaded under `New/Drywall-Join-Detect-*` or your preferred location.
- The notebook auto‑discovers the dataset root, parses YOLOv8 labels, and builds masks.

## Model & Training
- CLIP text encoder (ViT-B/32) for prompt embeddings.
- Lightweight UNet‑style decoder conditioned on text features.
- Focal loss to mitigate background dominance; basic augmentations.
- Threshold sweep selects the best operating point by validation mIoU/Dice.

## Outputs
- Improved masks: `pred_masks_improved/` named `imageid__prompt.png`.
- Visual QA overlays and a structured `REPORT.md` summarizing splits, metrics, and runtime.

## Notes
- Cracks dataset integration pending a published version; once available, add similar entries collection and retrain.
- See `PROJECT_REPORT.md` and `REPORT.md` for a detailed write‑up.
# Prompted Segmentation for Drywall QA

Text-conditioned segmentation model for drywall quality assurance tasks.

## 🎯 Goal

Train/fine-tune a text-conditioned segmentation model that produces binary masks given:
- **Input**: RGB image + natural language prompt
- **Output**: Binary mask (0/255) for the specified region

### Supported Prompts
- `"segment crack"`, `"segment wall crack"` → Crack detection
- `"segment taping area"`, `"segment joint/tape"`, `"segment drywall seam"` → Taping area detection

## 📁 Project Structure

```
drywall_qa/
├── data/
│   ├── raw/                    # Raw Roboflow datasets
│   └── processed/              # Preprocessed masks and splits
├── outputs/
│   ├── masks/                  # Generated prediction masks
│   └── visualizations/         # Visual comparison images
├── models/                     # Saved model checkpoints
├── drywall_segmentation.ipynb  # Main pipeline notebook
├── requirements.txt
└── README.md
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Roboflow API Key

Get your API key from [Roboflow](https://roboflow.com/) and set it as an environment variable:

```bash
# Windows PowerShell
$env:ROBOFLOW_API_KEY="your_api_key_here"

# Or add to notebook directly
```

### 3. Run the Notebook

Open and run `drywall_segmentation.ipynb` to:
1. Download datasets from Roboflow
2. Preprocess and create train/val/test splits
3. Run baseline CLIPSeg inference
4. Evaluate with mIoU and Dice metrics
5. Generate prediction masks and visualizations

## 📊 Datasets

- **Dataset 1 (Taping Area)**: [drywall-join-detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- **Dataset 2 (Cracks)**: [cracks-3ii36](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)

## 📈 Evaluation Metrics

- **mIoU** (mean Intersection over Union)
- **Dice Coefficient** (F1 for segmentation)
- Per-class breakdown (crack vs. taping area)

## 🔧 Reproducibility

- **Random seed**: 42 (set in all random operations)
- **Data splits**: 70% train, 15% val, 15% test
- **Model**: CLIPSeg (CLIP + segmentation decoder)

## 📝 Output Format

Prediction masks are saved as:
- Format: PNG, single-channel, values {0, 255}
- Filename: `{image_id}__{prompt_slug}.png`
- Example: `123__segment_crack.png`

## 🎓 Grading Rubric (100 pts)

- **Correctness (50 pts)**: mIoU & Dice on both prompt types
- **Consistency (30 pts)**: Stable across varied scenes
- **Presentation (20 pts)**: Clear documentation, seeds noted, report with tables & visuals

## 📄 License

Educational project for AI Research Engineer assignment.
