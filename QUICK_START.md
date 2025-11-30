# Quick Start Guide - Drywall QA Segmentation

## Run the Complete Pipeline in 5 Minutes

### Prerequisites
```bash
# Install Python 3.8+
# Ensure pip is up to date
python -m pip install --upgrade pip
```

### Step 1: Install Dependencies
```bash
cd drywall_qa
pip install -r requirements.txt
```

### Step 2: Set Up Roboflow API Key
Open `drywall_segmentation.ipynb` and set your API key:
```python
ROBOFLOW_API_KEY = "your_api_key_here"
```

### Step 3: Run All Cells
Execute the notebook sequentially (Shift+Enter or Run All).

**Expected Runtime:**
- Dataset download: ~30 seconds
- Preprocessing: ~3 seconds
- Model loading: ~1 minute
- Validation inference: ~5 minutes
- Total: **~7 minutes**

### Step 4: View Results

**Outputs:**
```
outputs/
├── validation_results.csv         # Metrics per image
├── evaluation_report.txt          # Summary statistics
├── visualizations/
│   └── predictions_comparison.png # Visual comparisons
└── masks/valid/*.png              # Predicted masks
```

**Key Metrics:**
- Mean IoU: 0.0156
- Mean Dice: 0.0251

---

## Alternative: Run Specific Sections Only

### Just Preprocessing
Run cells 1-12:
- Setup → Download → Preprocess
- Output: `data/processed/` with 1680 binary masks

### Just Inference
Assuming preprocessing is done, run cells 13-26:
- Load model → Run inference → Calculate metrics
- Output: `outputs/validation_results.csv`

### Just Visualization
Assuming inference is done, run cells 27-30:
- Generate plots → Create report
- Output: `outputs/visualizations/` and report

---

## Troubleshooting

### Issue: "CUDA not available"
**Solution:** This is normal. The code runs on CPU by default.

### Issue: "Dataset directory not found"
**Solution:** Check `ROBOFLOW_API_KEY` is set correctly and rerun download cell.

### Issue: "No samples processed"
**Solution:** Ensure preprocessing cell (cell 11) completed successfully. Check `data/processed/taping/` exists.

### Issue: Slow inference
**Solution:** Expected on CPU. For faster processing:
- Use GPU if available (code will auto-detect)
- Reduce validation set size in cell 22

---

## File Structure After Running

```
drywall_qa/
├── data/
│   ├── raw/taping/              # Downloaded (32 MB)
│   │   ├── train/ (1386 images)
│   │   └── valid/ (294 images)
│   └── processed/               # Generated (15 MB)
│       └── dataset_metadata.csv
├── outputs/                     # Generated (10 MB)
│   ├── validation_results.csv
│   ├── evaluation_report.txt
│   ├── masks/valid/ (294 PNGs)
│   └── visualizations/
│       └── predictions_comparison.png
├── drywall_segmentation.ipynb   # Main notebook
├── requirements.txt
└── PROJECT_REPORT.md            # Full documentation
```

Total disk space: **~60 MB**

---

## Next Steps

### Experiment with the Model
1. **Try different prompts** (Cell 5):
   ```python
   DATASETS = {
       'taping': {
           'prompts': ['segment tape', 'find taping area', 'drywall joint']
       }
   }
   ```

2. **Adjust binary threshold** (Cell 19):
   ```python
   def predict_mask(..., threshold=0.3):  # Try 0.3 instead of 0.5
   ```

3. **Visualize more samples** (Cell 29):
   ```python
   visualize_predictions(..., n_samples=10)  # Show 10 instead of 4
   ```

### Improve Performance
See `PROJECT_REPORT.md` → "Recommendations for Improvement" section.

---

## Common Questions

**Q: Why are metrics so low?**
A: CLIPSeg is not fine-tuned on drywall images. Expected behavior for baseline.

**Q: How to fine-tune the model?**
A: Would require training loop (not in scope for this assignment). See transformers `Trainer` API.

**Q: Can I use my own images?**
A: Yes! Add them to `data/raw/custom/` with COCO format annotations, update `DATASETS` config.

**Q: What about crack detection?**
A: Crack dataset was unavailable. You can add it by:
1. Finding a crack segmentation dataset
2. Adding to `DATASETS` config
3. Rerunning preprocessing and inference

---

## Performance Benchmarks

**Hardware:** CPU (no GPU)
**Model:** CLIPSeg (150M params)

| Operation          | Time      | Memory  |
|--------------------|-----------|---------|
| Model Loading      | 60s       | 600 MB  |
| Preprocessing      | 3s        | 100 MB  |
| Inference (1 img)  | 1s        | 200 MB  |
| Full Validation    | 5 min     | 800 MB  |

**With GPU (estimate):**
- Inference: ~0.1s per image
- Full validation: ~30 seconds

---

## Support

For issues, check:
1. Python version ≥3.8
2. All requirements installed: `pip list | grep torch`
3. Disk space available: ~100 MB free
4. Internet connection (for model download)

**Still stuck?** Check error messages in notebook output cells.
