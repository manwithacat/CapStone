# CSV to MLflow Import Guide

## Overview

When training on Kaggle without MLflow (the default), all training metrics are saved to CSV and JSON files. This guide explains how to convert those files into MLflow runs for unified tracking.

## Why This Matters

**Problem:** Kaggle doesn't have MLflow installed by default
**Solution:** Log to CSV during training, convert to MLflow afterward

**Benefits:**
- ✅ Kaggle training works without MLflow dependency
- ✅ Local tracking integrates Kaggle results seamlessly
- ✅ Compare Kaggle vs local runs in MLflow UI
- ✅ Full experiment history in one place

## Automatic Import (Recommended)

The easiest way is to use the download script, which handles everything:

```bash
# After Kaggle training completes
./scripts/kaggle_download_results.sh
```

This script automatically:
1. Downloads results from Kaggle
2. Detects CSV/JSON files
3. Converts to MLflow run
4. Tags with `platform=kaggle`
5. Imports all metrics and parameters

**Expected output:**
```
═══════════════════════════════════════════════════════════
Importing MLflow Experiment Tracking
═══════════════════════════════════════════════════════════

📊 No mlruns directory - checking for CSV/JSON...
   Found CSV: 06c_optimized_training_history.csv
   Found JSON: 06c_training_summary.json

🔄 Converting CSV to MLflow run...
📊 Reading training history: ...
   Found 50 epochs
📋 Reading training summary: ...

🔬 Creating MLflow run in experiment: cnn-custom

📝 Logging 15 parameters...
📈 Logging final metrics...
📊 Logging 50 epochs of metrics...
🏷️  Adding tags...
📎 Logging artifacts...

✅ MLflow run created successfully!
   Run ID: abc123...
   Run Name: kaggle-imported-06c_optimized_training_history
   Experiment: cnn-custom

📊 Logged:
   - 15 parameters
   - 4 final metrics
   - 50 epochs of training metrics
   - 10 tags
   - 2+ artifacts (CSV, JSON, model if available)
```

## Manual Import

If you need to import manually (e.g., re-import with different settings):

```bash
python3 scripts/csv_to_mlflow.py \
    outputs/reports/06c_optimized_training_history.csv \
    outputs/reports/06c_training_summary.json
```

**With custom experiment:**
```bash
python3 scripts/csv_to_mlflow.py \
    outputs/reports/06c_optimized_training_history.csv \
    outputs/reports/06c_training_summary.json \
    --experiment "kaggle-experiments"
```

**From downloaded archive:**
```bash
python3 scripts/csv_to_mlflow.py \
    ~/Downloads/results_20251101/outputs/reports/06c_optimized_training_history.csv \
    ~/Downloads/results_20251101/outputs/reports/06c_training_summary.json
```

## What Gets Imported

### Parameters (from JSON config)
```json
{
  "img_height": 224,
  "img_width": 224,
  "batch_size": 128,
  "epochs": 50,
  "learning_rate": 0.001,
  "filters": "[64, 128, 256]",
  ...
}
```

### Final Metrics (from JSON summary)
```
final_train_loss: 0.234
final_val_loss: 0.256
final_train_auc: 0.891
final_val_auc: 0.854
training_time_hours: 3.2
time_per_epoch_minutes: 3.84
epochs_completed: 50
```

### Per-Epoch Metrics (from CSV)
```
epoch_loss (50 values, step 0-49)
epoch_val_loss (50 values, step 0-49)
epoch_auc (50 values, step 0-49)
epoch_val_auc (50 values, step 0-49)
epoch_accuracy (50 values, step 0-49)
epoch_val_accuracy (50 values, step 0-49)
...
```

### Tags
```
mlflow.source.type: NOTEBOOK
mlflow.source.name: kaggle/06c_cnn_optimized
mlflow.user: kaggle
platform: kaggle
environment: kaggle-p100-gpu
execution_mode: cloud
imported_from: csv
framework: tensorflow
gpu: P100
notebook: 06c
version: optimized
batch_size: 128
```

### Artifacts
- Training history CSV
- Training summary JSON
- Model file (if available)

## Viewing Results

**Start MLflow UI:**
```bash
make mlflow-ui
# OR
./scripts/mlflow_start.sh
```

**Open browser:**
```
http://localhost:5001
```

**Filter Kaggle runs:**
```
tags.platform = "kaggle"
```

**Filter imported CSV runs specifically:**
```
tags.imported_from = "csv"
```

**Compare with local runs:**
1. Select Kaggle run and local run
2. Click "Compare"
3. View metrics side-by-side

## CSV Format Requirements

The script expects:

**Training History CSV:**
```csv
epoch,loss,accuracy,auc,val_loss,val_accuracy,val_auc,...
0,0.5123,0.7234,0.6543,0.5234,0.7123,0.6432,...
1,0.4987,0.7456,0.6789,0.5123,0.7234,0.6543,...
...
```

**Training Summary JSON:**
```json
{
  "config": {
    "img_height": 224,
    "batch_size": 128,
    ...
  },
  "training_time_hours": 3.2,
  "epochs_completed": 50,
  "time_per_epoch_minutes": 3.84,
  "final_metrics": {
    "train_loss": 0.234,
    "val_loss": 0.256,
    "train_auc": 0.891,
    "val_auc": 0.854
  }
}
```

Both files are automatically created by notebook 06c when running on Kaggle.

## Troubleshooting

### Issue: "CSV file not found"

**Check file location:**
```bash
ls outputs/reports/*training_history*.csv
ls ~/Downloads/results_*/outputs/reports/*training_history*.csv
```

**Fix:** Use full path to CSV file

### Issue: "JSON file not found"

**Check file location:**
```bash
ls outputs/reports/*training_summary*.json
ls ~/Downloads/results_*/outputs/reports/*training_summary*.json
```

**Fix:** Use full path to JSON file

### Issue: "MLflow experiment not found"

**Create experiment:**
```bash
python3 -c "import mlflow; mlflow.set_experiment('cnn-custom')"
```

### Issue: "Import creates duplicate runs"

**Check existing runs:**
```bash
# Start MLflow UI and check for existing imports
make mlflow-ui
```

**Note:** Each import creates a NEW run. Don't re-import the same CSV multiple times unless you want duplicates.

### Issue: "Model not logged"

**Check model location:**
```bash
ls models/saved_models/cnn_optimized_best.keras
ls models/saved_models_kaggle/cnn_optimized_best.keras
```

**Note:** Model is optional. CSV/JSON import works without it.

## Advanced Usage

### Import Multiple Kaggle Runs

```bash
# Loop through multiple downloaded results
for dir in ~/Downloads/results_*/; do
    csv="$dir/outputs/reports/06c_optimized_training_history.csv"
    json="$dir/outputs/reports/06c_training_summary.json"

    if [ -f "$csv" ] && [ -f "$json" ]; then
        echo "Importing from $dir"
        python3 scripts/csv_to_mlflow.py "$csv" "$json"
    fi
done
```

### Custom Run Names

Edit the script (`scripts/csv_to_mlflow.py`) line 45:

```python
# Change from:
run_name = f"kaggle-imported-{Path(csv_path).stem}"

# To:
run_name = f"my-custom-name-{Path(csv_path).stem}"
```

### Add Custom Tags

Edit the script, add to the `tags` dictionary (line 76):

```python
tags = {
    # ... existing tags ...
    'my_custom_tag': 'my_value',
    'experiment_group': 'baseline',
}
```

## Script Reference

**Location:** `scripts/csv_to_mlflow.py`

**Dependencies:**
- `mlflow`
- `pandas`
- Standard library (json, pathlib, argparse)

**Usage:**
```bash
python3 scripts/csv_to_mlflow.py <csv_path> <json_path> [--experiment NAME]
```

**Help:**
```bash
python3 scripts/csv_to_mlflow.py --help
```

## Workflow Summary

**On Kaggle:**
1. Notebook runs without MLflow
2. Saves to CSV: `06c_optimized_training_history.csv`
3. Saves to JSON: `06c_training_summary.json`

**After Download:**
```bash
./scripts/kaggle_download_results.sh  # Auto-imports to MLflow
```

**Result:**
- Kaggle run appears in MLflow UI
- Tagged with `platform=kaggle`
- Full training history preserved
- Compare with local runs

**Alternative Manual:**
```bash
python3 scripts/csv_to_mlflow.py outputs/reports/06c*.csv outputs/reports/06c*.json
```

## References

- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
- Kaggle Workflow: `docs/KAGGLE_WORKFLOW.md`
- MLflow Integration: `docs/MLFLOW_KAGGLE_INTEGRATION.md`
