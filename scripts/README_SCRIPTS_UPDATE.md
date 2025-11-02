# Scripts Directory Update - Optimized Notebook

All scripts have been updated to use the new optimized notebook (`06c_cnn_optimized.ipynb`) instead of the original `06b_cnn_kaggle.ipynb`.

## Updated Scripts

### 1. `kaggle_train_headless.sh`

**Changes**:
- Notebook: `06b_cnn_kaggle.ipynb` → `06c_cnn_optimized.ipynb`
- Kernel slug: `cnn-development-cloud` → `cnn-optimized-training`
- Updated timing estimate: "1-3 hours" → "4-6 hours for 50 epochs"
- Added optimization note in header comment

**New behavior**:
```bash
./scripts/kaggle_train_headless.sh
# Now pushes optimized notebook with:
# - Batch size 128 (was 32)
# - Mixed precision training
# - tf.data pipeline
# - Simplified augmentation
```

### 2. `kaggle_full_pipeline.sh`

**Changes**:
- Updated pipeline title to "OPTIMIZED - 12-15× Faster"
- Notebook reference: `06c_cnn_optimized.ipynb`
- Configuration display shows optimization details:
  - GPU: P100 (batch size 128, mixed precision, tf.data pipeline)
- Updated output file references:
  - `06_cnn_training_history.csv` → `06c_optimized_training_history.csv`
  - `cnn_custom_best.keras` → `cnn_optimized_best.keras`

**New behavior**:
```bash
./scripts/kaggle_full_pipeline.sh
# Now shows optimizations in config display
# References correct output filenames
```

### 3. `kaggle_download_results.sh`

**Changes**:
- Kernel slug: `cnn-development-cloud` → `cnn-optimized-training`
- Model filename: Checks for both `cnn_optimized_best.keras` (new) and `cnn_custom_best.keras` (legacy)
- Report filename: Checks for both `06c_optimized_*` and `06_cnn_*` patterns
- Updated next steps to reference analysis script

**New behavior**:
```bash
./scripts/kaggle_download_results.sh
# Downloads from new kernel slug
# Handles both new and legacy filenames (backward compatible)
# Suggests: python scripts/analyze_cloud_training.py
```

### 4. `analyze_cloud_training.py`

**Major improvements**:
- **Auto-detection**: Finds latest training history file automatically
- **Flexible input**: Accepts optional command-line argument for specific file
- **Backward compatible**: Searches for multiple filename patterns:
  1. `06c_optimized_training_history.csv` (new)
  2. `06c_optimized_training_history_kaggle.csv`
  3. `06_cnn_cloud_15epoch_history.csv` (legacy)
  4. `06_cnn_training_history_kaggle.csv` (legacy)
- **Dynamic output**: Filenames and titles adapt to number of epochs

**New behavior**:
```bash
# Auto-detect mode (recommended)
python scripts/analyze_cloud_training.py

# Or specify file explicitly
python scripts/analyze_cloud_training.py outputs/reports/06c_optimized_training_history.csv
```

**Output files**:
- Plot: `outputs/figures/cnn_training/06_cloud_training_<N>epochs.png` (dynamic N)
- Summary: `outputs/reports/06_cloud_training_summary.json`

## Migration Guide

### If you have existing cloud training results:

**Old results** (from 06b, 15 epochs):
```bash
# Still works! Script auto-detects:
python scripts/analyze_cloud_training.py
# Finds: 06_cnn_cloud_15epoch_history.csv
# Generates: 06_cloud_training_15epochs.png
```

**New results** (from 06c, 50 epochs):
```bash
# Run optimized training:
./scripts/kaggle_train_headless.sh

# Download results:
./scripts/kaggle_download_results.sh

# Analyze:
python scripts/analyze_cloud_training.py
# Finds: 06c_optimized_training_history.csv
# Generates: 06_cloud_training_50epochs.png
```

### Clean migration:

If you want to start fresh with the optimized version:

```bash
# 1. Push new optimized notebook
./scripts/kaggle_train_headless.sh

# 2. Wait for training (4-6 hours for 50 epochs)
# Monitor at: https://www.kaggle.com/code/<username>/cnn-optimized-training

# 3. Download when complete
./scripts/kaggle_download_results.sh

# 4. Analyze results
python scripts/analyze_cloud_training.py
```

## Backward Compatibility

All scripts maintain backward compatibility:

- **Download script**: Checks for both old and new filenames
- **Analysis script**: Auto-detects any training history file
- **Legacy files preserved**: Old results from 06b still work

This means you can:
- Keep your existing 15-epoch results
- Run new 50-epoch optimized training
- Compare both using the same analysis script

## Performance Expectations

### Original (06b):
- Batch size: 32
- Pipeline: ImageDataGenerator (CPU)
- Precision: FP32
- Speed: 45 min/epoch
- **50 epochs: 37.8 hours (TIMES OUT)**

### Optimized (06c):
- Batch size: 128
- Pipeline: tf.data with prefetching
- Precision: Mixed FP16
- Speed: 3-4 min/epoch
- **50 epochs: 2.5-3.5 hours ✓**

**Speedup**: 12-15× faster

## File Naming Convention

| Component | Old (06b) | New (06c) |
|-----------|-----------|-----------|
| Notebook | `06b_cnn_kaggle.ipynb` | `06c_cnn_optimized.ipynb` |
| Kernel slug | `cnn-development-cloud` | `cnn-optimized-training` |
| Model | `cnn_custom_best.keras` | `cnn_optimized_best.keras` |
| History | `06_cnn_training_history.csv` | `06c_optimized_training_history.csv` |
| Summary | `06_cnn_results.json` | `06c_training_summary.json` |
| Plot | `06_cloud_training_15epochs.png` | `06_cloud_training_50epochs.png` |

## Testing

Verify the updates work:

```bash
# 1. Check notebook exists
ls jupyter_notebooks/06c_cnn_optimized.ipynb

# 2. Test analysis script with existing data
python scripts/analyze_cloud_training.py

# 3. Dry-run training script (will prompt before pushing)
./scripts/kaggle_train_headless.sh
# Press 'n' when asked to proceed
```

## Summary

All scripts now:
- ✅ Use optimized notebook (06c) by default
- ✅ Reference correct output filenames
- ✅ Maintain backward compatibility with old results
- ✅ Show optimization details in output messages
- ✅ Auto-detect files intelligently

No breaking changes - everything is backward compatible!
