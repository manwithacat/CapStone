# Complete Kaggle Training Workflow

## Quick Start (TL;DR)

```bash
# 1. Start training on Kaggle (one command)
./scripts/kaggle_full_pipeline.sh

# 2. Wait for completion (check Kaggle UI or email notification)

# 3. Download results + import MLflow tracking
./scripts/kaggle_download_results.sh

# 4. View results in MLflow UI
make mlflow-ui
# Open: http://localhost:5001
# Filter by: tags.platform = "kaggle"
```

## Complete Workflow

### Prerequisites

**One-Time Setup:**

1. **Kaggle API credentials**
   ```bash
   # Should already exist at:
   ls .kaggle/kaggle.json
   ```

2. **Kaggle username in .env**
   ```bash
   # Edit .env file:
   KAGGLE_USERNAME=your-actual-username
   ```

3. **Dataset uploaded** (first time only)
   ```bash
   # If not already uploaded:
   ./scripts/kaggle_upload_dataset.sh
   ```

### Step 1: Start Training on Kaggle

**Run the full pipeline script:**

```bash
./scripts/kaggle_full_pipeline.sh
```

**What it does:**
1. ✅ Checks configuration (username, credentials)
2. ✅ Uploads dataset to Kaggle (if needed)
3. ✅ Pushes notebook `06c_cnn_optimized.ipynb` to Kaggle
4. ✅ Starts GPU training (P100, batch_size=128, mixed precision)
5. ✅ Provides Kaggle URL to monitor progress

**Expected output:**
```
╔═══════════════════════════════════════════════════════════╗
║  Kaggle Headless Training Pipeline - OPTIMIZED           ║
║  CNN Training on P100 GPU - 12-15× Faster               ║
╚═══════════════════════════════════════════════════════════╝

Configuration:
  Kaggle Username: your-username
  Notebook: jupyter_notebooks/06c_cnn_optimized.ipynb (OPTIMIZED)
  GPU: P100 (batch size 128, mixed precision, tf.data pipeline)

Proceed with training? [Y/n]
```

**Training time:**
- **Projected:** 2.5-3.5 hours for 50 epochs (12-15× faster than baseline)
- **Kaggle limit:** 12 hours (you'll finish in ~30% of the time limit)

### Step 2: Monitor Training

**Option 1: Kaggle UI (Recommended)**

1. Open the URL provided by the script:
   ```
   https://www.kaggle.com/code/your-username/cnn-optimized-training
   ```

2. Watch the output logs in real-time

3. Check status:
   - 🟢 **Running**: Training in progress
   - 🔵 **Complete**: Ready to download results
   - 🔴 **Failed**: Check logs for errors

**Option 2: Email Notification**

Kaggle will email you when the kernel completes.

**Option 3: CLI Status Check**

```bash
# Check status from command line
export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"
kaggle kernels status your-username/cnn-optimized-training
```

### Step 3: Download Results

**After training completes (2.5-3.5 hours):**

```bash
./scripts/kaggle_download_results.sh
```

**What it does:**
1. ✅ Checks kernel status (complete/running/failed)
2. ✅ Downloads all outputs from Kaggle
3. ✅ Copies files to correct locations:
   - Model → `models/saved_models/`
   - Reports → `outputs/reports/`
   - Figures → `outputs/figures/`
4. ✅ **Imports MLflow tracking** → `./mlruns/`
5. ✅ Creates archive in `~/Downloads/` for backup

**Expected output:**
```
📥 Kaggle Results Downloader
============================

🔍 Checking kernel status...
   Status: complete
   URL: https://www.kaggle.com/code/...

✅ Kernel execution complete - ready to download

📥 Downloading results...
✓ Downloaded to: /tmp/...

📂 Copying files to repository...
   ✓ Model: models/saved_models/cnn_optimized_best.keras
   ✓ Reports: outputs/reports/06c_optimized_*.{json,csv}
   ✓ Figures: outputs/figures/06_cnn_*.png
   ✓ Raw archive: ~/Downloads/results_20251101_...

═══════════════════════════════════════════════════════════
Importing MLflow Experiment Tracking
═══════════════════════════════════════════════════════════

✓ Found mlruns directory: /tmp/.../mlruns
📋 Import Summary:
  Source: /tmp/.../mlruns
  Destination: ./mlruns

🔄 Merging Kaggle runs into local MLflow tracking...

  📁 cnn-custom: Merging with existing experiment...
     ✓ Imported run abc123...

✅ MLflow runs imported successfully!
   View in UI: make mlflow-ui
   Filter by: tags.platform = 'kaggle'

📊 Imported Runs Summary:
=========================

Found 1 Kaggle run(s):

  🚀 cnn-optimized-kaggle-50epochs
     Experiment: cnn-custom
     Status: FINISHED
     Platform: kaggle
     GPU: P100
     Source: kaggle/06c_cnn_optimized

✅ Download complete!
```

### Step 4: View Results in MLflow

**Start MLflow UI:**

```bash
make mlflow-ui
# OR
./scripts/mlflow_start.sh
```

**Open in browser:**
```
http://localhost:5001
```

**Filter Kaggle runs:**

In the MLflow UI, add filter:
```
tags.platform = "kaggle"
```

**Compare with local runs:**

1. Select both Kaggle and local runs
2. Click **"Compare"**
3. View side-by-side:
   - Parameters (batch_size, learning_rate, etc.)
   - Metrics (val_auc, training_time, etc.)
   - Training curves
   - Model artifacts

## Troubleshooting

### Issue: Training Failed on Kaggle

**Check:**
1. Open Kaggle notebook URL
2. View error logs at the bottom
3. Common issues:
   - Dataset not found (re-run `kaggle_upload_dataset.sh`)
   - Out of memory (reduce batch_size in notebook)
   - Timeout (unlikely with optimized version)

**Fix:**
```bash
# Re-upload dataset
./scripts/kaggle_upload_dataset.sh

# Re-run training
./scripts/kaggle_train_headless.sh
```

### Issue: Download Script Can't Find Results

**Check:**
```bash
# Verify kernel exists
export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"
kaggle kernels list --mine | grep cnn-optimized
```

**Fix:**
```bash
# Check status
kaggle kernels status your-username/cnn-optimized-training

# If "error" or "canceled", check Kaggle UI for details
```

### Issue: MLflow Import Failed

**Check:**
```bash
# Verify mlruns directory exists in download
ls ~/Downloads/results_*/mlruns
```

**Manual import:**
```bash
./scripts/import_kaggle_mlflow.sh ~/Downloads/results_*/mlruns
```

### Issue: Can't See Kaggle Runs in MLflow UI

**Check:**
1. MLflow server is running: `make mlflow-ui`
2. Import completed successfully (see output above)
3. Using correct filter: `tags.platform = "kaggle"`

**Fix:**
```bash
# Restart MLflow server
pkill -f "mlflow server"
./scripts/mlflow_start.sh

# Open http://localhost:5001
```

## Advanced Usage

### Re-run Training with Different Parameters

**Edit the notebook:**

```bash
# Edit in jupytext format (easier)
vim jupyter_notebooks/06c_cnn_optimized.py

# Change CONFIG dictionary
CONFIG = {
    'batch_size': 256,  # ← Increase batch size
    'epochs': 100,      # ← More epochs
    # ...
}

# Sync back to .ipynb
jupytext --sync jupyter_notebooks/06c_cnn_optimized.py

# Re-run
./scripts/kaggle_train_headless.sh
```

**Update run name:**

In `06c_cnn_optimized.py`, change:
```python
run_name="cnn-optimized-kaggle-100epochs-batch256"
```

This creates a new MLflow run you can compare with the original.

### Download Partial Results

If training is still running but you want to see progress:

```bash
./scripts/kaggle_download_results.sh
# When prompted: "Download partial results anyway? (y/N)"
# Enter: y
```

### Manual Kaggle API Commands

```bash
# Set config directory
export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"

# List your kernels
kaggle kernels list --mine

# Check status
kaggle kernels status your-username/cnn-optimized-training

# Download outputs
kaggle kernels output your-username/cnn-optimized-training -p ./downloads

# Cancel running kernel
kaggle kernels push --id your-username/cnn-optimized-training --kernel-type notebook --cancel
```

## Performance Expectations

### Baseline (Notebook 06b - NOT Optimized)

- Time per epoch: ~45 minutes
- 50 epochs: ~37.8 hours ❌ (exceeds 12-hour limit)
- Only 15-16 epochs possible

### Optimized (Notebook 06c - WITH Optimizations)

- Time per epoch: ~3-4 minutes ✅
- 50 epochs: ~2.5-3.5 hours ✅ (well within limit)
- Speedup: **12-15×**

### Optimizations Applied

1. **Batch size:** 32 → 128 (4× speedup)
2. **tf.data pipeline:** Prefetching + parallel loading (2-3× speedup)
3. **Mixed precision:** FP16 on P100 (1.5-2× speedup)
4. **Simplified augmentation:** Medical-appropriate only (1.2× speedup)
5. **Reduced model complexity:** 3 blocks vs 4 (1.15-1.25× speedup)

**Combined:** 12-15× total speedup

## Files and Directories

### Local Files Modified/Used

```
scripts/
  ├── kaggle_full_pipeline.sh       # Full end-to-end pipeline
  ├── kaggle_upload_dataset.sh      # Upload splits to Kaggle
  ├── kaggle_train_headless.sh      # Start training
  ├── kaggle_download_results.sh    # Download + import MLflow
  └── import_kaggle_mlflow.sh       # Import MLflow runs

jupyter_notebooks/
  ├── 06c_cnn_optimized.ipynb       # Optimized notebook (for Kaggle)
  └── 06c_cnn_optimized.py          # Jupytext paired format (for editing)

.kaggle/
  └── kaggle.json                    # API credentials (DO NOT COMMIT)

.env
  └── KAGGLE_USERNAME=...            # Your username
```

### Downloaded from Kaggle

```
~/Downloads/results_YYYYMMDD_HHMMSS/
  ├── mlruns/                        # MLflow tracking (imported to ./mlruns)
  ├── models/
  │   └── cnn_optimized_best.keras   # Trained model
  ├── outputs/
  │   ├── reports/
  │   │   ├── 06c_training_summary.json
  │   │   └── 06c_optimized_training_history.csv
  │   └── figures/
  │       └── *.png
  └── download/                       # Raw Kaggle output
```

## Summary

**Complete workflow in 3 commands:**

```bash
# 1. Start training (one-time: ~5 min, training: ~3 hrs)
./scripts/kaggle_full_pipeline.sh

# 2. Download results (after completion)
./scripts/kaggle_download_results.sh

# 3. View in MLflow UI
make mlflow-ui
```

**Total time:** ~3-4 hours from start to viewing results

**Cost:** $0 (free Kaggle GPU)

**Result:** Fully trained CNN model with complete MLflow tracking, ready for evaluation and comparison with local experiments!

## References

- [Kaggle Kernels Documentation](https://www.kaggle.com/docs/kernels)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- MLflow Integration: `docs/MLFLOW_KAGGLE_INTEGRATION.md`
- Jupytext Workflow: `docs/JUPYTEXT_WORKFLOW.md`
