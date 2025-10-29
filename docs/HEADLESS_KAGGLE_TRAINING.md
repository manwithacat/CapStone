# Headless Kaggle Training Workflow

Train your CNN on Kaggle's P100 GPU **without ever opening a browser** - fully automated via Kaggle API.

## 🎯 What This Does

1. ✅ Packages your splits/metadata
2. ✅ Uploads to Kaggle as a dataset
3. ✅ Pushes notebook 06b to Kaggle
4. ✅ Starts training on P100 GPU
5. ✅ Monitors execution progress
6. ✅ Downloads trained model when complete

**All from your terminal. No browser needed.**

---

## 📋 Prerequisites

### 1. Kaggle API Key

Already set up at `.kaggle/kaggle.json`

To verify:
```bash
ls -la .kaggle/kaggle.json
# Should show: -rw------- (permissions 600)
```

### 2. Set Your Kaggle Username

**Option 1: Via .env file (recommended)**

```bash
# Edit .env file in project root
# Uncomment and set:
KAGGLE_USERNAME=your-actual-username
```

**Option 2: Via environment variable**

```bash
export KAGGLE_USERNAME="your-actual-username"  # Get from https://www.kaggle.com/settings
```

The .env method is recommended as it persists across terminal sessions.

---

## 🚀 Quick Start (One Command)

```bash
# Complete end-to-end automated training
./scripts/kaggle_full_pipeline.sh
```

This runs all steps automatically:
- Upload dataset
- Push notebook
- Train on P100 GPU
- Monitor and download

---

## 📖 Step-by-Step (Manual Control)

### Step 1: Upload Dataset (One-Time Setup)

```bash
# Set your username (if not already in .env)
export KAGGLE_USERNAME="your-username"

# Or edit .env file and uncomment KAGGLE_USERNAME

# Run upload script
./scripts/kaggle_upload_dataset.sh
```

**What it does:**
- Packages train/val/test CSVs + metadata (~34 MB)
- Uploads to Kaggle as dataset: `your-username/nih-chest-xray-splits`
- Creates new version if dataset already exists

**Note:** Only need to do this once (or when splits change)

### Step 2: Train Model on Kaggle

```bash
./scripts/kaggle_train_headless.sh
```

**What it does:**
1. Creates kernel metadata (GPU enabled, datasets linked)
2. Pushes notebook `06b_cnn_kaggle.ipynb` to Kaggle
3. Kaggle automatically starts execution
4. Script polls for completion every 30 seconds
5. Downloads results to `models/saved_models_kaggle/`

**Duration:**
- Sample (5K images): ~3-5 minutes
- Full (78K images): ~30-45 minutes

**Output:**
```
models/saved_models_kaggle/
├── cnn_custom_best.keras      (trained model, ~308 MB)
├── reports/
│   ├── 06_cnn_results.json
│   └── 06_cnn_training_history.csv
└── figures/
    └── 06_cnn_training_history.png
```

---

## 📊 Monitoring

### Check Status Manually

```bash
kaggle kernels status your-username/cnn-development-cloud
```

### View Logs

```bash
kaggle kernels output your-username/cnn-development-cloud --path /tmp/kaggle-logs
cat /tmp/kaggle-logs/*.log
```

### Stop Training

Training cannot be stopped via API once started. It will:
- Auto-stop on completion
- Auto-stop on error
- Auto-stop after 9-hour timeout (Kaggle limit)

---

## 🔍 Troubleshooting

### "Dataset not found"

```bash
# List your datasets
kaggle datasets list --user your-username

# If missing, re-run upload
./scripts/kaggle_upload_dataset.sh
```

### "Kernel failed"

Check logs:
```bash
kaggle kernels output your-username/cnn-development-cloud
```

Common issues:
- Dataset not linked correctly → Check `kernel-metadata.json`
- Out of memory → Reduce `batch_size` or `sample_size` in notebook
- Timeout → For full dataset, increase monitoring timeout in script

### "Permission denied"

```bash
# Check Kaggle API key permissions
chmod 600 .kaggle/kaggle.json
```

---

## 🆚 Comparison: Local vs Kaggle

| Aspect | Local (M2 Pro) | Kaggle (P100 GPU) |
|--------|----------------|-------------------|
| **Sample (5K)** | 30-40 min | 3-5 min |
| **Full (78K)** | 6-8 hours | 30-45 min |
| **Setup** | Install tensorflow-metal | Upload dataset once |
| **Data upload** | None (local) | 34 MB (splits only) |
| **Cost** | Free | Free (30 hrs/week) |
| **Automation** | N/A | ✅ Fully headless |

---

## 🎓 What Happens on Kaggle

1. **Notebook pushed** → Creates new kernel version
2. **Auto-execution starts** → GPU allocated, dependencies installed
3. **Data loading** → Loads from two datasets:
   - `nih-chest-xrays` (112K images, already on Kaggle)
   - `your-username/nih-chest-xray-splits` (your uploaded metadata)
4. **Training** → 50 epochs with callbacks (early stopping, checkpoints)
5. **Results saved** → Outputs to `/kaggle/working/`
6. **Kernel completes** → Status changes to "complete"
7. **Download ready** → Results available via API

---

## 🔄 Re-Running Training

To train again with different hyperparameters:

1. Edit `jupyter_notebooks/06b_cnn_kaggle.ipynb` locally
2. Change `CONFIG` values (e.g., `epochs`, `learning_rate`, `batch_size`)
3. Re-run:
```bash
./scripts/kaggle_train_headless.sh
```

This creates a new kernel version and re-trains.

---

## 💡 Advanced: Parallel Training

Train multiple configurations in parallel:

```bash
# Terminal 1: Default config
./scripts/kaggle_train_headless.sh

# Terminal 2: High learning rate (edit notebook first)
# Change kernel slug in script to avoid conflict
KERNEL_SLUG="cnn-development-lr-high" ./scripts/kaggle_train_headless.sh
```

Kaggle allows multiple kernels running simultaneously.

---

## 📦 Downloaded Model Usage

After download completes:

```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('models/saved_models_kaggle/cnn_custom_best.keras')

# Use for predictions
predictions = model.predict(X_test)
```

---

## ✨ Benefits of This Workflow

✅ **Reproducible** - Same environment every time
✅ **Scalable** - Train on GPU without owning one
✅ **Automated** - No manual clicking in UI
✅ **Version controlled** - Notebook in git
✅ **Cost effective** - Free P100 GPU access
✅ **Professional** - Production-ready ML pipeline

This is how real ML teams train models in the cloud!
