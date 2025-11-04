# Kaggle Progressive Training Workflow

## Overview

Complete workflow for iterative model development on Kaggle with:
- ✅ **Parameterized notebooks** (adjust hyperparameters without editing code)
- ✅ **Pre-flight testing** (validate before committing GPU hours)
- ✅ **Progressive training** (warm-start from previous best models)
- ✅ **MLflow integration** (automatic experiment tracking)
- ✅ **Dataset versioning** (track which data was used)

## Quick Start

### 1. Pre-Flight Test (Local, ~2 minutes)

```bash
# Test with 1000 samples, 2 epochs
python3 -c "
import papermill as pm
pm.execute_notebook(
    'jupyter_notebooks/06c_cnn_optimized.ipynb',
    '/tmp/test_06c_preflight.ipynb',
    parameters={
        'USE_SAMPLE': True,
        'SAMPLE_SIZE': 1000,
        'CONFIG': {'epochs': 2, 'batch_size': 32},
        'RUN_NAME': 'preflight-test'
    }
)
"
```

**Expected**: Completes in <2 minutes, validates pipeline works

### 2. Submit to Kaggle (Production Training)

```bash
# Push updated notebook to Kaggle
kaggle kernels push -p kaggle_kernel/
```

### 3. Monitor & Download Results

```bash
# Check status
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status manwithacat/cnn-optimized-training

# Download when complete
./scripts/kaggle_import_full.sh manwithacat/cnn-optimized-training
```

## Progressive Training Workflow

### Iteration 1: Baseline

**Goal**: Establish baseline performance

```json
{
  "CONFIG": {
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 0.001,
    "filters": [64, 128, 256]
  },
  "WARM_START": false,
  "RUN_NAME": "run-01-baseline"
}
```

**Expected**: val_auc ~0.69 (as per previous run)

### Iteration 2: Increased Capacity

**Goal**: Test if model is underfitting

```json
{
  "CONFIG": {
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 0.001,
    "filters": [64, 128, 256, 512],  ← Added layer
    "dense_units": 1024               ← Doubled
  },
  "WARM_START": false,
  "RUN_NAME": "run-02-larger-model"
}
```

**Decision**: If val_auc improves → model was underfitting
           If val_auc same/worse → model capacity is fine

### Iteration 3: Warm-Start from Best

**Goal**: Continue training from best previous model

```json
{
  "CONFIG": {
    "batch_size": 128,
    "epochs": 20,                     ← Additional 20 epochs
    "learning_rate": 0.0001,          ← Lower LR for fine-tuning
    "filters": [64, 128, 256, 512]
  },
  "WARM_START": true,
  "PREVIOUS_MODEL_PATH": "/kaggle/input/run-02-model/models/run-02-larger-model_best.keras",
  "RUN_NAME": "run-03-continued"
}
```

**Expected**: Marginal improvement from extended training

### Iteration 4: Hyperparameter Search

**Goal**: Optimize learning rate and regularization

```json
{
  "CONFIG": {
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 0.002,           ← Doubled
    "dropout_rate": 0.6,               ← Increased
    "l2_reg": 0.0005,                  ← Increased
    "filters": [64, 128, 256, 512]
  },
  "WARM_START": false,
  "RUN_NAME": "run-04-higher-lr"
}
```

**Decision**: Compare all runs in MLflow to find best configuration

## Parameter Reference

### CONFIG Dictionary

```python
CONFIG = {
    # Image dimensions
    'img_height': 224,        # Image height (px)
    'img_width': 224,         # Image width (px)
    'channels': 3,            # RGB channels

    # Training
    'batch_size': 128,        # Samples per batch (32-256)
    'epochs': 30,             # Training epochs (10-50)
    'learning_rate': 0.001,   # Learning rate (0.0001-0.01)

    # Architecture
    'filters': [64, 128, 256],         # Conv filters per block
    'dense_units': 512,                # Dense layer size
    'dropout_rate': 0.5,               # Dropout (0.3-0.7)
    'l2_reg': 0.0001,                  # L2 regularization

    # Callbacks
    'early_stopping_patience': 10,     # Early stop patience
    'reduce_lr_patience': 5,           # LR reduction patience

    # Data
    'num_classes': 14,                 # Disease classes
    'random_state': 42                 # Random seed
}
```

### Progressive Training Parameters

```python
WARM_START = False          # Load previous model?
PREVIOUS_MODEL_PATH = None  # Path to previous .keras file
RUN_NAME = "run-name"       # Unique identifier for this run
```

### Testing Parameters

```python
USE_SAMPLE = False          # Use subset of data?
SAMPLE_SIZE = 1000          # Samples when USE_SAMPLE=True
```

## Kaggle Setup

### 1. Create Kaggle Dataset with Previous Model

After a successful run, create a dataset with the trained model:

```bash
# 1. Download model from completed run
./scripts/kaggle_import_full.sh manwithacat/run-01-baseline

# 2. Create Kaggle dataset
mkdir /tmp/kaggle_dataset_run01
cp models/saved_models_kaggle/run-01-baseline_best.keras /tmp/kaggle_dataset_run01/

# 3. Create dataset metadata
cat > /tmp/kaggle_dataset_run01/dataset-metadata.json <<EOF
{
  "title": "NIH X-Ray CNN Model - Run 01 Baseline",
  "id": "manwithacat/run-01-model",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# 4. Push to Kaggle
cd /tmp/kaggle_dataset_run01
KAGGLE_CONFIG_DIR="/path/to/.kaggle" kaggle datasets create
```

### 2. Add Dataset as Input to Kernel

In your Kaggle kernel settings:
- **Add Input** → **Datasets** → Search for "manwithacat/run-01-model"
- Model will be available at: `/kaggle/input/run-01-model/run-01-baseline_best.keras`

### 3. Update Kernel Metadata

Edit `kaggle_kernel/kernel-metadata.json`:

```json
{
  "id": "manwithacat/cnn-optimized-training",
  "title": "CNN Optimized Training - Run 02",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [
    "nih-chest-xrays/data",
    "manwithacat/nih-chest-xray-splits",
    "manwithacat/run-01-model"         ← Add this
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

## Pre-Flight Testing

### Local Test (Recommended)

```bash
# Test with papermill
papermill \
  jupyter_notebooks/06c_cnn_optimized.ipynb \
  /tmp/test_06c_preflight.ipynb \
  -f /tmp/test_params.json

# Where test_params.json contains:
{
  "USE_SAMPLE": true,
  "SAMPLE_SIZE": 1000,
  "CONFIG": {
    "epochs": 2,
    "batch_size": 32
  },
  "RUN_NAME": "preflight-test"
}
```

**Validates**:
- ✅ Data loading works
- ✅ Model builds correctly
- ✅ Training loop runs
- ✅ No OOM errors
- ✅ Callbacks save files

**Time**: ~2 minutes

### Kaggle Test Run

For Kaggle-specific issues:

```json
{
  "USE_SAMPLE": true,
  "SAMPLE_SIZE": 5000,
  "CONFIG": {
    "epochs": 3,
    "batch_size": 128
  },
  "RUN_NAME": "kaggle-preflight"
}
```

**Time**: ~10 minutes on Kaggle P100

## MLflow Integration

### Automatic Tracking

Every Kaggle run is automatically imported to MLflow with:

**Parameters**:
- All CONFIG values
- Dataset hashes (train/val/test)
- WARM_START status
- RUN_NAME

**Metrics**:
- Per-epoch: loss, auc, accuracy, precision, recall
- Final: best validation metrics
- Training time: total and per-epoch

**Artifacts**:
- Model (.keras file, 1-2GB)
- Training history CSV
- Training summary JSON
- Kaggle execution log

**Tags**:
- platform: kaggle
- gpu: P100
- run_name: run-01-baseline
- warm_start: true/false

### Comparing Runs in MLflow

```bash
# Start MLflow UI
make mlflow-ui

# Or
./scripts/mlflow_start.sh
```

**Navigate to**: http://localhost:5001

**Compare runs**:
1. Select multiple runs (checkbox)
2. Click "Compare"
3. View side-by-side metrics, parameters, and charts

**Find best run**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = mlflow.get_experiment_by_name("cnn-custom")

# Get best run by val_auc
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.platform = 'kaggle'",
    order_by=["metrics.final_val_auc DESC"],
    max_results=1
)

best_run = runs[0]
print(f"Best run: {best_run.data.tags['run_name']}")
print(f"Val AUC: {best_run.data.metrics['final_val_auc']:.4f}")
```

## Example Progressive Workflow

### Week 1: Baseline Exploration

**Run 01** (baseline):
- batch_size=128, epochs=30, filters=[64,128,256]
- Result: val_auc=0.692
- Time: 6 hours

**Run 02** (larger):
- batch_size=128, epochs=30, filters=[64,128,256,512], dense=1024
- Result: val_auc=0.705 ← **Improvement!**
- Time: 8 hours

**Decision**: Model benefits from increased capacity

### Week 2: Fine-Tuning

**Run 03** (warm-start):
- WARM_START=True from run-02
- epochs=20, learning_rate=0.0001
- Result: val_auc=0.712 ← **Marginal improvement**
- Time: 5 hours

**Run 04** (higher LR):
- batch_size=128, epochs=30, learning_rate=0.002
- Result: val_auc=0.698 ← **Worse, too high**
- Time: 8 hours

**Run 05** (optimal):
- batch_size=128, epochs=40, learning_rate=0.001, dropout=0.6
- Result: val_auc=0.718 ← **Best so far!**
- Time: 10 hours

**Decision**: Run 05 is production candidate

### Week 3: Production Model

**Run 06** (production):
- WARM_START=True from run-05
- epochs=10 (additional fine-tuning)
- learning_rate=0.00005 (very low for stability)
- Result: val_auc=0.720
- Time: 2.5 hours

**Total**: 6 runs, ~40 GPU hours, val_auc improved from 0.692 → 0.720 (4% gain)

## Tips & Best Practices

### 1. Name Runs Descriptively

```python
RUN_NAME = "run-03-large-model-dropout06"  # Good
RUN_NAME = "test123"                       # Bad
```

### 2. Document Each Run

Create a runs log:

```
| Run | Goal | Config Changes | Val AUC | Notes |
|-----|------|----------------|---------|-------|
| 01  | Baseline | - | 0.692 | Good starting point |
| 02  | Larger model | +1 layer, 2x dense | 0.705 | Helps! |
| 03  | Continue 02 | Warm-start, low LR | 0.712 | Marginal |
```

### 3. One Change at a Time

Don't change multiple things simultaneously:
- ❌ Changed batch_size AND learning_rate AND architecture
- ✅ Only changed learning_rate

### 4. Save Successful Models as Datasets

After each successful run:
- Download model
- Create Kaggle dataset
- Easy to load for warm-starting

### 5. Monitor GPU Usage

In Kaggle kernel, add:

```python
# Check GPU utilization
!nvidia-smi
```

Target: 80-90% GPU utilization

## Troubleshooting

### OOM Error with Warm-Start

**Problem**: Loading previous large model causes OOM

**Solution**:
```python
# Clear session before loading
from tensorflow.keras import backend as K
K.clear_session()

# Then load model
model = keras.models.load_model(PREVIOUS_MODEL_PATH)
```

### Model Not Loading

**Problem**: `Unable to load model from /kaggle/input/...`

**Check**:
1. Dataset added to kernel inputs?
2. Path correct? Use `!ls /kaggle/input/` to verify
3. Model saved with correct format? (should be `.keras` not `.h5`)

### Parameters Not Applying

**Problem**: papermill parameters not taking effect

**Check**:
1. Parameters cell tagged with `tags=["parameters"]`?
2. Using `-f params.json` (file) not `-p` (inline)?
3. JSON syntax correct?

### Run Names Colliding

**Problem**: Multiple runs overwriting each other

**Solution**: Always use unique RUN_NAME:

```python
import datetime
RUN_NAME = f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

## Next Steps

1. **Run baseline** (establish performance floor)
2. **Experiment systematically** (one change at a time)
3. **Compare in MLflow** (visualize progress)
4. **Iterate** (warm-start from best models)
5. **Deploy best** (promote to production in MLflow registry)

## See Also

- [KAGGLE_MLFLOW_IMPORT.md](./KAGGLE_MLFLOW_IMPORT.md) - MLflow import details
- [MLFLOW_QUICKSTART.md](./MLFLOW_QUICKSTART.md) - MLflow basics
- [NOTEBOOK_07B_PYTORCH_SUMMARY.md](./NOTEBOOK_07B_PYTORCH_SUMMARY.md) - Transfer learning workflow
