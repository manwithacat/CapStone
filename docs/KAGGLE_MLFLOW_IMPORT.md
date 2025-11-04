# Kaggle → MLflow Import Workflow

## Overview

Complete workflow for downloading Kaggle training results and importing them into MLflow with full tracking, model registration, and dataset versioning.

## Quick Start

### Single Command Import

```bash
./scripts/kaggle_import_full.sh manwithacat/cnn-optimized-training
```

This will:
1. ✅ Check kernel completion status
2. ✅ Download all output files (model, CSV, JSON, logs)
3. ✅ Import to MLflow with full metadata
4. ✅ Track dataset versions (train/val/test splits)
5. ✅ Register model for versioning
6. ✅ Set correct run duration from Kaggle execution time

## Enhanced Features

### 1. ⏱️ Accurate Run Duration

**Problem**: Previous imports used "now" as the run timestamp
**Solution**: Parses Kaggle log timestamps to set correct start/end times

```python
# MLflow run will show actual Kaggle execution time
start_time: 2025-11-01 23:03:39
end_time:   2025-11-02 03:12:15
duration:   4.14 hours  # Matches Kaggle GPU time
```

**Benefits**:
- Accurate cost tracking
- True training time comparisons
- Correct timeline in MLflow UI

### 2. 📦 Dataset Versioning

**Problem**: No way to know which dataset version was used for training
**Solution**: Logs file hashes and sample counts for all splits

```python
Parameters logged:
  dataset_train_samples: 78831
  dataset_val_samples: 16383
  dataset_test_samples: 16890
  dataset_train_file_hash: 7d89e05bf9676c09
  dataset_val_file_hash: 6c83bf844146a323
  dataset_test_file_hash: 6b8046e19433350f
  dataset_preprocessing_config_hash: bd777177f49834f7
```

**Benefits**:
- Reproducibility: Know exact dataset used
- Version tracking: Detect when data changes
- Comparison: Filter runs by dataset version
- Debugging: Identify if issues are data-related

### 3. 🤖 Model Registration

**Problem**: Models logged as artifacts aren't versioned
**Solution**: Registers models in MLflow Model Registry

```python
Model: cnn-chest-xray-classifier
  Version 1: Kaggle P100 GPU (val_auc: 0.692)
  Version 2: Local training (val_auc: 0.705)
  Version 3: Kaggle with augmentation (val_auc: 0.718)
```

**Benefits**:
- Model versioning (v1, v2, v3...)
- Stage management (Staging → Production)
- Model comparison across runs
- Deployment tracking

## Workflow Steps

### Step 1: Train on Kaggle

Upload and run your notebook on Kaggle with GPU:

```python
# In your notebook (06c_cnn_optimized.ipynb)
# Make sure to save:
# - Training history CSV
# - Training summary JSON
# - Best model (.keras)
```

### Step 2: Check Status

```bash
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status manwithacat/cnn-optimized-training
```

Output:
```
manwithacat/cnn-optimized-training has status "KernelWorkerStatus.COMPLETE"
```

### Step 3: Download & Import

**Option A: Automated Script (Recommended)**

```bash
./scripts/kaggle_import_full.sh manwithacat/cnn-optimized-training
```

**Option B: Manual Steps**

```bash
# 1. Download
mkdir -p /tmp/kaggle_output
cd /tmp/kaggle_output
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels output manwithacat/cnn-optimized-training

# 2. Import
cd /Users/james/CodeInstitute/CapStone
python3 scripts/kaggle_import_enhanced.py \
    /tmp/kaggle_output/outputs/reports/06c_optimized_training_history.csv \
    /tmp/kaggle_output/outputs/reports/06c_training_summary.json \
    /tmp/kaggle_output/models/cnn_optimized_best.keras \
    /tmp/kaggle_output/cnn-optimized-training.log \
    --experiment cnn-custom \
    --kernel-slug manwithacat/cnn-optimized-training \
    --model-name cnn-chest-xray-classifier
```

### Step 4: Verify Import

```bash
# Start MLflow UI
make mlflow-ui

# Or
./scripts/mlflow_start.sh
```

Then visit:
- **Runs**: http://localhost:5001/#/experiments/cnn-custom
- **Models**: http://localhost:5001/#/models/cnn-chest-xray-classifier

Filter runs by:
- `tags.platform = 'kaggle'`
- `tags.gpu = 'P100'`
- `params.dataset_train_file_hash = '7d89e05b'`

## What Gets Logged

### Parameters (14 + 8 = 22 total)

**Model Configuration:**
- `img_height`, `img_width`, `channels`
- `batch_size`, `epochs`, `learning_rate`
- `filters`, `dense_units`, `dropout_rate`
- `l2_reg`, `optimizer`, etc.

**Dataset Information:**
- `dataset_train_samples` (78,831)
- `dataset_val_samples` (16,383)
- `dataset_test_samples` (16,890)
- `dataset_train_file_hash` (version tracking)
- `dataset_val_file_hash`
- `dataset_test_file_hash`
- `dataset_preprocessing_config_hash`
- `dataset_num_classes` (14)

### Metrics (19 total)

**Final Metrics:**
- `final_train_auc`, `final_val_auc`
- `final_train_loss`, `final_val_loss`

**Training Time:**
- `training_time_hours` (3.83)
- `time_per_epoch_minutes` (9.18)
- `epochs_completed` (25)
- `actual_duration_hours` (from log timestamps)

**Per-Epoch Metrics (25 epochs × 11 metrics):**
- `epoch_auc`, `epoch_val_auc`
- `epoch_loss`, `epoch_val_loss`
- `epoch_accuracy`, `epoch_val_accuracy`
- `epoch_precision`, `epoch_val_precision`
- `epoch_recall`, `epoch_val_recall`
- `epoch_learning_rate`

### Tags (15 total)

**Execution Environment:**
- `platform`: kaggle
- `gpu`: P100
- `environment`: kaggle-p100-gpu
- `framework`: tensorflow

**Tracking:**
- `kernel_slug`: manwithacat/cnn-optimized-training
- `notebook`: cnn-optimized-training
- `dataset_version`: 7d89e05b (first 8 chars of hash)
- `imported_from`: kaggle-api

**Model Registry:**
- `model_name`: cnn-chest-xray-classifier
- `model_version`: 1

### Artifacts

```
📁 model/
   📄 cnn_optimized_best.keras (1,982 MB)

📁 reports/
   📄 06c_optimized_training_history.csv (4.6 KB)
   📄 06c_training_summary.json (0.7 KB)

📁 logs/
   📄 cnn-optimized-training.log (26 KB)
```

## Comparing Runs

### By Dataset Version

```python
# Find all runs using the same dataset
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="params.dataset_train_file_hash = '7d89e05bf9676c09'"
)
```

### By Platform

```python
# Compare Kaggle vs Local training
kaggle_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.platform = 'kaggle'"
)

local_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.platform = 'local'"
)
```

### By Performance

```python
# Find best performing model
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.final_val_auc DESC"],
    max_results=1
)
best_run = runs[0]
```

## Model Registry Workflow

### 1. Automatic Registration (on import)

```bash
# Model automatically registered during import
Model: cnn-chest-xray-classifier
  Version 1: Created
  Tags: platform=kaggle, gpu=P100, val_auc=0.692
```

### 2. Promote to Staging

```python
import mlflow
client = mlflow.tracking.MlflowClient()

# Transition model to Staging
client.transition_model_version_stage(
    name="cnn-chest-xray-classifier",
    version=1,
    stage="Staging"
)
```

### 3. Promote to Production

```python
# After validation, promote to Production
client.transition_model_version_stage(
    name="cnn-chest-xray-classifier",
    version=1,
    stage="Production",
    archive_existing_versions=True  # Archive old Production versions
)
```

### 4. Load Model by Stage

```python
# Load current Production model
model = mlflow.keras.load_model(
    f"models:/cnn-chest-xray-classifier/Production"
)

# Or load specific version
model = mlflow.keras.load_model(
    f"models:/cnn-chest-xray-classifier/1"
)
```

## Troubleshooting

### Issue: Kernel still running

```bash
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status user/kernel
# Status: RUNNING

# Wait for completion or cancel
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels cancel user/kernel
```

### Issue: Model registration fails

**Cause**: Model artifact path doesn't match expected format

**Solution**: Model is still logged as artifact, register manually:

```python
import mlflow
client = mlflow.tracking.MlflowClient()

# Register model from run
result = client.create_model_version(
    name="cnn-chest-xray-classifier",
    source=f"runs:/RUN_ID/model/cnn_optimized_best.keras",
    run_id="RUN_ID"
)
```

### Issue: Dataset hashes not found

**Cause**: `data/processed/` splits not present

**Solution**: Run notebook 03 (preprocessing) first to create splits

### Issue: Log timestamps not parsed

**Cause**: Kaggle log format changed or is JSON-formatted

**Solution**: Script falls back to using training_time_hours from JSON

## Best Practices

### 1. Consistent Naming

Use consistent kernel names for easy tracking:
- `cnn-development-cloud`
- `cnn-optimized-training`
- `cnn-augmented-v2`

### 2. Tag Your Experiments

Add meaningful tags when importing:
```bash
--model-name "cnn-chest-xray-aug-v2"
```

### 3. Document Changes

Add notes to MLflow runs after import:
```python
client.set_tag(run_id, "notes", "Increased dropout to 0.6, added augmentation")
```

### 4. Archive Old Runs

Keep MLflow clean by archiving test runs:
```python
# Delete test experiment runs
mlflow.delete_experiment("cnn-custom-test")
```

## Next Steps

1. **Compare Models**: Use MLflow UI to compare Kaggle vs local runs
2. **Deploy Best Model**: Promote best model to Production stage
3. **Track Datasets**: Monitor when dataset changes affect performance
4. **Automate**: Set up GitHub Actions to auto-import completed Kaggle runs

## See Also

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Kaggle API Docs](https://github.com/Kaggle/kaggle-api)
