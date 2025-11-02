# MLflow + Kaggle Integration Guide

## Overview

This guide explains how to track machine learning experiments that run on Kaggle using MLflow, and how to import those tracked experiments into your local MLflow instance for analysis and comparison.

## Why Track Kaggle Runs with MLflow?

**Benefits:**
- **Unified Tracking**: Compare Kaggle GPU runs with local CPU/GPU runs in one place
- **Rich Metadata**: Track hyperparameters, metrics, artifacts, and environment details
- **Source Attribution**: Clear tags showing which runs came from Kaggle vs local
- **Reproducibility**: Full experiment history with configurations and results
- **Cost Tracking**: See which runs were free (Kaggle) vs paid (cloud GPU)

## How It Works

### 1. On Kaggle: MLflow Logs to File System

When you run a notebook on Kaggle with MLflow:

```python
import mlflow

# MLflow automatically uses file-based tracking
# Default: /kaggle/working/mlruns
mlflow.set_experiment("cnn-custom")

with mlflow.start_run(run_name="my-kaggle-run"):
    # Your training code
    mlflow.log_param("batch_size", 128)
    mlflow.log_metric("val_auc", 0.85)
```

**What happens:**
- MLflow creates `/kaggle/working/mlruns/` directory
- Logs parameters, metrics, artifacts to local files
- Everything is included in Kaggle's output download

### 2. Download Kaggle Outputs

After your Kaggle notebook completes:

1. Go to your Kaggle notebook
2. Click **"Output"** tab
3. Click **"Download All"** (downloads a .zip file)
4. Contains:
   - `mlruns/` - MLflow tracking data
   - `models/` - Saved model files
   - `outputs/` - Training logs, CSVs, figures

### 3. Import to Local MLflow

Use the provided script to merge Kaggle runs into your local tracking:

```bash
# From zip file
./scripts/import_kaggle_mlflow.sh ~/Downloads/kaggle-output.zip

# From extracted directory
./scripts/import_kaggle_mlflow.sh ~/Downloads/kaggle-output/mlruns
```

**The script:**
- ✅ Automatically finds `mlruns/` in the download
- ✅ Merges with existing local experiments (no conflicts)
- ✅ Preserves all tags, parameters, metrics, artifacts
- ✅ Shows summary of imported runs

### 4. View in MLflow UI

```bash
make mlflow-ui
# OR
./scripts/mlflow_start.sh

# Open: http://localhost:5001
```

**Filter Kaggle runs:**
- In the UI, add filter: `tags.platform = "kaggle"`
- Or filter by: `tags.gpu = "P100"`
- Or by source: `tags.mlflow.source.name LIKE "kaggle%"`

## Tagging Strategy

### Standard MLflow Tags

These are recognized by MLflow UI for source tracking:

```python
tags = {
    'mlflow.source.type': 'NOTEBOOK',       # Run type
    'mlflow.source.name': 'kaggle/06c_cnn', # Source identifier
    'mlflow.user': 'kaggle',                # User/platform
}
```

### Custom Environment Tags

Add these for filtering and context:

```python
tags = {
    # Platform identification
    'platform': 'kaggle',
    'environment': 'kaggle-p100-gpu',
    'execution_mode': 'cloud',

    # Hardware specs
    'gpu': 'P100',
    'gpu_memory': '16GB',
    'cpu_cores': '4',

    # Model metadata
    'framework': 'tensorflow',
    'model_type': 'CNN',
    'dataset': 'full',
    'optimization': 'tf.data+mixed_precision',

    # Performance tracking
    'batch_size': '128',
    'mixed_precision': 'float16',
    'target_speedup': '12-15x',
}
```

### Example: Full Kaggle Run Tracking

```python
import mlflow
from src.utils.mlflow_utils import MLflowExperimentTracker

# Set experiment (same as local)
mlflow.set_experiment("cnn-custom")

# Track Kaggle run with comprehensive tags
with MLflowExperimentTracker(
    experiment_name="cnn-custom",
    run_name="cnn-optimized-kaggle-50epochs",
    params=CONFIG,  # All hyperparameters
    tags={
        # Standard MLflow tags
        'mlflow.source.type': 'NOTEBOOK',
        'mlflow.source.name': 'kaggle/06c_cnn_optimized',
        'mlflow.user': 'kaggle',

        # Platform tags
        'platform': 'kaggle',
        'environment': 'kaggle-p100-gpu',
        'execution_mode': 'cloud',

        # Hardware
        'gpu': 'P100',

        # Model
        'framework': 'tensorflow',
        'notebook': '06c',
        'version': 'optimized',
    },
    description="Optimized CNN trained on Kaggle P100 GPU"
) as tracker:
    # Training code
    history = model.fit(...)

    # Log metrics
    tracker.log_training_history(history)

    # Log model
    tracker.log_model(model, "cnn_model")

    # Log custom metrics
    tracker.log_metrics_dict({
        'training_time_hours': training_time / 3600,
        'time_per_epoch_minutes': time_per_epoch
    })
```

## Comparing Kaggle vs Local Runs

### In MLflow UI

**View Side-by-Side:**
1. Select multiple runs (Kaggle + local)
2. Click "Compare"
3. See parameter differences, metric plots, etc.

**Create Custom Filters:**
```
# All Kaggle runs
tags.platform = "kaggle"

# Kaggle P100 runs only
tags.platform = "kaggle" AND tags.gpu = "P100"

# Optimized versions
tags.version = "optimized"

# Specific notebook
tags.notebook = "06c"
```

### Programmatic Comparison

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs from experiment
runs = client.search_runs(
    experiment_ids=["<experiment_id>"],
    order_by=["metrics.val_auc DESC"]
)

# Filter Kaggle runs
kaggle_runs = [
    run for run in runs
    if run.data.tags.get('platform') == 'kaggle'
]

# Filter local runs
local_runs = [
    run for run in runs
    if run.data.tags.get('platform') != 'kaggle'
]

# Compare best from each
best_kaggle = kaggle_runs[0]
best_local = local_runs[0]

print(f"Best Kaggle AUC: {best_kaggle.data.metrics['val_auc']}")
print(f"Best Local AUC:  {best_local.data.metrics['val_auc']}")
```

## Workflow Best Practices

### 1. Consistent Experiment Names

Use the **same experiment name** on Kaggle and locally:

```python
# Both Kaggle and local
mlflow.set_experiment("cnn-custom")
```

This allows runs to merge into the same experiment.

### 2. Descriptive Run Names

Include platform and key details:

```python
# Kaggle
run_name = "cnn-optimized-kaggle-50epochs"

# Local
run_name = "cnn-optimized-local-10epochs-test"
```

### 3. Complete Parameter Logging

Log **all** hyperparameters, not just model params:

```python
mlflow.log_params({
    # Model architecture
    'img_height': 224,
    'filters': [64, 128, 256],
    'dropout': 0.5,

    # Training
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,

    # Data
    'dataset_size': len(train_df),
    'num_classes': 14,

    # Environment (important!)
    'platform': 'kaggle',
    'gpu': 'P100'
})
```

### 4. Artifact Organization

Structure artifacts consistently:

```python
# Log model
mlflow.tensorflow.log_model(model, "model")

# Log training history
mlflow.log_artifact("training_history.csv", artifact_path="reports")

# Log figures
mlflow.log_artifact("training_curves.png", artifact_path="figures")
mlflow.log_artifact("confusion_matrix.png", artifact_path="figures")

# Log configs
mlflow.log_dict(CONFIG, "config.json")
```

## Troubleshooting

### Issue: Runs Not Showing in UI

**Check:**
1. MLflow server is running: `make mlflow-ui`
2. Import completed successfully: `ls mlruns/`
3. Experiment exists: `mlflow experiments list`

**Fix:**
```bash
# Restart MLflow server
pkill -f "mlflow server"
./scripts/mlflow_start.sh
```

### Issue: Duplicate Runs After Import

**Cause:** Running import script multiple times

**Prevention:** The script checks for existing run IDs and skips duplicates

**Fix:** Delete duplicate runs in UI or:
```bash
mlflow gc --backend-store-uri ./mlruns
```

### Issue: Missing Artifacts

**Check:** Artifacts exist in Kaggle output:
```bash
# After extracting
ls kaggle-output/mlruns/<exp_id>/<run_id>/artifacts/
```

**If missing:** Re-download from Kaggle (may have been deleted)

### Issue: Tags Not Visible in UI

**Check:** Tags are set correctly:
```python
# Correct
mlflow.set_tags({'platform': 'kaggle'})

# Incorrect (param, not tag)
mlflow.log_param('platform', 'kaggle')
```

## Advanced: Real-Time Kaggle Tracking

For real-time tracking (advanced users):

### Option 1: MLflow Tracking Server on Cloud

1. Deploy MLflow server (e.g., Heroku, Railway)
2. Set tracking URI in Kaggle:
```python
mlflow.set_tracking_uri("https://your-mlflow-server.com")
```

**Pros:** Real-time tracking during Kaggle run
**Cons:** Requires cloud server, authentication setup

### Option 2: Git-Based Sync

1. Kaggle notebook commits to GitHub
2. Local cron job pulls and imports
3. Near real-time updates

**Pros:** No cloud server needed
**Cons:** Delayed updates, complex setup

### Recommendation

For this project, **file-based tracking + manual import** is best:
- ✅ Simple, no infrastructure
- ✅ Kaggle supports it natively
- ✅ Good enough for assessment/development

## Summary

**On Kaggle:**
```python
# Set experiment
mlflow.set_experiment("cnn-custom")

# Track run with tags
with mlflow.start_run():
    mlflow.set_tags({
        'platform': 'kaggle',
        'mlflow.source.type': 'NOTEBOOK',
        'mlflow.source.name': 'kaggle/notebook_name'
    })

    # Train and log
    mlflow.log_params(CONFIG)
    history = model.fit(...)
    mlflow.log_metrics({...})
```

**After Download:**
```bash
# Import to local
./scripts/import_kaggle_mlflow.sh ~/Downloads/kaggle-output.zip

# View in UI
make mlflow-ui

# Filter: tags.platform = "kaggle"
```

**Result:** Unified view of all experiments, local + Kaggle, with clear source attribution!

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Experiment Tags](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)
- Project docs: `docs/MLFLOW_QUICKSTART.md`
