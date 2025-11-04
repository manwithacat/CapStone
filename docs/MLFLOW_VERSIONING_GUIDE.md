# MLflow Model Versioning & Dataset Tracking Guide

**Created**: November 3, 2025
**Status**: ✅ Active (4 Kaggle runs updated with v7/v8/v9 metadata)

---

## Overview

This guide covers the model versioning and dataset tracking system for MLflow.

**Benefits**:
- Track model evolution (v7 → v8 → v9 → v10)
- Understand what changed between versions
- Record dataset splits and preprocessing
- Query by version: "Show me all v7 runs"
- Compare versions side-by-side

---

## Quick Start

### In Notebooks/Scripts

```python
import mlflow
from src.utils.mlflow_utils import log_model_version, log_dataset_info, log_training_time

# Set tracking URI (if not already set)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cnn-custom")

with mlflow.start_run(run_name="v11-training") as run:
    # ... training code ...

    # Log model version
    log_model_version(
        version="v11",
        parent_version="v7",
        changes="Conservative fine-tune: LR=0.0001, all other params from v7",
        warm_start=True,
        best_epoch=8
    )

    # Log dataset info
    log_dataset_info(
        train_size=78566,
        val_size=17063,
        test_size=16491,
        preprocessing={
            "img_height": 224,
            "img_width": 224,
            "augmentation": True,
            "normalization": "imagenet"
        }
    )

    # Log training time (actual, not import time)
    log_training_time(hours=3.5)
```

---

## Available Functions

### `log_model_version()`
Logs model version and lineage metadata.

**Parameters**:
- `version`: Version identifier (e.g., "v11", "v1-test")
- `parent_version`: Parent model version if warm-start
- `parent_run_id`: MLflow run_id of parent (optional)
- `changes`: Description of what changed
- `warm_start`: Whether pre-trained model was loaded
- `best_epoch`: Epoch with best validation metric

**Metadata stored**:
- `model_version` (tag): "v11"
- `parent_model_version` (tag): "v7"
- `model_lineage` (tag): "v7 → v11"
- `architecture_changes` (tag): Description of changes
- `warm_start` (param): "true"/"false"
- `best_epoch` (tag): Epoch number

### `log_dataset_info()`
Logs dataset split sizes and preprocessing.

**Parameters**:
- `train_size`, `val_size`, `test_size`: Image counts
- `dataset_name`: Name of dataset (default: "NIH-Chest-Xrays-112k")
- `preprocessing`: Dict of preprocessing params

**Metadata stored**:
- `dataset_name` (tag)
- `dataset_train_size`, `dataset_val_size`, `dataset_test_size` (tags)
- `dataset_total_size` (tag)
- `dataset_split_ratio` (tag): e.g., "70.00%/15.20%/14.70%"
- `data_*` (params): Each preprocessing parameter

### `log_training_time()`
Logs actual training time (separate from MLflow's automatic duration).

**Parameters**:
- `hours`: Training time in hours (can be fractional)

**Metrics stored**:
- `training_time_hours`: e.g., 3.5
- `training_time_minutes`: e.g., 210

### `get_best_run()`
Find the run_id of the best run in an experiment.

```python
from src.utils.mlflow_utils import get_best_run

best_run_id = get_best_run("cnn-custom", metric="val_auc")
# Output:
# ✅ Best run: kaggle-06c_optimized_training_history-20251102-192906
#    val_auc: 0.7030
#    run_id: abc123...
```

### `query_runs_by_version()`
Find all runs with a specific version.

```python
from src.utils.mlflow_utils import query_runs_by_version

runs = query_runs_by_version("v7")
# Output: Found 1 runs with version 'v7'

for run in runs:
    print(run.data.tags.get('mlflow.runName'))
```

### `compare_model_lineage()`
Compare two model versions side-by-side.

```python
from src.utils.mlflow_utils import compare_model_lineage

compare_model_lineage("v7", "v9")
```

**Output**:
```
================================================================================
Comparing v7 vs v9
================================================================================

v7 changes:
  Kaggle GPU training: 64 filters, 0.5 dropout, 0.0001 L2, LR=0.001

v9 changes:
  Warm-start from v7, over-regularized (LR=0.0001, dropout=0.6, L2=0.0003, batch=128)

Metrics comparison:
  val_auc              0.7030 → 0.6917 (-0.0113) 📉
  val_loss             0.3841 → 0.3915 (+0.0074) 📉
  val_precision        0.5215 → 0.0000 (-0.5215) 📉
  val_recall           0.4087 → 0.0000 (-0.4087) 📉

Parameter changes:
  batch_size                    64 → 128
  dropout_rate                  0.5 → 0.6
  l2_reg                        0.0001 → 0.0003
  learning_rate                 0.001 → 0.0001
================================================================================
```

---

## SQL Queries

With SQLite backend, you can query versioning metadata directly:

### Find All Versions
```bash
sqlite3 mlflow.db "
SELECT DISTINCT value as version
FROM tags
WHERE key = 'model_version'
ORDER BY value;
"
```

### Model Lineage
```bash
sqlite3 mlflow.db "
SELECT
    r.name,
    t1.value as version,
    t2.value as parent,
    t3.value as lineage
FROM runs r
JOIN tags t1 ON r.run_uuid = t1.run_uuid AND t1.key = 'model_version'
LEFT JOIN tags t2 ON r.run_uuid = t2.run_uuid AND t2.key = 'parent_model_version'
LEFT JOIN tags t3 ON r.run_uuid = t3.run_uuid AND t3.key = 'model_lineage'
WHERE t1.value IS NOT NULL
ORDER BY t1.value;
"
```

### Best Model for Each Version
```bash
sqlite3 mlflow.db "
SELECT
    t.value as version,
    r.name,
    m.value as val_auc
FROM runs r
JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'model_version'
JOIN latest_metrics m ON r.run_uuid = m.run_uuid AND m.key = 'val_auc'
ORDER BY t.value, m.value DESC;
"
```

### Dataset Usage by Model
```bash
sqlite3 mlflow.db "
SELECT
    t1.value as version,
    t2.value as train_size,
    t3.value as val_size,
    t4.value as test_size
FROM runs r
JOIN tags t1 ON r.run_uuid = t1.run_uuid AND t1.key = 'model_version'
LEFT JOIN tags t2 ON r.run_uuid = t2.run_uuid AND t2.key = 'dataset_train_size'
LEFT JOIN tags t3 ON r.run_uuid = t3.run_uuid AND t3.key = 'dataset_val_size'
LEFT JOIN tags t4 ON r.run_uuid = t4.run_uuid AND t4.key = 'dataset_test_size'
WHERE t1.value IS NOT NULL;
"
```

---

## Current Model Versions

| Version | Parent | Status | Key Changes | Best Metric |
|---------|--------|--------|-------------|-------------|
| v7 | - | ✅ Stable | Kaggle training: 64 filters, 0.5 dropout, LR=0.001, stopped at epoch 15 | val_auc=0.703 |
| v8 | v7 | ❌ Failed | Warm-start with default params, model collapsed | val_auc=0.690 |
| v9 | v7 | ❌ Failed | Over-regularized (4 params changed), model collapsed | val_auc=0.692 |
| v10 | v7 | 🚀 Running | Conservative: Only LR changed (0.001→0.0001) | TBD |

---

## Versioning Best Practices

### 1. Semantic Versioning
- `v1`, `v2`, `v3`: Major architecture changes
- `v7-test`: Test/experimental variants
- `v7-finetune`: Fine-tuning variants

### 2. Always Log Parent for Warm-Start
```python
log_model_version(
    version="v11",
    parent_version="v7",  # ✅ Always specify parent
    warm_start=True
)
```

### 3. Describe What Changed
Be specific about what parameters/architecture changed:
```python
changes="Conservative fine-tune: LR 0.001→0.0001, revert dropout 0.6→0.5, L2 0.0003→0.0001"
```

### 4. Log Dataset Info Every Time
Even if using same splits, logging helps future analysis:
```python
log_dataset_info(
    train_size=78566,
    val_size=17063,
    test_size=16491
)
```

### 5. Actual Training Time
Separate from MLflow's import time:
```python
import time
start_time = time.time()
# ... training ...
training_hours = (time.time() - start_time) / 3600
log_training_time(training_hours)
```

---

## Maintenance Scripts

### Update Run Durations
Update MLflow run durations to reflect actual training time:
```bash
python3 scripts/mlflow_update_durations.py --dry-run  # Preview
python3 scripts/mlflow_update_durations.py            # Execute
```

### Add Versioning to Existing Runs
Retroactively add versioning metadata:
```bash
python3 scripts/mlflow_add_versioning.py --dry-run --verbose  # Preview
python3 scripts/mlflow_add_versioning.py                      # Execute
```

### View Runs with Training Times
```bash
python3 scripts/mlflow_view_runs.py --platform kaggle
python3 scripts/mlflow_view_runs.py --top 20
```

---

## Integration with Kaggle

When importing Kaggle results, versioning is automatically added if run name matches pattern:

```bash
python3 scripts/kaggle_import_enhanced.py \
  /tmp/kaggle_latest/outputs/reports/v10-*_training_history.csv \
  /tmp/kaggle_latest/outputs/reports/06c_training_summary.json \
  /tmp/kaggle_latest/models/v10-*_best.keras \
  /tmp/kaggle_latest/cnn-optimized-training.log \
  --experiment cnn-custom \
  --kernel-slug manwithacat/cnn-optimized-training
```

The import script detects "v10" in filename and adds metadata automatically.

---

## Viewing in MLflow UI

1. Start MLflow UI:
   ```bash
   make mlflow-ui
   # OR
   ./scripts/mlflow_start.sh
   ```

2. Navigate to http://localhost:5001

3. Click on any run to see versioning metadata in the "Tags" section:
   - `model_version`
   - `parent_model_version`
   - `model_lineage`
   - `architecture_changes`
   - `dataset_train_size`, `dataset_val_size`, `dataset_test_size`

4. Filter runs by version:
   - In search box: `tags.model_version = "v7"`

---

## Summary

✅ **Implemented**:
- Model versioning with parent lineage tracking
- Dataset split recording
- Actual training time (separate from import time)
- Utility functions in `src/utils/mlflow_utils.py`
- Maintenance scripts for updating existing runs
- SQL query support via SQLite backend

✅ **Updated Runs**:
- 4 Kaggle runs (v7, v8, v9) with full versioning metadata
- 5 runs with accurate training durations

🎯 **Next Steps**:
- v10 Kaggle run completing (will auto-add metadata on import)
- Future runs use `log_model_version()` in notebooks
- Compare model lineage as versions evolve
