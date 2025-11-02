# MLflow Implementation Strategy - NIH Chest X-Ray Project

**Purpose**: Implement MLflow for experiment tracking, model versioning, and comparison dashboard.

**Status**: 🟡 Planning Phase

---

## Table of Contents

1. [Why MLflow?](#why-mlflow)
2. [Architecture](#architecture)
3. [Implementation Plan](#implementation-plan)
4. [Features to Track](#features-to-track)
5. [Project Integration](#project-integration)
6. [Usage Guide](#usage-guide)
7. [Best Practices](#best-practices)

---

## Why MLflow?

### Problems MLflow Solves

**Current Challenges**:
- ❌ Training results scattered across CSV files and notebooks
- ❌ No easy way to compare different model architectures
- ❌ Manual tracking of hyperparameters and metrics
- ❌ Difficult to reproduce specific experiments
- ❌ Model versions not clearly documented
- ❌ No visual dashboard for browsing experiments

**MLflow Solutions**:
- ✅ Centralized experiment tracking with UI dashboard
- ✅ Automatic logging of parameters, metrics, and artifacts
- ✅ Easy comparison of runs side-by-side
- ✅ Model registry with versioning and staging
- ✅ Reproducibility through environment tracking
- ✅ Integration with TensorFlow/Keras callbacks

### What You Get

1. **Tracking Server**: Web UI at `http://localhost:5000`
2. **Experiment Organization**: Group runs by experiment type
3. **Metrics Visualization**: Interactive plots and comparisons
4. **Model Registry**: Version management with staging (Dev, Staging, Production)
5. **Artifact Storage**: Models, plots, reports automatically saved
6. **Search & Filter**: Find experiments by parameters or metrics

---

## Architecture

### Directory Structure

```
CapStone/
├── mlruns/                    # MLflow tracking data (local)
│   ├── 0/                     # Default experiment
│   ├── 1/                     # Baseline Models
│   ├── 2/                     # CNN Development
│   └── 3/                     # Transfer Learning
├── mlflow/
│   ├── mlflow.db              # SQLite backend (optional)
│   └── artifacts/             # Model artifacts
├── src/
│   └── utils/
│       └── mlflow_utils.py    # Helper functions
└── scripts/
    ├── mlflow_start.sh        # Start MLflow server
    └── mlflow_compare.py      # Compare runs
```

### Experiments Organization

| Experiment ID | Name | Models |
|---------------|------|---------|
| 1 | `baseline-models` | Logistic Regression, Random Forest, XGBoost |
| 2 | `cnn-custom` | Custom CNN (06, 06b, 06c) |
| 3 | `cnn-transfer-learning` | ResNet50, DenseNet121, EfficientNet |
| 4 | `model-optimization` | Threshold tuning, ensemble methods |

---

## Implementation Plan

### Phase 1: Setup & Infrastructure (30 min)

**Files to Create**:
- `src/utils/mlflow_utils.py` - Helper functions
- `scripts/mlflow_start.sh` - Server startup script
- `mlflow_config.py` - Configuration
- `.mlflowignore` - Exclude patterns

**Tasks**:
1. Install MLflow: `pip install mlflow`
2. Create tracking directory structure
3. Set up SQLite backend (optional, for persistence)
4. Configure artifact storage location
5. Update `.gitignore` to exclude mlruns/

### Phase 2: Notebook Integration (1-2 hours)

**Notebooks to Update**:
1. ✅ `05_baseline_models.ipynb` - Track XGBoost, RF, LR
2. ✅ `06_cnn_development.ipynb` - Track CNN training
3. ✅ `06c_cnn_optimized.ipynb` - Track optimized runs
4. ⏳ `07_transfer_learning.ipynb` - Track transfer learning

**Integration Pattern**:
```python
import mlflow
import mlflow.keras

# Start run
with mlflow.start_run(run_name="cnn-custom-v1"):
    # Log parameters
    mlflow.log_params(CONFIG)

    # Train model
    history = model.fit(...)

    # Log metrics
    for epoch, metrics in enumerate(history.history):
        mlflow.log_metrics({
            'train_loss': metrics['loss'],
            'val_loss': metrics['val_loss'],
            'val_auc': metrics['val_auc']
        }, step=epoch)

    # Log model
    mlflow.keras.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("training_curves.png")
```

### Phase 3: Enhanced Tracking (1 hour)

**Custom Metrics**:
- Per-disease AUC scores
- Threshold-tuned F1 scores
- Training time and GPU utilization
- Model size and parameter count

**Artifacts to Log**:
- Training history CSV
- ROC curves (per disease)
- Confusion matrices
- Training plots
- Model architecture diagram
- Preprocessing config

### Phase 4: Model Registry (30 min)

**Setup**:
```python
# Register model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="cnn-chest-xray-classifier"
)

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="cnn-chest-xray-classifier",
    version=1,
    stage="Staging"
)
```

**Stages**:
- **None**: Initial development
- **Staging**: Testing/validation
- **Production**: Deployed to dashboard
- **Archived**: Old versions

---

## Features to Track

### Parameters (Hyperparameters)

**All Models**:
- `model_type`: "baseline", "cnn", "transfer_learning"
- `dataset_size`: Number of training samples
- `random_seed`: For reproducibility

**CNN-Specific**:
- `batch_size`: 32, 128, 256
- `learning_rate`: 0.001, 0.0001
- `epochs`: 50, 100
- `img_size`: 224, 299
- `filters`: [32, 64, 128, 256]
- `dropout_rate`: 0.5
- `l2_reg`: 0.0001
- `optimizer`: "adam", "sgd"
- `augmentation`: "basic", "heavy", "medical"
- `mixed_precision`: True/False

**Transfer Learning**:
- `base_model`: "resnet50", "densenet121"
- `freeze_layers`: Number of frozen layers
- `fine_tune_from`: Layer to start fine-tuning

### Metrics (Performance)

**Per Epoch**:
- `train_loss`, `val_loss`, `test_loss`
- `train_auc`, `val_auc`, `test_auc`
- `train_accuracy`, `val_accuracy`
- `learning_rate` (if scheduled)

**Final Metrics**:
- `best_val_auc`: Best validation AUC across epochs
- `best_epoch`: Epoch number of best model
- `final_test_auc`: Test set performance
- `per_disease_auc`: Dict of AUC per disease class
- `training_time_seconds`: Total training duration
- `time_per_epoch_seconds`: Average epoch time

**Threshold-Tuned**:
- `f1_scores`: Per-disease F1 after threshold tuning
- `optimal_thresholds`: Best thresholds per disease
- `precision`, `recall`, `specificity`

### Tags (Metadata)

```python
mlflow.set_tags({
    'experiment_type': 'cnn_optimized',
    'gpu': 'P100',
    'framework': 'tensorflow',
    'dataset': 'nih-chest-xray-full',
    'notes': 'Optimized pipeline with mixed precision',
    'status': 'completed'
})
```

### Artifacts (Files)

**Automatically Logged**:
- Model file: `model.keras` (via `mlflow.keras.log_model`)
- Training history: `history.json`
- Model summary: `model_summary.txt`

**Manually Logged**:
- `training_curves.png`: Loss/AUC plots
- `roc_curves.png`: Per-disease ROC curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `config.json`: Full configuration
- `preprocessing_params.json`: Data pipeline config
- `per_disease_metrics.json`: Detailed results

---

## Project Integration

### 1. Baseline Models (Notebook 05)

```python
# Start experiment
mlflow.set_experiment("baseline-models")

# Log each model type
for model_name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
    with mlflow.start_run(run_name=f"baseline-{model_name}"):
        mlflow.log_params({
            'model_type': model_name,
            'n_estimators': 100,  # if applicable
            'max_depth': 10
        })

        # Train and log metrics
        model.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test))
        mlflow.log_metric('test_auc', test_auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")
```

### 2. CNN Development (Notebooks 06, 06c)

```python
# Start experiment
mlflow.set_experiment("cnn-custom")

with mlflow.start_run(run_name="cnn-optimized-50epochs"):
    # Log hyperparameters
    mlflow.log_params(CONFIG)

    # Create MLflow callback
    class MLflowCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metrics(logs, step=epoch)

    # Train with callback
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=[MLflowCallback()]
    )

    # Log best model
    mlflow.keras.log_model(model, "model")

    # Log training plot
    plot_training_history(history)
    mlflow.log_artifact("training_history.png")
```

### 3. Transfer Learning (Notebook 07)

```python
mlflow.set_experiment("cnn-transfer-learning")

for base_model_name in ['ResNet50', 'DenseNet121']:
    with mlflow.start_run(run_name=f"transfer-{base_model_name}"):
        mlflow.log_params({
            'base_model': base_model_name,
            'freeze_layers': 100,
            'fine_tune': True
        })

        # Train and log...
```

---

## Usage Guide

### Starting MLflow UI

**Option 1: Local Filesystem** (Simplest)
```bash
# Start server
mlflow ui

# Access dashboard
open http://localhost:5000
```

**Option 2: SQLite Backend** (Recommended for project)
```bash
# Start with backend
mlflow server \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root ./mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000

# Access dashboard
open http://localhost:5000
```

**Option 3: Use Provided Script**
```bash
# Use convenience script
./scripts/mlflow_start.sh
```

### Viewing Experiments

1. **Navigate to UI**: `http://localhost:5000`
2. **Select Experiment**: Click on experiment name (e.g., "cnn-custom")
3. **Compare Runs**: Select multiple runs, click "Compare"
4. **View Details**: Click run name to see parameters, metrics, artifacts

### Comparing Models

**In UI**:
1. Select 2+ runs
2. Click "Compare" button
3. View side-by-side:
   - Parameters diff
   - Metrics comparison (table and charts)
   - Tags and notes

**Via API**:
```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["cnn-custom"],
    filter_string="metrics.val_auc > 0.75",
    order_by=["metrics.val_auc DESC"]
)

print(runs[['run_id', 'params.batch_size', 'metrics.val_auc']])
```

### Loading Saved Models

```python
# Load model from registry
model_uri = "models:/cnn-chest-xray-classifier/Production"
model = mlflow.keras.load_model(model_uri)

# Or load from specific run
run_id = "abc123..."
model = mlflow.keras.load_model(f"runs:/{run_id}/model")

# Make predictions
predictions = model.predict(X_test)
```

---

## Best Practices

### 1. Naming Conventions

**Run Names**:
- `baseline-{model_type}-v{version}`: e.g., "baseline-xgboost-v1"
- `cnn-custom-{epochs}ep-{notes}`: e.g., "cnn-custom-50ep-optimized"
- `transfer-{base}-{layers}frozen`: e.g., "transfer-resnet50-100frozen"

**Experiment Names**:
- Use lowercase with hyphens
- Be descriptive: "cnn-custom" not "experiment1"
- Group related runs: "cnn-ablation-study"

### 2. What to Track

**Always Track**:
- ✅ All hyperparameters (even defaults)
- ✅ Best validation metric
- ✅ Training time
- ✅ Dataset version/size
- ✅ Random seed

**Optional but Useful**:
- Hardware used (GPU type)
- Git commit hash
- Data preprocessing steps
- Failed runs (to learn from)

### 3. Organization

**Use Tags Liberally**:
```python
mlflow.set_tags({
    'stage': 'development',  # development, testing, production
    'priority': 'high',
    'dataset': 'full',        # full, sample, balanced
    'status': 'completed',    # running, completed, failed
    'notes': 'First optimized run with mixed precision'
})
```

**Nested Runs** (for complex experiments):
```python
with mlflow.start_run(run_name="hyperparameter-search"):
    for lr in [0.001, 0.0001]:
        with mlflow.start_run(nested=True, run_name=f"lr-{lr}"):
            mlflow.log_param('learning_rate', lr)
            # Train and log...
```

### 4. Artifact Management

**Keep Artifacts Organized**:
```python
# Log to subdirectories
mlflow.log_artifact("plot.png", "figures")
mlflow.log_artifact("report.json", "reports")
mlflow.log_artifact("model_weights.h5", "checkpoints")
```

**Large Files**:
- Don't log every epoch checkpoint (just best)
- Compress large artifacts before logging
- Use artifact URI instead of downloading repeatedly

### 5. Model Registry

**Version Workflow**:
1. Train model → Log to MLflow
2. Register as new version
3. Test on validation set
4. Transition to "Staging" if good
5. Human review + approve
6. Transition to "Production"
7. Archive old production model

```python
# Register new version
result = mlflow.register_model(
    f"runs:/{run.info.run_id}/model",
    "cnn-chest-xray-classifier"
)

# Add description
client.update_model_version(
    name="cnn-chest-xray-classifier",
    version=result.version,
    description="Optimized CNN with mixed precision, Val AUC: 0.792"
)
```

---

## Integration Checklist

### Setup Phase
- [ ] Install MLflow: `pip install mlflow`
- [ ] Add to `requirements.txt`
- [ ] Create `mlflow/` directory
- [ ] Update `.gitignore` to exclude `mlruns/` and `mlflow/mlflow.db`
- [ ] Create utility functions in `src/utils/mlflow_utils.py`
- [ ] Create startup script `scripts/mlflow_start.sh`

### Notebook Updates
- [ ] Add MLflow imports to all training notebooks
- [ ] Create experiment for each notebook
- [ ] Add `mlflow.start_run()` context managers
- [ ] Log parameters with `mlflow.log_params()`
- [ ] Log metrics with `mlflow.log_metrics()` or callbacks
- [ ] Log models with `mlflow.{framework}.log_model()`
- [ ] Log artifacts (plots, reports)
- [ ] Add tags for organization

### Testing
- [ ] Start MLflow UI: `mlflow ui`
- [ ] Run one notebook end-to-end
- [ ] Verify experiment appears in UI
- [ ] Check parameters, metrics, artifacts logged correctly
- [ ] Test model loading from UI
- [ ] Compare multiple runs

### Documentation
- [ ] Update project README with MLflow instructions
- [ ] Add MLflow section to setup guide
- [ ] Document experiment organization
- [ ] Create example queries/comparisons
- [ ] Screenshot UI for documentation

---

## Expected Benefits

### For Development
- 🎯 **Faster Experimentation**: Easily see what worked
- 📊 **Visual Comparison**: Spot trends in hyperparameter effects
- 🔄 **Reproducibility**: Re-run any experiment exactly
- 📝 **Documentation**: Experiments self-document

### For Assessment
- 🏆 **Professional Presentation**: Industry-standard tool
- 📈 **Clear Progress**: Show iterative improvement
- 🔍 **Transparency**: All experiments visible, not just best
- 📦 **Deliverable**: Assessors can browse your experiments

### For Portfolio
- 💼 **Industry Tool**: MLflow used at Databricks, Netflix, etc.
- 🌟 **Differentiator**: Shows understanding of MLOps
- 📚 **Storytelling**: Visual journey of model development

---

## Estimated Time Investment

| Phase | Time | Complexity |
|-------|------|------------|
| Setup & Installation | 30 min | Low |
| Utility Functions | 30 min | Medium |
| Notebook Integration | 2 hours | Medium |
| Testing & Validation | 1 hour | Low |
| Documentation | 30 min | Low |
| **Total** | **4-5 hours** | **Medium** |

**ROI**: High - Saves time in long run, impressive for assessment

---

## Next Steps

1. **Immediate** (today):
   - [ ] Install MLflow
   - [ ] Create basic utilities
   - [ ] Update one notebook (06c) as test

2. **Short-term** (this week):
   - [ ] Integrate all training notebooks
   - [ ] Run experiments and populate UI
   - [ ] Test model registry workflow

3. **Before Submission**:
   - [ ] Clean up experiments (remove failed runs)
   - [ ] Add descriptions to important runs
   - [ ] Screenshot UI for documentation
   - [ ] Update project README

---

**Status**: Ready to implement
**Priority**: High (significant assessment value)
**Risk**: Low (doesn't affect existing code)
