# MLflow Quick Start Guide

**Get started with experiment tracking in 5 minutes!**

---

## Step 1: Install MLflow

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
mlflow --version
```

## Step 2: Start MLflow UI

```bash
# Start server (from project root)
./scripts/mlflow_start.sh

# Open dashboard
open http://localhost:5000
```

You should see the MLflow UI with no experiments yet.

## Step 3: Run Your First Tracked Experiment

### Option A: Jupyter Notebook

```python
import mlflow
import mlflow.keras
from src.utils.mlflow_utils import MLflowExperimentTracker

# Set experiment
mlflow.set_experiment("my-first-experiment")

# Track a simple run
with mlflow.start_run(run_name="test-run"):
    # Log parameters
    mlflow.log_params({
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    })

    # Simulate training
    for epoch in range(10):
        loss = 1.0 / (epoch + 1)  # Fake decreasing loss
        mlflow.log_metrics({'loss': loss, 'accuracy': 1 - loss}, step=epoch)

    print("✅ Tracked experiment!")
```

### Option B: Python Script

```python
# example_tracking.py
import mlflow
from src.utils.mlflow_utils import MLflowExperimentTracker

# Use convenient context manager
with MLflowExperimentTracker(
    experiment_name="example-experiment",
    run_name="my-run",
    params={'param1': 'value1'},
    tags={'framework': 'keras'}
) as tracker:
    # Your training code here
    tracker.log_metrics_dict({'accuracy': 0.95})
    print("Done!")
```

Run it:
```bash
python example_tracking.py
```

## Step 4: View Results in UI

1. Go to http://localhost:5000
2. Click on your experiment name
3. See your runs with parameters and metrics
4. Click a run to see details and plots

## Step 5: Compare Runs

**In UI**:
1. Select 2+ runs (checkboxes)
2. Click "Compare" button
3. View side-by-side comparison

**In Code**:
```python
from src.utils.mlflow_utils import search_best_runs

# Find best runs
best = search_best_runs(
    "my-first-experiment",
    metric="metrics.accuracy",
    max_results=5
)
print(best[['run_id', 'params.learning_rate', 'metrics.accuracy']])
```

---

## Real Example: Track CNN Training

Here's how to integrate MLflow into your CNN training:

```python
import mlflow
import mlflow.keras
from tensorflow import keras
from src.utils.mlflow_utils import MLflowExperimentTracker, MLflowKerasCallback

# Configuration
CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 50
}

# Start tracking
with MLflowExperimentTracker(
    experiment_name="cnn-custom",
    run_name="cnn-optimized-v1",
    params=CONFIG,
    tags={'gpu': 'P100', 'framework': 'tensorflow'}
) as tracker:
    # Build model
    model = build_cnn_model()

    # Train with MLflow callback
    history = model.fit(
        train_dataset,
        epochs=CONFIG['epochs'],
        validation_data=val_dataset,
        callbacks=[MLflowKerasCallback()]  # Auto-log metrics each epoch
    )

    # Log model
    tracker.log_model(model, "cnn_model")

    # Log training plot
    fig = plot_training_curves(history)
    tracker.log_figure(fig, "training_curves.png")

    print("✅ Experiment tracked!")
```

---

## Common Commands

### Server Management

```bash
# Start server
./scripts/mlflow_start.sh

# Start on different port
./scripts/mlflow_start.sh 5001

# Stop server
./scripts/mlflow_stop.sh
```

### Compare Experiments

```bash
# Compare top 5 runs
python scripts/mlflow_compare.py --experiment cnn-custom --top 5

# Compare specific runs
python scripts/mlflow_compare.py \
    --experiment cnn-custom \
    --runs run1,run2,run3

# Save comparison to CSV
python scripts/mlflow_compare.py \
    --experiment cnn-custom \
    --top 10 \
    --output comparison.csv
```

### Search Runs

```python
import mlflow

# Find runs with specific parameters
runs = mlflow.search_runs(
    experiment_names=["cnn-custom"],
    filter_string="params.batch_size = '128'",
    order_by=["metrics.val_auc DESC"]
)

# Find runs with good performance
runs = mlflow.search_runs(
    experiment_names=["cnn-custom"],
    filter_string="metrics.val_auc > 0.75"
)
```

### Load Saved Models

```python
import mlflow.keras

# Load model from specific run
run_id = "abc123..."
model = mlflow.keras.load_model(f"runs:/{run_id}/model")

# Make predictions
predictions = model.predict(X_test)
```

---

## Best Practices

### 1. Always Track These

```python
# Parameters (hyperparameters)
mlflow.log_params({
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 50,
    'optimizer': 'adam'
})

# Best metrics (not just last epoch)
mlflow.log_metric('best_val_auc', best_auc)
mlflow.log_metric('best_epoch', best_epoch)

# Training time
mlflow.log_metric('training_time_seconds', elapsed)

# Tags for organization
mlflow.set_tags({
    'gpu': 'P100',
    'dataset': 'full',
    'status': 'completed'
})
```

### 2. Use Descriptive Names

```python
# Good run names
"cnn-optimized-50ep-mixed-precision"
"transfer-resnet50-100frozen"
"baseline-xgboost-v2"

# Bad run names
"run1"
"test"
"asdf"
```

### 3. Log Important Artifacts

```python
# Training curves
mlflow.log_artifact("training_curves.png", "figures")

# Model architecture
with open("model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
mlflow.log_artifact("model_summary.txt")

# Configuration
mlflow.log_dict(CONFIG, "config.json")
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000

# Kill it
kill -9 <PID>

# Or use different port
./scripts/mlflow_start.sh 5001
```

### Cannot Connect to UI

1. Check server is running: `ps aux | grep mlflow`
2. Check port: `lsof -i :5000`
3. Try: `http://127.0.0.1:5000` instead of `localhost`

### Runs Not Appearing

```python
# Check tracking URI
print(mlflow.get_tracking_uri())

# Should be: file:///path/to/project/mlruns
```

### Database Locked

```bash
# Stop all MLflow processes
./scripts/mlflow_stop.sh

# Restart
./scripts/mlflow_start.sh
```

---

## Next Steps

1. **Read the strategy**: `docs/MLFLOW_IMPLEMENTATION_STRATEGY.md`
2. **Update notebooks**: Add tracking to your training notebooks
3. **Run experiments**: Train models and track results
4. **Compare**: Find what works best
5. **Document**: Screenshot comparisons for your report

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [Project Strategy](MLFLOW_IMPLEMENTATION_STRATEGY.md)
- [Utilities Reference](../src/utils/mlflow_utils.py)

---

**Ready to track experiments!** 🚀

Start the server and run your first tracked experiment.
