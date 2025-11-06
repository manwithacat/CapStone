#!/usr/bin/env python3
"""
Update Colab MLflow runs with final test metrics.

Reads model_results.csv and adds test metrics to existing runs.
"""

import mlflow
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
EXPERIMENT_NAME = "NIH-XRay-Transfer-Learning"
METRICS_CSV = PROJECT_ROOT / "colab/results/artifacts/metrics/2025-11-05_063105_model_results.csv"

def main():
    print("🔧 Updating Colab runs with final test metrics...")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Read metrics
    df = pd.read_csv(METRICS_CSV, index_col=0)
    print(f"\n📊 Loaded metrics for {len(df)} models:")
    print(df)

    # Map model names in CSV to run names
    model_mapping = {
        "ResNet50": "resnet50-transfer",
        "DenseNet121": "densenet121-transfer",
        "EfficientNetB3": "efficientnetb3-transfer"
    }

    # Get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    print(f"\n✓ Found {len(runs)} runs in MLflow")

    # Update each run
    for model_csv, model_mlflow in model_mapping.items():
        if model_csv not in df.index:
            print(f"  ⚠️  {model_csv} not found in metrics CSV")
            continue

        # Find run
        matching_runs = runs[runs['tags.model_architecture'] == model_mlflow.split('-')[0]]

        if len(matching_runs) == 0:
            print(f"  ⚠️  No run found for {model_mlflow}")
            continue

        run_id = matching_runs.iloc[0]['run_id']
        metrics = df.loc[model_csv]

        # Update run with final test metrics
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_loss", float(metrics['loss']))
            mlflow.log_metric("test_accuracy", float(metrics['accuracy']))
            mlflow.log_metric("test_auc", float(metrics['auc']))

        print(f"  ✅ Updated {model_mlflow}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}, auc={metrics['auc']:.4f}")

    print("\n✅ Update complete!")
    print(f"   View: http://localhost:5001")

if __name__ == "__main__":
    main()
