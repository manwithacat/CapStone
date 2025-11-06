#!/usr/bin/env python3
"""
Import Colab training results into MLflow.

This script:
1. Scans colab/results/ for trained models and artifacts
2. Creates MLflow runs for each model
3. Logs models, metrics, params, and artifacts to MLflow
4. Tags runs with platform=colab, gpu=a100, etc.

Usage:
    python3 scripts/import_colab_mlflow.py
"""

import os
import json
import mlflow
import mlflow.tensorflow
from pathlib import Path
from datetime import datetime
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
COLAB_RESULTS_DIR = PROJECT_ROOT / "colab" / "results"

# MLflow settings
EXPERIMENT_NAME = "NIH-XRay-Transfer-Learning"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"

def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"✓ Created experiment: {EXPERIMENT_NAME}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✓ Using experiment: {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id

def find_model_runs(results_dir: Path):
    """
    Find all model training runs in results directory.

    Expected structure:
        colab/results/
            models/
                resnet50-transfer/
                    runs/YYYY-MM-DD_HHMMSS/
                        resnet50_transfer_best.keras
                        config.json
                densenet121-transfer/
                    runs/YYYY-MM-DD_HHMMSS/
                        densenet121_transfer_best.keras
                        config.json
            artifacts/
                metrics/
                    YYYY-MM-DD_HHMMSS_model_results.csv
                plots/
                    YYYY-MM-DD_HHMMSS_resnet50_training_history.png
    """
    model_runs = []

    models_dir = results_dir / "models"
    if not models_dir.exists():
        return model_runs

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name  # e.g., "resnet50-transfer"
        runs_dir = model_dir / "runs"

        if not runs_dir.exists():
            continue

        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Find model file
            model_files = list(run_dir.glob("*.keras")) + list(run_dir.glob("*.h5"))
            if not model_files:
                continue

            model_file = model_files[0]
            config_file = run_dir / "config.json"

            model_runs.append({
                "model_name": model_name,
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "model_file": model_file,
                "config_file": config_file if config_file.exists() else None
            })

    return model_runs

def find_artifacts(results_dir: Path, run_name: str, model_name: str):
    """Find artifacts (plots, metrics) for a specific run."""
    artifacts = {}

    artifacts_dir = results_dir / "artifacts"
    if not artifacts_dir.exists():
        return artifacts

    # Look for metrics CSV
    metrics_dir = artifacts_dir / "metrics"
    if metrics_dir.exists():
        for csv_file in metrics_dir.glob(f"{run_name}*.csv"):
            artifacts["metrics_csv"] = csv_file

        # Also look for any CSV with model name
        for csv_file in metrics_dir.glob(f"*{model_name.split('-')[0]}*.csv"):
            if "metrics_csv" not in artifacts:
                artifacts["metrics_csv"] = csv_file

    # Look for training history plots
    plots_dir = artifacts_dir / "plots"
    if plots_dir.exists():
        for plot_file in plots_dir.glob(f"{run_name}*.png"):
            artifacts.setdefault("plots", []).append(plot_file)

        # Also look for plots with model name
        for plot_file in plots_dir.glob(f"*{model_name.split('-')[0]}*.png"):
            if plot_file not in artifacts.get("plots", []):
                artifacts.setdefault("plots", []).append(plot_file)

    return artifacts

def import_run(run_info: dict, results_dir: Path):
    """Import a single training run into MLflow."""
    model_name = run_info["model_name"]
    run_name = run_info["run_name"]
    model_file = run_info["model_file"]
    config_file = run_info["config_file"]

    print(f"\n📦 Importing {model_name} - {run_name}")

    # Load config
    config = {}
    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"   ✓ Loaded config: {len(config)} parameters")

    # Find artifacts
    artifacts = find_artifacts(results_dir, run_name, model_name)
    print(f"   ✓ Found {len(artifacts)} artifact types")

    # Create MLflow run
    with mlflow.start_run(run_name=f"{model_name}_{run_name}"):
        # Log parameters
        params = {
            "model_architecture": model_name.split("-")[0],
            "training_strategy": "transfer-learning",
            "platform": "colab",
            "gpu": "a100",
            **config
        }

        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)

        print(f"   ✓ Logged {len(params)} parameters")

        # Log metrics from config (final values)
        metrics_logged = 0
        if "final_metrics" in config:
            for key, value in config["final_metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                    metrics_logged += 1

        # Log metrics from CSV (training history)
        if "metrics_csv" in artifacts:
            try:
                df = pd.read_csv(artifacts["metrics_csv"])

                # Log each row as a step
                for idx, row in df.iterrows():
                    epoch = row.get("epoch", idx + 1)
                    for col in df.columns:
                        if col != "epoch" and pd.notna(row[col]):
                            mlflow.log_metric(col, row[col], step=int(epoch))
                            metrics_logged += 1

                print(f"   ✓ Logged {metrics_logged} metrics from training history")
            except Exception as e:
                print(f"   ⚠️  Could not load metrics CSV: {e}")
        else:
            print(f"   ✓ Logged {metrics_logged} final metrics")

        # Log model
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file)
            mlflow.tensorflow.log_model(model, "model")
            print(f"   ✓ Logged model: {model_file.name}")
        except Exception as e:
            print(f"   ⚠️  Could not log model: {e}")
            # Log as artifact instead
            mlflow.log_artifact(str(model_file), "model")
            print(f"   ✓ Logged model as artifact")

        # Log artifacts
        if "plots" in artifacts:
            for plot_file in artifacts["plots"]:
                mlflow.log_artifact(str(plot_file), "plots")
            print(f"   ✓ Logged {len(artifacts['plots'])} plot(s)")

        if "metrics_csv" in artifacts:
            mlflow.log_artifact(str(artifacts["metrics_csv"]), "metrics")
            print(f"   ✓ Logged metrics CSV")

        if config_file:
            mlflow.log_artifact(str(config_file))
            print(f"   ✓ Logged config.json")

        # Tag run
        mlflow.set_tag("platform", "colab")
        mlflow.set_tag("gpu", "a100")
        mlflow.set_tag("source", "gcs")
        mlflow.set_tag("model_architecture", model_name.split("-")[0])
        mlflow.set_tag("run_timestamp", run_name)

        run = mlflow.active_run()
        print(f"   ✅ Created MLflow run: {run.info.run_id}")

def main():
    """Main import function."""
    print("🚀 Colab Results → MLflow Import")
    print("=" * 50)

    # Check if results directory exists
    if not COLAB_RESULTS_DIR.exists():
        print(f"\n❌ Results directory not found: {COLAB_RESULTS_DIR}")
        print("\n💡 Run this first:")
        print("   ./scripts/download_colab_results.sh")
        return 1

    # Setup MLflow
    print(f"\n🔧 MLflow Configuration")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    experiment_id = setup_mlflow()

    # Find all model runs
    print(f"\n🔍 Scanning for model runs...")
    model_runs = find_model_runs(COLAB_RESULTS_DIR)

    if not model_runs:
        print(f"\n⚠️  No model runs found in {COLAB_RESULTS_DIR}")
        print("\n💡 Expected structure:")
        print("   colab/results/models/<model-name>/runs/<run-timestamp>/")
        return 1

    print(f"   ✓ Found {len(model_runs)} model run(s)")

    # Import each run
    print(f"\n📥 Importing runs into MLflow...")
    for run_info in model_runs:
        try:
            import_run(run_info, COLAB_RESULTS_DIR)
        except Exception as e:
            print(f"   ❌ Failed to import {run_info['model_name']}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("✅ Import Complete!")
    print("=" * 50)
    print(f"\n📊 View results:")
    print(f"   MLflow UI: http://localhost:5001")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print(f"\n💾 Data stored in:")
    print(f"   {MLFLOW_TRACKING_URI}")

    return 0

if __name__ == "__main__":
    exit(main())
