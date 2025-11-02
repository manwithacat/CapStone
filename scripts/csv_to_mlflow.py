#!/usr/bin/env python3
"""
Convert Kaggle training results (CSV + JSON) to MLflow run

Usage:
    python scripts/csv_to_mlflow.py <training_history.csv> <training_summary.json>

Example:
    python scripts/csv_to_mlflow.py \
        outputs/reports/06c_optimized_training_history.csv \
        outputs/reports/06c_training_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd


def create_mlflow_run_from_csv(csv_path, json_path, experiment_name="cnn-custom"):
    """
    Create an MLflow run from Kaggle training CSV and JSON.

    Args:
        csv_path: Path to training history CSV
        json_path: Path to training summary JSON
        experiment_name: MLflow experiment name
    """

    # Read training history CSV
    print(f"📊 Reading training history: {csv_path}")
    history_df = pd.read_csv(csv_path)
    print(f"   Found {len(history_df)} epochs")

    # Read training summary JSON
    print(f"📋 Reading training summary: {json_path}")
    with open(json_path, 'r') as f:
        summary = json.load(f)

    config = summary.get('config', {})
    final_metrics = summary.get('final_metrics', {})
    training_time = summary.get('training_time_hours', 0)
    epochs_completed = summary.get('epochs_completed', len(history_df))
    time_per_epoch = summary.get('time_per_epoch_minutes', 0)

    # Set experiment
    mlflow.set_experiment(experiment_name)
    print(f"\n🔬 Creating MLflow run in experiment: {experiment_name}")

    # Create run name
    run_name = f"kaggle-imported-{Path(csv_path).stem}"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Log all config parameters
        print(f"\n📝 Logging {len(config)} parameters...")
        for key, value in config.items():
            # Convert lists/dicts to strings for MLflow
            if isinstance(value, (list, dict)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)

        # Log training time metrics
        mlflow.log_metric("training_time_hours", training_time)
        mlflow.log_metric("training_time_minutes", training_time * 60)
        mlflow.log_metric("time_per_epoch_minutes", time_per_epoch)
        mlflow.log_metric("epochs_completed", epochs_completed)

        # Log final metrics
        print(f"📈 Logging final metrics...")
        for key, value in final_metrics.items():
            mlflow.log_metric(f"final_{key}", value)

        # Log per-epoch metrics
        print(f"📊 Logging {len(history_df)} epochs of metrics...")
        for idx, row in history_df.iterrows():
            epoch = idx  # 0-indexed

            # Log all metrics from the CSV
            for col in history_df.columns:
                if col == 'epoch':
                    continue

                value = row[col]
                if pd.notna(value):  # Skip NaN values
                    # Convert numpy types to Python types
                    if hasattr(value, 'item'):
                        value = value.item()

                    # Create metric name (e.g., "epoch_train_loss", "epoch_val_auc")
                    metric_name = f"epoch_{col}"
                    mlflow.log_metric(metric_name, value, step=epoch)

        # Add tags
        print(f"🏷️  Adding tags...")
        tags = {
            # MLflow standard tags
            'mlflow.source.type': 'NOTEBOOK',
            'mlflow.source.name': 'kaggle/06c_cnn_optimized',
            'mlflow.user': 'kaggle',

            # Execution environment
            'platform': 'kaggle',
            'environment': 'kaggle-p100-gpu',
            'execution_mode': 'cloud',
            'imported_from': 'csv',

            # Model metadata
            'framework': 'tensorflow',
            'gpu': 'P100',
            'notebook': '06c',
            'version': 'optimized',
        }

        # Add optional tags from config
        if config.get('batch_size'):
            tags['batch_size'] = str(config['batch_size'])
        if config.get('epochs'):
            tags['epochs_requested'] = str(config['epochs'])

        mlflow.set_tags(tags)

        # Log artifacts (CSV and JSON)
        print(f"📎 Logging artifacts...")
        mlflow.log_artifact(csv_path, artifact_path="reports")
        mlflow.log_artifact(json_path, artifact_path="reports")

        # Try to log model if it exists
        model_paths = [
            Path('models/saved_models/cnn_optimized_best.keras'),
            Path('models/saved_models_kaggle/cnn_optimized_best.keras'),
        ]

        for model_path in model_paths:
            if model_path.exists():
                print(f"🤖 Logging model: {model_path}")
                mlflow.log_artifact(str(model_path), artifact_path="model")
                break

        print(f"\n✅ MLflow run created successfully!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Run Name: {run_name}")
        print(f"   Experiment: {experiment_name}")
        print(f"\n📊 Logged:")
        print(f"   - {len(config)} parameters")
        print(f"   - {len(final_metrics)} final metrics")
        print(f"   - {len(history_df)} epochs of training metrics")
        print(f"   - {len(tags)} tags")
        print(f"   - 2+ artifacts (CSV, JSON, model if available)")

        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle training results to MLflow run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import from default Kaggle output locations
  python scripts/csv_to_mlflow.py \\
      outputs/reports/06c_optimized_training_history.csv \\
      outputs/reports/06c_training_summary.json

  # Import from downloaded Kaggle results
  python scripts/csv_to_mlflow.py \\
      ~/Downloads/results_20251101/outputs/reports/06c_optimized_training_history.csv \\
      ~/Downloads/results_20251101/outputs/reports/06c_training_summary.json

  # Specify custom experiment name
  python scripts/csv_to_mlflow.py \\
      outputs/reports/06c_optimized_training_history.csv \\
      outputs/reports/06c_training_summary.json \\
      --experiment "cnn-kaggle-experiments"
        """
    )

    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to training history CSV file'
    )

    parser.add_argument(
        'json_path',
        type=str,
        help='Path to training summary JSON file'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='cnn-custom',
        help='MLflow experiment name (default: cnn-custom)'
    )

    args = parser.parse_args()

    # Validate files exist
    csv_path = Path(args.csv_path)
    json_path = Path(args.json_path)

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)

    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        sys.exit(1)

    # Create MLflow run
    print("="*70)
    print("CSV to MLflow Importer")
    print("="*70)
    print()

    try:
        run_id = create_mlflow_run_from_csv(
            csv_path,
            json_path,
            experiment_name=args.experiment
        )

        print(f"\n💡 View in MLflow UI:")
        print(f"   make mlflow-ui")
        print(f"   Filter by: tags.platform = 'kaggle'")
        print(f"   Or direct link (once UI is open):")
        print(f"   http://localhost:5001/#/experiments/{args.experiment}/runs/{run_id}")

    except Exception as e:
        print(f"\n❌ Error creating MLflow run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
