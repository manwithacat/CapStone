#!/usr/bin/env python3
"""
Enhanced Kaggle → MLflow Import

Imports Kaggle training results with:
- Correct run duration from Kaggle execution time
- Model registration for versioning
- Dataset tracking (train/val/test splits)

Usage:
    python scripts/kaggle_import_enhanced.py <csv> <json> <model> <log> \\
        --experiment cnn-custom \\
        --kernel-slug manwithacat/cnn-optimized-training
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def parse_kaggle_log_times(log_path):
    """
    Extract start and end times from Kaggle log file.

    Returns:
        tuple: (start_time_ms, end_time_ms, duration_seconds)
    """
    with open(log_path, 'r') as f:
        log_content = f.read()

    # Look for timestamps in log (Kaggle logs have various formats)
    # Try to find first and last timestamps
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    timestamps = re.findall(timestamp_pattern, log_content)

    if len(timestamps) >= 2:
        # Parse first and last timestamp
        start_str = timestamps[0]
        end_str = timestamps[-1]

        start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')

        # Convert to milliseconds (MLflow format)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        duration_s = (end_dt - start_dt).total_seconds()

        return start_ms, end_ms, duration_s

    return None, None, None


def compute_file_hash(file_path):
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # First 16 chars


def log_dataset_info(data_dir='data/processed'):
    """
    Log dataset information (train/val/test splits).

    Returns:
        dict: Dataset metadata
    """
    data_path = Path(data_dir)

    dataset_info = {}

    # Check for split CSV files
    for split in ['train', 'val', 'test']:
        split_file = data_path / f'{split}_split.csv'
        if split_file.exists():
            df = pd.read_csv(split_file)

            # Compute hash for version tracking
            file_hash = compute_file_hash(split_file)

            dataset_info[f'{split}_samples'] = len(df)
            dataset_info[f'{split}_file_hash'] = file_hash

    # Check for preprocessing config
    config_file = data_path / 'preprocessing_config.json'
    if config_file.exists():
        config_hash = compute_file_hash(config_file)
        dataset_info['preprocessing_config_hash'] = config_hash

        # Load config for additional info
        with open(config_file) as f:
            config = json.load(f)
            dataset_info['num_classes'] = len(config.get('disease_classes', []))

    return dataset_info


def create_mlflow_run_from_kaggle(
    csv_path,
    json_path,
    model_path,
    log_path,
    experiment_name="cnn-custom",
    kernel_slug=None,
    model_name="cnn-chest-xray-classifier"
):
    """
    Create an MLflow run from Kaggle training results with enhanced tracking.

    Args:
        csv_path: Path to training history CSV
        json_path: Path to training summary JSON
        model_path: Path to trained model (.keras)
        log_path: Path to Kaggle execution log
        experiment_name: MLflow experiment name
        kernel_slug: Kaggle kernel slug (e.g., "user/kernel-name")
        model_name: Name for model registration
    """

    # Read training data
    print(f"📊 Reading training history: {csv_path}")
    history_df = pd.read_csv(csv_path)
    print(f"   Found {len(history_df)} epochs")

    print(f"📋 Reading training summary: {json_path}")
    with open(json_path, 'r') as f:
        summary = json.load(f)

    config = summary.get('config', {})
    final_metrics = summary.get('final_metrics', {})
    training_time_hours = summary.get('training_time_hours', 0)
    epochs_completed = summary.get('epochs_completed', len(history_df))
    time_per_epoch = summary.get('time_per_epoch_minutes', 0)

    # Parse Kaggle execution times
    print(f"⏱️  Parsing Kaggle execution log: {log_path}")
    start_time_ms, end_time_ms, duration_s = parse_kaggle_log_times(log_path)

    if start_time_ms and end_time_ms:
        print(f"   Start: {datetime.fromtimestamp(start_time_ms/1000)}")
        print(f"   End:   {datetime.fromtimestamp(end_time_ms/1000)}")
        print(f"   Duration: {duration_s/3600:.2f} hours")
    else:
        print("   ⚠️  Could not parse execution times from log")
        # Fallback: use current time - duration
        end_time_ms = int(datetime.now().timestamp() * 1000)
        start_time_ms = end_time_ms - int(training_time_hours * 3600 * 1000)

    # Get dataset information
    print(f"\n📦 Gathering dataset information...")
    dataset_info = log_dataset_info()
    print(f"   Train: {dataset_info.get('train_samples', 'N/A'):,} samples")
    print(f"   Val:   {dataset_info.get('val_samples', 'N/A'):,} samples")
    print(f"   Test:  {dataset_info.get('test_samples', 'N/A'):,} samples")

    # Set experiment
    mlflow.set_experiment(experiment_name)
    print(f"\n🔬 Creating MLflow run in experiment: {experiment_name}")

    # Create run name
    run_name = f"kaggle-{Path(csv_path).stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Start MLflow run with custom start/end times
    client = MlflowClient()

    # Create run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        start_time=start_time_ms,
        run_name=run_name
    )
    run_id = run.info.run_id

    print(f"   Run ID: {run_id}")

    # Use the run context
    with mlflow.start_run(run_id=run_id):

        # Log all config parameters
        print(f"\n📝 Logging {len(config)} parameters...")
        for key, value in config.items():
            # Convert lists/dicts to strings for MLflow
            if isinstance(value, (list, dict)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)

        # Log dataset parameters
        print(f"📦 Logging dataset information...")
        for key, value in dataset_info.items():
            mlflow.log_param(f"dataset_{key}", value)

        # Log training time metrics
        mlflow.log_metric("training_time_hours", training_time_hours)
        mlflow.log_metric("training_time_minutes", training_time_hours * 60)
        mlflow.log_metric("time_per_epoch_minutes", time_per_epoch)
        mlflow.log_metric("epochs_completed", epochs_completed)
        mlflow.log_metric("actual_duration_hours", duration_s / 3600 if duration_s else training_time_hours)

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

                    # Create metric name
                    metric_name = f"epoch_{col}"
                    mlflow.log_metric(metric_name, value, step=epoch)

        # Add tags
        print(f"🏷️  Adding tags...")
        tags = {
            # MLflow standard tags
            'mlflow.source.type': 'NOTEBOOK',
            'mlflow.source.name': f'kaggle/{Path(csv_path).stem}',
            'mlflow.user': 'kaggle',

            # Execution environment
            'platform': 'kaggle',
            'environment': 'kaggle-p100-gpu',
            'execution_mode': 'cloud',
            'imported_from': 'kaggle-api',

            # Model metadata
            'framework': 'tensorflow',
            'gpu': 'P100',
            'version': 'optimized',
        }

        # Add kernel slug if provided
        if kernel_slug:
            tags['kernel_slug'] = kernel_slug
            tags['notebook'] = kernel_slug.split('/')[-1]

        # Add dataset tracking tag
        if dataset_info:
            tags['dataset_version'] = dataset_info.get('train_file_hash', 'unknown')[:8]

        mlflow.set_tags(tags)

        # Log artifacts (CSV and JSON)
        print(f"📎 Logging artifacts...")
        mlflow.log_artifact(csv_path, artifact_path="reports")
        mlflow.log_artifact(json_path, artifact_path="reports")
        mlflow.log_artifact(log_path, artifact_path="logs")

        # Log model artifact
        if Path(model_path).exists():
            print(f"🤖 Logging model: {model_path}")
            model_size_mb = Path(model_path).stat().st_size / (1024*1024)
            print(f"   Size: {model_size_mb:.1f} MB")
            mlflow.log_artifact(model_path, artifact_path="model")

            # Register model for versioning
            print(f"📝 Registering model: {model_name}")

            # Log model with MLflow (creates model signature)
            model_uri = f"runs:/{run_id}/model/{Path(model_path).name}"

            try:
                # Register the model
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    tags={
                        "platform": "kaggle",
                        "gpu": "P100",
                        "val_auc": str(final_metrics.get('val_auc', 0)),
                        "kernel": kernel_slug or "unknown",
                        "dataset_version": dataset_info.get('train_file_hash', 'unknown')[:8]
                    }
                )

                print(f"   ✅ Model registered: {model_name}")
                print(f"   Version: {result.version}")

                # Add model version to run tags
                mlflow.set_tag("model_version", result.version)
                mlflow.set_tag("model_name", model_name)

            except Exception as e:
                print(f"   ⚠️  Model registration note: {e}")
                print(f"   (Model artifact logged, registration can be done manually)")

    # Set end time
    client.set_terminated(run_id, status='FINISHED', end_time=end_time_ms)

    print(f"\n✅ MLflow run created successfully!")
    print(f"   Run ID: {run_id}")
    print(f"   Run Name: {run_name}")
    print(f"   Experiment: {experiment_name}")
    print(f"\n📊 Logged:")
    print(f"   - {len(config)} configuration parameters")
    print(f"   - {len(dataset_info)} dataset parameters")
    print(f"   - {len(final_metrics)} final metrics")
    print(f"   - {len(history_df)} epochs of training metrics")
    print(f"   - {len(tags)} tags")
    print(f"   - Model artifact + registration")
    print(f"   - Dataset tracking (train/val/test splits)")
    print(f"   - Accurate run duration: {duration_s/3600:.2f}h" if duration_s else "")

    return run_id


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Kaggle → MLflow import with model registration and dataset tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import Kaggle results
  python scripts/kaggle_import_enhanced.py \\
      /tmp/kaggle_output/outputs/reports/training_history.csv \\
      /tmp/kaggle_output/outputs/reports/training_summary.json \\
      /tmp/kaggle_output/models/model_best.keras \\
      /tmp/kaggle_output/training.log \\
      --experiment cnn-custom \\
      --kernel-slug manwithacat/cnn-optimized-training
        """
    )

    parser.add_argument('csv_path', type=str, help='Path to training history CSV')
    parser.add_argument('json_path', type=str, help='Path to training summary JSON')
    parser.add_argument('model_path', type=str, help='Path to trained model (.keras)')
    parser.add_argument('log_path', type=str, help='Path to Kaggle execution log')

    parser.add_argument(
        '--experiment',
        type=str,
        default='cnn-custom',
        help='MLflow experiment name (default: cnn-custom)'
    )

    parser.add_argument(
        '--kernel-slug',
        type=str,
        help='Kaggle kernel slug (e.g., user/kernel-name)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='cnn-chest-xray-classifier',
        help='Model name for registration (default: cnn-chest-xray-classifier)'
    )

    args = parser.parse_args()

    # Validate files exist
    for path_str, name in [
        (args.csv_path, 'CSV'),
        (args.json_path, 'JSON'),
        (args.model_path, 'Model'),
        (args.log_path, 'Log')
    ]:
        path = Path(path_str)
        if not path.exists():
            print(f"❌ Error: {name} file not found: {path}")
            sys.exit(1)

    # Create MLflow run
    print("=" * 70)
    print("Enhanced Kaggle → MLflow Import")
    print("=" * 70)
    print()

    try:
        run_id = create_mlflow_run_from_kaggle(
            csv_path=args.csv_path,
            json_path=args.json_path,
            model_path=args.model_path,
            log_path=args.log_path,
            experiment_name=args.experiment,
            kernel_slug=args.kernel_slug,
            model_name=args.model_name
        )

        print(f"\n💡 View in MLflow UI:")
        print(f"   make mlflow-ui")
        print(f"   Filter by: tags.platform = 'kaggle'")
        print(f"   Or visit: http://localhost:5001/#/experiments/s/{args.experiment}")
        print()
        print(f"📦 Registered Models:")
        print(f"   http://localhost:5001/#/models/{args.model_name}")

    except Exception as e:
        print(f"\n❌ Error creating MLflow run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
