#!/usr/bin/env python3
"""
MLflow Run Viewer - Show runs with actual training time (not import time)

Usage:
    python scripts/mlflow_view_runs.py [--experiment EXPERIMENT]
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def format_duration(hours):
    """Format hours into human-readable duration."""
    if hours is None:
        return "N/A"
    if hours < 1:
        return f"{hours*60:.1f}m"
    elif hours < 24:
        return f"{hours:.2f}h"
    else:
        days = int(hours / 24)
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="View MLflow runs with actual training time")
    parser.add_argument("--experiment", default="cnn-custom", help="Experiment name")
    parser.add_argument("--platform", choices=["kaggle", "local", "all"], default="all",
                       help="Filter by platform")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs")
    args = parser.parse_args()

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)

    if not experiment:
        print(f"❌ Experiment '{args.experiment}' not found")
        return

    # Build filter
    filter_str = ""
    if args.platform != "all":
        filter_str = f"tags.platform = '{args.platform}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=["start_time DESC"],
        max_results=args.top
    )

    print("=" * 120)
    print(f"MLflow Runs - {args.experiment.upper()}")
    if args.platform != "all":
        print(f"Platform: {args.platform}")
    print("=" * 120)

    print(f"\n{'#':<4} {'Run Name':<35} {'Platform':<10} {'Training Time':<15} {'val_auc':<10} {'Epochs':<8} {'Status':<15}")
    print("-" * 120)

    for i, run in enumerate(runs, 1):
        run_name = run.data.tags.get('run_name', run.data.tags.get('mlflow.runName', 'unnamed'))[:35]
        platform = run.data.tags.get('platform', 'local')

        # Get actual training time from metric
        training_hours = run.data.metrics.get('training_time_hours', None)
        training_time = format_duration(training_hours)

        # Get performance metrics
        val_auc = run.data.metrics.get('val_auc', 0.0)
        epochs = run.data.params.get('epochs_completed', run.data.params.get('epochs', 'N/A'))

        # Determine status
        val_prec = run.data.metrics.get('val_precision', 0.0)
        val_rec = run.data.metrics.get('val_recall', 0.0)

        if val_prec == 0 and val_rec == 0 and val_auc > 0:
            status = "⚠️ COLLAPSED"
        elif val_auc > 0.70:
            status = "✅ GOOD"
        elif val_auc > 0.65:
            status = "⚠️ WEAK"
        elif val_auc > 0:
            status = "❌ POOR"
        else:
            status = "🏃 RUNNING"

        print(f"{i:<4} {run_name:<35} {platform:<10} {training_time:<15} {val_auc:<10.4f} {str(epochs):<8} {status:<15}")

    print("\n" + "=" * 120)
    print("💡 TIP: Training Time shows ACTUAL execution time, not MLflow import duration")
    print("=" * 120)


if __name__ == "__main__":
    main()
