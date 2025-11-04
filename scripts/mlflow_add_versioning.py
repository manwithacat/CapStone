#!/usr/bin/env python3
"""
Add model versioning and dataset tracking metadata to existing MLflow runs.

Updates runs with:
- Model version tags (v7, v8, v9, etc.)
- Parent model lineage for warm-start runs
- Dataset split sizes
- Architecture change descriptions
"""

import argparse
import sqlite3
import json
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow


# Model version metadata
MODEL_VERSIONS = {
    # Format: run_name_pattern: {version, parent, changes, dataset}
    "baseline": {
        "version": "baseline-v1",
        "parent": None,
        "changes": "Initial baseline models (LR, RF, XGBoost) on extracted features",
        "dataset": {"train": 78566, "val": 17063, "test": 16491}
    },
    "cnn-custom-v1": {
        "version": "v1",
        "parent": None,
        "changes": "Initial custom CNN from scratch",
        "dataset": {"train": 78566, "val": 17063, "test": 16491}
    },
    "cnn-custom-test": {
        "version": "v1-test",
        "parent": None,
        "changes": "Test run with 10 samples for pipeline validation",
        "dataset": {"train": 10, "val": 10, "test": 10}
    },
    "kaggle-06c_optimized_training_history": {
        "version": "v7",
        "parent": None,
        "changes": "Kaggle GPU training: 64 filters, 0.5 dropout, 0.0001 L2, LR=0.001, stopped at epoch 15 (best val_auc=0.703)",
        "dataset": {"train": 78566, "val": 17063, "test": 16491},
        "platform": "kaggle",
        "epochs": 25,
        "best_epoch": 15
    },
    "kaggle-v8-default-params": {
        "version": "v8",
        "parent": "v7",
        "changes": "Warm-start from v7, default params, failed (model collapse)",
        "dataset": {"train": 78566, "val": 17063, "test": 16491},
        "platform": "kaggle",
        "epochs": 26,
        "warm_start": True,
        "parent_model": "cnn-v7-model/cnn_optimized_best.keras"
    },
    "kaggle-v9-finetune-regularized": {
        "version": "v9",
        "parent": "v7",
        "changes": "Warm-start from v7, over-regularized (LR=0.0001, dropout=0.6, L2=0.0003, batch=128), failed (model collapse)",
        "dataset": {"train": 78566, "val": 17063, "test": 16491},
        "platform": "kaggle",
        "epochs": 9,
        "warm_start": True,
        "parent_model": "cnn-v7-model/cnn_optimized_best.keras"
    }
}


def get_run_version_info(run_name):
    """Get version metadata for a run based on its name."""
    for pattern, info in MODEL_VERSIONS.items():
        if pattern in run_name:
            return info
    return None


def add_versioning_to_run(client, run_id, run_name, verbose=False):
    """Add versioning and dataset metadata to a run."""

    # Get version info
    version_info = get_run_version_info(run_name)

    if not version_info:
        if verbose:
            print(f"  ⚠️  No version info for: {run_name}")
        return False

    if verbose:
        print(f"  📦 Updating: {run_name}")
        print(f"     Version: {version_info['version']}")

    # Set model version tags
    client.set_tag(run_id, "model_version", version_info["version"])

    if version_info.get("parent"):
        client.set_tag(run_id, "parent_model_version", version_info["parent"])
        lineage = f"{version_info['parent']} → {version_info['version']}"
        client.set_tag(run_id, "model_lineage", lineage)

    client.set_tag(run_id, "architecture_changes", version_info["changes"])

    # Set dataset tags
    dataset = version_info["dataset"]
    client.set_tag(run_id, "dataset_name", "NIH-Chest-Xrays-112k")
    client.set_tag(run_id, "dataset_train_size", str(dataset["train"]))
    client.set_tag(run_id, "dataset_val_size", str(dataset["val"]))
    client.set_tag(run_id, "dataset_test_size", str(dataset["test"]))
    client.set_tag(run_id, "dataset_total_size", str(sum(dataset.values())))

    # Set additional metadata
    if version_info.get("platform"):
        # Already set, but ensure consistency
        client.set_tag(run_id, "platform", version_info["platform"])

    if version_info.get("warm_start"):
        # Log as param (more queryable than tag)
        try:
            client.log_param(run_id, "warm_start", "true")
        except:
            pass  # Param might already exist

    if version_info.get("parent_model"):
        try:
            client.log_param(run_id, "parent_model_path", version_info["parent_model"])
        except:
            pass

    if version_info.get("best_epoch"):
        client.set_tag(run_id, "best_epoch", str(version_info["best_epoch"]))

    if verbose:
        print(f"     ✅ Updated")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add model versioning and dataset tracking to MLflow runs"
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed update information"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MLflow Model Versioning & Dataset Tracking")
    print("=" * 80)
    print(f"\nTracking URI: {args.tracking_uri}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPDATE'}")
    print()

    # Connect to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Get all runs
    experiments = client.search_experiments()

    updated_count = 0
    total_runs = 0

    for exp in experiments:
        if exp.name == "Default":
            continue

        runs = client.search_runs(exp.experiment_id)

        if not runs:
            continue

        print(f"\n📊 Experiment: {exp.name}")
        print(f"   {len(runs)} runs")
        print()

        for run in runs:
            total_runs += 1
            run_name = run.data.tags.get('mlflow.runName', 'unnamed')

            if args.dry_run:
                version_info = get_run_version_info(run_name)
                if version_info:
                    print(f"  Would update: {run_name}")
                    print(f"    → Version: {version_info['version']}")
                    if version_info.get('parent'):
                        print(f"    → Parent: {version_info['parent']}")
                    print(f"    → Dataset: {sum(version_info['dataset'].values())} images")
                    updated_count += 1
                else:
                    print(f"  No version info: {run_name}")
            else:
                if add_versioning_to_run(client, run.info.run_id, run_name, args.verbose):
                    updated_count += 1

    print("\n" + "=" * 80)

    if args.dry_run:
        print(f"DRY RUN - No changes made")
        print(f"Would update: {updated_count}/{total_runs} runs")
    else:
        print(f"✅ Updated {updated_count}/{total_runs} runs")
        print("\nAdded metadata:")
        print("  • Model versions (v1, v7, v8, v9, etc.)")
        print("  • Parent model lineage for warm-start runs")
        print("  • Architecture change descriptions")
        print("  • Dataset split sizes (train/val/test)")
        print("  • Dataset name and total size")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
