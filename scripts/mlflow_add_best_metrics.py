#!/usr/bin/env python3
"""
Add best_val_auc and best_val_auc_epoch metrics to MLflow runs.

Looks through epoch-by-epoch metrics to find the best validation AUC
and which epoch it occurred.
"""

import argparse
import sqlite3
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def add_best_metrics(db_path, dry_run=False, verbose=False):
    """Add best_val_auc and best_val_auc_epoch to runs."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all runs
    cursor.execute("SELECT run_uuid, name FROM runs ORDER BY start_time DESC")
    runs = cursor.fetchall()

    print(f"Found {len(runs)} runs")
    print()

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()

    updated_count = 0

    for run_uuid, run_name in runs:
        # Get all val_auc metrics for this run (with step numbers)
        cursor.execute("""
            SELECT step, value
            FROM metrics
            WHERE run_uuid = ? AND key = 'val_auc'
            ORDER BY step
        """, (run_uuid,))

        val_aucs = cursor.fetchall()

        if not val_aucs:
            if verbose:
                print(f"⚠️  {run_name}: No val_auc metrics")
            continue

        # Find best val_auc
        best_step, best_auc = max(val_aucs, key=lambda x: x[1])

        # Check if already has best_val_auc
        cursor.execute("""
            SELECT value FROM latest_metrics
            WHERE run_uuid = ? AND key = 'best_val_auc'
        """, (run_uuid,))

        existing = cursor.fetchone()

        if existing:
            if verbose:
                print(f"✓  {run_name}: Already has best_val_auc={existing[0]:.4f}")
            continue

        # Add the metrics
        if dry_run:
            print(f"Would add: {run_name}")
            print(f"  best_val_auc: {best_auc:.4f}")
            print(f"  best_val_auc_epoch: {best_step}")
        else:
            client.log_metric(run_uuid, "best_val_auc", best_auc)
            client.log_metric(run_uuid, "best_val_auc_epoch", best_step)

            if verbose or len(val_aucs) > 1:  # Show if multiple epochs
                print(f"✅ {run_name}")
                print(f"   best_val_auc: {best_auc:.4f} (epoch {best_step})")
                if len(val_aucs) > 1:
                    final_auc = val_aucs[-1][1]
                    if final_auc < best_auc:
                        print(f"   final_val_auc: {final_auc:.4f} (degraded by {best_auc - final_auc:.4f})")

            updated_count += 1

    conn.close()

    print()
    print("=" * 80)
    if dry_run:
        print(f"DRY RUN - Would update {updated_count} runs")
    else:
        print(f"✅ Updated {updated_count} runs")
        print("\nAdded metrics:")
        print("  • best_val_auc: Best validation AUC across all epochs")
        print("  • best_val_auc_epoch: Epoch number where best AUC occurred")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Add best_val_auc and best_val_auc_epoch to MLflow runs"
    )
    parser.add_argument(
        "--db",
        default="mlflow.db",
        help="Path to MLflow SQLite database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print("=" * 80)
    print("MLflow Best Metrics Updater")
    print("=" * 80)
    print(f"\nDatabase: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPDATE'}")
    print()

    add_best_metrics(db_path, args.dry_run, args.verbose)

    return 0


if __name__ == "__main__":
    exit(main())
