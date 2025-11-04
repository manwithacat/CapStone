#!/usr/bin/env python3
"""
Update MLflow run durations to reflect actual training time (not import time).

Uses training_time_hours metric to calculate accurate start/end times.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def update_run_duration(db_path, run_uuid, training_hours, verbose=False):
    """Update a run's start_time and end_time based on actual training duration."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get current times
    cursor.execute(
        "SELECT start_time, end_time FROM runs WHERE run_uuid = ?",
        (run_uuid,)
    )
    result = cursor.fetchone()

    if not result:
        if verbose:
            print(f"  ⚠️  Run {run_uuid} not found")
        return False

    old_start, old_end = result

    # Calculate new end time based on training duration
    # Keep start time, but adjust end time to match actual training duration
    training_ms = int(training_hours * 3600 * 1000)  # hours to milliseconds
    new_end = old_start + training_ms

    # Update
    cursor.execute(
        "UPDATE runs SET end_time = ? WHERE run_uuid = ?",
        (new_end, run_uuid)
    )

    conn.commit()
    conn.close()

    if verbose:
        old_duration = (old_end - old_start) / 1000 / 3600
        print(f"  ✅ Updated: {old_duration:.2f}h → {training_hours:.2f}h")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update MLflow run durations based on training_time_hours metric"
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
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print("=" * 80)
    print("MLflow Run Duration Updater")
    print("=" * 80)
    print(f"\nDatabase: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPDATE'}")
    print()

    # Connect and find runs with training_time_hours
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all runs with training_time_hours metric
    query = """
    SELECT
        r.run_uuid,
        r.name,
        m.value as training_hours,
        (r.end_time - r.start_time) / 1000.0 / 3600.0 as recorded_hours
    FROM runs r
    JOIN latest_metrics m ON r.run_uuid = m.run_uuid
    WHERE m.key = 'training_time_hours'
    ORDER BY r.start_time DESC
    """

    cursor.execute(query)
    runs = cursor.fetchall()
    conn.close()

    if not runs:
        print("No runs found with training_time_hours metric")
        return 0

    print(f"Found {len(runs)} runs with training_time_hours:\n")
    print(f"{'Run Name':<45} {'Recorded':<12} {'Actual':<12} {'Status':<15}")
    print("-" * 80)

    updated_count = 0

    for run_uuid, name, training_hours, recorded_hours in runs:
        # Truncate name if too long
        display_name = name[:45] if len(name) > 45 else name

        # Check if needs update (tolerance of 1 minute)
        diff = abs(training_hours - recorded_hours)
        needs_update = diff > 0.017  # 1 minute in hours

        status = "✅ OK" if not needs_update else "🔄 UPDATE"

        print(f"{display_name:<45} {recorded_hours:<12.2f}h {training_hours:<12.2f}h {status:<15}")

        if needs_update and not args.dry_run:
            update_run_duration(db_path, run_uuid, training_hours, verbose=False)
            updated_count += 1

    print("\n" + "=" * 80)

    if args.dry_run:
        print("DRY RUN - No changes made")
        print(f"Would update: {sum(1 for _, _, t, r in runs if abs(t-r) > 0.017)} runs")
    else:
        print(f"✅ Updated {updated_count} runs")
        print("\nDurations now reflect actual training time (not import time)")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
