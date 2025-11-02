#!/usr/bin/env python3
"""
Compare MLflow runs and generate comparison report.

Usage:
    python scripts/mlflow_compare.py --experiment cnn-custom --top 5
    python scripts/mlflow_compare.py --runs run1,run2,run3
"""

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mlflow_utils import search_best_runs, compare_runs


def main():
    parser = argparse.ArgumentParser(description="Compare MLflow runs")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to compare runs from"
    )
    parser.add_argument(
        "--runs",
        type=str,
        help="Comma-separated list of run IDs to compare"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top runs to show (when using --experiment)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_auc",
        help="Metric to sort by (default: val_auc)"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="batch_size,learning_rate,epochs",
        help="Comma-separated list of parameters to include"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="val_auc,val_loss,test_auc",
        help="Comma-separated list of metrics to include"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path"
    )

    args = parser.parse_args()

    # Set MLflow tracking URI if needed
    # mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

    if args.runs:
        # Compare specific runs
        run_ids = [r.strip() for r in args.runs.split(',')]
        params = [p.strip() for p in args.params.split(',')]
        metrics = [m.strip() for m in args.metrics.split(',')]

        if not args.experiment:
            print("Error: --experiment required when using --runs")
            sys.exit(1)

        print(f"Comparing {len(run_ids)} runs from experiment '{args.experiment}'...")
        comparison = compare_runs(
            args.experiment,
            run_ids,
            metrics=metrics,
            params=params
        )

    elif args.experiment:
        # Get top runs from experiment
        print(f"Finding top {args.top} runs from experiment '{args.experiment}'...")
        comparison = search_best_runs(
            args.experiment,
            metric=f"metrics.{args.metric}",
            max_results=args.top
        )

        # Select relevant columns
        cols = ['run_id', 'start_time', 'status']
        for p in args.params.split(','):
            col = f'params.{p.strip()}'
            if col in comparison.columns:
                cols.append(col)
        for m in args.metrics.split(','):
            col = f'metrics.{m.strip()}'
            if col in comparison.columns:
                cols.append(col)

        comparison = comparison[cols]

    else:
        print("Error: Either --experiment or --runs must be specified")
        parser.print_help()
        sys.exit(1)

    # Display results
    print("\n" + "="*80)
    print("RUN COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))
    print("="*80)

    # Save to file if requested
    if args.output:
        comparison.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")

    # Summary statistics
    if not comparison.empty:
        print("\nSummary Statistics:")
        numeric_cols = comparison.select_dtypes(include=['float64', 'int64']).columns
        print(comparison[numeric_cols].describe().round(4).to_string())


if __name__ == "__main__":
    main()
