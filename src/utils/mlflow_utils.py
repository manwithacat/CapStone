"""
MLflow Utilities for NIH Chest X-Ray Project

Helper functions for experiment tracking, model logging, and metrics visualization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
import pandas as pd
from tensorflow import keras


class MLflowExperimentTracker:
    """
    Context manager for MLflow experiment tracking.

    Usage:
        with MLflowExperimentTracker(
            experiment_name="cnn-custom",
            run_name="cnn-optimized-v1",
            params=CONFIG
        ) as tracker:
            # Train model
            history = model.fit(...)

            # Log metrics
            tracker.log_training_history(history)
            tracker.log_model(model, "cnn_model")
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        params: Optional[Dict] = None,
        tags: Optional[Dict] = None,
        description: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self.description = description
        self.run = None
        self.start_time = None

    def __enter__(self):
        # Set experiment
        mlflow.set_experiment(self.experiment_name)

        # Start run
        self.run = mlflow.start_run(run_name=self.run_name)
        self.start_time = time.time()

        # Log parameters
        if self.params:
            # Flatten nested dicts
            flat_params = self._flatten_dict(self.params)
            mlflow.log_params(flat_params)

        # Set tags
        if self.tags:
            mlflow.set_tags(self.tags)

        # Set description
        if self.description:
            mlflow.set_tag("mlflow.note.content", self.description)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log total time
        if self.start_time:
            total_time = time.time() - self.start_time
            mlflow.log_metric("total_time_seconds", total_time)

        # End run
        mlflow.end_run()

        # Don't suppress exceptions
        return False

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for MLflow parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    MLflowExperimentTracker._flatten_dict(v, new_key, sep=sep).items()
                )
            elif isinstance(v, (list, tuple)):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def log_training_history(
        self,
        history: keras.callbacks.History,
        prefix: str = ""
    ):
        """
        Log Keras training history to MLflow.

        Args:
            history: Keras History object from model.fit()
            prefix: Prefix to add to metric names (e.g., "train_")
        """
        for epoch, metrics in enumerate(history.epoch):
            step_metrics = {}
            for key, values in history.history.items():
                metric_name = f"{prefix}{key}" if prefix else key
                step_metrics[metric_name] = values[epoch]

            mlflow.log_metrics(step_metrics, step=epoch)

        # Log best epoch
        if 'val_loss' in history.history:
            best_epoch = np.argmin(history.history['val_loss'])
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("best_val_loss", history.history['val_loss'][best_epoch])

        if 'val_auc' in history.history:
            best_epoch = np.argmax(history.history['val_auc'])
            mlflow.log_metric("best_val_auc", history.history['val_auc'][best_epoch])

    def log_model(
        self,
        model,
        artifact_path: str,
        framework: str = "keras",
        **kwargs
    ):
        """
        Log model to MLflow.

        Args:
            model: Model object (Keras, sklearn, etc.)
            artifact_path: Path within artifact store
            framework: "keras" or "sklearn"
            **kwargs: Additional arguments for model logging
        """
        if framework == "keras":
            mlflow.keras.log_model(model, artifact_path, **kwargs)
        elif framework == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def log_metrics_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log dictionary of metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, fig, filename: str, artifact_path: Optional[str] = None):
        """
        Save matplotlib figure and log as artifact.

        Args:
            fig: Matplotlib figure object
            filename: Filename to save as
            artifact_path: Subdirectory in artifact store
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(str(filepath), artifact_path)


class MLflowKerasCallback(keras.callbacks.Callback):
    """
    Keras callback for logging metrics to MLflow during training.

    Usage:
        mlflow.start_run()
        model.fit(
            X_train, y_train,
            callbacks=[MLflowKerasCallback()]
        )
        mlflow.end_run()
    """

    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if logs and epoch % self.log_every_n_epochs == 0:
            mlflow.log_metrics(logs, step=epoch)


def log_per_disease_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disease_classes: List[str],
    prefix: str = ""
):
    """
    Log per-disease performance metrics to MLflow.

    Args:
        y_true: True labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        disease_classes: List of disease class names
        prefix: Prefix for metric names (e.g., "test_")
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    metrics = {}

    for i, disease in enumerate(disease_classes):
        y_true_disease = y_true[:, i]
        y_pred_disease = y_pred[:, i]

        # AUC
        if len(np.unique(y_true_disease)) > 1:
            auc = roc_auc_score(y_true_disease, y_pred_disease)
            metrics[f"{prefix}auc_{disease}"] = auc

        # F1, Precision, Recall (with threshold 0.5)
        y_pred_binary = (y_pred_disease > 0.5).astype(int)
        f1 = f1_score(y_true_disease, y_pred_binary, zero_division=0)
        precision = precision_score(y_true_disease, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_disease, y_pred_binary, zero_division=0)

        metrics[f"{prefix}f1_{disease}"] = f1
        metrics[f"{prefix}precision_{disease}"] = precision
        metrics[f"{prefix}recall_{disease}"] = recall

    mlflow.log_metrics(metrics)

    # Log average metrics
    avg_metrics = {
        f"{prefix}avg_auc": np.mean([v for k, v in metrics.items() if 'auc_' in k]),
        f"{prefix}avg_f1": np.mean([v for k, v in metrics.items() if 'f1_' in k]),
        f"{prefix}avg_precision": np.mean([v for k, v in metrics.items() if 'precision_' in k]),
        f"{prefix}avg_recall": np.mean([v for k, v in metrics.items() if 'recall_' in k])
    }
    mlflow.log_metrics(avg_metrics)


def search_best_runs(
    experiment_name: str,
    metric: str = "metrics.val_auc",
    max_results: int = 10,
    filter_string: Optional[str] = None
) -> pd.DataFrame:
    """
    Search for best runs in an experiment.

    Args:
        experiment_name: Name of MLflow experiment
        metric: Metric to sort by (e.g., "metrics.val_auc")
        max_results: Maximum number of runs to return
        filter_string: Optional filter (e.g., "params.batch_size = '128'")

    Returns:
        DataFrame with run information

    Example:
        >>> best_runs = search_best_runs(
        ...     "cnn-custom",
        ...     metric="metrics.val_auc",
        ...     max_results=5,
        ...     filter_string="params.optimizer = 'adam'"
        ... )
    """
    return mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        order_by=[f"{metric} DESC"],
        max_results=max_results
    )


def load_model_from_registry(
    model_name: str,
    stage: str = "Production"
):
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Name of registered model
        stage: Model stage ("None", "Staging", "Production", "Archived")

    Returns:
        Loaded model

    Example:
        >>> model = load_model_from_registry(
        ...     "cnn-chest-xray-classifier",
        ...     stage="Production"
        ... )
    """
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.keras.load_model(model_uri)


def load_model_from_run(
    run_id: str,
    artifact_path: str = "model"
):
    """
    Load model from specific run.

    Args:
        run_id: MLflow run ID
        artifact_path: Path to model artifact

    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.keras.load_model(model_uri)


def compare_runs(
    experiment_name: str,
    run_ids: List[str],
    metrics: Optional[List[str]] = None,
    params: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple runs side-by-side.

    Args:
        experiment_name: Name of experiment
        run_ids: List of run IDs to compare
        metrics: List of metrics to include
        params: List of parameters to include

    Returns:
        DataFrame with comparison

    Example:
        >>> comparison = compare_runs(
        ...     "cnn-custom",
        ...     run_ids=["abc123", "def456"],
        ...     metrics=["val_auc", "test_auc"],
        ...     params=["batch_size", "learning_rate"]
        ... )
    """
    # Get all runs
    all_runs = mlflow.search_runs(experiment_names=[experiment_name])

    # Filter to specified runs
    runs = all_runs[all_runs['run_id'].isin(run_ids)]

    # Select columns
    cols = ['run_id', 'start_time', 'status']

    if params:
        param_cols = [f'params.{p}' for p in params]
        cols.extend([c for c in param_cols if c in runs.columns])

    if metrics:
        metric_cols = [f'metrics.{m}' for m in metrics]
        cols.extend([c for c in metric_cols if c in runs.columns])

    return runs[cols].sort_values('start_time', ascending=False)


def get_artifact_path(run_id: str, artifact_path: str = "") -> str:
    """
    Get local path to artifact from a run.

    Args:
        run_id: MLflow run ID
        artifact_path: Path within artifacts

    Returns:
        Local filesystem path to artifact
    """
    client = mlflow.tracking.MlflowClient()
    return client.download_artifacts(run_id, artifact_path)


def register_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "val_auc",
    description: Optional[str] = None
) -> Any:
    """
    Find best run and register its model.

    Args:
        experiment_name: Name of experiment
        model_name: Name to register model as
        metric: Metric to optimize
        description: Model version description

    Returns:
        ModelVersion object
    """
    # Find best run
    best_run = search_best_runs(
        experiment_name,
        metric=f"metrics.{metric}",
        max_results=1
    ).iloc[0]

    run_id = best_run['run_id']
    metric_value = best_run[f'metrics.{metric}']

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    # Update description
    if description is None:
        description = f"Best model from {experiment_name} (val_{metric}={metric_value:.4f})"

    client = mlflow.tracking.MlflowClient()
    client.update_model_version(
        name=model_name,
        version=result.version,
        description=description
    )

    return result


# Example usage
if __name__ == "__main__":
    # Example: Track a simple experiment
    with MLflowExperimentTracker(
        experiment_name="example-experiment",
        run_name="test-run",
        params={'learning_rate': 0.001, 'batch_size': 32},
        tags={'framework': 'keras', 'dataset': 'nih-chest-xray'}
    ) as tracker:
        print("Tracking experiment...")
        tracker.log_metrics_dict({'accuracy': 0.85, 'loss': 0.3})
        print("Done!")
