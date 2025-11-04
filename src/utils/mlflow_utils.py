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


def log_model_version(
    version: str,
    parent_version: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    changes: str = "",
    warm_start: bool = False,
    best_epoch: Optional[int] = None
):
    """
    Log model version metadata to current MLflow run.

    Args:
        version: Model version identifier (e.g., "v11", "v1-test")
        parent_version: Version of parent model if warm-start (e.g., "v7")
        parent_run_id: MLflow run_id of parent model (optional)
        changes: Description of architecture/parameter changes
        warm_start: Whether this run loaded a pre-trained model
        best_epoch: Epoch number where best validation metric was achieved

    Example:
        >>> with mlflow.start_run(run_name="v11-training"):
        >>>     # ... training code ...
        >>>     log_model_version(
        >>>         version="v11",
        >>>         parent_version="v7",
        >>>         changes="Increased filters to 128, added batch norm",
        >>>         warm_start=True
        >>>     )
    """
    # Set version tag
    mlflow.set_tag("model_version", version)

    # Set parent lineage if warm-start
    if parent_version:
        mlflow.set_tag("parent_model_version", parent_version)
        lineage = f"{parent_version} → {version}"
        mlflow.set_tag("model_lineage", lineage)

    if parent_run_id:
        mlflow.set_tag("parent_model_run_id", parent_run_id)

    # Log architecture changes
    if changes:
        mlflow.set_tag("architecture_changes", changes)

    # Log warm-start flag as param (more queryable)
    if warm_start:
        mlflow.log_param("warm_start", "true")

    # Log best epoch
    if best_epoch is not None:
        mlflow.set_tag("best_epoch", str(best_epoch))

    print(f"✅ Logged model version: {version}")
    if parent_version:
        print(f"   Parent: {parent_version}")


def log_dataset_info(
    train_size: int,
    val_size: int,
    test_size: int,
    dataset_name: str = "NIH-Chest-Xrays-112k",
    preprocessing: Optional[Dict[str, Any]] = None
):
    """
    Log dataset metadata to current MLflow run.

    Args:
        train_size: Number of training images
        val_size: Number of validation images
        test_size: Number of test images
        dataset_name: Name of the dataset
        preprocessing: Dictionary of preprocessing parameters

    Example:
        >>> with mlflow.start_run():
        >>>     log_dataset_info(
        >>>         train_size=78566,
        >>>         val_size=17063,
        >>>         test_size=16491,
        >>>         preprocessing={
        >>>             "img_height": 224,
        >>>             "augmentation": True
        >>>         }
        >>>     )
    """
    # Set dataset tags
    mlflow.set_tag("dataset_name", dataset_name)
    mlflow.set_tag("dataset_train_size", str(train_size))
    mlflow.set_tag("dataset_val_size", str(val_size))
    mlflow.set_tag("dataset_test_size", str(test_size))
    mlflow.set_tag("dataset_total_size", str(train_size + val_size + test_size))

    # Log split ratios
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total
    mlflow.set_tag("dataset_split_ratio", f"{train_ratio:.2%}/{val_ratio:.2%}/{test_ratio:.2%}")

    # Log preprocessing params
    if preprocessing:
        for key, value in preprocessing.items():
            mlflow.log_param(f"data_{key}", value)

    print(f"✅ Logged dataset info: {train_size:,} train, {val_size:,} val, {test_size:,} test")


def log_training_time(hours: float):
    """
    Log actual training time in hours.

    This is separate from MLflow's automatic duration tracking, which includes
    import/setup time. Use this to record actual model training time.

    Args:
        hours: Training time in hours (can be fractional)
    """
    mlflow.log_metric("training_time_hours", hours)

    # Also log in different units for convenience
    mlflow.log_metric("training_time_minutes", hours * 60)

    if hours >= 1:
        print(f"✅ Logged training time: {hours:.2f} hours")
    else:
        print(f"✅ Logged training time: {hours*60:.1f} minutes")


def get_best_run(
    experiment_name: str,
    metric: str = "val_auc",
    ascending: bool = False
) -> Optional[str]:
    """
    Get the run_id of the best run in an experiment.

    Useful for finding parent models for warm-start training.

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to use for ranking (default: "val_auc")
        ascending: If True, lower is better (default: False, higher is better)

    Returns:
        run_id of best run, or None if no runs found

    Example:
        >>> best_run_id = get_best_run("cnn-custom", "val_auc")
        >>> print(f"Best run: {best_run_id}")
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found")
        return None

    # Search runs
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1
    )

    if not runs:
        print(f"❌ No runs found in experiment '{experiment_name}'")
        return None

    best_run = runs[0]
    best_value = best_run.data.metrics.get(metric)
    run_name = best_run.data.tags.get('mlflow.runName', 'unnamed')

    print(f"✅ Best run: {run_name}")
    print(f"   {metric}: {best_value:.4f}")
    print(f"   run_id: {best_run.info.run_id}")

    return best_run.info.run_id


def query_runs_by_version(version: str) -> list:
    """
    Find all runs with a specific model version.

    Args:
        version: Model version to search for (e.g., "v7", "v11")

    Returns:
        List of MLflow Run objects

    Example:
        >>> runs = query_runs_by_version("v7")
        >>> for run in runs:
        >>>     print(run.data.tags.get('mlflow.runName'))
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Search all runs with the version tag
    runs = client.search_runs(
        experiment_ids=[],  # Search all experiments
        filter_string=f"tags.model_version = '{version}'"
    )

    print(f"Found {len(runs)} runs with version '{version}'")
    return runs


def compare_model_lineage(version1: str, version2: str):
    """
    Compare two model versions and their metadata.

    Useful for understanding what changed between versions.

    Args:
        version1: First model version (e.g., "v7")
        version2: Second model version (e.g., "v10")

    Example:
        >>> compare_model_lineage("v7", "v10")
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get runs for both versions
    runs1 = query_runs_by_version(version1)
    runs2 = query_runs_by_version(version2)

    if not runs1 or not runs2:
        print("❌ One or both versions not found")
        return

    run1 = runs1[0]
    run2 = runs2[0]

    print("\n" + "=" * 80)
    print(f"Comparing {version1} vs {version2}")
    print("=" * 80)

    # Compare architecture changes
    changes1 = run1.data.tags.get('architecture_changes', 'N/A')
    changes2 = run2.data.tags.get('architecture_changes', 'N/A')

    print(f"\n{version1} changes:")
    print(f"  {changes1}")
    print(f"\n{version2} changes:")
    print(f"  {changes2}")

    # Compare key metrics
    print("\nMetrics comparison:")
    metrics_to_compare = ['val_auc', 'val_loss', 'val_precision', 'val_recall']

    for metric in metrics_to_compare:
        val1 = run1.data.metrics.get(metric, 0)
        val2 = run2.data.metrics.get(metric, 0)
        diff = val2 - val1
        emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"  {metric:<20} {val1:.4f} → {val2:.4f} ({diff:+.4f}) {emoji}")

    # Compare parameters
    print("\nParameter changes:")
    all_params = set(run1.data.params.keys()) | set(run2.data.params.keys())

    for param in sorted(all_params):
        val1 = run1.data.params.get(param, 'N/A')
        val2 = run2.data.params.get(param, 'N/A')
        if val1 != val2:
            print(f"  {param:<25} {val1} → {val2}")

    print("=" * 80)


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
