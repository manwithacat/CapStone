"""
Test suite for Notebook 06: CNN Development
Validates notebook can run successfully before long training runs.
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestNotebook06Setup:
    """Test notebook 06 setup and configuration."""

    def test_data_splits_exist(self):
        """Verify train/val/test split files exist."""
        data_dir = PROJECT_ROOT / "data" / "processed"
        assert (data_dir / "train_split.csv").exists(), "train_split.csv not found"
        assert (data_dir / "val_split.csv").exists(), "val_split.csv not found"
        assert (data_dir / "test_split.csv").exists(), "test_split.csv not found"

    def test_data_splits_valid(self):
        """Verify data splits have correct structure."""
        data_dir = PROJECT_ROOT / "data" / "processed"

        train_df = pd.read_csv(data_dir / "train_split.csv")
        val_df = pd.read_csv(data_dir / "val_split.csv")
        test_df = pd.read_csv(data_dir / "test_split.csv")

        # Check required columns
        required_cols = ["Image Index", "full_path"]
        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            for col in required_cols:
                assert col in df.columns, f"{col} missing from {name}_split.csv"

        # Check sizes
        assert len(train_df) > 0, "Train split is empty"
        assert len(val_df) > 0, "Val split is empty"
        assert len(test_df) > 0, "Test split is empty"

        print(f"✓ Data splits valid: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")

    def test_preprocessing_config_exists(self):
        """Verify preprocessing config with class weights exists."""
        config_path = PROJECT_ROOT / "data" / "processed" / "preprocessing_config.json"
        assert config_path.exists(), "preprocessing_config.json not found"

        with open(config_path) as f:
            config = json.load(f)

        assert "disease_classes" in config, "disease_classes missing"
        assert "class_weights" in config, "class_weights missing"
        assert len(config["disease_classes"]) == 14, "Should have 14 disease classes"

        print(f"✓ Preprocessing config valid with {len(config['disease_classes'])} classes")

    def test_mlflow_setup(self):
        """Verify MLflow can create experiments."""
        # Clean up any existing test experiment
        try:
            exp = mlflow.get_experiment_by_name("test-notebook-06")
            if exp:
                mlflow.delete_experiment(exp.experiment_id)
        except:
            pass

        # Create new experiment
        mlflow.set_experiment("test-notebook-06")
        exp = mlflow.get_experiment_by_name("test-notebook-06")
        assert exp is not None, "Failed to create MLflow experiment"

        print(f"✓ MLflow working (experiment_id={exp.experiment_id})")


class TestNotebook06Model:
    """Test model building and compilation."""

    @pytest.fixture
    def sample_config(self):
        """Minimal config for testing."""
        return {
            "img_height": 224,
            "img_width": 224,
            "channels": 3,
            "filters": [32, 64, 128, 256],
            "dense_units": 512,
            "dropout_rate": 0.5,
            "l2_reg": 0.0001,
            "num_classes": 14,
            "batch_size": 32,
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "use_cosine_decay": True,
            "gradient_clip_norm": None,
        }

    def test_model_architecture(self, sample_config):
        """Verify model can be built."""
        input_shape = (
            sample_config["img_height"],
            sample_config["img_width"],
            sample_config["channels"],
        )

        # Build model (simplified version of notebook function)
        l2_reg = keras.regularizers.l2(sample_config["l2_reg"])
        model = keras.models.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                # Conv Block 1
                keras.layers.Conv2D(
                    sample_config["filters"][0],
                    (3, 3),
                    activation="relu",
                    padding="same",
                    kernel_regularizer=l2_reg,
                ),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.BatchNormalization(),
                # Conv Block 2
                keras.layers.Conv2D(
                    sample_config["filters"][1],
                    (3, 3),
                    activation="relu",
                    padding="same",
                    kernel_regularizer=l2_reg,
                ),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.BatchNormalization(),
                # Dense layers
                keras.layers.Flatten(),
                keras.layers.Dense(
                    sample_config["dense_units"],
                    activation="relu",
                    kernel_regularizer=l2_reg,
                ),
                keras.layers.Dropout(sample_config["dropout_rate"]),
                # Output
                keras.layers.Dense(sample_config["num_classes"], activation="sigmoid"),
            ]
        )

        assert model is not None, "Model build failed"
        assert len(model.layers) > 0, "Model has no layers"
        assert model.output_shape == (None, 14), f"Wrong output shape: {model.output_shape}"

        print(f"✓ Model built successfully with {model.count_params():,} parameters")

    def test_adamw_optimizer(self, sample_config):
        """Verify AdamW optimizer can be created."""
        optimizer = keras.optimizers.AdamW(
            learning_rate=sample_config["learning_rate"],
            weight_decay=sample_config["weight_decay"],
        )

        assert optimizer is not None, "AdamW optimizer creation failed"
        assert optimizer.learning_rate == sample_config["learning_rate"]
        assert optimizer.weight_decay == sample_config["weight_decay"]

        print(f"✓ AdamW optimizer configured (lr={optimizer.learning_rate}, wd={optimizer.weight_decay})")

    def test_cosine_decay_schedule(self, sample_config):
        """Verify cosine decay schedule works."""
        steps_per_epoch = 100  # Mock value
        total_steps = steps_per_epoch * sample_config["epochs"]

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=sample_config["learning_rate"],
            decay_steps=total_steps,
            alpha=0.0,
        )

        # Test LR at different steps
        lr_start = lr_schedule(0).numpy()
        lr_mid = lr_schedule(total_steps // 2).numpy()
        lr_end = lr_schedule(total_steps).numpy()

        assert lr_start == sample_config["learning_rate"], "Initial LR incorrect"
        assert lr_mid < lr_start, "Mid LR should be less than start"
        assert lr_end < lr_mid, "End LR should be less than mid"
        assert lr_end < 0.0001, "Final LR should be near 0"

        print(f"✓ Cosine decay: {lr_start:.6f} → {lr_mid:.6f} → {lr_end:.6f}")

    def test_label_smoothing_loss(self):
        """Verify label smoothing loss function works."""
        loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.1, dtype="float32")

        # Create dummy data
        y_true = np.array([[0, 1, 0, 1]])
        y_pred = np.array([[0.1, 0.9, 0.2, 0.8]])

        loss = loss_fn(y_true, y_pred).numpy()

        assert not np.isnan(loss), "Loss is NaN"
        assert loss > 0, "Loss should be positive"

        print(f"✓ Label smoothing loss function works (loss={loss:.4f})")


class TestNotebook06DataPipeline:
    """Test tf.data pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for testing."""
        data_dir = PROJECT_ROOT / "data" / "processed"
        df = pd.read_csv(data_dir / "train_split.csv").head(100)  # Use only 100 samples

        config_path = data_dir / "preprocessing_config.json"
        with open(config_path) as f:
            config = json.load(f)

        return df, config["disease_classes"]

    def test_dataset_creation(self, sample_data):
        """Verify tf.data.Dataset can be created."""
        df, disease_classes = sample_data

        # Create dataset (simplified version)
        paths = df["full_path"].values[:10]
        labels = df[disease_classes].values[:10].astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.batch(4)

        # Test iteration
        for batch_paths, batch_labels in dataset.take(1):
            assert batch_paths.shape[0] <= 4, "Batch size incorrect"
            assert batch_labels.shape == (batch_paths.shape[0], 14), "Labels shape incorrect"

        print(f"✓ Dataset created successfully from {len(paths)} samples")

    def test_image_loading(self, sample_data):
        """Verify images can be loaded."""
        df, _ = sample_data

        # Get first valid image path
        for _, row in df.head(10).iterrows():
            img_path = row["full_path"]
            if Path(img_path).exists():
                # Load image
                img = tf.io.read_file(img_path)
                img = tf.image.decode_png(img, channels=3)
                img = tf.image.resize(img, [224, 224])

                assert img.shape == (224, 224, 3), f"Wrong image shape: {img.shape}"
                print(f"✓ Image loading works ({img_path})")
                return

        pytest.skip("No valid images found in sample")


def test_notebook_06_quick_run():
    """
    Full integration test: Run notebook with minimal config.
    This is the main pre-flight check before long training runs.
    """
    import papermill as pm

    # Create test config
    test_params = {
        "CONFIG": {
            "img_height": 224,
            "img_width": 224,
            "channels": 3,
            "batch_size": 32,
            "epochs": 1,  # Only 1 epoch for testing
            "learning_rate": 0.001,
            "filters": [32, 64, 128, 256],
            "dense_units": 512,
            "dropout_rate": 0.5,
            "l2_reg": 0.0001,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "use_cosine_decay": True,
            "gradient_clip_norm": None,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "num_classes": 14,
            "use_sample": True,  # Use sample for fast test
            "sample_size": 100,  # Minimal sample
            "random_state": 42,
        },
        "RETRAIN_MODEL": True,  # Actually train (1 epoch)
    }

    notebook_path = PROJECT_ROOT / "jupyter_notebooks" / "06_cnn_development.ipynb"
    output_path = Path("/tmp") / "test_06_integration.ipynb"

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=test_params,
            kernel_name="python3",
        )
        print("✓ Integration test PASSED - Notebook executed successfully")
        return True
    except Exception as e:
        print(f"✗ Integration test FAILED: {e}")
        raise


if __name__ == "__main__":
    # Run with: python tests/test_notebook_06.py
    pytest.main([__file__, "-v", "-s"])
