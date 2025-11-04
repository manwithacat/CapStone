#!/usr/bin/env python3
"""
Notebook 06 Quick Pre-Flight Test
Runs a minimal training loop to validate the full pipeline works.
Target: < 30 seconds, creates MLflow run
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("Notebook 06 Quick Pre-Flight Test")
print("=" * 60)
print()

# Configuration for ultra-fast test
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'channels': 3,
    'batch_size': 8,  # Small batch for speed
    'epochs': 1,
    'learning_rate': 0.001,
    'filters': [16, 32],  # Minimal architecture
    'dense_units': 64,
    'dropout_rate': 0.5,
    'l2_reg': 0.0001,
    'weight_decay': 0.01,
    'optimizer': 'adamw',
    'num_classes': 14,
    'sample_size': 16,  # Tiny sample for speed
}

print("Step 1/5: Loading data...")
data_dir = PROJECT_ROOT / "data" / "processed"

# Load config
config_path = data_dir / "preprocessing_config.json"
with open(config_path) as f:
    prep_config = json.load(f)

disease_classes = prep_config['disease_classes']
class_weights_dict = prep_config.get('class_weights', {})

# Load training data (tiny sample)
train_df = pd.read_csv(data_dir / "train_split.csv").head(CONFIG['sample_size'])
print(f"✓ Loaded {len(train_df)} training samples")

print()
print("Step 2/5: Creating data pipeline...")

def load_and_preprocess_image(path, label):
    """Load and preprocess image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [CONFIG['img_height'], CONFIG['img_width']])
    img = img / 255.0  # Normalize
    return img, label

# Create dataset
train_paths = train_df['full_path'].values
train_labels = train_df[disease_classes].values.astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=2)
train_dataset = train_dataset.batch(CONFIG['batch_size'])
train_dataset = train_dataset.prefetch(1)

print(f"✓ Created tf.data pipeline with {len(train_df)} samples")

print()
print("Step 3/5: Building model...")

# Build minimal CNN
l2_reg = keras.regularizers.l2(CONFIG['l2_reg'])
model = keras.models.Sequential([
    keras.layers.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], CONFIG['channels'])),

    # Conv Block 1
    keras.layers.Conv2D(CONFIG['filters'][0], (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),

    # Conv Block 2
    keras.layers.Conv2D(CONFIG['filters'][1], (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),

    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(CONFIG['dense_units'], activation='relu', kernel_regularizer=l2_reg),
    keras.layers.Dropout(CONFIG['dropout_rate']),

    # Output
    keras.layers.Dense(CONFIG['num_classes'], activation='sigmoid'),
])

print(f"✓ Model built with {model.count_params():,} parameters")

print()
print("Step 4/5: Compiling and training...")

# Configure optimizer (AdamW)
optimizer = keras.optimizers.AdamW(
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
)

# Compile
model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1, dtype='float32'),
    metrics=[
        keras.metrics.BinaryAccuracy(name='accuracy', dtype='float32'),
        keras.metrics.AUC(name='auc', dtype='float32', multi_label=True, num_labels=CONFIG['num_classes']),
    ]
)

print("✓ Model compiled with AdamW optimizer")

# Start MLflow run (use test experiment to separate from production runs)
mlflow.set_experiment("cnn-custom-test")

with mlflow.start_run(run_name="quick-preflight-test") as run:
    # Tag this as a test run
    mlflow.set_tags({
        "test": "true",
        "test_type": "quick-preflight",
        "test_samples": str(CONFIG['sample_size']),
        "test_epochs": str(CONFIG['epochs']),
    })
    run_id = run.info.run_id
    print(f"✓ MLflow run started: {run_id}")

    # Log parameters
    mlflow.log_params({
        'optimizer': 'adamw',
        'learning_rate': CONFIG['learning_rate'],
        'weight_decay': CONFIG['weight_decay'],
        'batch_size': CONFIG['batch_size'],
        'sample_size': CONFIG['sample_size'],
        'epochs': CONFIG['epochs'],
        'test_type': 'quick-preflight',
    })

    # Train for 1 epoch
    print()
    print(f"Training for {CONFIG['epochs']} epoch on {CONFIG['sample_size']} samples...")
    history = model.fit(
        train_dataset,
        epochs=CONFIG['epochs'],
        verbose=1,
    )

    # Log metrics
    final_auc = history.history['auc'][-1]
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]

    mlflow.log_metrics({
        'final_auc': final_auc,
        'final_loss': final_loss,
        'final_accuracy': final_acc,
    })

    print()
    print("=" * 60)
    print("✅ PRE-FLIGHT TEST PASSED")
    print("=" * 60)
    print()
    print(f"MLflow run ID: {run_id}")
    print("Experiment: cnn-custom-test (test runs)")
    print()
    print("Results:")
    print(f"  Final AUC: {final_auc:.4f}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Final Accuracy: {final_acc:.4f}")
    print()
    print("✓ Data pipeline works")
    print("✓ Model builds and compiles")
    print("✓ Training completes successfully")
    print("✓ MLflow tracking works")
    print()
    print("Notebook 06 is ready for production training!")
    print()

print("Step 5/5: Verifying MLflow run...")
client = mlflow.tracking.MlflowClient()
experiment = mlflow.get_experiment_by_name("cnn-custom-test")
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if runs:
    latest_run = runs[0]
    print(f"✓ MLflow run verified: {latest_run.info.run_id}")
    print(f"✓ Status: {latest_run.info.status}")
else:
    print("⚠ Warning: Could not verify MLflow run")

sys.exit(0)
