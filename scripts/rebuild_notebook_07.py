#!/usr/bin/env python3
"""
Rebuild notebook 07 with clean structure following notebook 06 pattern.
This creates a working transfer learning notebook with proper two-stage training.
"""

import json
from pathlib import Path

# Create clean notebook structure
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        },
        "jupytext": {
            "formats": "ipynb,py:percent"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown_cell(content):
    """Add a markdown cell to the notebook."""
    # Split by \n and add \n back to each line except last
    lines = content.split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # Add last line without \n if it's not empty
        source.append(lines[-1])
    nb['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code_cell(content):
    """Add a code cell to the notebook."""
    # Split by \n and add \n back to each line except last
    lines = content.split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # Add last line without \n if it's not empty
        source.append(lines[-1])
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# Title and Overview
add_markdown_cell("""# Notebook 07: Transfer Learning

---

## Overview

This notebook implements **transfer learning** using pre-trained models for multi-label chest X-ray disease classification.

**Objectives:**
1. Load pre-trained models (ResNet50, DenseNet121, EfficientNetB3)
2. Adapt for multi-label classification (14 diseases)
3. Implement two-stage training: feature extraction → fine-tuning
4. Track experiments with MLflow
5. Compare transfer learning vs custom CNN

**Why Transfer Learning?**

Pre-trained models learned from **1.4 million ImageNet images**:
- Low-level features (edges, textures) transfer well to medical imaging
- Saves weeks of training time
- Often outperforms custom CNNs when data is limited

**Outputs:**
- Fine-tuned models saved to `models/saved_models/`
- Training metrics logged to MLflow
- Performance comparison with baseline and CNN models

---""")

# Setup
add_markdown_cell("## 1. Setup and Configuration")

add_code_cell("""# Import libraries
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    EfficientNetB3,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# MLflow tracking
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("✓ Libraries imported successfully")
print(f"TensorFlow version: {tf.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPUs")""")

add_code_cell("""# Define paths
current_path = Path.cwd()

if current_path.name == 'jupyter_notebooks':
    PROJECT_ROOT = current_path.parent
elif (current_path / 'setup.py').exists() or (current_path / 'README.md').exists():
    PROJECT_ROOT = current_path
else:
    PROJECT_ROOT = current_path.parent

DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved_models'

# Create directories if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Models directory: {MODELS_DIR}")""")

add_code_cell("""# Transfer Learning Configuration
CONFIG = {
    # Image parameters
    'img_height': 224,
    'img_width': 224,
    'channels': 3,

    # Training parameters - Stage 1 (Feature extraction)
    'batch_size': 32,
    'epochs_stage1': 10,  # Feature extraction
    'learning_rate_stage1': 0.001,

    # Training parameters - Stage 2 (Fine-tuning)
    'epochs_stage2': 20,  # Fine-tuning
    'learning_rate_stage2': 0.0001,  # Lower LR for fine-tuning
    'unfreeze_layers': 20,  # Number of layers to unfreeze from top

    # Model architecture
    'dense_units': 512,
    'dropout_rate': 0.5,

    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,

    # Data
    'num_classes': 14,
    'use_sample': False,
    'sample_size': 5000,

    'random_state': 42
}

print("Transfer Learning Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")""")

# Parameters cell for papermill
add_code_cell("""# Parameters (can be overridden by papermill)
MODELS_TO_TRAIN = ['resnet50', 'densenet121', 'efficientnetb3']  # Which models to train
RUN_NAME = None  # Custom run name for MLflow (auto-generated if None)
USE_MLFLOW = True  # Enable MLflow tracking
RETRAIN_MODELS = False  # Set to True to retrain even if models exist

print(f"Models to train: {MODELS_TO_TRAIN}")
print(f"Run name: {RUN_NAME if RUN_NAME else 'Auto-generated'}")
print(f"MLflow tracking: {'Enabled' if USE_MLFLOW else 'Disabled'}")
print(f"Retrain models: {RETRAIN_MODELS}")""")

# Type conversion cell for papermill string parameters
add_code_cell("""# Convert papermill string parameters to correct types
import ast

# Handle MODELS_TO_TRAIN (papermill passes as string "['densenet121']")
if isinstance(MODELS_TO_TRAIN, str):
    try:
        MODELS_TO_TRAIN = ast.literal_eval(MODELS_TO_TRAIN)
        print(f"✓ Converted MODELS_TO_TRAIN from string: {MODELS_TO_TRAIN}")
    except:
        # If it's a single model name, wrap in list
        MODELS_TO_TRAIN = [MODELS_TO_TRAIN]
        print(f"✓ Wrapped MODELS_TO_TRAIN in list: {MODELS_TO_TRAIN}")

# Handle USE_MLFLOW (papermill passes as string "true")
if isinstance(USE_MLFLOW, str):
    USE_MLFLOW = USE_MLFLOW.lower() in ('true', 'yes', '1')
    print(f"✓ Converted USE_MLFLOW from string: {USE_MLFLOW}")

# Handle RETRAIN_MODELS (papermill passes as string "true")
if isinstance(RETRAIN_MODELS, str):
    RETRAIN_MODELS = RETRAIN_MODELS.lower() in ('true', 'yes', '1')
    print(f"✓ Converted RETRAIN_MODELS from string: {RETRAIN_MODELS}")

# Handle CONFIG if passed as string
if 'CONFIG' in globals() and isinstance(CONFIG, str):
    CONFIG = ast.literal_eval(CONFIG)
    print(f"✓ Converted CONFIG from string")

print(f"\\n📋 Final parameter values:")
print(f"  MODELS_TO_TRAIN: {MODELS_TO_TRAIN} (type: {type(MODELS_TO_TRAIN).__name__})")
print(f"  USE_MLFLOW: {USE_MLFLOW} (type: {type(USE_MLFLOW).__name__})")
print(f"  RETRAIN_MODELS: {RETRAIN_MODELS} (type: {type(RETRAIN_MODELS).__name__})")""")

# MLflow setup
add_markdown_cell("## 2. MLflow Setup")

add_code_cell("""# Setup MLflow tracking
if USE_MLFLOW:
    # Set tracking URI to local SQLite database
    mlflow_db_path = PROJECT_ROOT / 'mlflow.db'
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

    # Set experiment
    experiment_name = "07-transfer-learning-tensorflow"
    mlflow.set_experiment(experiment_name)

    # Generate run name if not provided
    if RUN_NAME is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        RUN_NAME = f"tf_transfer_{timestamp}"

    print(f"✓ MLflow tracking configured")
    print(f"  Tracking URI: sqlite:///{mlflow_db_path}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Run name: {RUN_NAME}")
else:
    print("⚠️  MLflow tracking disabled")""")

# Load data
add_markdown_cell("## 3. Load Data")

add_code_cell("""# Load split files
train_df = pd.read_csv(PROCESSED_DIR / 'train_split.csv')
val_df = pd.read_csv(PROCESSED_DIR / 'val_split.csv')
test_df = pd.read_csv(PROCESSED_DIR / 'test_split.csv')

# Load preprocessing config
with open(PROCESSED_DIR / 'preprocessing_config.json', 'r') as f:
    prep_config = json.load(f)

disease_classes = prep_config['disease_classes']

print(f"✓ Loaded data splits and configuration")
print(f"  Train: {len(train_df):,} images")
print(f"  Val:   {len(val_df):,} images")
print(f"  Test:  {len(test_df):,} images")
print(f"  Disease classes: {len(disease_classes)}")""")

add_code_cell("""# Sample if configured
if CONFIG['use_sample']:
    sample_size = CONFIG['sample_size']
    train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(sample_size // 5, len(val_df)), random_state=42)
    test_df = test_df.sample(n=min(sample_size // 5, len(test_df)), random_state=42)

    print(f"⚠️ Using sample mode:")
    print(f"  Train: {len(train_df):,} images")
    print(f"  Val:   {len(val_df):,} images")
    print(f"  Test:  {len(test_df):,} images")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)""")

# Check existing models
add_markdown_cell("## 4. Check Existing Models")

add_code_cell("""# Check if all models exist
model_paths = {
    'resnet50': MODELS_DIR / "resnet50_transfer_best.keras",
    'densenet121': MODELS_DIR / "densenet121_transfer_best.keras",
    'efficientnetb3': MODELS_DIR / "efficientnetb3_transfer_best.keras"
}

all_models_exist = all([path.exists() for path in model_paths.values()])

if not RETRAIN_MODELS and all_models_exist:
    print("="*60)
    print("LOADING EXISTING MODELS")
    print("="*60)
    print("\\nAll 3 transfer learning models found:")
    for name, path in model_paths.items():
        print(f"  ✓ {path.name}")
    print("\\nSkipping training (RETRAIN_MODELS=False)")
    print("\\n💡 Set RETRAIN_MODELS=True to retrain models\\n")
    SKIP_TRAINING = True
else:
    if RETRAIN_MODELS:
        print("="*60)
        print("RETRAINING ALL MODELS")
        print("="*60)
        print("\\n⚠️  RETRAIN_MODELS=True, training all models from scratch...")
    else:
        print("="*60)
        print("TRAINING REQUIRED")
        print("="*60)
        print("\\n⚠️  Some models missing, will train:")
        missing = [name for name, path in model_paths.items() if not path.exists()]
        for name in missing:
            print(f"     - {name}")

    SKIP_TRAINING = False
    print("\\n▶️  Proceeding with training...\\n")""")

# Data generators
add_markdown_cell("## 5. Data Generators")

add_code_cell("""if not SKIP_TRAINING:
    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='full_path',
        y_col=disease_classes,
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='raw',  # Multi-label
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='full_path',
        y_col=disease_classes,
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='raw',
        shuffle=False
    )

    print("✓ Data generators created")
    print(f"  Train batches: {len(train_generator)}")
    print(f"  Val batches: {len(val_generator)}")
else:
    print("✓ Skipping data generator creation (loading existing models)")""")

# Model building function
add_markdown_cell("## 6. Model Building Function")

add_code_cell("""def build_transfer_model(base_model_class, model_name, input_shape, num_classes, config):
    \"\"\"
    Build transfer learning model with pre-trained base.

    Args:
        base_model_class: Keras application (ResNet50, DenseNet121, etc.)
        model_name: String name for logging
        input_shape: (height, width, channels)
        num_classes: Number of output classes
        config: Configuration dict

    Returns:
        model: Complete Keras model
        base_model: Pre-trained base model
    \"\"\"
    # Load pre-trained base model (without top classification layer)
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,  # Remove ImageNet classification layer
        input_shape=input_shape
    )

    # Freeze base model initially
    base_model.trainable = False

    # Build complete model
    inputs = keras.Input(shape=input_shape)

    # Pre-trained base
    x = base_model(inputs, training=False)

    # Custom top layers for chest X-ray classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config['dense_units'], activation='relu', name='dense_features')(x)
    x = layers.Dropout(config['dropout_rate'], name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)

    model = keras.Model(inputs, outputs, name=model_name)

    return model, base_model


print("✓ Transfer learning model builder defined")""")

# Training function
add_markdown_cell("## 7. Training Function")

add_code_cell("""def train_transfer_learning_model(
    model_class,
    model_name,
    train_gen,
    val_gen,
    config,
    models_to_train,
    use_mlflow,
    run_name
):
    \"\"\"
    Train a transfer learning model with two-stage training.

    Stage 1: Feature extraction (base frozen)
    Stage 2: Fine-tuning (unfreeze top layers)

    Args:
        model_class: Keras application class
        model_name: String name for the model
        train_gen: Training data generator
        val_gen: Validation data generator
        config: Configuration dict
        models_to_train: List of model names to train
        use_mlflow: Whether to log to MLflow
        run_name: MLflow run name prefix

    Returns:
        model: Trained Keras model
        history: Training history
    \"\"\"
    # Check if we should train this model
    if model_name.lower() not in [m.lower() for m in models_to_train]:
        print(f"\\n⏭️  Skipping {model_name} (not in MODELS_TO_TRAIN)")
        return None, None

    print("\\n" + "="*60)
    print(f"TRAINING: {model_name}")
    print("="*60)

    # Start MLflow run
    if use_mlflow:
        mlflow_run = mlflow.start_run(run_name=f"{run_name}_{model_name.lower()}")
        mlflow.log_params({
            'model': model_name,
            'framework': 'tensorflow',
            'batch_size': config['batch_size'],
            'epochs_stage1': config['epochs_stage1'],
            'epochs_stage2': config['epochs_stage2'],
            'lr_stage1': config['learning_rate_stage1'],
            'lr_stage2': config['learning_rate_stage2'],
            'unfreeze_layers': config['unfreeze_layers'],
            'dense_units': config['dense_units'],
            'dropout_rate': config['dropout_rate'],
            'use_sample': config['use_sample'],
        })
        # Enable autologging
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    # Build model
    input_shape = (config['img_height'], config['img_width'], config['channels'])
    model, base_model = build_transfer_model(
        model_class, f"{model_name.lower()}_transfer",
        input_shape, config['num_classes'], config
    )

    print(f"\\n{model_name} architecture:")
    print(f"  Base layers: {len(base_model.layers)} (frozen)")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    # ===== STAGE 1: Feature Extraction =====
    print("\\n" + "-"*60)
    print("STAGE 1: Feature Extraction (base frozen)")
    print("-"*60)
    print(f"Learning rate: {config['learning_rate_stage1']}")
    print(f"Epochs: {config['epochs_stage1']}")
    print("\\nTraining top layers only...\\n")

    # Compile for Stage 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate_stage1']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
    )

    # Train Stage 1
    history_s1 = model.fit(
        train_gen,
        epochs=config['epochs_stage1'],
        validation_data=val_gen,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=config['early_stopping_patience'] // 2,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config['reduce_lr_patience'] // 2,
                verbose=1
            )
        ],
        verbose=1
    )

    print("\\n✓ Stage 1 complete")

    # ===== STAGE 2: Fine-Tuning =====
    print("\\n" + "-"*60)
    print("STAGE 2: Fine-Tuning (unfreeze top layers)")
    print("-"*60)

    # Unfreeze last N layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-config['unfreeze_layers']]:
        layer.trainable = False

    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_layers} / {len(base_model.layers)}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate_stage2']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
    )

    print(f"Learning rate: {config['learning_rate_stage2']} (10× lower)")
    print(f"Epochs: {config['epochs_stage2']}")
    print("\\nFine-tuning...\\n")

    # Define model save path
    model_save_path = MODELS_DIR / f"{model_name.lower()}_transfer_best.keras"

    # Train Stage 2
    history_s2 = model.fit(
        train_gen,
        epochs=config['epochs_stage2'],
        validation_data=val_gen,
        callbacks=[
            callbacks.ModelCheckpoint(
                str(model_save_path),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config['reduce_lr_patience'],
                verbose=1
            )
        ],
        verbose=1
    )

    # Log model artifact to MLflow
    if use_mlflow:
        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(str(model_save_path))
        mlflow.end_run()

    print(f"\\n✓ {model_name} training complete")
    print(f"  Model saved to: {model_save_path}")

    # Combine histories
    history_combined = {
        'stage1': history_s1.history,
        'stage2': history_s2.history
    }

    return model, history_combined


print("✓ Training function defined")""")

# Train models
add_markdown_cell("## 8. Train Models")

add_code_cell("""# Train all models
trained_models = {}
training_histories = {}

if not SKIP_TRAINING:
    for model_name, model_class in [
        ('ResNet50', ResNet50),
        ('DenseNet121', DenseNet121),
        ('EfficientNetB3', EfficientNetB3)
    ]:
        model, history = train_transfer_learning_model(
            model_class=model_class,
            model_name=model_name,
            train_gen=train_generator,
            val_gen=val_generator,
            config=CONFIG,
            models_to_train=MODELS_TO_TRAIN,
            use_mlflow=USE_MLFLOW,
            run_name=RUN_NAME
        )

        if model is not None:
            trained_models[model_name] = model
            training_histories[model_name] = history

    print("\\n" + "="*60)
    print("ALL TRAINING COMPLETE")
    print("="*60)
    print(f"\\nTrained {len(trained_models)} models:")
    for name in trained_models.keys():
        print(f"  ✓ {name}")
else:
    print("\\n⏭️  Skipping training (loading existing models)")""")

# Summary
add_markdown_cell("## 9. Summary")

add_code_cell("""print("="*60)
print("  ✅ Notebook 07 Complete: Transfer Learning")
print("="*60)

if not SKIP_TRAINING:
    print(f"\\n🏗️  Models Trained: {len(trained_models)}")
    for name in trained_models.keys():
        print(f"  - {name}")

    print("\\n📊 Training Strategy:")
    print(f"  Stage 1: Feature extraction ({CONFIG['epochs_stage1']} epochs, base frozen)")
    print(f"  Stage 2: Fine-tuning ({CONFIG['epochs_stage2']} epochs, last {CONFIG['unfreeze_layers']} layers unfrozen)")

    print("\\n📁 Generated Files:")
    for name in trained_models.keys():
        model_file = MODELS_DIR / f"{name.lower()}_transfer_best.keras"
        print(f"  {model_file}")

    if USE_MLFLOW:
        print("\\n📈 MLflow:")
        print(f"  Experiment: 07-transfer-learning-tensorflow")
        print(f"  Runs: {len(trained_models)}")
        print(f"  View at: http://localhost:5000")
else:
    print("\\n✓ Loaded existing models (no training performed)")

print("\\n💡 Next Steps:")
print("  - View MLflow UI to compare models")
print("  - Use notebooks 08-09 for evaluation and interpretation")
print("  - Compare with baseline and CNN models")""")

# Save notebook
notebook_path = Path(__file__).parent.parent / 'jupyter_notebooks' / '07_transfer_learning.ipynb'
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"✓ Created clean notebook: {notebook_path}")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Markdown cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}")
