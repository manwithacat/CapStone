# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transfer Learning - NIH Chest X-Ray Classification (GCS)
#
# **🌐 Single Source of Truth: Google Cloud Storage Medallion Architecture**
#
# This notebook uses the GCS bucket `gs://nih-xrays` with medallion architecture:
# - **00_raw/**: Immutable NIH X-ray images (~47 GB)
# - **10_bronze/**: Train/val/test CSV splits
# - **40_models/**: Trained model checkpoints
# - **50_artifacts/**: Metrics, plots, reports
# - **60_logs/**: TensorBoard logs
#
# **Runtime**: GPU (T4, A100, or V100 recommended)

# %% [markdown]
# ## 1. Check GPU

# %%
!nvidia-smi

# %% [markdown]
# ## 2. Setup GCS Authentication
#
# **Two options for authentication:**
#
# ### Option A: Service Account (Recommended for Production)
# - Upload your service account JSON key
# - No OAuth prompts
# - Works in automated pipelines
#
# ### Option B: User OAuth (Quick Start)
# - Uses your Google account
# - One-time OAuth prompt
# - Good for interactive development

# %%
from google.colab import auth, files
from google.cloud import storage
import os
from pathlib import Path
import json

# Detect if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

BUCKET_NAME = 'nih-xrays'
PROJECT_ID = None  # Auto-detected from credentials

print("🔐 GCS Authentication Setup")
print("=" * 60)

# Check for existing authentication
auth_method = input("\nChoose authentication method:\n  1. Service Account (JSON key)\n  2. User OAuth (Google account)\n\nEnter 1 or 2: ").strip()

if auth_method == '1':
    # Service Account Authentication
    print("\n📁 Upload your service account JSON key file")
    print("   (Download from Google Cloud Console > IAM > Service Accounts)")

    uploaded = files.upload()

    if not uploaded:
        raise ValueError("No file uploaded. Please upload service account JSON key.")

    # Get the uploaded file name (should be .json)
    key_file = list(uploaded.keys())[0]

    # Set environment variable for authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file

    # Load project ID from key file
    with open(key_file, 'r') as f:
        key_data = json.load(f)
        PROJECT_ID = key_data.get('project_id')

    print(f"✓ Service account authenticated")
    print(f"✓ Project ID: {PROJECT_ID}")

elif auth_method == '2':
    # User OAuth Authentication
    print("\n🔓 Authenticating with your Google account...")
    print("   This will open an OAuth consent screen.")

    if IN_COLAB:
        auth.authenticate_user()
    else:
        # For local testing, use application default credentials
        os.system('gcloud auth application-default login')

    # Project ID should be set manually for OAuth
    PROJECT_ID = input("\nEnter your Google Cloud Project ID: ").strip()

    print(f"✓ User OAuth authenticated")
    print(f"✓ Project ID: {PROJECT_ID}")

else:
    raise ValueError("Invalid choice. Please enter 1 or 2.")

# Create storage client
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    # Test access by listing top-level directories
    blobs = list(storage_client.list_blobs(BUCKET_NAME, max_results=5, delimiter='/'))
    prefixes = [prefix for prefix in storage_client.list_blobs(BUCKET_NAME, max_results=100, delimiter='/').prefixes]

    print(f"\n✓ Successfully connected to gs://{BUCKET_NAME}")
    print(f"✓ Top-level directories: {prefixes[:10]}")

except Exception as e:
    print(f"\n❌ Error connecting to bucket: {e}")
    print(f"\nTroubleshooting:")
    print(f"  1. Verify bucket exists: gs://{BUCKET_NAME}")
    print(f"  2. Check IAM permissions (Storage Object Viewer minimum)")
    print(f"  3. Confirm project ID: {PROJECT_ID}")
    raise

print("\n" + "=" * 60)
print("✅ GCS Authentication Complete")
print("=" * 60)

# %% [markdown]
# ## 3. Install Dependencies

# %%
!pip install -q tensorflow matplotlib seaborn scikit-learn google-cloud-storage

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPUs")

# %% [markdown]
# ## 4. Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import tempfile
import shutil

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("✓ Libraries imported")

# %% [markdown]
# ## 5. Configuration

# %%
# GCS Medallion Paths
GCS_BUCKET = f"gs://{BUCKET_NAME}"
GCS_PATHS = {
    'raw_images': f"{GCS_BUCKET}/00_raw/nih-cxr/images/",
    'manifests': f"{GCS_BUCKET}/10_bronze/nih-cxr/manifests/",
    'models': f"{GCS_BUCKET}/40_models/nih-cxr/",
    'artifacts': f"{GCS_BUCKET}/50_artifacts/nih-cxr/",
    'logs': f"{GCS_BUCKET}/60_logs/nih-cxr/",
}

# Local working directories
LOCAL_ROOT = Path('/content')
LOCAL_DATA = LOCAL_ROOT / 'data'
LOCAL_MODELS = LOCAL_ROOT / 'models'
LOCAL_OUTPUTS = LOCAL_ROOT / 'outputs'

LOCAL_DATA.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)

# Training configuration
CONFIG = {
    # Image parameters
    'img_height': 224,
    'img_width': 224,
    'channels': 3,

    # Training parameters - Stage 1 (Feature extraction)
    'batch_size': 64,  # Increase to 128 for A100
    'epochs_stage1': 5,
    'learning_rate_stage1': 0.001,

    # Training parameters - Stage 2 (Fine-tuning)
    'epochs_stage2': 10,
    'learning_rate_stage2': 0.0001,
    'unfreeze_layers': 20,

    # Model architecture
    'dense_units': 512,
    'dropout_rate': 0.5,

    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,

    # Data
    'num_classes': 14,
    'use_sample': True,  # Set to False for full training
    'sample_size': 1000,
    'random_state': 42
}

# Models to train
MODELS_TO_TRAIN = ['resnet50', 'densenet121', 'efficientnetb3']
RUN_NAME = f"gcs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print("⚙️  Configuration:")
print(f"\n📂 GCS Paths:")
for key, path in GCS_PATHS.items():
    print(f"  {key}: {path}")

print(f"\n📁 Local Paths:")
print(f"  data: {LOCAL_DATA}")
print(f"  models: {LOCAL_MODELS}")
print(f"  outputs: {LOCAL_OUTPUTS}")

print(f"\n🎯 Training:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

print(f"\n🏗️  Models to train: {MODELS_TO_TRAIN}")
print(f"📝 Run name: {RUN_NAME}")

# %% [markdown]
# ## 6. Download Manifests from GCS
#
# Download train/val/test CSV splits from the bronze layer

# %%
def download_gcs_file(bucket_name, source_blob_name, destination_file_path):
    """Download a file from GCS."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_path)
    size_mb = Path(destination_file_path).stat().st_size / 1024 / 1024
    print(f"  ✓ Downloaded {Path(destination_file_path).name} ({size_mb:.2f} MB)")

print("📥 Downloading manifests from GCS bronze layer...")
print(f"   Source: {GCS_PATHS['manifests']}")
print(f"   Destination: {LOCAL_DATA}\n")

manifest_files = [
    'train_split.csv',
    'val_split.csv',
    'test_split.csv',
    'preprocessing_config.json'
]

for filename in manifest_files:
    source_blob = f"10_bronze/nih-cxr/manifests/{filename}"
    dest_path = LOCAL_DATA / filename

    print(f"📄 {filename}...", end=' ', flush=True)
    download_gcs_file(BUCKET_NAME, source_blob, dest_path)

print(f"\n✅ All manifests downloaded to {LOCAL_DATA}")

# %% [markdown]
# ## 7. Load Data Splits

# %%
# Load CSV splits
train_df = pd.read_csv(LOCAL_DATA / 'train_split.csv')
val_df = pd.read_csv(LOCAL_DATA / 'val_split.csv')
test_df = pd.read_csv(LOCAL_DATA / 'test_split.csv')

# Load preprocessing config
with open(LOCAL_DATA / 'preprocessing_config.json', 'r') as f:
    prep_config = json.load(f)

disease_classes = prep_config['disease_classes']

print(f"✓ Loaded data splits:")
print(f"  Train: {len(train_df):,} images")
print(f"  Val:   {len(val_df):,} images")
print(f"  Test:  {len(test_df):,} images")

print(f"\n🏥 Disease classes ({len(disease_classes)}):")
for i, disease in enumerate(disease_classes, 1):
    print(f"  {i:2d}. {disease}")

# %% [markdown]
# ## 8. Build GCS Image Paths
#
# Update CSV paths to point to GCS raw layer

# %%
import re

def build_gcs_image_path(filename, gcs_base_path):
    """
    Build GCS path for an image file.

    Args:
        filename: Image filename (e.g., '00000001_000.png')
        gcs_base_path: Base GCS path (e.g., 'gs://nih-xrays/00_raw/nih-cxr/images/')

    Returns:
        Full GCS path to the image
    """
    # Extract patient ID from filename (first 8 digits)
    # Images are organized in subdirectories: images_001/, images_002/, etc.
    # We need to determine which subdirectory based on the filename

    # For now, we'll construct paths and let TensorFlow's gfile handle GCS access
    # The images are in: gs://nih-xrays/00_raw/nih-cxr/images/images_001/images/00000001_000.png

    # Pattern: Extract images_XXX directory from existing full_path if available
    # Otherwise, we'll need to search or use a mapping

    # For this implementation, we'll assume images are organized the same way as local
    # and construct paths by searching the known subdirectories

    # Since we can't efficiently search GCS for each image, we'll use a different approach:
    # Download a manifest of image locations from GCS, or construct based on known structure

    # For now, let's construct the path assuming standard NIH structure
    # images_001 through images_012

    # We'll create paths but note that TensorFlow will handle GCS access via gfile
    return f"{gcs_base_path}{{subdir}}/images/{filename}"

# Actually, let's use TensorFlow's GCS integration
# TensorFlow can read directly from GCS using tf.io.gfile

# We need to find which subdirectory each image is in
# Let's create a mapping by listing GCS bucket contents

print("🔍 Building image path index from GCS...")
print("   This may take a few minutes...")

# List all image subdirectories
storage_client = storage.Client(project=PROJECT_ID)
image_subdirs = []
prefix = "00_raw/nih-cxr/images/"

# Get subdirectories (images_001, images_002, etc.)
blobs = storage_client.list_blobs(BUCKET_NAME, prefix=prefix, delimiter='/')
for page in blobs.pages:
    image_subdirs.extend([p for p in page.prefixes if 'images_' in p])

print(f"✓ Found {len(image_subdirs)} image subdirectories")

# For efficiency, we'll assume the CSV already has correct structure
# and just update the base path to GCS

# Alternative: Check if the CSV has full_path column with local paths
if 'full_path' in train_df.columns:
    print("\n✓ Using existing full_path from CSV")
    print("   Converting local paths to GCS paths...")

    def convert_to_gcs_path(local_path):
        """Convert local path to GCS path."""
        if pd.isna(local_path):
            return None

        # Extract: images_XXX/images/filename.png
        match = re.search(r'(images_\d+/images/[^/]+\.png)$', str(local_path))
        if match:
            rel_path = match.group(1)
            return f"{GCS_PATHS['raw_images']}{rel_path}"
        return None

    train_df['full_path'] = train_df['full_path'].apply(convert_to_gcs_path)
    val_df['full_path'] = val_df['full_path'].apply(convert_to_gcs_path)
    test_df['full_path'] = test_df['full_path'].apply(convert_to_gcs_path)

    # Verify paths
    sample_path = train_df['full_path'].iloc[0]
    print(f"\n✓ Sample GCS path: {sample_path}")

    # Check for missing paths
    missing_train = train_df['full_path'].isna().sum()
    missing_val = val_df['full_path'].isna().sum()
    missing_test = test_df['full_path'].isna().sum()

    if missing_train + missing_val + missing_test > 0:
        print(f"\n⚠️  Warning: Some paths could not be converted:")
        print(f"   Train: {missing_train} missing")
        print(f"   Val: {missing_val} missing")
        print(f"   Test: {missing_test} missing")
    else:
        print(f"\n✅ All {len(train_df) + len(val_df) + len(test_df):,} image paths converted to GCS")

else:
    print("❌ Error: full_path column not found in CSV")
    raise ValueError("CSV must contain full_path column")

# %% [markdown]
# ## 9. Sample Data (Optional)

# %%
if CONFIG['use_sample']:
    sample_size = CONFIG['sample_size']
    train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(sample_size // 5, len(val_df)), random_state=42)
    test_df = test_df.sample(n=min(sample_size // 5, len(test_df)), random_state=42)

    print(f"⚠️  Using SAMPLE mode:")
    print(f"  Train: {len(train_df):,} images")
    print(f"  Val:   {len(val_df):,} images")
    print(f"  Test:  {len(test_df):,} images")
    print(f"\n💡 Set CONFIG['use_sample'] = False for full training")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# %% [markdown]
# ## 10. Data Generators with GCS Support
#
# **TensorFlow supports reading directly from GCS using `tf.io.gfile`**
#
# ImageDataGenerator can read GCS paths if we use TensorFlow's file I/O

# %%
# Custom data generator that reads from GCS
import tensorflow as tf

def gcs_compatible_data_generator(dataframe, disease_classes, target_size, batch_size,
                                   augment=True, shuffle=True):
    """
    Create a TensorFlow dataset that reads images from GCS.

    Args:
        dataframe: DataFrame with 'full_path' (GCS paths) and disease labels
        disease_classes: List of disease column names
        target_size: Tuple (height, width)
        batch_size: Batch size
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset
    """
    # Get image paths and labels
    image_paths = dataframe['full_path'].values
    labels = dataframe[disease_classes].values.astype('float32')

    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    # Map function to load and preprocess images
    def load_and_preprocess_image(path, label):
        # Read image from GCS using tf.io.gfile
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0  # Normalize to [0, 1]

        if augment:
            # Data augmentation
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            # Random rotation (approx)
            if tf.random.uniform([]) > 0.5:
                img = tf.image.rot90(img, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))

        return img, label

    # Apply preprocessing
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

print("🔧 Creating GCS-compatible data generators...")

target_size = (CONFIG['img_height'], CONFIG['img_width'])

train_dataset = gcs_compatible_data_generator(
    train_df,
    disease_classes,
    target_size,
    CONFIG['batch_size'],
    augment=True,
    shuffle=True
)

val_dataset = gcs_compatible_data_generator(
    val_df,
    disease_classes,
    target_size,
    CONFIG['batch_size'],
    augment=False,
    shuffle=False
)

print(f"✓ Train dataset: ~{len(train_df) // CONFIG['batch_size']} batches")
print(f"✓ Val dataset: ~{len(val_df) // CONFIG['batch_size']} batches")

# Test that we can read from GCS
print(f"\n🧪 Testing GCS data pipeline...")
sample_batch = next(iter(train_dataset.take(1)))
images, labels = sample_batch
print(f"✓ Successfully loaded batch from GCS")
print(f"  Batch shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")

# %% [markdown]
# ## 11. Model Building Functions

# %%
def build_transfer_model(base_model_class, model_name, config):
    """
    Build transfer learning model with pre-trained base.
    """
    input_shape = (config['img_height'], config['img_width'], config['channels'])

    # Load pre-trained base model
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model initially
    base_model.trainable = False

    # Build complete model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config['dense_units'], activation='relu', name='dense_features')(x)
    x = layers.Dropout(config['dropout_rate'], name='dropout')(x)
    outputs = layers.Dense(config['num_classes'], activation='sigmoid', name='predictions')(x)

    model = keras.Model(inputs, outputs, name=model_name)

    return model, base_model

print("✓ Model building function defined")

# %% [markdown]
# ## 12. Training Function with GCS Model Saving

# %%
def upload_to_gcs(local_path, gcs_path):
    """Upload a file to GCS."""
    storage_client = storage.Client(project=PROJECT_ID)

    # Parse GCS path
    if gcs_path.startswith('gs://'):
        gcs_path = gcs_path[5:]  # Remove 'gs://'

    parts = gcs_path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ''

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_path)
    size_mb = Path(local_path).stat().st_size / 1024 / 1024
    print(f"    ✓ Uploaded to {gcs_path} ({size_mb:.2f} MB)")

def train_transfer_learning_model(model_class, model_name, train_dataset, val_dataset,
                                    config, models_to_train, run_name):
    """
    Train a transfer learning model with two-stage training.
    Saves models to GCS 40_models/ layer.
    """
    # Check if we should train this model
    if model_name.lower() not in [m.lower() for m in models_to_train]:
        print(f"\n⏭️  Skipping {model_name} (not in MODELS_TO_TRAIN)")
        return None, None

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    # Build model
    model, base_model = build_transfer_model(
        model_class,
        f"{model_name.lower()}_transfer",
        config
    )

    print(f"\n{model_name} architecture:")
    print(f"  Base layers: {len(base_model.layers)} (frozen)")
    print(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")

    # ===== STAGE 1: Feature Extraction =====
    print("\n" + "-" * 60)
    print("STAGE 1: Feature Extraction (base frozen)")
    print("-" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate_stage1']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
    )

    history_s1 = model.fit(
        train_dataset,
        epochs=config['epochs_stage1'],
        validation_data=val_dataset,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=config['early_stopping_patience'] // 2,
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=2
    )

    print("\n✓ Stage 1 complete")

    # ===== STAGE 2: Fine-Tuning =====
    print("\n" + "-" * 60)
    print("STAGE 2: Fine-Tuning (unfreeze top layers)")
    print("-" * 60)

    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-config['unfreeze_layers']]:
        layer.trainable = False

    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfrozen layers: {trainable_layers} / {len(base_model.layers)}")

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate_stage2']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
    )

    # Local checkpoint path
    local_model_path = LOCAL_MODELS / f"{model_name.lower()}_transfer_best.keras"

    # GCS destination path
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    gcs_model_dir = f"{GCS_PATHS['models']}{model_name.lower()}-transfer/runs/{timestamp}/"
    gcs_model_path = f"{gcs_model_dir}{model_name.lower()}_transfer_best.keras"

    history_s2 = model.fit(
        train_dataset,
        epochs=config['epochs_stage2'],
        validation_data=val_dataset,
        callbacks=[
            callbacks.ModelCheckpoint(
                str(local_model_path),
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
            )
        ],
        verbose=2
    )

    print(f"\n✓ Model trained and saved locally: {local_model_path}")

    # Upload model to GCS
    print(f"\n📤 Uploading model to GCS 40_models/ layer...")
    print(f"   Destination: {gcs_model_path}")
    upload_to_gcs(str(local_model_path), gcs_model_path)

    # Also save training config
    config_local = LOCAL_MODELS / f"{model_name.lower()}_config.json"
    with open(config_local, 'w') as f:
        json.dump(config, f, indent=2)

    config_gcs = f"{gcs_model_dir}config.json"
    upload_to_gcs(str(config_local), config_gcs)

    print(f"\n✅ {model_name} complete")
    print(f"   GCS model directory: {gcs_model_dir}")

    # Combine histories
    history_combined = {
        'stage1': history_s1.history,
        'stage2': history_s2.history
    }

    return model, history_combined

print("✓ Training function defined")

# %% [markdown]
# ## 13. Train All Models

# %%
import time

trained_models = {}
training_histories = {}

start_time = time.time()

for model_name, model_class in [
    ('ResNet50', ResNet50),
    ('DenseNet121', DenseNet121),
    ('EfficientNetB3', EfficientNetB3)
]:
    model_start = time.time()

    model, history = train_transfer_learning_model(
        model_class=model_class,
        model_name=model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=CONFIG,
        models_to_train=MODELS_TO_TRAIN,
        run_name=RUN_NAME
    )

    if model is not None:
        trained_models[model_name] = model
        training_histories[model_name] = history

        model_duration = time.time() - model_start
        print(f"\n⏱️  {model_name} training time: {model_duration / 60:.1f} minutes")

total_duration = time.time() - start_time

print(f"\n{'=' * 60}")
print("🎉 ALL TRAINING COMPLETE")
print("=" * 60)
print(f"\n✓ Trained {len(trained_models)} models:")
for name in trained_models.keys():
    print(f"  • {name}")
print(f"\n⏱️  Total time: {total_duration / 60:.1f} minutes ({total_duration / 3600:.2f} hours)")

# %% [markdown]
# ## 14. Visualize Training History

# %%
for model_name, history in training_histories.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{model_name} Training History', fontsize=16, fontweight='bold')

    s1_epochs = len(history['stage1']['loss'])
    s2_epochs = len(history['stage2']['loss'])

    epochs_s1 = list(range(1, s1_epochs + 1))
    epochs_s2 = list(range(s1_epochs + 1, s1_epochs + s2_epochs + 1))

    for idx, metric in enumerate(['loss', 'accuracy', 'auc']):
        ax = axes[idx]

        # Stage 1
        ax.plot(epochs_s1, history['stage1'][metric], 'b-', label='Train S1', linewidth=2)
        ax.plot(epochs_s1, history['stage1'][f'val_{metric}'], 'b--', label='Val S1', linewidth=2)

        # Stage 2
        ax.plot(epochs_s2, history['stage2'][metric], 'r-', label='Train S2', linewidth=2)
        ax.plot(epochs_s2, history['stage2'][f'val_{metric}'], 'r--', label='Val S2', linewidth=2)

        # Mark transition
        ax.axvline(x=s1_epochs, color='gray', linestyle=':', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save locally
    fig_path = LOCAL_OUTPUTS / f'{model_name.lower()}_training_history.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    # Upload to GCS artifacts layer
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    gcs_fig_path = f"{GCS_PATHS['artifacts']}plots/{timestamp}_{model_name.lower()}_training_history.png"
    upload_to_gcs(str(fig_path), gcs_fig_path)

    print(f"✓ {model_name} plot saved to GCS: {gcs_fig_path}")

    plt.show()

# %% [markdown]
# ## 15. Evaluate on Test Set

# %%
test_dataset = gcs_compatible_data_generator(
    test_df,
    disease_classes,
    (CONFIG['img_height'], CONFIG['img_width']),
    CONFIG['batch_size'],
    augment=False,
    shuffle=False
)

results = {}

print("📊 Evaluating models on test set...\n")

for model_name, model in trained_models.items():
    print(f"Evaluating {model_name}...", end=' ', flush=True)

    test_loss, test_acc, test_auc = model.evaluate(test_dataset, verbose=0)

    results[model_name] = {
        'loss': test_loss,
        'accuracy': test_acc,
        'auc': test_auc
    }

    print(f"✓ Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

# Save results to GCS
results_df = pd.DataFrame(results).T
results_csv = LOCAL_OUTPUTS / 'model_results.csv'
results_df.to_csv(results_csv)

timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
gcs_results_path = f"{GCS_PATHS['artifacts']}metrics/{timestamp}_model_results.csv"
upload_to_gcs(str(results_csv), gcs_results_path)

print(f"\n✓ Results saved to GCS: {gcs_results_path}")

print(f"\n{'=' * 60}")
print("✅ EVALUATION COMPLETE")
print("=" * 60)

# %% [markdown]
# ## 16. Summary

# %%
print("\n" + "=" * 60)
print("📋 TRAINING SUMMARY - GCS Medallion Architecture")
print("=" * 60)

print(f"\n🎯 Configuration:")
print(f"  Dataset: NIH Chest X-Ray (14 disease classes)")
print(f"  Data source: {GCS_PATHS['raw_images']}")
print(f"  Train images: {len(train_df):,}")
print(f"  Val images: {len(val_df):,}")
print(f"  Test images: {len(test_df):,}")

print(f"\n🏆 Model Performance:")
results_df_sorted = results_df.sort_values('auc', ascending=False)
print(results_df_sorted.to_string())

print(f"\n📦 GCS Artifacts:")
print(f"  Models: {GCS_PATHS['models']}")
print(f"  Metrics: {GCS_PATHS['artifacts']}metrics/")
print(f"  Plots: {GCS_PATHS['artifacts']}plots/")

print(f"\n⏱️  Training Time: {total_duration / 60:.1f} minutes ({total_duration / 3600:.2f} hours)")

print(f"\n🎯 Next Steps:")
print(f"  1. View artifacts in GCS console:")
print(f"     https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/40_models")
print(f"  2. Download models for local evaluation")
print(f"  3. Move to Vertex AI for production training")
print(f"  4. Consider PyTorch migration for more flexibility")

print("=" * 60)
print("✅ Notebook 07 Complete - GCS Integration")
print("=" * 60)
