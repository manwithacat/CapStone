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
# ## 2. Mount Google Drive (Optional - for persistent credentials)
#
# **Recommended**: Store service account key in Google Drive to avoid re-uploading each session

# %%
from google.colab import drive
from pathlib import Path

# Mount Google Drive
print("📂 Mounting Google Drive...")
drive.mount('/content/drive', force_remount=False)
print("✓ Drive mounted at /content/drive")

# Check if service account key exists in Drive
DRIVE_KEY_PATH = Path('/content/drive/MyDrive/colab_credentials/nih-xrays-service-account.json')

if DRIVE_KEY_PATH.exists():
    print(f"✓ Found service account key in Drive: {DRIVE_KEY_PATH}")
else:
    print(f"\n⚠️  Service account key not found at: {DRIVE_KEY_PATH}")
    print("   To save for next time:")
    print("   1. Upload your JSON key using the file picker below")
    print("   2. I'll copy it to your Drive automatically")

# %% [markdown]
# ## 3. Setup GCS Authentication (Service Account)
#
# **Option A**: Use key from Google Drive (recommended)
# **Option B**: Upload key file (first time only)

# %%
from google.colab import files
from google.cloud import storage
import os
import json
import shutil

BUCKET_NAME = 'nih-xrays'
PROJECT_ID = None  # Auto-detected from credentials
key_file = None

print("🔐 GCS Service Account Authentication")
print("=" * 60)
print()

# Try to load from Drive first
if DRIVE_KEY_PATH.exists():
    print(f"📦 Loading service account key from Google Drive...")
    key_file = str(DRIVE_KEY_PATH)
    print(f"✓ Using: {DRIVE_KEY_PATH.name}")
else:
    # Upload and save to Drive
    print("📁 Upload your service account JSON key file")
    print("   (Download from Google Cloud Console > IAM > Service Accounts)")
    print()

    uploaded = files.upload()

    if not uploaded:
        raise ValueError("No file uploaded. Please upload service account JSON key.")

    # Get the uploaded file name
    key_file = list(uploaded.keys())[0]

    # Save to Drive for next time
    DRIVE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(key_file, DRIVE_KEY_PATH)
    print(f"\n💾 Saved to Drive: {DRIVE_KEY_PATH}")
    print(f"   Next session will load automatically!")

# Set environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file

# Load project ID from key file
with open(key_file, 'r') as f:
    key_data = json.load(f)
    PROJECT_ID = key_data.get('project_id')

print(f"\n✓ Service account authenticated")
print(f"✓ Project ID: {PROJECT_ID}")

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
# ## 4. Install Dependencies

# %%
!pip install -q tensorflow matplotlib seaborn scikit-learn google-cloud-storage

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPUs")

# %% [markdown]
# ## 5. Enable Mixed Precision (A100 Optimization)

# %%
from tensorflow.keras import mixed_precision

# Enable mixed precision for A100 tensor cores
# This uses float16 for computation but float32 for variables
# Result: ~2-3x faster training + lower memory usage
mixed_precision.set_global_policy('mixed_float16')

print("⚡ Mixed Precision Training Enabled")
print(f"   Policy: {mixed_precision.global_policy()}")
print(f"   Compute dtype: {mixed_precision.global_policy().compute_dtype}")
print(f"   Variable dtype: {mixed_precision.global_policy().variable_dtype}")
print()
print("   Benefits:")
print("   • 2-3x faster training on A100")
print("   • Lower GPU memory usage")
print("   • Same model accuracy")

# %% [markdown]
# ## 6. Import Libraries

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
    'batch_size': 1024,  # Optimized for A100 80GB (was 64)
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
    'use_sample': False,  # Full training on A100!
    'random_state': 42
}

# GPU-specific adjustments
print("\n🔍 Detecting GPU...")
import subprocess
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], encoding='utf-8')
    print(f"GPU: {gpu_info.strip()}")

    # Auto-adjust batch size based on GPU
    if 'A100' in gpu_info:
        CONFIG['batch_size'] = 1024
        print("✓ A100 detected: Using batch_size=1024")
    elif 'V100' in gpu_info:
        CONFIG['batch_size'] = 512
        print("✓ V100 detected: Using batch_size=512")
    elif 'T4' in gpu_info:
        CONFIG['batch_size'] = 256
        print("✓ T4 detected: Using batch_size=256")
    else:
        CONFIG['batch_size'] = 128
        print(f"⚠️  Unknown GPU: Using batch_size=128")
except:
    print("⚠️  No GPU detected or nvidia-smi unavailable")
    CONFIG['batch_size'] = 64

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
# ## 8. Build GCS Image Path Index
#
# The CSVs contain only `Image Index` (filename), not full paths.
# We need to build a mapping: filename → GCS path

# %%
print("🔍 Loading image path index...")
print()

from google.cloud import storage
import time
import json

# Create storage client
storage_client = storage.Client(project=PROJECT_ID)

# Build mapping: filename -> GCS path
filename_to_gcs_path = {}

# Cache location in medallion structure
INDEX_CACHE_PATH = "10_bronze/nih-cxr/image_path_index.json"
INDEX_CACHE_GCS = f"gs://{BUCKET_NAME}/{INDEX_CACHE_PATH}"

start_time = time.time()

# Try to load cached index
try:
    print(f"📦 Checking for cached index at {INDEX_CACHE_GCS}...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(INDEX_CACHE_PATH)

    if blob.exists():
        # Download and parse JSON
        index_json = blob.download_as_text()
        filename_to_gcs_path = json.loads(index_json)
        total_images = len(filename_to_gcs_path)
        elapsed = time.time() - start_time

        print(f"✅ Loaded cached index: {total_images:,} images in {elapsed:.1f} seconds")
        print(f"   (Much faster than rebuilding!)")
        print()
    else:
        raise FileNotFoundError("Cache not found, will build from scratch")

except Exception as e:
    # Cache not found or error - build from scratch
    print(f"⚠️  Cached index not available: {e}")
    print(f"   Building index from scratch (this may take 2-3 minutes)...")
    print()

    # List all images in the bucket
    prefix = "00_raw/nih-cxr/images/"
    print(f"📂 Scanning bucket: gs://{BUCKET_NAME}/{prefix}")

    build_start_time = time.time()
    total_images = 0

    # Get all subdirectories (images_001, images_002, etc.)
    iterator = storage_client.list_blobs(BUCKET_NAME, prefix=prefix, delimiter='/')
    # Force iteration to populate prefixes
    _ = list(iterator)
    # Now get prefixes
    subdirs = [p for p in iterator.prefixes if 'images_' in p]

    print(f"✓ Found {len(subdirs)} image subdirectories")
    print()

    # For each subdirectory, list images
    for subdir_idx, subdir in enumerate(sorted(subdirs), 1):
        # Each subdir is like: 00_raw/nih-cxr/images/images_001/
        # Images are in: 00_raw/nih-cxr/images/images_001/images/
        images_prefix = f"{subdir}images/"

        print(f"[{subdir_idx}/{len(subdirs)}] Scanning {subdir}...", end=' ', flush=True)

        # List all .png files in this subdirectory (no delimiter to see actual files)
        image_blobs = storage_client.list_blobs(
            BUCKET_NAME,
            prefix=images_prefix
        )

        count = 0
        for blob in image_blobs:
            if blob.name.endswith('.png'):
                # Extract filename (e.g., "00000001_000.png")
                filename = blob.name.split('/')[-1]

                # Store GCS path
                gcs_path = f"gs://{BUCKET_NAME}/{blob.name}"
                filename_to_gcs_path[filename] = gcs_path
                count += 1

        total_images += count
        print(f"✓ {count:,} images")

    elapsed = time.time() - build_start_time
    print()
    print(f"✅ Index built: {total_images:,} images in {elapsed:.1f} seconds")

    # Save to GCS for next time
    print()
    print(f"💾 Saving index cache to {INDEX_CACHE_GCS}...")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(INDEX_CACHE_PATH)

        # Upload JSON
        index_json = json.dumps(filename_to_gcs_path, indent=2)
        blob.upload_from_string(index_json, content_type='application/json')

        print(f"✅ Index cached for next run!")
        print(f"   Next time this will load in ~1 second instead of {elapsed:.0f} seconds")
    except Exception as save_error:
        print(f"⚠️  Failed to cache index: {save_error}")
        print(f"   (Not critical - will rebuild next time)")

    print()

elapsed = time.time() - start_time
print(f"⏱️  Total time: {elapsed:.1f} seconds")
print()

# Verify we found all expected images
expected_images = len(train_df) + len(val_df) + len(test_df)
print(f"📊 Images in CSVs: {expected_images:,}")
print(f"📊 Images in index: {total_images:,}")

if total_images < expected_images:
    print(f"⚠️  Warning: Index has fewer images than CSVs!")
    print(f"   Missing: {expected_images - total_images:,} images")
else:
    print(f"✓ Index covers all CSV images!")

# %% [markdown]
# ## 9. Build full_path Column

# %%
print("🔧 Building full_path column for all splits...")
print()

def build_gcs_path_from_index(filename):
    """Look up GCS path from filename using the index."""
    if pd.isna(filename):
        return None

    gcs_path = filename_to_gcs_path.get(filename)

    if gcs_path is None:
        # Try with .png extension if not included
        if not filename.endswith('.png'):
            gcs_path = filename_to_gcs_path.get(f"{filename}.png")

    return gcs_path

# Build paths for all splits
print("  Building train paths...", end=' ', flush=True)
train_df['full_path'] = train_df['Image Index'].apply(build_gcs_path_from_index)
print("✓")

print("  Building val paths...", end=' ', flush=True)
val_df['full_path'] = val_df['Image Index'].apply(build_gcs_path_from_index)
print("✓")

print("  Building test paths...", end=' ', flush=True)
test_df['full_path'] = test_df['Image Index'].apply(build_gcs_path_from_index)
print("✓")

print()

# Verify paths
print("🔍 Verifying image paths...")
all_dfs = [('train', train_df), ('val', val_df), ('test', test_df)]
total_missing = 0

for split_name, df in all_dfs:
    missing_in_split = df['full_path'].isna().sum()
    total_missing += missing_in_split

    if missing_in_split > 0:
        print(f"  ❌ {split_name}: {missing_in_split:,} missing paths")
        # Show a few examples
        missing_files = df[df['full_path'].isna()]['Image Index'].head(3).tolist()
        for fname in missing_files:
            print(f"     - {fname}")
    else:
        print(f"  ✓ {split_name}: All {len(df):,} paths found")

if total_missing > 0:
    print(f"\n⚠️  Total {total_missing:,} images not found in GCS!")
    print("   This might indicate:")
    print("   - CSV files from different dataset version")
    print("   - Incomplete GCS upload")
    print("   - Incorrect bucket structure")
    raise ValueError(f"{total_missing} image files not found in gs://{BUCKET_NAME}")
else:
    print(f"\n✅ All {len(train_df) + len(val_df) + len(test_df):,} image paths verified!")

    # Show sample path
    sample_path = train_df['full_path'].iloc[0]
    print(f"\n📝 Sample GCS path:")
    print(f"   {sample_path}")

# %% [markdown]
# ## 10. Sample Data (Optional)

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
# ## 11. Data Generators with GCS Support
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
# ## 12. Cache Pretrained Weights

# %%
print("🔍 Setting up pretrained model weights cache...")
print()

# Keras stores downloaded weights in ~/.keras/models/
KERAS_CACHE_DIR = Path.home() / '.keras' / 'models'
KERAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# GCS cache location for pretrained weights
WEIGHTS_CACHE_GCS = f"{GCS_BUCKET}/70_cfg/pretrained_weights/"

# Weight files we expect to download
EXPECTED_WEIGHTS = {
    'resnet50': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'densenet121': 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'efficientnetb3': 'efficientnetb3_notop.h5'
}

# Try to download cached weights from GCS
print(f"📦 Checking for cached weights in {WEIGHTS_CACHE_GCS}...")
bucket = storage_client.bucket(BUCKET_NAME)
weights_downloaded = 0

for model_name, weight_file in EXPECTED_WEIGHTS.items():
    local_path = KERAS_CACHE_DIR / weight_file
    gcs_path = f"70_cfg/pretrained_weights/{weight_file}"
    blob = bucket.blob(gcs_path)

    # Only download if we don't already have it locally
    if not local_path.exists():
        if blob.exists():
            print(f"  ⬇️  Downloading {weight_file}...", end=' ', flush=True)
            try:
                blob.download_to_filename(str(local_path))
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
                weights_downloaded += 1
            except Exception as e:
                print(f"⚠️  Failed: {e}")
        else:
            print(f"  ⏭️  {weight_file} not in cache (will download from ImageNet)")
    else:
        print(f"  ✓ {weight_file} already cached locally")

if weights_downloaded > 0:
    print(f"\n✅ Downloaded {weights_downloaded} weight file(s) from GCS cache")
else:
    print(f"\n✓ All weights available (will be cached to GCS after first download)")

print()

# %% [markdown]
# ## 13. Model Building Functions

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
# ## 13. Training Function with GCS Model Saving

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
# ## 14. Train All Models (Parallel Execution)
#
# **Parallel Training on A100:**
# - Trains all 3 models simultaneously
# - Uses ~50-60 GB GPU RAM (still leaves 20-30 GB free)
# - Expected time: ~30-40 minutes (vs 90-120 minutes sequential)
# - 2.5-3x faster than sequential training

# %%
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Training mode selection
PARALLEL_TRAINING = True  # Set to False for sequential training

trained_models = {}
training_histories = {}
model_timings = {}

print("\n" + "=" * 60)
if PARALLEL_TRAINING:
    print("🚀 PARALLEL TRAINING MODE")
    print("=" * 60)
    print(f"\nTraining all {len(MODELS_TO_TRAIN)} models simultaneously")
    print(f"Expected GPU RAM: ~50-60 GB / 80 GB")
    print(f"Expected time: ~30-40 minutes")
else:
    print("📝 SEQUENTIAL TRAINING MODE")
    print("=" * 60)
    print(f"\nTraining {len(MODELS_TO_TRAIN)} models one by one")
    print(f"Expected GPU RAM: ~17 GB / 80 GB per model")
    print(f"Expected time: ~90-120 minutes")

print()

# Model configurations
MODEL_CONFIGS = [
    ('ResNet50', ResNet50),
    ('DenseNet121', DenseNet121),
    ('EfficientNetB3', EfficientNetB3)
]

def train_model_wrapper(model_config):
    """Wrapper function for parallel training"""
    model_name, model_class = model_config
    thread_name = threading.current_thread().name

    print(f"\n[{thread_name}] 🚀 Starting {model_name}...")
    model_start = time.time()

    try:
        model, history = train_transfer_learning_model(
            model_class=model_class,
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=CONFIG,
            models_to_train=MODELS_TO_TRAIN,
            run_name=RUN_NAME
        )

        model_duration = time.time() - model_start

        if model is not None:
            print(f"\n[{thread_name}] ✅ {model_name} complete in {model_duration / 60:.1f} minutes")
            return model_name, model, history, model_duration
        else:
            print(f"\n[{thread_name}] ⏭️  {model_name} skipped")
            return model_name, None, None, 0

    except Exception as e:
        print(f"\n[{thread_name}] ❌ {model_name} failed: {e}")
        return model_name, None, None, 0

start_time = time.time()

if PARALLEL_TRAINING:
    # Parallel training using ThreadPoolExecutor
    print("⏳ Launching parallel training threads...\n")

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="Model") as executor:
        # Submit all training jobs
        futures = {
            executor.submit(train_model_wrapper, config): config[0]
            for config in MODEL_CONFIGS
        }

        # Collect results as they complete
        for future in as_completed(futures):
            model_name, model, history, duration = future.result()

            if model is not None:
                trained_models[model_name] = model
                training_histories[model_name] = history
                model_timings[model_name] = duration

else:
    # Sequential training (fallback)
    print("⏳ Starting sequential training...\n")

    for model_config in MODEL_CONFIGS:
        model_name, model, history, duration = train_model_wrapper(model_config)

        if model is not None:
            trained_models[model_name] = model
            training_histories[model_name] = history
            model_timings[model_name] = duration

total_duration = time.time() - start_time

print(f"\n{'=' * 60}")
print("🎉 ALL TRAINING COMPLETE")
print("=" * 60)
print(f"\n✓ Trained {len(trained_models)} models:")

# Show individual model timings
if model_timings:
    print(f"\n⏱️  Individual model times:")
    for name, duration in sorted(model_timings.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {name}: {duration / 60:.1f} minutes")

    if PARALLEL_TRAINING and len(model_timings) > 1:
        sequential_time = sum(model_timings.values()) / 60
        parallel_time = total_duration / 60
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        print(f"\n📊 Parallel efficiency:")
        print(f"  Sequential time (estimated): {sequential_time:.1f} minutes")
        print(f"  Parallel time (actual): {parallel_time:.1f} minutes")
        print(f"  Speedup: {speedup:.1f}x faster!")
else:
    for name in trained_models.keys():
        print(f"  • {name}")

print(f"\n⏱️  Total wallclock time: {total_duration / 60:.1f} minutes ({total_duration / 3600:.2f} hours)")

# %%
# Cache pretrained weights to GCS for next time
print(f"\n💾 Caching pretrained weights to GCS...")
print()

weights_uploaded = 0
for model_name, weight_file in EXPECTED_WEIGHTS.items():
    local_path = KERAS_CACHE_DIR / weight_file
    gcs_path = f"70_cfg/pretrained_weights/{weight_file}"
    blob = bucket.blob(gcs_path)

    # Upload if file exists locally but not in GCS
    if local_path.exists() and not blob.exists():
        print(f"  ⬆️  Uploading {weight_file}...", end=' ', flush=True)
        try:
            blob.upload_from_filename(str(local_path))
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.1f} MB)")
            weights_uploaded += 1
        except Exception as e:
            print(f"⚠️  Failed: {e}")
    elif blob.exists():
        print(f"  ✓ {weight_file} already in GCS cache")
    else:
        print(f"  ⏭️  {weight_file} not found locally")

if weights_uploaded > 0:
    print(f"\n✅ Uploaded {weights_uploaded} weight file(s) to GCS cache")
    print(f"   Next session will load these from GCS instead of downloading from ImageNet!")
else:
    print(f"\n✓ All weights already cached in GCS")

print()

# %% [markdown]
# ## 15. Visualize Training History

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
# ## 16. Evaluate on Test Set

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
# ## 17. Summary

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
