# Google Cloud Storage Setup Guide for Colab

This guide explains how to use notebook `07_transfer_learning_gcs.ipynb` with the GCS medallion architecture.

## Overview

The notebook uses **Google Cloud Storage (GCS)** as the single source of truth:

```
gs://nih-xrays/
├── 00_raw/           # Immutable NIH X-ray images (~47 GB)
├── 10_bronze/        # Train/val/test CSV splits
├── 40_models/        # Trained model checkpoints
├── 50_artifacts/     # Metrics, plots, reports
└── 60_logs/          # TensorBoard logs
```

## Prerequisites

### 1. GCS Bucket Setup

Follow the medallion architecture setup:

```bash
# From project root
chmod +x scripts/setup_gcs_medallion.sh
./scripts/setup_gcs_medallion.sh
```

This creates the medallion structure and uploads your data.

### 2. Service Account Authentication

**Setup (one-time):**

1. Create service account in Google Cloud Console:
   ```
   IAM & Admin > Service Accounts > Create Service Account
   ```

2. Grant permissions:
   - `Storage Object Admin` (for full read/write to GCS)
   - OR `Storage Object Viewer` (for read-only access)

3. Create JSON key:
   ```
   Actions > Manage Keys > Add Key > Create New Key > JSON
   ```

4. Save locally as: `.colab/service-account-key.json` (gitignored)

**In Colab:**
- Upload this JSON file when prompted (each session)
- Project ID will be auto-detected from the key

## Running the Notebook

### Step 1: Open in Google Colab

Upload `colab/07_transfer_learning_gcs.ipynb` to Colab:

**Quick link:**
```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab/07_transfer_learning_gcs.ipynb
```

Or manually:
1. Go to [Google Colab](https://colab.research.google.com)
2. File > Upload notebook
3. Select `07_transfer_learning_gcs.ipynb`

### Step 2: Select Runtime

**Recommended:**
- Runtime type: Python 3
- Hardware accelerator: **GPU (T4 or better)**
- For Colab Pro: Use A100 for fastest training

### Step 3: Run Authentication Cell

When you run cell 2, you'll see:

```
🔐 GCS Service Account Authentication
============================================================

📁 Upload your service account JSON key file
   (Download from Google Cloud Console > IAM > Service Accounts)
```

Click "Choose Files" and upload your service account JSON key.

The notebook will:
- Auto-detect your Project ID from the key
- Test connection to `gs://nih-xrays`
- Show confirmation with project ID

### Step 4: Configure Training

Edit the `CONFIG` cell:

```python
CONFIG = {
    'batch_size': 64,  # Increase to 128 for A100
    'epochs_stage1': 5,
    'epochs_stage2': 10,
    'use_sample': True,  # Set False for full training
    'sample_size': 1000,
    ...
}

# Choose models to train
MODELS_TO_TRAIN = ['resnet50', 'densenet121', 'efficientnetb3']
```

**Sample mode (testing):**
- `use_sample: True`, `sample_size: 1000`
- ~15-20 minutes per model
- Good for testing the pipeline

**Full training:**
- `use_sample: False`
- ~2-3 hours per model (with T4 GPU)
- ~45-60 minutes per model (with A100 GPU)

### Step 5: Run Training Cells

Runtime > Run all (or run cells sequentially)

The notebook will:
1. Authenticate with GCS
2. Download CSV manifests (7 MB)
3. Load or build image path index
   - **First run**: Builds index from 112K images (2-3 min), caches to GCS
   - **Subsequent runs**: Loads cached index (~1 second)
4. Load or download pretrained model weights
   - **First run**: Downloads from ImageNet (~178 MB), caches to GCS
   - **Subsequent runs**: Loads from GCS cache (~10 seconds)
5. Create data generators
6. Train selected models
7. Upload results back to GCS

## What the Notebook Does

### 1. Data Flow

```
GCS Bronze Layer              Local Colab VM              GCS Models Layer
─────────────────             ──────────────              ────────────────
10_bronze/manifests/    →     /content/data/
  train_split.csv
  val_split.csv
  test_split.csv

GCS Raw Layer
─────────────
00_raw/images/          →     TensorFlow reads       →   40_models/
  images_001/...              directly from GCS!         resnet50-transfer/
  images_002/...                                           runs/2025-11-05/
  ...                                                        model.keras
```

### 2. Training Stages

**Stage 1: Feature Extraction**
- Base model frozen
- Train only top layers
- 5 epochs (configurable)
- Learning rate: 0.001

**Stage 2: Fine-Tuning**
- Unfreeze top 20 layers
- Lower learning rate: 0.0001
- 10 epochs (configurable)

### 3. Output Artifacts

All saved to GCS medallion layers:

```
40_models/nih-cxr/
  resnet50-transfer/
    runs/2025-11-05_143022/
      resnet50_transfer_best.keras
      config.json
  densenet121-transfer/
    runs/2025-11-05_151534/
      densenet121_transfer_best.keras
      config.json

50_artifacts/nih-cxr/
  metrics/
    2025-11-05_160112_model_results.csv
  plots/
    2025-11-05_160112_resnet50_training_history.png
    2025-11-05_160112_densenet121_training_history.png
```

## Performance Optimizations

### Caching Strategy

The notebook includes intelligent caching to minimize repeated downloads:

**1. Image Path Index Cache**
- **Location**: `gs://nih-xrays/10_bronze/nih-cxr/image_path_index.json`
- **First run**: Scans bucket for 112K images (2-3 minutes), saves JSON index
- **Subsequent runs**: Loads JSON index (~1 second)
- **Speedup**: ~100x faster startup

**2. Pretrained Weights Cache**
- **Location**: `gs://nih-xrays/70_cfg/pretrained_weights/`
- **Cached files**:
  - `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` (98 MB)
  - `densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5` (33 MB)
  - `efficientnetb3_notop.h5` (47 MB)
- **First run**: Downloads from ImageNet (~3-5 min), uploads to GCS cache
- **Subsequent runs**: Downloads from GCS (~10 seconds)
- **Speedup**: ~20x faster model initialization

### Cache Management

**To rebuild the image path index** (if images added/removed):
```bash
# From local machine
gsutil rm gs://nih-xrays/10_bronze/nih-cxr/image_path_index.json
```

**To clear pretrained weights cache** (to force re-download):
```bash
# From local machine
gsutil -m rm gs://nih-xrays/70_cfg/pretrained_weights/*.h5
```

**Storage impact:**
- Image path index: ~3 MB (JSON)
- Pretrained weights: ~178 MB (3 models)
- Total cache overhead: ~181 MB (~$0.004/month)

### First vs. Subsequent Runs

**First Run** (cold start):
- Download manifests: 10 seconds
- Build image index: 2-3 minutes
- Download weights: 3-5 minutes
- **Total overhead**: ~6-8 minutes

**Subsequent Runs** (warm start):
- Download manifests: 10 seconds
- Load cached index: 1 second
- Load cached weights: 10 seconds
- **Total overhead**: ~20 seconds

**Time saved**: ~5-7 minutes per session after first run!

## Troubleshooting

### "Permission Denied" Error

**Problem:** Cannot access `gs://nih-xrays`

**Solution:**
1. Verify bucket exists: https://console.cloud.google.com/storage/browser/nih-xrays
2. Check IAM permissions:
   - Service account needs `Storage Object Viewer` role
   - User account needs same role
3. Verify project ID matches

```bash
# Check via gcloud CLI
gsutil ls gs://nih-xrays/
```

### "File Not Found" in GCS

**Problem:** Image paths incorrect

**Solution:**
1. Verify medallion structure exists:
   ```bash
   gsutil ls -r gs://nih-xrays/00_raw/
   gsutil ls -r gs://nih-xrays/10_bronze/
   ```

2. Re-run setup script:
   ```bash
   ./scripts/setup_gcs_medallion.sh
   ```

### Slow Training

**Problem:** Training is very slow

**Solution:**
1. Verify GPU is enabled: Runtime > Change runtime type > GPU
2. Check GPU type: `!nvidia-smi` (T4 is good, A100 is best)
3. Increase batch size for A100: `CONFIG['batch_size'] = 128`
4. Use sample mode for testing: `CONFIG['use_sample'] = True`

### Out of Memory

**Problem:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**
1. Reduce batch size: `CONFIG['batch_size'] = 32`
2. Use smaller model first: `MODELS_TO_TRAIN = ['resnet50']`
3. Enable mixed precision:
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

## Cost Optimization

### GCS Storage Costs
- Raw images (47 GB): ~$0.94/month
- Models (10 GB): ~$0.20/month
- Artifacts (1 GB): ~$0.02/month
- **Total: ~$1.20/month**

### Colab Costs
- **Free tier:**
  - T4 GPU: Limited hours per week
  - Session timeout: 12 hours

- **Colab Pro ($10/month):**
  - More GPU hours
  - Better GPUs (A100 sometimes)
  - Longer sessions (24 hours)
  - **Persistent disk**: 200 GB (data survives sessions!)

### Tips to Minimize Costs
1. Use sample mode for testing (`use_sample: True`)
2. Train one model at a time
3. Download artifacts from GCS, then delete old runs
4. Set lifecycle rules for temp data (already configured for `90_tmp/`)

## Next Steps After Training

### 1. Download Models for Local Use

```bash
# From local machine
gsutil -m rsync -r gs://nih-xrays/40_models/ ./models/saved_models_gcs/
```

### 2. Integrate with MLflow

```python
# Load model into MLflow
import mlflow.tensorflow

model = tf.keras.models.load_model('models/saved_models_gcs/resnet50_transfer_best.keras')

with mlflow.start_run():
    mlflow.tensorflow.log_model(model, "model")
```

### 3. Move to Vertex AI

The GCS medallion structure is **Vertex AI-ready**:

```python
from google.cloud import aiplatform

aiplatform.init(
    project='your-project-id',
    location='us-central1',
    staging_bucket='gs://nih-xrays/90_tmp'
)

# Custom training job using same GCS paths
job = aiplatform.CustomTrainingJob(
    display_name='nih-xray-transfer-learning',
    script_path='vertex_training_script.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-gpu.2-12:latest',
)
```

### 4. PyTorch Migration

Consider PyTorch for more flexibility:
- Better multi-GPU support
- More research community support
- Easier model customization

Create `07_transfer_learning_pytorch.py` using:
- `torch.utils.data.Dataset` for GCS data loading
- `torchvision.models` for pre-trained models
- Same GCS medallion structure

## Support

**Issues with:**
- **GCS setup**: See `docs/GCS_MEDALLION_ARCHITECTURE.md`
- **Medallion architecture**: See `scripts/setup_gcs_medallion.sh`
- **Authentication**: See Google Cloud [IAM docs](https://cloud.google.com/iam/docs)
- **Colab**: See [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## References

- [GCS Medallion Architecture](../docs/GCS_MEDALLION_ARCHITECTURE.md)
- [TensorFlow GCS Integration](https://www.tensorflow.org/io/tutorials/gcs)
- [Google Cloud Storage Python Client](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python)
- [Vertex AI Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
