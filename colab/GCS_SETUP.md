# Google Cloud Storage Setup for Colab

## Benefits

✅ **One-time upload** - 47 GB uploaded once, use forever
✅ **Fast access** - GCS → Colab is blazing fast (same network)
✅ **No file manipulation** - Data stays in original structure
✅ **Version control** - Easy to update data without re-uploading everything
✅ **Cost-effective** - ~$1/month for 50 GB storage

## Setup Steps

### 1. Enable Cloud Storage API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (**mlq425** from your client_secrets.json)
3. **APIs & Services → Library**
4. Search: "Cloud Storage API"
5. Click **Enable**

### 2. Grant Yourself Permissions

1. Go to [Storage Browser](https://console.cloud.google.com/storage/browser)
2. Click on `nih-xrays` bucket
3. **Permissions** tab → **Grant Access**
4. Add your Google account email
5. Role: **Storage Object Admin**
6. **Save**

### 3. Install gcloud CLI (Local Machine)

```bash
# macOS
brew install --cask google-cloud-sdk

# Or download from:
# https://cloud.google.com/sdk/docs/install
```

### 4. Authenticate

```bash
gcloud auth login
gcloud config set project mlq425
```

### 5. Upload Data

```bash
# Make script executable
chmod +x scripts/upload_to_gcs.sh

# Upload (this will take 30-60 minutes for images)
./scripts/upload_to_gcs.sh
```

This uploads:
- `gs://nih-xrays/data_splits/` - Your Colab CSV files (6.6 MB)
- `gs://nih-xrays/images/` - NIH X-ray images (47 GB)

### 6. Verify Upload

```bash
# Check bucket contents
gsutil ls -lh gs://nih-xrays/

# Check total size
gsutil du -sh gs://nih-xrays/*
```

## Colab Notebook Changes

The updated notebook will:

1. **Authenticate with GCS** (using your client_secrets.json)
2. **Download data splits** (6.6 MB, <5 seconds)
3. **Stream images from GCS** (no download needed!)
4. **Upload results back to GCS**

### Option A: Stream from GCS (Recommended)

```python
# Read images directly from GCS
# Keras ImageDataGenerator supports GCS paths
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='gcs_path',  # gs://nih-xrays/images/images_001/...
    target_size=(224, 224),
    ...
)
```

**Benefits:**
- ✅ No download time
- ✅ No local storage needed
- ✅ GCS → Colab is fast (same network)

### Option B: Download to Persistent Disk

```python
# Download once to persistent disk
gsutil -m rsync -r gs://nih-xrays/images/ /content/data/images/
```

**Benefits:**
- ✅ Slightly faster I/O during training
- ✅ Works offline after download

## Cost Estimate

**Google Cloud Storage Pricing:**
- Storage: $0.020 per GB/month
- 50 GB = **~$1/month**
- Download (egress): First 1 GB/month free, then $0.12/GB
  - Colab → GCS is same region = **FREE** or very cheap

**Total cost: ~$1-2/month**

Compare to:
- Colab Pro persistent disk: $10/month (includes compute)
- Re-downloading from Kaggle: Free but 30 min every time

## Security

**Bucket Permissions:**
- Set to **private** (not public)
- Only your Google account has access
- Colab authenticates with your OAuth credentials

**Best Practices:**
1. Never make bucket public
2. Use IAM roles (not legacy ACLs)
3. Enable versioning for data protection
4. Set lifecycle rules to delete old versions

## Bucket Structure

```
gs://nih-xrays/
├── data_splits/          # Colab CSV files (6.6 MB)
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── preprocessing_config.json
├── images/               # NIH Chest X-Rays (47 GB)
│   ├── images_001/
│   │   └── images/
│   │       ├── 00000001_000.png
│   │       └── ...
│   ├── images_002/
│   └── ...
└── results/              # Training outputs
    ├── models/
    └── reports/
```

## Troubleshooting

### "Access Denied" Error

```bash
# Re-authenticate
gcloud auth login

# Verify project
gcloud config get-value project

# Check bucket permissions
gsutil iam get gs://nih-xrays
```

### Slow Upload

```bash
# Use parallel uploads
gsutil -m -o "GSUtil:parallel_process_count=8" \
    rsync -r /path/to/data/ gs://nih-xrays/images/
```

### Colab Can't Access GCS

In Colab:
```python
# Check authentication
from google.colab import auth
auth.authenticate_user()

# Test access
!gsutil ls gs://nih-xrays/
```

## MLflow Integration

Download training results to local MLflow:

```python
# After training in Colab
# Upload results to GCS
!gsutil -m cp -r /content/models/*.keras gs://nih-xrays/results/models/
!gsutil -m cp -r /content/outputs/*.csv gs://nih-xrays/results/reports/

# Then locally:
# Download to MLflow
gsutil -m cp -r gs://nih-xrays/results/ ./outputs/
python scripts/csv_to_mlflow.py outputs/reports/training_history.csv
```

## Next Steps

1. ✅ Enable Cloud Storage API
2. ✅ Grant yourself permissions
3. ✅ Install gcloud CLI
4. ✅ Authenticate
5. ✅ Upload data (`./scripts/upload_to_gcs.sh`)
6. 🔄 Update Colab notebook (I'll do this next)
7. 🚀 Run training in Colab
8. 📊 Download results to MLflow

Ready to proceed with step 1-5? Then I'll update the notebook.
