# Colab Setup Without Google Drive

## Problem

Google Colab's `drive.mount()` requires OAuth authentication every session, which is annoying for repeated runs.

## Solution: Use Kaggle Datasets API Instead

Instead of mounting Google Drive, download data directly from Kaggle using the Kaggle API.

### Setup Steps

#### 1. One-Time Kaggle API Setup

In Colab, run this cell ONCE at the start:

```python
import os
from google.colab import files

# Upload kaggle.json (one-time per session)
print("Upload your kaggle.json file:")
uploaded = files.upload()

# Create .kaggle directory
os.makedirs('/root/.kaggle', exist_ok=True)

# Move kaggle.json to correct location
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(uploaded['kaggle.json'].decode('utf-8'))

# Set permissions
os.chmod('/root/.kaggle/kaggle.json', 0o600)

print("✓ Kaggle API configured")
```

#### 2. Download Data from Kaggle

Replace the Drive mount section with:

```python
# Install Kaggle API if needed
!pip install -q kaggle

# Download your data splits dataset
!kaggle datasets download -d YOUR_USERNAME/nih-chest-xray-splits -p /content/data --unzip

# Or download the full NIH dataset
# !kaggle datasets download -d nih-chest-xrays/data -p /content/data --unzip

# Verify download
from pathlib import Path
data_dir = Path('/content/data')
print(f"Data directory: {data_dir}")
print(f"Files: {list(data_dir.glob('*'))}")
```

#### 3. Update Paths

Replace:

```python
# OLD: Uses Google Drive
DRIVE_DATA = Path('/content/drive/MyDrive/capstone_data')
PROCESSED_DIR = DRIVE_DATA / 'nih-chest-xray-splits'
```

With:

```python
# NEW: Uses downloaded Kaggle data
PROCESSED_DIR = Path('/content/data/nih-chest-xray-splits')
IMAGE_DIR = Path('/content/data/images')  # If using full dataset
```

### Complete Modified Setup Section

```python
# ============================================================================
# SETUP (No Google Drive Required)
# ============================================================================

# 1. Install and configure Kaggle API (one-time per session)
from google.colab import files
import os

if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("⚠️  Upload your kaggle.json file:")
    uploaded = files.upload()
    os.makedirs('/root/.kaggle', exist_ok=True)
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        f.write(uploaded['kaggle.json'].decode('utf-8'))
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    print("✓ Kaggle API configured")
else:
    print("✓ Kaggle API already configured")

# 2. Download data from Kaggle
!pip install -q kaggle
!kaggle datasets download -d YOUR_USERNAME/nih-chest-xray-splits -p /content/data --unzip

# 3. Set up paths
from pathlib import Path

PROJECT_ROOT = Path('/content')
PROCESSED_DIR = Path('/content/data')
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Data directory: {PROCESSED_DIR}")
print(f"✓ Files available: {list(PROCESSED_DIR.glob('*.csv'))}")
```

## Alternative: Persistent Storage (Still Faster than Drive)

If you want to avoid re-downloading every session, use Colab's persistent storage:

### Option 1: Use /content/sample_data (Persists Between Runs)

Colab's `/content/sample_data` persists for ~12 hours:

```python
DATA_CACHE = Path('/content/sample_data/nih_splits')

if not DATA_CACHE.exists():
    print("Downloading data (one-time)...")
    !kaggle datasets download -d YOUR_USERNAME/nih-chest-xray-splits \
        -p /content/sample_data/nih_splits --unzip
else:
    print("✓ Using cached data")

PROCESSED_DIR = DATA_CACHE
```

### Option 2: Use Google Cloud Storage (Best for Production)

For frequent use:

```python
# Download to GCS bucket
!gsutil cp -r gs://your-bucket/nih-splits /content/data

# Or use GCS directly
PROCESSED_DIR = 'gs://your-bucket/nih-splits'
```

## Benefits

✅ **No OAuth prompts** - One kaggle.json upload per session
✅ **Faster** - Local /content is faster than Drive
✅ **Reproducible** - Same setup for all users
✅ **Versioned** - Use specific dataset versions
✅ **Shareable** - Anyone with Kaggle API can run

## Trade-offs

⚠️ **Session storage** - Data deleted when runtime disconnects (but fast to re-download)
⚠️ **Initial upload** - Must upload kaggle.json once per session

## Recommendation

**For development**: Use Kaggle datasets download to `/content/data`
**For production**: Use Google Cloud Storage bucket
**Avoid**: Google Drive mount (slow, requires OAuth)

## Updated Colab Cell Order

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Configure Kaggle API (if not already done)
from google.colab import files
import os

if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("Upload kaggle.json:")
    uploaded = files.upload()
    os.makedirs('/root/.kaggle', exist_ok=True)
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        f.write(uploaded['kaggle.json'].decode('utf-8'))
    os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Cell 3: Download data
!pip install -q kaggle
!kaggle datasets download -d YOUR_USERNAME/nih-chest-xray-splits -p /content/data --unzip

# Cell 4: Imports and setup
import tensorflow as tf
from pathlib import Path

PROCESSED_DIR = Path('/content/data')
# ... rest of setup

# Cell 5+: Continue with training
```

## Testing

To verify the setup works:

```python
# Should NOT require Google Drive authentication
assert Path('/content/data/train_split.csv').exists()
assert Path('/content/data/val_split.csv').exists()
assert Path('/content/data/test_split.csv').exists()
print("✓ Data files verified")
```
