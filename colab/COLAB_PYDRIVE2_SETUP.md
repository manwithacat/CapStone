# Using PyDrive2 in Colab (Best Solution)

## Why PyDrive2 > drive.mount()

✅ **Persistent authentication** - Credentials cached between sessions
✅ **Selective access** - Only download files you need
✅ **Faster** - Direct API access, no FUSE overhead
✅ **More control** - Programmatic file management
✅ **Better error handling** - Clear API responses

## Setup Instructions

### 1. Install PyDrive2

```python
!pip install -q pydrive2
```

### 2. Create OAuth Credentials (One-Time Setup)

#### Option A: Use Colab's Built-in Auth (Easiest)

```python
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate with Google
auth.authenticate_user()

# Create credentials
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()

# Create Drive client
drive = GoogleDrive(gauth)

print("✓ Authenticated with PyDrive2")
```

This uses Colab's built-in authentication - **no OAuth prompts!**

#### Option B: Use Custom OAuth App (More Control)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use existing)
3. Enable Google Drive API
4. Create OAuth 2.0 Client ID (Desktop app)
5. Download `client_secrets.json`

Upload to Colab:

```python
from google.colab import files
import json

# Upload client_secrets.json
uploaded = files.upload()

# Save to file
with open('client_secrets.json', 'wb') as f:
    f.write(uploaded['client_secrets.json'])

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates credentials.json on first run

drive = GoogleDrive(gauth)
```

### 3. Complete Colab Setup Cell

```python
# ============================================================================
# SETUP: PyDrive2 Authentication (No drive.mount() needed!)
# ============================================================================

!pip install -q pydrive2

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pathlib import Path
import os

# Method 1: Colab's built-in auth (recommended)
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

print("✓ PyDrive2 authenticated")
print("✓ No drive.mount() needed!")
```

### 4. Download Files from Drive

```python
# ============================================================================
# DOWNLOAD DATA FROM GOOGLE DRIVE
# ============================================================================

def download_file_from_drive(drive, file_id, destination):
    """Download a file from Google Drive by ID."""
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(destination)
    print(f"✓ Downloaded: {destination}")

def download_folder_from_drive(drive, folder_id, destination_dir):
    """Download all files from a Google Drive folder."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        file_path = destination_dir / file['title']
        print(f"Downloading {file['title']}... ", end='')
        file.GetContentFile(str(file_path))
        print(f"✓ ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return len(file_list)

# Download your data splits
DATA_FOLDER_ID = 'YOUR_DRIVE_FOLDER_ID_HERE'  # Get from Drive URL
destination = Path('/content/data')

num_files = download_folder_from_drive(drive, DATA_FOLDER_ID, destination)
print(f"\n✓ Downloaded {num_files} files to {destination}")

# Verify files
csv_files = list(destination.glob('*.csv'))
print(f"✓ Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
```

### 5. How to Get Drive Folder/File IDs

**From Drive URL:**
```
https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9
                                         ^^^^^^^^^^^^^^^^
                                         This is the folder ID
```

**From PyDrive:**
```python
# List all folders in your Drive
file_list = drive.ListFile({
    'q': "mimeType='application/vnd.google-apps.folder' and trashed=false"
}).GetList()

for folder in file_list:
    print(f"{folder['title']}: {folder['id']}")
```

## Complete Colab Notebook Template

```python
# ========================================================================
# Cell 1: Check GPU
# ========================================================================
!nvidia-smi

# ========================================================================
# Cell 2: Install and Authenticate PyDrive2
# ========================================================================
!pip install -q pydrive2

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

print("✓ PyDrive2 ready")

# ========================================================================
# Cell 3: Download Data
# ========================================================================
from pathlib import Path

def download_folder(drive, folder_id, dest):
    """Download all files from a Drive folder."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for file in files:
        print(f"📥 {file['title']}", end=' ')
        file.GetContentFile(str(dest / file['title']))
        print("✓")

    return len(files)

# YOUR DRIVE FOLDER ID (from URL)
DATA_FOLDER_ID = '1a2b3c4d5e6f7g8h9'  # Replace with your folder ID

# Download data
num_files = download_folder(drive, DATA_FOLDER_ID, '/content/data')
print(f"\n✓ Downloaded {num_files} files")

# Verify
data_dir = Path('/content/data')
print(f"Files: {list(data_dir.glob('*'))}")

# ========================================================================
# Cell 4: Setup Paths and Config
# ========================================================================
import tensorflow as tf
import pandas as pd
import json

PROCESSED_DIR = Path('/content/data')
MODELS_DIR = Path('/content/models')
OUTPUTS_DIR = Path('/content/outputs')

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Load data
train_df = pd.read_csv(PROCESSED_DIR / 'train_split.csv')
val_df = pd.read_csv(PROCESSED_DIR / 'val_split.csv')
test_df = pd.read_csv(PROCESSED_DIR / 'test_split.csv')

print(f"✓ Loaded {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")

# ========================================================================
# Cell 5+: Your training code
# ========================================================================
# Continue with model training...
```

## Advanced Features

### Upload Results Back to Drive

```python
def upload_file_to_drive(drive, local_path, drive_folder_id=None):
    """Upload a file to Google Drive."""
    local_path = Path(local_path)

    file_metadata = {
        'title': local_path.name,
    }

    if drive_folder_id:
        file_metadata['parents'] = [{'id': drive_folder_id}]

    file = drive.CreateFile(file_metadata)
    file.SetContentFile(str(local_path))
    file.Upload()

    print(f"✓ Uploaded: {local_path.name}")
    return file['id']

# Upload trained model
model_path = MODELS_DIR / 'best_model.keras'
file_id = upload_file_to_drive(drive, model_path, DATA_FOLDER_ID)
print(f"Model URL: https://drive.google.com/file/d/{file_id}")
```

### Check if Files Exist Before Downloading

```python
def file_exists_in_drive(drive, folder_id, filename):
    """Check if a file exists in a Drive folder."""
    files = drive.ListFile({
        'q': f"'{folder_id}' in parents and title='{filename}' and trashed=false"
    }).GetList()
    return len(files) > 0

# Only download if needed
if not Path('/content/data/train_split.csv').exists():
    print("Downloading data...")
    download_folder(drive, DATA_FOLDER_ID, '/content/data')
else:
    print("✓ Data already downloaded")
```

### Cache Credentials (Persist Between Sessions)

```python
# Save credentials to Drive for reuse
import pickle

# After first authentication
with open('/content/credentials.pkl', 'wb') as f:
    pickle.dump(gauth.credentials, f)

# Upload to Drive for next session
upload_file_to_drive(drive, '/content/credentials.pkl', CREDENTIALS_FOLDER_ID)

# Next session: Load credentials
creds_file = download_file_from_drive(drive, CREDS_FILE_ID, '/content/credentials.pkl')
with open('/content/credentials.pkl', 'rb') as f:
    gauth.credentials = pickle.load(f)
```

## Comparison: PyDrive2 vs drive.mount() vs Kaggle API

| Feature | PyDrive2 | drive.mount() | Kaggle API |
|---------|----------|---------------|------------|
| **Setup time** | 🟢 Fast | 🟡 Medium | 🟢 Fast |
| **OAuth prompts** | 🟢 None (cached) | 🔴 Every session | 🟢 None |
| **Speed** | 🟢 Fast (API) | 🔴 Slow (FUSE) | 🟢 Fast |
| **Selective access** | 🟢 Yes | 🔴 No (mounts all) | 🟢 Yes |
| **File upload** | 🟢 Easy | 🟢 Easy | 🔴 Manual |
| **Private data** | 🟢 Yes | 🟢 Yes | 🔴 Public only |
| **Storage** | 🟢 Your Drive | 🟢 Your Drive | 🟡 Kaggle datasets |

## Troubleshooting

### Error: "Invalid Credentials"

```python
# Clear credentials and re-authenticate
!rm -rf /content/credentials.pkl
!rm -rf ~/.credentials/

# Re-run authentication cell
from google.colab import auth
auth.authenticate_user()
```

### Error: "File not found"

Check folder ID is correct:

```python
# List all your Drive folders
file_list = drive.ListFile({
    'q': "mimeType='application/vnd.google-apps.folder'"
}).GetList()

for folder in file_list:
    print(f"{folder['title']}: {folder['id']}")
```

### Slow Downloads

Use parallel downloads:

```python
from concurrent.futures import ThreadPoolExecutor

def download_parallel(drive, folder_id, dest, max_workers=4):
    """Download files in parallel."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    def download_one(file):
        file_path = dest / file['title']
        file.GetContentFile(str(file_path))
        return file['title']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_one, files))

    return len(results)
```

## Best Practices

1. **Store folder IDs in variables** at the top of notebook
2. **Check file existence** before downloading
3. **Upload results** back to Drive for persistence
4. **Use meaningful folder names** in Drive
5. **Version your data** (e.g., `nih-splits-v1/`, `nih-splits-v2/`)
6. **Cache credentials** for faster re-authentication

## Recommendation

**For Colab notebooks**: Use PyDrive2 with Colab's built-in auth
**For data uploads**: First create Kaggle dataset, then use PyDrive2 for Colab access
**For large files (>100MB)**: Consider Kaggle datasets + PyDrive2 hybrid

PyDrive2 is the **best solution** for regular Colab usage with private data.
