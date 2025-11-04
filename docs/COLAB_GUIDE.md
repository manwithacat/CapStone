# Google Colab Platform Guide

**Directory**: `colab/`

## Overview

Google Colab is used for interactive development and testing with powerful GPUs (T4, A100).

## Directory Structure

```
colab/
├── 07_transfer_learning_colab.ipynb  # Clean, minimal notebook
├── COLAB_REFERENCE.md                # Quick reference
├── COLAB_SETUP_GUIDE.md              # Setup instructions
└── scripts/
    └── colab_workflow.sh             # Opens stable notebook
```

## Your Stable Notebook

**URL**: https://colab.research.google.com/drive/1FJdto9vlXuvtofDpjIN9Vb5DzyPAlLEH

**Bookmark this!** It auto-saves to your Google Drive.

## Workflow

### 1. Open Notebook

```bash
# From local
./colab/scripts/colab_workflow.sh

# Or just open the URL in browser
```

### 2. Select Runtime

- **CPU + High RAM**: Free, for testing pipeline
- **T4 GPU**: Uses compute units, moderate speed
- **A100 GPU**: Uses more compute units, 6-8x faster than Kaggle P100

### 3. Run Notebook

- Edit directly in browser
- Run cells interactively
- Auto-saves to Google Drive

### 4. (Optional) Sync Back

```bash
# Pull latest version from Colab
colab-cli pull-nb colab/07_transfer_learning_colab.ipynb
```

## Key Differences from Local/Kaggle

### vs Local
- **No MLflow**: Colab notebooks don't track experiments
- **No Papermill**: Just run cells manually
- **Ephemeral**: Files in `/content/` deleted when runtime stops
- **Google Drive**: Mount for persistent storage

### vs Kaggle
- **Interactive**: Can run cells individually, see outputs immediately
- **Longer runtimes**: 12-24 hours vs 9 hours
- **Better GPUs**: A100 available vs T4 only
- **Internet access**: Can download ImageNet weights directly

## Setup Data in Colab

### Option A: Test with Dummy Data (CPU Testing)

Add this cell after Drive mount:

```python
# Create minimal test dataset
import numpy as np
from PIL import Image
import json

DATA_DIR = Path('/content/test_data')
IMAGES_DIR = DATA_DIR / 'images'
SPLITS_DIR = DATA_DIR / 'splits'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

disease_classes = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Create 1000 dummy images
image_paths = []
for i in range(1000):
    img = Image.fromarray(np.random.randint(50, 200, (256, 256), dtype=np.uint8), mode='L')
    img_path = IMAGES_DIR / f'dummy_xray_{i:04d}.png'
    img.save(img_path)
    image_paths.append(str(img_path))

# Create splits
def create_split_df(paths, start_idx, size):
    df_data = {
        'Image Index': [f'dummy_{i:04d}.png' for i in range(start_idx, start_idx + size)],
        'full_path': paths[start_idx:start_idx + size],
        'Patient ID': [f'P{i // 10}' for i in range(size)],
    }
    for disease in disease_classes:
        df_data[disease] = np.random.binomial(1, np.random.uniform(0.1, 0.3), size)
    return pd.DataFrame(df_data)

train_df = create_split_df(image_paths, 0, 700)
val_df = create_split_df(image_paths, 700, 200)
test_df = create_split_df(image_paths, 900, 100)

train_df.to_csv(SPLITS_DIR / 'train_split.csv', index=False)
val_df.to_csv(SPLITS_DIR / 'val_split.csv', index=False)
test_df.to_csv(SPLITS_DIR / 'test_split.csv', index=False)

with open(SPLITS_DIR / 'preprocessing_config.json', 'w') as f:
    json.dump({'disease_classes': disease_classes}, f)

PROCESSED_DIR = SPLITS_DIR
print(f"✓ Test dataset ready: {PROCESSED_DIR}")
```

Then update Configuration cell:
```python
PROCESSED_DIR = Path('/content/test_data/splits')
```

### Option B: Upload to Google Drive (GPU Training)

1. Upload splits to Google Drive: `/MyDrive/capstone_data/nih-chest-xray-splits/`
2. (Optional) Upload images for full training
3. Notebook will mount and access automatically

## Managing Compute Resources

### Check Usage

```python
# In Colab notebook
!nvidia-smi
```

### View Active Sessions

1. Runtime → Manage sessions
2. See all notebooks with GPU/RAM usage
3. Terminate old sessions

### Auto-Shutdown

- Colab auto-disconnects after **90 minutes idle**
- Manually disconnect: Runtime → Disconnect and delete runtime

## Colab Pro Benefits

- 100 compute units per month
- A100 GPU access (40GB VRAM)
- 12-24 hour session limits
- Priority access to GPUs

## Tips

1. **Test on CPU first**: Use dummy data to verify pipeline works before using GPU
2. **Edit in place**: Don't push new notebooks, edit the stable one
3. **Manage sessions**: Terminate old runtimes to free resources
4. **Monitor compute units**: Check usage in Colab UI
5. **Save checkpoints**: Models in `/content/` are ephemeral

## Comparison: Kaggle vs Colab

| Feature          | Kaggle           | Colab (Free)    | Colab Pro      |
|------------------|------------------|-----------------|----------------|
| GPU              | T4 (16GB)        | T4 (16GB)       | A100 (40GB)    |
| Time Limit       | 9 hours          | 12 hours        | 24 hours       |
| Compute Units    | Free unlimited   | Free limited    | 100/month      |
| Internet         | ❌ No            | ✅ Yes          | ✅ Yes         |
| Interactive      | ❌ Batch only    | ✅ Yes          | ✅ Yes         |
| Data Storage     | Datasets only    | Google Drive    | Google Drive   |
| Best For         | Long batch runs  | Quick tests     | Heavy training |

## Common Issues

### Issue: FileNotFoundError for data
**Solution**: Create test data (Option A) or upload to Google Drive (Option B)

### Issue: Runtime disconnects
**Solution**: Colab auto-disconnects after 90 min idle. Reconnect and continue.

### Issue: Out of compute units
**Solution**: Wait for monthly reset or switch to CPU/Kaggle

### Issue: Notebook changes not saving
**Solution**: Check you're editing the stable URL, not a new copy

## Current Status

- **Notebook**: Clean version (no Kaggle/MLflow code)
- **Runtime**: Set up manually (CPU or GPU)
- **Data**: Use test data generator for CPU testing
- **Purpose**: Interactive development and A100 training
