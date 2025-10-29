# Changes Needed for Kaggle Notebooks

## Cell 2: Path Setup (CRITICAL)

**Replace this:**
```python
# Define paths
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
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved_models'
```

**With this:**
```python
# Kaggle paths
DATA_DIR = Path('/kaggle/input/nih-chest-xrays')
PROCESSED_DIR = Path('/kaggle/input/nih-chest-xray-splits')  # Your uploaded splits
OUTPUTS_DIR = Path('/kaggle/working/outputs')
MODELS_DIR = Path('/kaggle/working/models')
FIGURES_DIR = OUTPUTS_DIR / 'figures'

# Create output directories
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
```

## Cell 7: Add Path Conversion Function

**Add this NEW cell after loading splits:**
```python
# Convert local paths to Kaggle paths
def update_kaggle_image_paths(df):
    """
    Update full_path column to point to Kaggle's image directory.
    
    Local path:  /Users/james/...CapStone/data/raw/images_001/00000001_000.png
    Kaggle path: /kaggle/input/nih-chest-xrays/images/00000001_000.png
    """
    def get_kaggle_path(local_path):
        # Extract just the filename
        filename = Path(local_path).name
        return f'/kaggle/input/nih-chest-xrays/images/{filename}'
    
    df['full_path'] = df['full_path'].apply(get_kaggle_path)
    return df

# Apply to all splits
train_df = update_kaggle_image_paths(train_df)
val_df = update_kaggle_image_paths(val_df)
test_df = update_kaggle_image_paths(test_df)

print("✓ Updated image paths for Kaggle environment")
print(f"Example path: {train_df['full_path'].iloc[0]}")
```

## Cell 6 or Config: Set RETRAIN_MODEL = True

**Change:**
```python
RETRAIN_MODEL = False  # ← Local default
```

**To:**
```python
RETRAIN_MODEL = True  # ← Always train on Kaggle (no cached model)
```

## Optional: Reduce Sample Size for Testing

**In CONFIG cell:**
```python
CONFIG = {
    # ...
    'use_sample': True,
    'sample_size': 1000,  # ← Start small to test (takes ~1-2 min)
    # ...
}
```

After testing, change to `5000` or set `use_sample: False` for full dataset.

## After Training: Download Results

**Add final cell:**
```python
# Package results for download
import shutil
print("📦 Packaging results...")

# Create download package
download_dir = Path('/kaggle/working/download')
download_dir.mkdir(exist_ok=True)

# Copy trained model
shutil.copy(MODELS_DIR / 'cnn_custom_best.h5', download_dir / 'cnn_custom_best.h5')

# Copy results
shutil.copytree(OUTPUTS_DIR / 'reports', download_dir / 'reports', dirs_exist_ok=True)
shutil.copytree(OUTPUTS_DIR / 'figures', download_dir / 'figures', dirs_exist_ok=True)

print("✅ Results ready for download in /kaggle/working/download/")
print("   Click 'Output' tab → Download files")
```

## Summary of Changes

| Component | Local | Kaggle |
|-----------|-------|--------|
| **Image dir** | `data/raw/images_*/` | `/kaggle/input/nih-chest-xrays/images/` |
| **Splits CSV** | `data/processed/*.csv` | `/kaggle/input/nih-chest-xray-splits/*.csv` |
| **Output dir** | `outputs/` | `/kaggle/working/outputs/` |
| **Model dir** | `models/saved_models/` | `/kaggle/working/models/` |
| **RETRAIN** | `False` (use cached) | `True` (always train) |

## Testing Workflow

1. **Test run (1K samples)**: ~1-2 minutes → verify paths work
2. **Medium run (5K samples)**: ~3-5 minutes → verify training works
3. **Full run (78K samples)**: ~30-45 minutes → production training
