# Kaggle GPU Training Guide

## Setup (One-time)

1. **Create Kaggle Account & API Token**
   - Go to https://www.kaggle.com/settings
   - Create New API Token → downloads `kaggle.json`
   - Already done: token at `.kaggle/kaggle.json`

2. **Create Kaggle Dataset with Splits**
   ```bash
   # Package your metadata
   cd data/processed
   kaggle datasets init -p .  # Creates dataset-metadata.json
   # Edit dataset-metadata.json:
   # - title: "NIH-Chest-XRay-Splits-and-Metadata"
   # - id: "yourusername/nih-chest-xray-splits"
   
   # Upload
   kaggle datasets create -p . -r zip
   ```

3. **Create Kaggle Notebook**
   - New Notebook → Settings:
     - Accelerator: **GPU P100**
     - Internet: **On**
     - Add Dataset: "nih-chest-xrays" (original images)
     - Add Dataset: "yourusername/nih-chest-xray-splits" (your splits)

## Notebook Adaptations

### Path Changes:
```python
# Local paths
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Kaggle paths
DATA_DIR = Path('/kaggle/input/nih-chest-xrays')  # Original images
PROCESSED_DIR = Path('/kaggle/input/nih-chest-xray-splits')  # Your splits
OUTPUTS_DIR = Path('/kaggle/working')  # Writeable area
```

### Key Differences:
1. **Read-only input**: `/kaggle/input/` is read-only
2. **Writeable output**: `/kaggle/working/` for models/results
3. **Download results**: After training, download `.h5` model files

## Training Workflow

```python
# 1. Load your splits (from your uploaded dataset)
train_df = pd.read_csv('/kaggle/input/nih-chest-xray-splits/train_split.csv')

# 2. Update image paths to point to Kaggle's images
def update_kaggle_paths(df):
    """Update full_path to use Kaggle's image directory."""
    df['full_path'] = df['Image Index'].apply(
        lambda x: f'/kaggle/input/nih-chest-xrays/images/{x}'
    )
    return df

train_df = update_kaggle_paths(train_df)
val_df = update_kaggle_paths(val_df)
test_df = update_kaggle_paths(test_df)

# 3. Train as normal (GPU automatically used)
history = model.fit(...)

# 4. Save model to /kaggle/working
model.save('/kaggle/working/cnn_custom_best.h5')

# 5. Download from Kaggle UI or CLI:
# kaggle kernels output yourusername/notebook-name -p ./models/saved_models/
```

## Tips

- **Save checkpoints**: Kaggle sessions can timeout (9 hours max)
- **Use versioning**: Commit notebook after each major change
- **Monitor GPU usage**: Check GPU memory in notebook output
- **Download models**: Don't rely on Kaggle storage long-term

## Estimated Training Times (P100 GPU)

| Dataset | Notebook 06 (Custom CNN) | Notebook 07 (Transfer Learning) |
|---------|--------------------------|----------------------------------|
| Sample (5K images) | ~3-5 min | ~2-3 min |
| Full (78K images) | ~30-45 min | ~45-60 min |

Compare to **M2 Pro Metal**: 30-40 min (sample), 6-8 hours (full)
**Speedup: 10-12x faster on Kaggle P100!**
