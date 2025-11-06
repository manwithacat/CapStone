# Colab Data Splits

These CSV files are optimized for Google Colab usage.

## Differences from Kaggle splits

**Kaggle splits** (`kaggle/datasets/data-splits/`):
- Include `full_path` column with absolute paths
- Larger file sizes (~23 MB for train)
- Designed for Kaggle kernels where dataset is pre-mounted

**Colab splits** (this directory):
- Only include `Image Index` (filename)
- Smaller file sizes (~5-8 MB for train)
- Notebook constructs paths based on downloaded dataset location

## Usage in Colab

```python
# Download NIH dataset with kagglehub
dataset_path = kagglehub.dataset_download("nih-chest-xrays/data")

# Load splits
train_df = pd.read_csv('train_split.csv')

# Construct full paths
def build_image_path(filename, base_dir):
    # Images are in subdirectories like images_001/images/
    for subdir in Path(base_dir).glob('images_*'):
        img_path = subdir / 'images' / filename
        if img_path.exists():
            return str(img_path)
    return None

train_df['full_path'] = train_df['Image Index'].apply(
    lambda x: build_image_path(x, dataset_path)
)
```

## Files

- `train_split.csv` - Training set
- `val_split.csv` - Validation set
- `test_split.csv` - Test set
- `preprocessing_config.json` - Disease classes and metadata
- `README.md` - This file

## Uploading to Google Drive

Upload these 4 files to a Google Drive folder, then use PyDrive2 in Colab to download them.

See `colab/COLAB_PYDRIVE2_SETUP.md` for instructions.
