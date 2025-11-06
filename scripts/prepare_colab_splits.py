#!/usr/bin/env python3
"""
Prepare Colab-specific data splits.

Creates lightweight CSV files for Google Colab that only contain:
- Image Index (filename)
- Disease label columns (14 classes)
- Metadata columns (age, gender, etc.)

No absolute paths - Colab notebook will construct paths based on downloaded images.
"""

from pathlib import Path
import pandas as pd
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
KAGGLE_SPLITS_DIR = PROJECT_ROOT / 'kaggle' / 'datasets' / 'data-splits'
COLAB_SPLITS_DIR = PROJECT_ROOT / 'colab' / 'data_splits'

# Disease classes
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

def create_colab_splits():
    """Create Colab-friendly data splits."""

    COLAB_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Columns to keep for Colab
    columns_to_keep = [
        'Image Index',
        'Patient ID',
        'Patient Age',
        'Patient Gender',
        'View Position',
    ] + DISEASE_CLASSES + ['No_Finding']

    print("📦 Creating Colab-specific data splits...\n")

    for split_name in ['train', 'val', 'test']:
        print(f"Processing {split_name}_split.csv... ", end='', flush=True)

        # Load original split
        original_file = KAGGLE_SPLITS_DIR / f'{split_name}_split.csv'
        df = pd.read_csv(original_file)

        # Select only necessary columns
        df_colab = df[columns_to_keep].copy()

        # Save to Colab directory
        output_file = COLAB_SPLITS_DIR / f'{split_name}_split.csv'
        df_colab.to_csv(output_file, index=False)

        # Calculate size reduction
        original_size = original_file.stat().st_size / 1024 / 1024
        colab_size = output_file.stat().st_size / 1024 / 1024
        reduction = ((original_size - colab_size) / original_size) * 100

        print(f"✓ {len(df_colab):,} rows | {colab_size:.2f} MB (was {original_size:.2f} MB, {reduction:.1f}% smaller)")

    # Copy preprocessing config (but update to remove path dependencies)
    print(f"\nCopying preprocessing_config.json... ", end='', flush=True)
    with open(KAGGLE_SPLITS_DIR / 'preprocessing_config.json') as f:
        config = json.load(f)

    # Create Colab-specific config
    colab_config = {
        'disease_classes': config['disease_classes'],
        'num_classes': config['num_classes'],
        'date_created': config.get('date_created', 'Unknown'),
        'note': 'Colab-specific version - use with kagglehub to download NIH dataset'
    }

    with open(COLAB_SPLITS_DIR / 'preprocessing_config.json', 'w') as f:
        json.dump(colab_config, f, indent=2)

    print("✓")

    # Create README
    print(f"Creating README.md... ", end='', flush=True)
    readme_content = """# Colab Data Splits

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
"""

    with open(COLAB_SPLITS_DIR / 'README.md', 'w') as f:
        f.write(readme_content)

    print("✓")

    print("\n" + "="*60)
    print("✅ Colab splits created successfully!")
    print("="*60)
    print(f"\n📂 Output directory: {COLAB_SPLITS_DIR}")
    print(f"\n📋 Files created:")
    for file in sorted(COLAB_SPLITS_DIR.iterdir()):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:<30} ({size_kb:>8.1f} KB)")

    print(f"\n💡 Next steps:")
    print(f"  1. Upload these files to a Google Drive folder")
    print(f"  2. Get the folder ID from the Drive URL")
    print(f"  3. Update DATA_FOLDER_ID in the Colab notebook")
    print(f"  4. Run the notebook!")

if __name__ == '__main__':
    create_colab_splits()
