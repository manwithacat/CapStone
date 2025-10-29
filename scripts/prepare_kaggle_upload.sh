#!/bin/bash
# Prepare data for Kaggle dataset upload

set -e

echo "📦 Preparing data for Kaggle upload..."

# Create staging directory
STAGING_DIR="kaggle_upload"
rm -rf $STAGING_DIR
mkdir -p $STAGING_DIR

# Copy essential files
echo "Copying CSV splits..."
cp data/processed/train_split.csv $STAGING_DIR/
cp data/processed/val_split.csv $STAGING_DIR/
cp data/processed/test_split.csv $STAGING_DIR/

echo "Copying metadata..."
cp data/processed/preprocessing_config.json $STAGING_DIR/
cp data/processed/class_weights.json $STAGING_DIR/

# Create dataset metadata
cat > $STAGING_DIR/dataset-metadata.json << 'METADATA'
{
  "title": "NIH Chest X-Ray Splits and Metadata",
  "id": "YOUR_USERNAME/nih-chest-xray-splits",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["health", "medical", "x-ray", "deep learning"],
  "description": "Train/val/test splits and preprocessing metadata for NIH Chest X-Ray dataset. Use with the original NIH Chest X-Ray Images dataset."
}
METADATA

# Create README
cat > $STAGING_DIR/README.md << 'README'
# NIH Chest X-Ray Splits and Metadata

This dataset contains train/validation/test splits and preprocessing metadata for the NIH Chest X-Ray dataset.

## Files

- `train_split.csv`: Training set (78,831 images)
- `val_split.csv`: Validation set (16,383 images)
- `test_split.csv`: Test set (16,890 images)
- `preprocessing_config.json`: Disease classes, normalization params
- `class_weights.json`: Class weights for handling imbalance

## Usage with Kaggle

Use alongside the original NIH Chest X-Ray Images dataset:

```python
import pandas as pd
from pathlib import Path

# Load splits
train_df = pd.read_csv('/kaggle/input/nih-chest-xray-splits/train_split.csv')

# Update paths to point to Kaggle images
train_df['full_path'] = train_df['Image Index'].apply(
    lambda x: f'/kaggle/input/nih-chest-xrays/images/{x}'
)
```

## Split Strategy

- Stratified by key diseases (Cardiomegaly, Effusion, etc.)
- Patient-level split (no patient appears in multiple splits)
- Maintains disease distribution across splits
README

echo ""
echo "✅ Prepared data in $STAGING_DIR/"
echo ""
echo "📊 File sizes:"
du -sh $STAGING_DIR/*
echo ""
echo "Total size:"
du -sh $STAGING_DIR
echo ""
echo "Next steps:"
echo "1. Edit $STAGING_DIR/dataset-metadata.json with your Kaggle username"
echo "2. Upload to Kaggle:"
echo "   cd $STAGING_DIR"
echo "   kaggle datasets create -p . -r zip"
echo "3. Or upload via Kaggle web UI: https://www.kaggle.com/datasets"
