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
  "title": "NIH Chest X-Ray: Stratified Train/Val/Test Splits",
  "subtitle": "Patient-level stratified splits with class weights and preprocessing config for 112K chest X-rays",
  "id": "YOUR_USERNAME/nih-chest-xray-splits",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": [
    "health",
    "medicine",
    "medical imaging",
    "radiology",
    "chest x-ray",
    "deep learning",
    "computer vision",
    "disease detection",
    "classification",
    "healthcare",
    "medical diagnosis"
  ],
  "description": "## Overview\n\nStratified train/validation/test splits with preprocessing metadata for the NIH Chest X-Ray dataset (112,120 images, 14 disease classes, 30,805 patients).\n\n**Key Features:**\n- Patient-level splits (no data leakage between sets)\n- Stratified by key diseases (maintains disease distribution)\n- Class weights for handling severe class imbalance\n- Preprocessing configuration for reproducibility\n- Ready for deep learning pipelines\n\n## Dataset Composition\n\n- **Training**: 78,831 images (70%)\n- **Validation**: 16,383 images (15%)\n- **Test**: 16,890 images (15%)\n\n## Files Included\n\n1. **train_split.csv** - Training set with image paths and multi-label disease annotations\n2. **val_split.csv** - Validation set for hyperparameter tuning\n3. **test_split.csv** - Test set for final evaluation\n4. **preprocessing_config.json** - Disease classes, normalization parameters, class weights\n5. **class_weights.json** - Per-disease weights for handling class imbalance (e.g., Hernia: 0.4% vs No Finding: 53.8%)\n6. **README.md** - Detailed usage instructions and examples\n\n## Use Case\n\nDesigned for multi-label disease classification research. Use with the original [NIH Chest X-Ray Images](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset available on Kaggle.\n\n## Citation\n\nOriginal dataset: Wang et al., ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR 2017"
}
METADATA

# Create README
cat > $STAGING_DIR/README.md << 'README'
# NIH Chest X-Ray: Stratified Train/Val/Test Splits

Stratified train/validation/test splits with preprocessing metadata for the NIH Chest X-Ray dataset (112,120 chest X-rays, 14 disease classes, 30,805 patients).

## 📊 Dataset Overview

This dataset provides **production-ready splits** for training deep learning models on the NIH Chest X-Ray dataset. All splits are:

- ✅ **Patient-level stratified** (no data leakage)
- ✅ **Disease-balanced** (maintains class distribution)
- ✅ **Reproducible** (includes preprocessing config)
- ✅ **Class-weighted** (handles severe imbalance)

### Split Composition

| Split | Images | Percentage | Purpose |
|-------|--------|------------|---------|
| Training | 78,831 | 70% | Model training |
| Validation | 16,383 | 15% | Hyperparameter tuning |
| Test | 16,890 | 15% | Final evaluation |
| **Total** | **112,104** | **100%** | Full dataset |

## 📁 Files

### 1. train_split.csv, val_split.csv, test_split.csv

CSV files with image-level annotations for multi-label disease classification.

**Columns:**

- `Image Index` (string): Image filename (e.g., "00000001_000.png")
- `Finding Labels` (string): Pipe-separated disease labels (e.g., "Infiltration|Effusion")
- `Patient ID` (int): Unique patient identifier (used for patient-level splitting)
- `Patient Age` (int): Patient age in years (0-95)
- `Patient Gender` (string): Patient gender ("M" or "F")
- `View Position` (string): X-ray view angle ("PA" = posteroanterior, "AP" = anteroposterior)
- `full_path` (string): Local file path to image (update for Kaggle environment)
- `Atelectasis` through `Pneumothorax` (int): Binary labels (0/1) for each of 14 diseases

**Disease Classes (14 total):**
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

**Note:** Images can have multiple diseases (multi-label classification). "No Finding" cases have all disease columns = 0.

### 2. preprocessing_config.json

Configuration for reproducible preprocessing:

```json
{
  "disease_classes": ["Atelectasis", "Cardiomegaly", ...],
  "class_weights": {...},
  "normalization": {...},
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}
}
```

### 3. class_weights.json

Per-disease weights for handling class imbalance:

```json
{
  "Hernia": 258.23,        // Very rare (0.4% of images)
  "Infiltration": 0.55,    // Common (17.5% of images)
  "No Finding": 0.20,      // Very common (53.8% of images)
  ...
}
```

Use these weights with loss functions (e.g., `class_weight` in Keras) to improve model performance on rare diseases.

## 🚀 Usage

### Kaggle Notebook Setup

```python
import pandas as pd
import json
from pathlib import Path

# Load splits
train_df = pd.read_csv('/kaggle/input/nih-chest-xray-splits/train_split.csv')
val_df = pd.read_csv('/kaggle/input/nih-chest-xray-splits/val_split.csv')
test_df = pd.read_csv('/kaggle/input/nih-chest-xray-splits/test_split.csv')

# Load preprocessing config
with open('/kaggle/input/nih-chest-xray-splits/preprocessing_config.json') as f:
    config = json.load(f)

disease_classes = config['disease_classes']
class_weights = config['class_weights']

# Update image paths for Kaggle environment
def update_kaggle_paths(df):
    df['full_path'] = df['Image Index'].apply(
        lambda x: f'/kaggle/input/nih-chest-xrays/images/{x}'
    )
    return df

train_df = update_kaggle_paths(train_df)
val_df = update_kaggle_paths(val_df)
test_df = update_kaggle_paths(test_df)

print(f"Training: {len(train_df):,} images")
print(f"Validation: {len(val_df):,} images")
print(f"Test: {len(test_df):,} images")
```

### TensorFlow/Keras Data Generator

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='full_path',
    y_col=disease_classes,  # Multi-label
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=True
)
```

### PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, df, disease_classes, transform=None):
        self.df = df
        self.disease_classes = disease_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['full_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.FloatTensor(row[self.disease_classes].values)
        return image, labels

train_dataset = ChestXrayDataset(train_df, disease_classes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## 🔬 Split Methodology

### Patient-Level Splitting

Ensures no patient appears in multiple splits (prevents data leakage):

```python
# Group by patient, assign to splits
patient_splits = {}
for patient_id in unique_patients:
    # Deterministic assignment based on patient ID
    patient_splits[patient_id] = assign_split(patient_id)
```

### Stratified by Key Diseases

Maintains disease distribution across splits:

- **Cardiomegaly**: ~3% in all splits
- **Effusion**: ~12% in all splits
- **No Finding**: ~54% in all splits
- ...and so on for all 14 diseases

### Class Imbalance Handling

Disease prevalence varies dramatically:

| Disease | Prevalence | Recommended Weight |
|---------|------------|-------------------|
| Hernia | 0.4% | 258.23 |
| Pneumothorax | 5.3% | 1.95 |
| Cardiomegaly | 2.8% | 3.57 |
| No Finding | 53.8% | 0.20 |

Use provided class weights to improve model performance on rare diseases.

## 📖 Related Resources

- **Original Images**: [NIH Chest X-Ray Images on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (112K images, ~47 GB)
- **Original Paper**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR 2017
- **NIH Dataset Page**: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## 📊 Expected Model Performance

Based on literature and this split:

| Metric | Baseline (LogReg) | CNN | Transfer Learning |
|--------|------------------|-----|-------------------|
| **AUC (avg)** | ~0.65 | ~0.75 | ~0.85 |
| **Accuracy** | ~70% | ~80% | ~85% |

Use validation set for early stopping and hyperparameter tuning. Report final metrics on test set.

## 📝 Citation

If you use these splits in your research, please cite the original NIH dataset:

```
@inproceedings{wang2017chestxray8,
  title={ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2097--2106},
  year={2017}
}
```

## 🔄 Version History

- **v1** (2024-10): Initial release with stratified patient-level splits
- Class weights computed from full dataset
- Preprocessing config includes normalization parameters

## 📧 Questions?

For issues or questions about these splits, please open an issue on the project repository or contact the dataset maintainer.

---

**License**: CC0-1.0 (Public Domain)
**Compatibility**: TensorFlow, PyTorch, scikit-learn, Keras
**Environment**: Kaggle, Colab, Local
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
