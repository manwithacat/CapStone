# Kaggle Platform Guide

**Directory**: `kaggle/`

## Overview

Kaggle is used for long training runs (up to 9 hours) on free T4 GPU.

## Directory Structure

```
kaggle/
├── kernels/                 # Kernel configurations
│   ├── 06_cnn/
│   │   ├── notebook.ipynb
│   │   └── kernel-metadata.json
│   └── 07_transfer_learning/
│       ├── notebook.ipynb
│       └── kernel-metadata.json
├── datasets/               # Uploaded datasets
│   ├── imagenet-weights/
│   └── nih-chest-xray-splits/
├── results/                # Downloaded results
│   ├── 06_cnn/
│   │   └── v12/
│   └── 07_transfer_learning/
│       ├── v8/  (sample 1000, no weights)
│       ├── v9/  (full data, no weights, TIMEOUT)
│       └── v10/ (full data, ImageNet weights, SUCCESS)
└── scripts/                # Kaggle automation
    ├── kaggle_push_*.sh
    ├── kaggle_monitor_*.sh
    └── kaggle_import_*.py
```

## Workflow

### 1. Prepare Kernel

```bash
cd kaggle/kernels/07_transfer_learning
# Edit notebook.ipynb and kernel-metadata.json
```

### 2. Push to Kaggle

```bash
cd kaggle/kernels/07_transfer_learning
KAGGLE_CONFIG_DIR="../../../.kaggle" kaggle kernels push
```

### 3. Monitor Progress

```bash
# Check status
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status manwithacat/nb07-transfer-learning-v2

# Monitor logs (if script exists)
./kaggle/scripts/kaggle_monitor_07.sh
```

### 4. Download Results

```bash
# Download to results directory
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels output manwithacat/nb07-transfer-learning-v2 \
  -p kaggle/results/07_transfer_learning/v11
```

### 5. Import to MLflow

```bash
# Import models and metrics to local MLflow
python kaggle/scripts/kaggle_import_v10.py
```

## Key Differences from Local

- **No MLflow**: Kaggle doesn't support MLflow tracking
- **No Internet**: Can't download ImageNet weights on-the-fly (use uploaded datasets)
- **verbose=2**: Use one-line-per-epoch output instead of progress bars
- **9-hour limit**: Training must complete within time limit
- **Parameterization**: Use papermill or environment detection

## Tips

1. **Test locally first**: Always test with sample data locally before pushing
2. **Use pre-flight tests**: Run `./scripts/test_notebook_07_preflight.sh`
3. **Upload large files as datasets**: ImageNet weights, data splits
4. **Monitor early**: Check first few epochs to catch errors early
5. **Version your kernels**: Track v8, v9, v10, etc. in results/

## Current Kaggle Datasets

- `manwithacat/nih-chest-xray-splits` - Preprocessed train/val/test splits (5MB)
- `manwithacat/imagenet-pretrained-weights` - ImageNet weights (160MB)
- `nih-chest-xrays/data` - Full NIH dataset (47GB)

## Results Summary

| Version | Data    | Weights   | Status   | Time   | Models |
|---------|---------|-----------|----------|--------|--------|
| v8      | 1000    | None      | SUCCESS  | ~2h    | 3      |
| v9      | 78,566  | None      | TIMEOUT  | 9.4h   | 0      |
| v10     | 78,566  | ImageNet  | SUCCESS  | ~6h    | 3      |

## Common Issues

### Issue: Notebook times out
**Solution**: Reduce epochs, use ImageNet weights, or reduce data size

### Issue: ImportError for MLflow
**Solution**: Remove MLflow imports or wrap in try/except (already handled)

### Issue: Can't download ImageNet weights
**Solution**: Upload weights as Kaggle dataset, load from `/kaggle/input/`

### Issue: Path not found
**Solution**: Fix paths in CSV files for Kaggle environment (`/kaggle/input/data/`)
