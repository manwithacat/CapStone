# Platform Organization - Quick Reference

**Last Updated**: November 4, 2024

## Directory Structure

```
/Volumes/SSD/Capstone/
│
├── jupyter_notebooks/              📁 LOCAL DEVELOPMENT
│   ├── 06_cnn_development.ipynb   ← Local CNN training (with MLflow)
│   └── 07_transfer_learning.ipynb ← Local transfer learning (with MLflow)
│
├── kaggle/                         📁 KAGGLE PLATFORM
│   ├── kernels/
│   │   ├── 06_cnn/
│   │   │   ├── notebook.ipynb
│   │   │   └── kernel-metadata.json
│   │   └── 07_transfer_learning/
│   │       ├── notebook.ipynb
│   │       └── kernel-metadata.json
│   ├── datasets/
│   │   └── imagenet-weights/      (ImageNet pretrained weights)
│   ├── results/
│   │   ├── 06_cnn/
│   │   │   └── v12/               (downloaded)
│   │   └── 07_transfer_learning/
│   │       ├── v8/                (sample 1000, no weights)
│   │       ├── v9/                (full data, no weights, TIMEOUT)
│   │       └── v10/               (full data, ImageNet weights, ✅ SUCCESS)
│   └── scripts/
│       ├── kaggle_push_*.sh
│       ├── kaggle_monitor_*.sh
│       └── kaggle_import_*.py
│
├── colab/                          📁 GOOGLE COLAB
│   ├── 07_transfer_learning_colab.ipynb  (clean, minimal)
│   ├── COLAB_REFERENCE.md
│   └── scripts/
│       └── colab_workflow.sh
│
├── models/
│   ├── saved_models/              (local training + Kaggle imports)
│   └── pretrained_weights/        (ImageNet weights)
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── mlruns/                        (MLflow local tracking)
├── mlflow.db                      (MLflow database)
│
└── docs/
    ├── KAGGLE_GUIDE.md           ← Kaggle workflow
    ├── COLAB_GUIDE.md            ← Colab workflow
    └── PLATFORM_ORGANIZATION.md  ← This file
```

## Platform Comparison

| Feature              | Local                    | Kaggle                  | Colab                    |
|----------------------|--------------------------|-------------------------|--------------------------|
| **Purpose**          | Development & tracking   | Long training runs      | Interactive testing      |
| **Notebook Location**| `jupyter_notebooks/`     | `kaggle/kernels/`       | `colab/`                 |
| **GPU**              | None (CPU only)          | T4 (16GB)               | T4 or A100 (16-40GB)     |
| **Time Limit**       | Unlimited                | 9 hours                 | 12-24 hours              |
| **Data Source**      | Local SSD (47GB)         | Kaggle datasets         | Google Drive             |
| **MLflow Tracking**  | ✅ Yes                   | ❌ No (import later)    | ❌ No                    |
| **Papermill**        | ✅ Yes                   | ✅ Yes                  | ❌ No                    |
| **Internet Access**  | ✅ Yes                   | ❌ No                   | ✅ Yes                   |
| **Interactive**      | ✅ Yes                   | ❌ Batch only           | ✅ Yes                   |
| **Best For**         | Experimentation, MLflow  | Free GPU training       | A100 GPU, quick tests    |

## Workflow by Platform

### 🖥️ LOCAL (Development)

```bash
# Work in jupyter_notebooks/
cd jupyter_notebooks

# Edit notebooks with full MLflow tracking
jupyter lab 07_transfer_learning.ipynb

# View MLflow UI
open http://localhost:5001
```

**Features:**
- Full MLflow experiment tracking
- Papermill parameterization
- Access to full 47GB dataset
- Unlimited time (but no GPU)

---

### ☁️ KAGGLE (Production Training)

```bash
# Work in kaggle/kernels/
cd kaggle/kernels/07_transfer_learning

# Push to Kaggle
KAGGLE_CONFIG_DIR="../../../.kaggle" kaggle kernels push

# Monitor
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status manwithacat/nb07-transfer-learning-v2

# Download results
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels output manwithacat/nb07-transfer-learning-v2 \
  -p kaggle/results/07_transfer_learning/v11

# Import to MLflow
python kaggle/scripts/kaggle_import_v11.py
```

**Features:**
- Free T4 GPU (9 hour limit)
- Batch execution
- No MLflow (import after)
- Internet restricted

**Latest Result:**
- ✅ **v10 SUCCESS**: 3 models trained with ImageNet weights (~6 hours)

---

### 🌐 COLAB (Interactive Testing)

```bash
# Open stable notebook
./colab/scripts/colab_workflow.sh
# Or: https://colab.research.google.com/drive/1FJdto9vlXuvtofDpjIN9Vb5DzyPAlLEH

# Edit in browser
# Select runtime: CPU (test) or A100 (training)
# Run cells interactively

# (Optional) Pull back
colab-cli pull-nb colab/07_transfer_learning_colab.ipynb
```

**Features:**
- Interactive development
- A100 GPU available (with Pro)
- Internet access (downloads ImageNet weights)
- No MLflow tracking

---

## Quick Commands

### Check Kaggle Status
```bash
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status manwithacat/nb07-transfer-learning-v2
```

### Download Latest Kaggle Results
```bash
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels output manwithacat/nb07-transfer-learning-v2 \
  -p kaggle/results/07_transfer_learning/v11
```

### Open Colab
```bash
./colab/scripts/colab_workflow.sh
```

### Start MLflow UI
```bash
./scripts/mlflow_start.sh 5001
open http://localhost:5001
```

---

## File Locations

### Notebooks
- **Local (with MLflow)**: `jupyter_notebooks/07_transfer_learning.ipynb`
- **Kaggle (no MLflow)**: `kaggle/kernels/07_transfer_learning/notebook.ipynb`
- **Colab (minimal)**: `colab/07_transfer_learning_colab.ipynb`

### Models
- **Local training**: `models/saved_models/`
- **Kaggle downloads**: `kaggle/results/07_transfer_learning/v*/models/saved_models/`
- **Imported to MLflow**: Tracked in `mlruns/` and `mlflow.db`

### Results
- **Local**: `outputs/reports/07_*.csv`
- **Kaggle**: `kaggle/results/07_transfer_learning/v*/`
- **Colab**: Ephemeral (download manually if needed)

---

## Documentation

| Guide | Purpose |
|-------|---------|
| `docs/KAGGLE_GUIDE.md` | Complete Kaggle workflow and troubleshooting |
| `docs/COLAB_GUIDE.md` | Complete Colab workflow and setup |
| `docs/PLATFORM_ORGANIZATION.md` | This file - quick reference |
| `colab/COLAB_REFERENCE.md` | One-page Colab quick reference |

---

## Migration Status

✅ **Completed**:
- Created new directory structure
- Moved Kaggle kernels to `kaggle/kernels/`
- Moved Colab notebook to `colab/`
- Moved Kaggle scripts to `kaggle/scripts/`
- Moved v10 results to `kaggle/results/07_transfer_learning/v10/`
- Created KAGGLE_GUIDE.md and COLAB_GUIDE.md
- Updated .gitignore

🔄 **Next Steps**:
- Move v8/v9 results if needed
- Clean up old directories (kaggle_kernel_07/, etc.)
- Update any remaining scripts with new paths

---

## Summary

**Each platform now has its own directory with clear purpose:**

- 🖥️ **jupyter_notebooks/**: Local development with full features
- ☁️ **kaggle/**: Cloud GPU training (batch jobs)
- 🌐 **colab/**: Interactive development (A100 access)

**No more confusion about which notebook is for which platform!**
