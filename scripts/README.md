# Kaggle Headless Training Scripts

Fully automated cloud training on Kaggle P100 GPU - no browser needed!

## 🚀 Quick Start

```bash
# Set your Kaggle username
export KAGGLE_USERNAME="your-kaggle-username"

# Run complete pipeline
./scripts/kaggle_full_pipeline.sh
```

That's it! The script will:
1. Upload your data splits to Kaggle
2. Push notebook 06b and start training
3. Monitor execution
4. Download trained model when complete

## 📜 Individual Scripts

### `kaggle_full_pipeline.sh` - Complete Automation ⭐
One command to do everything. **Start here.**

### `prepare_kaggle_upload.sh` - Prepare Data
Packages CSVs and metadata into `kaggle_upload/` directory.

### `kaggle_upload_dataset.sh` - Upload Dataset
Uploads prepared data to Kaggle as a dataset (one-time setup).

### `kaggle_train_headless.sh` - Train Model
Pushes notebook, monitors training, downloads results.

## 📖 Documentation

See `docs/HEADLESS_KAGGLE_TRAINING.md` for complete guide.

## ⏱️ Expected Runtime

| Dataset Size | Training Time (P100 GPU) |
|--------------|-------------------------|
| Sample (5K) | 3-5 minutes |
| Full (78K) | 30-45 minutes |

Compare to **6-8 hours on M2 Pro** for full dataset!

## 🔧 Requirements

- Kaggle API key at `.kaggle/kaggle.json` ✓ (already configured)
- `kaggle` CLI tool ✓ (in requirements.txt)
- Your Kaggle username

## 💡 Tips

**Re-run with different hyperparameters:**
1. Edit `jupyter_notebooks/06b_cnn_kaggle.ipynb`
2. Run `./scripts/kaggle_train_headless.sh`

**Check status manually:**
```bash
kaggle kernels status your-username/cnn-development-cloud
```

**Download results later:**
```bash
kaggle kernels output your-username/cnn-development-cloud -p ./models/
```
