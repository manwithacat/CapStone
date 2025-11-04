# Google Colab Setup Guide - Notebook 07 Transfer Learning

## 🎯 GPU Recommendations

Based on your screenshot options:

### Free Tier (Recommended for Testing)
- **T4 GPU**: 16GB VRAM
  - ✅ Perfect for your use case
  - ✅ Comparable to Kaggle's P100
  - ✅ Should complete training in ~2-3 hours
  - Cost: **FREE**

### Paid Tier (Colab Pro - $10/month)
- **L4 GPU**: 24GB VRAM
  - Newer architecture, faster than T4
  - Good for larger batch sizes
  
- **A100 GPU**: 40GB VRAM
  - Fastest option available
  - Overkill for this project but nice to have

**Recommendation**: Start with **T4 GPU** (free). Upgrade to Pro only if you need:
- Longer runtime (24 hours vs 12 hours)
- Faster iteration
- Bigger models in the future

---

## 🚀 Quick Start (Manual Upload)

**Easiest method - No CLI setup needed:**

1. **Open Colab**: https://colab.research.google.com/
2. **Upload notebook**: 
   - File → Upload notebook
   - Select: `jupyter_notebooks/07_transfer_learning_colab.ipynb`
3. **Set GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator → **T4 GPU**
   - High RAM → **ON** (recommended)
   - Save
4. **Upload data to Google Drive** (one-time):
   ```
   /content/drive/MyDrive/
   ├── capstone_data/
   │   ├── nih-chest-xray-splits/
   │   │   ├── train_split.csv
   │   │   ├── val_split.csv
   │   │   ├── test_split.csv
   │   │   └── preprocessing_config.json
   │   └── nih-chest-xrays/
   │       └── (your image data)
   ```
5. **Run the notebook!**

---

## 🔧 CLI Setup (Advanced - Optional)

### Step 1: Get Google Cloud Credentials

1. Go to: https://console.cloud.google.com/
2. Create new project (or select existing)
3. Enable **Google Drive API**:
   - APIs & Services → Enable APIs and Services
   - Search "Google Drive API"
   - Click Enable
4. Create credentials:
   - APIs & Services → Credentials
   - Create Credentials → OAuth client ID
   - Application type: **Desktop app**
   - Download JSON → Save as `client_secrets.json`

### Step 2: Configure colab-cli

```bash
# Set credentials
colab-cli set-config

# When prompted, provide path to client_secrets.json
# Then authenticate in browser

# Open notebook in Colab
colab-cli open-nb jupyter_notebooks/07_transfer_learning_colab.ipynb

# After editing in Colab, pull changes
colab-cli pull-nb jupyter_notebooks/07_transfer_learning_colab.ipynb
```

---

## 📊 Performance Comparison

| Platform | GPU | VRAM | Runtime Limit | Cost | Internet | ImageNet Weights |
|----------|-----|------|---------------|------|----------|------------------|
| **Kaggle** | P100 | 16GB | 9 hours | FREE | ❌ | ✅ (uploaded) |
| **Colab Free** | T4 | 16GB | 12 hours | FREE | ✅ | ✅ (download) |
| **Colab Pro** | A100/L4 | 40GB/24GB | 24 hours | $10/mo | ✅ | ✅ |

---

## ⚡ Colab Advantages for This Project

1. **Longer runtime**: 12 hours (vs 9 on Kaggle)
2. **Internet access**: Download ImageNet weights directly (no upload needed)
3. **Interactive debugging**: Edit and run cells immediately
4. **Better for iteration**: Faster feedback cycle

---

## 📝 Notebook Modifications for Colab

The Colab version (`07_transfer_learning_colab.ipynb`) includes:

1. ✅ Google Drive mounting
2. ✅ GPU detection and display
3. ✅ Automatic environment detection
4. ✅ ImageNet weights download (can use internet!)
5. ✅ Path handling for Colab's `/content/drive/MyDrive/`

---

## 🎬 Recommended Workflow

1. **Upload data to Google Drive** (one-time, ~30 min)
2. **Open notebook in Colab** (manual or CLI)
3. **Select T4 GPU + High RAM**
4. **Run all cells**
5. **Download artifacts** when complete (or save to Drive)
6. **Import to MLflow locally**

---

## 💡 Pro Tips

- **High RAM**: Always enable for deep learning
- **Keep browser open**: Colab disconnects if idle too long
- **Save checkpoints**: Save to Drive periodically
- **Use `%%time`**: Add to cells to track execution time
- **Monitor GPU**: Run `!nvidia-smi` to check GPU usage

