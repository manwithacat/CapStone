# Google Colab - Quick Reference

## 📌 Your Stable Notebook

**URL**: https://colab.research.google.com/drive/1FJdto9vlXuvtofDpjIN9Vb5DzyPAlLEH

**Bookmark this!** - It's your permanent notebook that auto-saves.

---

## ✅ Recommended Workflow

### For Active Development (Edit in Colab):
1. **Open**: Click the URL above (or run `./scripts/colab_workflow.sh`)
2. **Edit**: Make changes directly in browser
3. **Run**: Execute cells interactively
4. **Auto-saves**: Colab saves to your Drive automatically
5. **Done**: Just close the tab - changes are saved

### To Sync Back to Local (Optional):
```bash
# Pull latest version from Colab to local
colab-cli pull-nb jupyter_notebooks/07_transfer_learning_colab.ipynb
```

**Don't use `push-nb` anymore** - it creates new notebooks!

---

## 🔌 Managing Compute Resources

### Check Active Runtimes:
1. In Colab, click **Runtime** menu
2. Select **Manage sessions**
3. See all active notebooks with GPU/RAM usage

### Shut Down Old Runtimes:
1. Runtime → Manage sessions
2. Click **Terminate** on any old sessions
3. Or just disconnect: Runtime → Disconnect and delete runtime

### Auto-Shutdown:
- Colab auto-disconnects after **90 minutes of idle**
- No need to manually manage unless you have many open tabs

---

## 🎯 Current Setup

- **Notebook**: 07_transfer_learning_colab.ipynb
- **Location**: Google Drive (auto-synced)
- **Runtime**: A100 GPU + High RAM
- **Mode**: Sample mode (use_sample: True)

---

## 🚀 Quick Start (Right Now)

1. **Open the URL above** (should be open already)
2. **Verify GPU**: Runtime → View runtime type → Should show A100
3. **Add data download cell** (see below)
4. **Run all**

### Quick Data Setup Cell:
Insert this as a new cell near the top:

```python
# Quick setup - download splits from Kaggle
!pip install -q kaggle

# Upload kaggle.json when prompted
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download splits (tiny, ~5MB)
!kaggle datasets download -d manwithacat/nih-chest-xray-splits -p /content
!unzip -q nih-chest-xray-splits.zip -d /content/nih-chest-xray-splits

# For sample mode, create dummy images
import os
from PIL import Image
import numpy as np

os.makedirs('/content/images', exist_ok=True)
for i in range(1000):
    img = Image.new('L', (224, 224), color=np.random.randint(0, 255))
    img.save(f'/content/images/img_{i}.png')

print("✓ Setup complete!")
```

---

## 🔑 Key Points

✅ **One notebook URL** - always use the same one
✅ **Edit in browser** - no CLI needed for development  
✅ **Auto-saves** - changes persist automatically
✅ **Manage sessions** - Runtime → Manage sessions to clean up
✅ **Pull when needed** - Use `colab-cli pull-nb` to sync back locally

---

## 📊 Alternative: Just Use Kaggle

**Reality check**: Since Kaggle v10 is already running with real data and ImageNet weights, you might not need Colab at all!

Check Kaggle status:
```bash
KAGGLE_CONFIG_DIR="/Volumes/SSD/Capstone/.kaggle" kaggle kernels status manwithacat/nb07-transfer-learning-v2-v10-imagenet-weights
```

Colab is great for:
- Interactive debugging
- Faster iteration
- Longer runtimes

But Kaggle is simpler for:
- Batch training
- Pre-configured datasets
- Set it and forget it

