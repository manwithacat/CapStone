# Colab Pro Persistent Disk Setup

## Benefits

With Colab Pro's persistent disk:
- ✅ Download NIH dataset **once** (10-30 min)
- ✅ Disconnect runtime when done (no charges)
- ✅ Reconnect anytime - data is still there
- ✅ No need to keep runtime running
- ✅ Faster GPUs (A100, V100)

## One-Time Setup

### 1. Enable Persistent Disk

When starting your runtime:

1. **Runtime → Change runtime type**
2. Select your GPU (T4, A100, or V100)
3. Click **Advanced settings**
4. **Persistent disk**:
   - Enable: ✅ ON
   - Size: **200 GB** (or 100 GB minimum for NIH dataset)
5. **Save**

### 2. First Run - Download Dataset

Run the notebook normally. Cell 13 will:

```
📥 Downloading NIH Chest X-Ray dataset from Kaggle...
⏱️  Takes 10-30 minutes
📦 Copying to persistent disk: /content/data/nih-chest-xrays
✓ Copied to persistent disk
```

**This happens ONCE.**

### 3. Disconnect Runtime

After training completes (or even during download):

1. **Runtime → Disconnect and delete runtime**
2. ✅ Your data stays on persistent disk
3. ✅ No charges while disconnected

## Using Cached Data (Every Time After First)

### 1. Start New Runtime

1. **Runtime → Change runtime type**
2. Select GPU
3. **Advanced settings → Persistent disk**: Select your existing disk
4. **Connect**

### 2. Run Notebook

Cell 13 will now show:

```
✅ Found cached NIH Chest X-Ray dataset!
📂 Location: /content/data/nih-chest-xrays
📊 Image directories: 12
💡 Using cached data - no download needed!
```

**Training starts immediately - no 30-minute wait!**

## Workflow Comparison

### Without Persistent Disk (Free Colab)
```
Start runtime → Download 47 GB (30 min) → Train → Results
          ↓
     Disconnect
          ↓
Start runtime → Download 47 GB AGAIN (30 min) → Train → Results
```

### With Persistent Disk (Colab Pro)
```
FIRST TIME:
Start runtime → Download 47 GB (30 min) → Train → Results
          ↓
     Disconnect (data saved to disk)
          ↓
EVERY TIME AFTER:
Start runtime → Train immediately (data cached) → Results
```

## Cost & Billing

**Colab Pro**: $10/month
- Faster GPUs
- Longer runtimes (24 hours vs 12 hours)
- More RAM
- 200 GB persistent disk included

**You're only charged when:**
- ✅ Runtime is running
- ❌ NOT when disconnected (even with persistent disk)

**So:**
- Download dataset (30 min running)
- Disconnect (no charges)
- Come back tomorrow
- Reconnect (charges resume)
- Data still there!

## Tips

### 1. Monitor Disk Usage

```python
!df -h /content/data
```

Shows how much of your 200 GB is used.

### 2. Clear Cache When Done

If you need to free up space:

```python
# Only do this when you're done with the project
!rm -rf /content/data/nih-chest-xrays
```

### 3. Multiple Projects

You can create **multiple persistent disks**:
- One for NIH Chest X-Ray project (200 GB)
- Another for different project (100 GB)
- Switch between them as needed

### 4. Verify Disk is Attached

At the start of your notebook:

```python
import os
print("Disk usage:")
os.system("df -h /content/data")

from pathlib import Path
if Path('/content/data/nih-chest-xrays').exists():
    print("✅ Persistent disk attached with cached data")
else:
    print("⚠️  No cached data - will download")
```

## Troubleshooting

### "No space left on device"

Your persistent disk is full. Options:
1. Increase disk size (Advanced settings)
2. Delete old data
3. Use a different disk

### "Cached data not found"

You might have:
- Selected a different persistent disk
- Created a new disk instead of reusing
- Data was in `/root/.cache/` (ephemeral) instead of `/content/data/` (persistent)

**Solution**: Check Advanced settings → Make sure you selected the **same disk**

### Dataset Downloaded to Wrong Location

The notebook copies kagglehub downloads to `/content/data/nih-chest-xrays` specifically because:
- `/root/.cache/` is ephemeral (deleted on disconnect)
- `/content/data/` persists (if disk attached)

## Best Practices

1. **Always use the same persistent disk** for this project
2. **Disconnect when not training** (save money)
3. **Verify cache exists** before starting long training runs
4. **Keep disk size reasonable** (200 GB is plenty for NIH dataset)
5. **Don't commit large files** (models go to Drive, not git)

## Comparison: Kaggle vs Colab Pro

| Feature | Kaggle Kernels | Colab Pro |
|---------|----------------|-----------|
| **Dataset** | Pre-mounted (instant) | Download once (30 min), then cached |
| **GPU** | P100, T4 (16 GB) | A100 (40 GB), V100, T4 |
| **Time Limit** | 9 hours | 24 hours |
| **Cost** | Free | $10/month |
| **Disconnect** | Stops kernel | Data persists |
| **Setup** | Use nbpush CLI | Manual upload |

**Recommendation:**
- **Kaggle**: For quick experiments (dataset pre-mounted)
- **Colab Pro**: For iterative development (persistent workspace)

Both work great - choose based on your workflow!
