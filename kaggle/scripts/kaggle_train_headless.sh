#!/bin/bash
# Train CNN on Kaggle P100 GPU - fully headless workflow (OPTIMIZED)
# Uses 06c_cnn_optimized.ipynb with 12-15× speedup optimizations

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

KAGGLE_USERNAME="${KAGGLE_USERNAME:-yourusername}"
KERNEL_SLUG="cnn-optimized-training"
NOTEBOOK_PATH="jupyter_notebooks/06c_cnn_optimized.ipynb"

echo "🚀 Kaggle Headless Training Pipeline"
echo "===================================="
echo ""

# Check prerequisites
if [ ! -f ".kaggle/kaggle.json" ]; then
    echo "❌ Error: .kaggle/kaggle.json not found"
    exit 1
fi

if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "❌ Error: $NOTEBOOK_PATH not found"
    exit 1
fi

export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"

# Use official NIH Chest X-Ray dataset
# https://www.kaggle.com/datasets/nih-chest-xrays/data
echo "🔍 Validating NIH Chest X-Ray dataset..."
NIH_DATASET_SLUG="${NIH_IMAGE_DATASET:-nih-chest-xrays/data}"

echo "✓ Using NIH dataset: $NIH_DATASET_SLUG"

# Validate user's splits dataset exists
echo "🔍 Validating your splits dataset..."
USER_DATASET_SLUG="$KAGGLE_USERNAME/nih-chest-xray-splits"

if kaggle datasets list --user "$KAGGLE_USERNAME" | grep -q "nih-chest-xray-splits"; then
    echo "✓ Found splits dataset: $USER_DATASET_SLUG"
else
    echo "❌ Error: Your splits dataset not found"
    echo "   Expected: $USER_DATASET_SLUG"
    echo ""
    echo "Run this first:"
    echo "  ./scripts/kaggle_upload_dataset.sh"
    exit 1
fi

# Create kernel metadata
echo ""
echo "📝 Creating kernel metadata..."
mkdir -p kaggle_kernel
cp "$NOTEBOOK_PATH" kaggle_kernel/notebook.ipynb

cat > kaggle_kernel/kernel-metadata.json << METADATA
{
  "id": "$KAGGLE_USERNAME/$KERNEL_SLUG",
  "title": "CNN Optimized Training",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "$NIH_DATASET_SLUG",
    "$USER_DATASET_SLUG"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
METADATA

echo "   Linked datasets:"
echo "   - NIH Images: $NIH_DATASET_SLUG"
echo "   - Your Splits: $USER_DATASET_SLUG"

# Push notebook to Kaggle
echo ""
echo "📤 Pushing notebook to Kaggle..."
cd kaggle_kernel
kaggle kernels push -p .
cd ..

KERNEL_URL="https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_SLUG"
echo ""
echo "✅ Notebook pushed and queued for execution!"
echo ""
echo "📊 Kernel Details:"
echo "   URL: $KERNEL_URL"
echo "   Username: $KAGGLE_USERNAME"
echo "   Kernel ID: $KERNEL_SLUG"
echo ""
echo "⏰ Training will run on Kaggle (~4-6 hours for 50 epochs with optimizations)"
echo ""
echo "🔍 To check status:"
echo "   kaggle kernels status $KAGGLE_USERNAME/$KERNEL_SLUG"
echo ""
echo "📥 To download results when complete:"
echo "   kaggle kernels output $KAGGLE_USERNAME/$KERNEL_SLUG -p ~/Downloads/results"
echo ""
echo "   Or use the download script:"
echo "   ./scripts/kaggle_download_results.sh"
echo ""
echo "💡 Tip: You can close this terminal - training will continue on Kaggle"
echo "   Check progress at: $KERNEL_URL"
