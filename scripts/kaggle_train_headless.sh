#!/bin/bash
# Train CNN on Kaggle P100 GPU - fully headless workflow

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

KAGGLE_USERNAME="${KAGGLE_USERNAME:-yourusername}"
KERNEL_SLUG="cnn-development-cloud"
NOTEBOOK_PATH="jupyter_notebooks/06b_cnn_kaggle.ipynb"

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
  "title": "CNN Development (Cloud)",
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
echo "✅ Notebook pushed!"
echo "   URL: $KERNEL_URL"
echo ""

# Monitor execution
echo "⏳ Monitoring execution (this may take 5-45 minutes)..."
echo "   Press Ctrl+C to stop monitoring (training will continue)"
echo ""

POLL_INTERVAL=30
MAX_WAIT=3600  # 1 hour timeout
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    STATUS=$(kaggle kernels status "$KAGGLE_USERNAME/$KERNEL_SLUG" 2>/dev/null || echo "error")

    case "$STATUS" in
        *"complete"*)
            # Check if it completed successfully or with errors
            echo ""
            echo "✅ Kernel execution complete"
            echo ""
            echo "📥 Downloading logs to check for errors..."
            kaggle kernels output "$KAGGLE_USERNAME/$KERNEL_SLUG" -p /tmp/kaggle_output 2>/dev/null

            if grep -qi "error\|exception\|failed\|traceback" /tmp/kaggle_output/*.log 2>/dev/null; then
                echo "❌ Errors detected in logs!"
                echo "   Log file: /tmp/kaggle_output/*.log"
                echo "   Kernel URL: $KERNEL_URL"
                exit 1
            else
                echo "✅ Training successful!"
            fi
            break
            ;;
        *"running"*)
            echo -n "."
            ;;
        *"error"*|*"failed"*|*"cancelAcknowledged"*)
            echo ""
            echo "❌ Kernel failed or was cancelled"
            echo "   Status: $STATUS"
            echo "   Check: $KERNEL_URL"
            exit 1
            ;;
        *)
            echo -n "."
            ;;
    esac

    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo ""
    echo "⏰ Timeout reached. Training may still be running."
    echo "   Check status: $KERNEL_URL"
    exit 1
fi

# Download results
echo ""
echo "📥 Downloading trained model and results..."

OUTPUT_DIR="models/saved_models_kaggle"
mkdir -p "$OUTPUT_DIR"

kaggle kernels output "$KAGGLE_USERNAME/$KERNEL_SLUG" -p "$OUTPUT_DIR"

echo ""
echo "✅ Headless training complete!"
echo ""
echo "📊 Results:"
echo "   Model: $OUTPUT_DIR/cnn_custom_best.keras"
echo "   Reports: $OUTPUT_DIR/reports/"
echo "   Figures: $OUTPUT_DIR/figures/"
echo ""
echo "🎯 Next steps:"
echo "   1. Evaluate model locally"
echo "   2. Compare with local training (notebook 06)"
echo "   3. Deploy or continue with transfer learning"
