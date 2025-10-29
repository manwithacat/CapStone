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

# Create kernel metadata
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
    "nih-chest-xrays",
    "$KAGGLE_USERNAME/nih-chest-xray-splits"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
METADATA

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
            echo ""
            echo "✅ Training complete!"
            break
            ;;
        *"running"*)
            echo -n "."
            ;;
        *"error"*|*"failed"*)
            echo ""
            echo "❌ Training failed. Check logs:"
            echo "   $KERNEL_URL"
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
