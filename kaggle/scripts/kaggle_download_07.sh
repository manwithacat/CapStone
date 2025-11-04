#!/bin/bash
# Download results from Kaggle Kernel for Notebook 07
#
# Usage: ./scripts/kaggle_download_07.sh [output_directory]

KERNEL_SLUG="manwithacat/transfer-learning-chest-x-ray-classification"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-/tmp/kaggle_07_output}"

export KAGGLE_CONFIG_DIR="$PROJECT_ROOT/.kaggle"

echo "=========================================="
echo "Downloading Kaggle Kernel 07 Results"
echo "=========================================="
echo ""
echo "Kernel: $KERNEL_SLUG"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check status first
echo "Checking kernel status..."
STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1)
echo "$STATUS"
echo ""

if [[ "$STATUS" == *"COMPLETE"* ]]; then
    echo "✅ Kernel completed! Downloading outputs..."
    echo ""

    # Download outputs
    kaggle kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR"

    echo ""
    echo "📥 Download complete!"
    echo ""
    echo "📁 Files downloaded to: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
    echo ""

    # Show what to do next
    echo "🎯 Next steps:"
    echo "   1. Check the trained models:"
    echo "      ls -lh $OUTPUT_DIR/*.keras"
    echo ""
    echo "   2. Copy models to your project:"
    echo "      cp $OUTPUT_DIR/*.keras $PROJECT_ROOT/models/saved_models/"
    echo ""
    echo "   3. Review training logs:"
    echo "      cat $OUTPUT_DIR/*.log"

elif [[ "$STATUS" == *"RUNNING"* ]]; then
    echo "⏳ Kernel is still running. Check back later."
    echo "   Estimated time: ~9 hours total"
    echo "   Monitor progress: ./scripts/kaggle_monitor_07.sh"

elif [[ "$STATUS" == *"ERROR"* ]] || [[ "$STATUS" == *"CANCEL"* ]]; then
    echo "❌ Kernel failed or was cancelled!"
    echo "   View logs at: https://www.kaggle.com/code/$KERNEL_SLUG"
    exit 1

else
    echo "❓ Unknown status: $STATUS"
    exit 1
fi
