#!/bin/bash
# Monitor Kaggle Kernel for Notebook 07 Transfer Learning
#
# Usage: ./scripts/kaggle_monitor_07.sh

KERNEL_SLUG="manwithacat/transfer-learning-chest-x-ray-classification"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export KAGGLE_CONFIG_DIR="$PROJECT_ROOT/.kaggle"

echo "=========================================="
echo "Monitoring Kaggle Kernel: Notebook 07"
echo "=========================================="
echo ""
echo "Kernel: $KERNEL_SLUG"
echo ""

# Check status
echo "Current Status:"
kaggle kernels status "$KERNEL_SLUG"
echo ""

# Show kernel URL
echo "🔗 View on Kaggle:"
echo "   https://www.kaggle.com/code/$KERNEL_SLUG"
echo ""

# Show output download command
echo "📥 Download outputs when complete:"
echo "   KAGGLE_CONFIG_DIR=.kaggle kaggle kernels output $KERNEL_SLUG -p /tmp/kaggle_07_output"
echo ""

# Provide monitoring tips
echo "💡 Monitoring Tips:"
echo "   • Estimated time: ~9 hours (3 models × ~3 hours each)"
echo "   • Check status periodically: ./scripts/kaggle_monitor_07.sh"
echo "   • View live logs on Kaggle website"
echo "   • Models will be saved automatically"
echo ""

echo "🎯 What's Training:"
echo "   • 3 models: ResNet50, DenseNet121, EfficientNetB3"
echo "   • Stage 1: 3 epochs (feature extraction, base frozen)"
echo "   • Stage 2: 7 epochs (fine-tuning, top 20 layers unfrozen)"
echo "   • Full dataset: 78,831 train / 16,383 val images"
echo "   • Batch size: 16"
echo ""

# Show what to do when complete
echo "✅ When complete, run:"
echo "   ./scripts/kaggle_download_07.sh"
