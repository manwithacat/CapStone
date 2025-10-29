#!/bin/bash
# Complete end-to-end Kaggle training pipeline - one command to rule them all

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Kaggle Headless Training Pipeline                       ║"
echo "║  CNN Training on P100 GPU - Fully Automated              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "📋 Loading configuration from .env..."
    # Export variables from .env (skip comments and empty lines)
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Check environment
KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"
if [ -z "$KAGGLE_USERNAME" ] || [ "$KAGGLE_USERNAME" = "yourusername" ]; then
    echo "⚠️  KAGGLE_USERNAME not set"
    echo ""
    echo "Please set your Kaggle username (choose one method):"
    echo ""
    echo "Option 1 - Via .env file (recommended):"
    echo "  1. Edit .env file"
    echo "  2. Uncomment: KAGGLE_USERNAME=your-actual-username"
    echo "  3. Re-run this script"
    echo ""
    echo "Option 2 - Via environment variable:"
    echo "  export KAGGLE_USERNAME=\"your-actual-username\""
    echo ""
    echo "Get your username from: https://www.kaggle.com/settings"
    echo ""
    read -p "Or enter now: " input_username
    if [ -n "$input_username" ]; then
        export KAGGLE_USERNAME="$input_username"
        echo "✓ Using username: $KAGGLE_USERNAME"
    else
        echo "❌ Cannot proceed without username"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Kaggle Username: $KAGGLE_USERNAME"
echo "  Notebook: jupyter_notebooks/06b_cnn_kaggle.ipynb"
echo "  Your Dataset: $KAGGLE_USERNAME/nih-chest-xray-splits"
echo "  NIH Images: ${NIH_IMAGE_DATASET:-nih-chest-xrays/data}"
echo "  GPU: P100"
echo ""
read -p "Proceed with training? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 1/3: Upload Dataset to Kaggle"
echo "═══════════════════════════════════════════════════════════"
echo ""

if ./scripts/kaggle_upload_dataset.sh; then
    echo "✅ Dataset ready"
else
    echo "❌ Dataset upload failed"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 2/3: Push Notebook and Start Training"
echo "═══════════════════════════════════════════════════════════"
echo ""

if ./scripts/kaggle_train_headless.sh; then
    echo "✅ Training complete"
else
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 3/3: Summary"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✨ Pipeline completed successfully!"
echo ""
echo "📊 Results available at:"
echo "   models/saved_models_kaggle/"
echo ""
echo "📈 Compare with local training:"
echo "   diff models/saved_models/cnn_custom_best.keras \\"
echo "        models/saved_models_kaggle/cnn_custom_best.keras"
echo ""
echo "🔍 View training history:"
echo "   cat models/saved_models_kaggle/reports/06_cnn_training_history.csv"
echo ""
echo "🎯 Next steps:"
echo "   1. Evaluate model performance"
echo "   2. Try transfer learning (notebook 07)"
echo "   3. Deploy to production"
echo ""
