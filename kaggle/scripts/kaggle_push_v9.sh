#!/bin/bash
set -e

echo "======================================"
echo "Pushing Notebook 07 v9 to Kaggle"
echo "Full Training - All Data"
echo "======================================"

# Load parameters
PARAMS_FILE="test_params_v9_kaggle.json"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "❌ Parameter file not found: $PARAMS_FILE"
    exit 1
fi

echo "✓ Using parameters from: $PARAMS_FILE"
cat "$PARAMS_FILE"

# Update kernel metadata with new title
cd kaggle_kernel_07

# Backup current metadata
cp kernel-metadata.json kernel-metadata.json.bak

# Update title and ID
cat kernel-metadata.json.bak | \
    python3 -c "import sys, json; d=json.load(sys.stdin); d['title']='NB07 Transfer Learning v2 v9 - Full Training'; d['id']='jm00e4/nb07-transfer-learning-v2'; print(json.dumps(d, indent=2))" \
    > kernel-metadata.json

echo ""
echo "Updated metadata:"
cat kernel-metadata.json

echo ""
echo "Pushing to Kaggle..."
KAGGLE_CONFIG_DIR=../.kaggle kaggle kernels push

echo ""
echo "✅ Version 9 pushed successfully!"
echo ""
echo "Monitor with:"
echo "  KAGGLE_CONFIG_DIR=\"/Volumes/SSD/Capstone/.kaggle\" kaggle kernels status jm00e4/nb07-transfer-learning-v2"
echo ""
echo "Training configuration:"
echo "  - Full dataset (78,831 train / 16,383 val images)"
echo "  - Batch size: 32"
echo "  - Stage 1: 10 epochs (feature extraction)"
echo "  - Stage 2: 10 epochs (fine-tuning)"
echo "  - Training from scratch (no ImageNet weights)"
echo "  - Estimated time: ~6-8 hours on P100"
