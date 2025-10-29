#!/bin/bash
# Upload train/val/test splits to Kaggle as a dataset (headless)

set -e

KAGGLE_USERNAME="${KAGGLE_USERNAME:-yourusername}"
DATASET_SLUG="nih-chest-xray-splits"

echo "📦 Preparing NIH Chest X-Ray splits for Kaggle..."
echo ""

# Check if kaggle API is configured
if [ ! -f ".kaggle/kaggle.json" ]; then
    echo "❌ Error: .kaggle/kaggle.json not found"
    echo "   Create it at: https://www.kaggle.com/settings → API → Create New Token"
    exit 1
fi

# Set Kaggle config directory
export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"

# Prepare staging directory
./scripts/prepare_kaggle_upload.sh

cd kaggle_upload

# Update metadata with actual username
echo "Updating dataset metadata..."
if [ "$KAGGLE_USERNAME" = "yourusername" ]; then
    echo "⚠️  Warning: Using placeholder username 'yourusername'"
    echo "   Set your username: export KAGGLE_USERNAME=your-kaggle-username"
fi

sed -i.bak "s/YOUR_USERNAME/$KAGGLE_USERNAME/" dataset-metadata.json
rm -f dataset-metadata.json.bak

# Check if dataset already exists
echo ""
echo "Checking if dataset exists..."
if kaggle datasets list --user "$KAGGLE_USERNAME" | grep -q "$DATASET_SLUG"; then
    echo "Dataset already exists. Creating new version..."
    kaggle datasets version -p . -m "Updated splits and metadata" -r zip
else
    echo "Creating new dataset..."
    kaggle datasets create -p . -r zip
fi

cd ..

echo ""
echo "✅ Dataset uploaded!"
echo ""
echo "Dataset URL: https://www.kaggle.com/datasets/$KAGGLE_USERNAME/$DATASET_SLUG"
echo ""
echo "Next: Run 'scripts/kaggle_train_headless.sh' to train model"
