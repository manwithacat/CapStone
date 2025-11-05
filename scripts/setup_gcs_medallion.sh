#!/usr/bin/env bash
#
# Setup GCS bucket with Medallion architecture
#
# Structure:
#   00_raw/       - Immutable source data (NIH images as downloaded)
#   10_bronze/    - Light cleaning (train/val/test splits)
#   20_silver/    - Preprocessed (resized, normalized - future)
#   30_gold/      - Feature-ready (embeddings - future)
#   40_models/    - Model registry (checkpoints, configs)
#   50_artifacts/ - Evaluation outputs (metrics, plots)
#   60_logs/      - Training logs (tensorboard, mlflow)
#   70_cfg/       - Environment configs (requirements.txt)
#   80_docs/      - Documentation (datasheets, ethics)
#   90_tmp/       - Scratch space (auto-cleaned)

set -e

BUCKET="gs://nih-xrays"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🏗️  Setting up Medallion Architecture in GCS"
echo "=============================================="
echo

# Check authentication
if ! gsutil ls "${BUCKET}" &> /dev/null; then
    echo "❌ Cannot access ${BUCKET}"
    echo "   Run: gcloud auth application-default login"
    exit 1
fi

echo "✓ Authenticated"
echo

# Create directory structure (GCS creates dirs on first write)
echo "📁 Creating directory structure..."

# Create placeholder files to establish structure
cat > /tmp/.gitkeep << 'EOF'
# Placeholder file to maintain directory structure
EOF

for dir in 00_raw 10_bronze 20_silver 30_gold 40_models 50_artifacts 60_logs 70_cfg 80_docs 90_tmp; do
    echo "  Creating ${dir}/"
    gsutil cp /tmp/.gitkeep "${BUCKET}/${dir}/.gitkeep" 2>/dev/null || true
done

rm /tmp/.gitkeep
echo "✓ Structure created"
echo

# Upload data to appropriate layers
echo "📤 Uploading data to medallion layers..."
echo

# 00_raw: NIH Chest X-Ray images (immutable originals)
if [ -d "${PROJECT_ROOT}/data/raw/images_001" ]; then
    echo "📦 00_raw/ - Uploading NIH images (this will take time)..."
    echo "   Source: ${PROJECT_ROOT}/data/raw/"
    echo "   Destination: ${BUCKET}/00_raw/nih-cxr/images/"
    echo

    read -p "Upload ~47 GB of images? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gsutil -m -o "GSUtil:parallel_process_count=8" \
            rsync -r -x '.*\.DS_Store$|.*\._.*$|.*\.Spotlight-V100.*$|.*\.Trashes.*$' \
            "${PROJECT_ROOT}/data/raw/" "${BUCKET}/00_raw/nih-cxr/images/"
        echo "✓ Images uploaded to 00_raw/"
    else
        echo "⏭️  Skipping image upload"
    fi
else
    echo "⚠️  No images found at ${PROJECT_ROOT}/data/raw/"
    echo "   Skipping 00_raw/ upload"
fi

echo

# 10_bronze: Data splits (lightly processed manifests)
echo "📦 10_bronze/ - Uploading train/val/test splits..."
if [ -d "${PROJECT_ROOT}/colab/data_splits" ]; then
    gsutil -m rsync -x '.*\.DS_Store$|.*\._.*$' \
        "${PROJECT_ROOT}/colab/data_splits/" "${BUCKET}/10_bronze/nih-cxr/manifests/"
    echo "✓ Manifests uploaded to 10_bronze/"
else
    echo "⚠️  No data splits found. Run: python3 scripts/prepare_colab_splits.py"
fi

echo

# 70_cfg: Environment configuration
echo "📦 70_cfg/ - Uploading environment configs..."
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    gsutil cp "${PROJECT_ROOT}/requirements.txt" "${BUCKET}/70_cfg/"
    echo "✓ requirements.txt uploaded"
fi

echo

# 80_docs: Documentation
echo "📦 80_docs/ - Uploading documentation..."
if [ -d "${PROJECT_ROOT}/docs" ]; then
    # Upload key documentation
    for doc in README.md KAGGLE_GUIDE.md COLAB_GUIDE.md; do
        if [ -f "${PROJECT_ROOT}/${doc}" ] || [ -f "${PROJECT_ROOT}/docs/${doc}" ]; then
            gsutil cp "${PROJECT_ROOT}/${doc}" "${BUCKET}/80_docs/" 2>/dev/null || \
                gsutil cp "${PROJECT_ROOT}/docs/${doc}" "${BUCKET}/80_docs/" 2>/dev/null || true
        fi
    done
    echo "✓ Documentation uploaded"
fi

echo

# Set lifecycle rules for 90_tmp/ (auto-delete after 7 days)
echo "🔧 Setting lifecycle rules..."
cat > /tmp/lifecycle.json << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 7,
          "matchesPrefix": ["90_tmp/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json "${BUCKET}"
rm /tmp/lifecycle.json
echo "✓ Lifecycle rules set (90_tmp/ auto-deletes after 7 days)"

echo

# Summary
echo "=========================================="
echo "✅ Medallion Architecture Setup Complete!"
echo "=========================================="
echo

echo "📊 Bucket structure:"
gsutil ls "${BUCKET}/"

echo
echo "💾 Storage usage:"
gsutil du -sh "${BUCKET}/"*

echo
echo "🔗 View in console:"
echo "   https://console.cloud.google.com/storage/browser/nih-xrays"

echo
echo "📝 Next steps:"
echo "   1. Verify structure: gsutil ls -r ${BUCKET}/"
echo "   2. Update Colab notebook to use new paths"
echo "   3. Run training and save to 40_models/"
echo "   4. Download artifacts to local MLflow"
