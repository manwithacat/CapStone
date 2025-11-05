#!/usr/bin/env bash
#
# Upload NIH Chest X-Ray data to Google Cloud Storage
#
# Prerequisites:
#   - gcloud CLI installed: https://cloud.google.com/sdk/docs/install
#   - Authenticated: gcloud auth login
#   - Bucket created: nih-xrays

set -e

BUCKET_NAME="nih-xrays"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "📦 Uploading NIH Chest X-Ray Data to GCS"
echo "=========================================="
echo

# Check gcloud is installed
if ! command -v gsutil &> /dev/null; then
    echo "❌ gsutil not found. Install Google Cloud SDK:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gsutil ls gs://${BUCKET_NAME} &> /dev/null; then
    echo "❌ Cannot access gs://${BUCKET_NAME}"
    echo "   Run: gcloud auth login"
    exit 1
fi

echo "✓ Authenticated and bucket accessible"
echo

# Upload data splits (small, fast)
echo "📤 Uploading data splits..."
gsutil -m rsync -r -x '.*\.DS_Store$|.*\._.*$' \
    "${PROJECT_ROOT}/colab/data_splits/" "gs://${BUCKET_NAME}/data_splits/"
echo "✓ Data splits uploaded"
echo

# Upload NIH images (large, slow - ~47 GB)
echo "📤 Uploading NIH Chest X-Ray images..."
echo "⚠️  This will take 30-60 minutes depending on your connection"
echo "   Size: ~47 GB (112,120 images)"
echo

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping image upload"
    exit 0
fi

# Upload images with parallel transfers (excluding macOS metadata)
gsutil -m -o "GSUtil:parallel_process_count=8" \
    -o "GSUtil:parallel_thread_count=16" \
    rsync -r -x '.*\.DS_Store$|.*\._.*$|.*\.Spotlight-V100.*$|.*\.Trashes.*$' \
    "${PROJECT_ROOT}/data/raw/" "gs://${BUCKET_NAME}/images/"

echo
echo "✅ Upload complete!"
echo
echo "📊 Bucket contents:"
gsutil du -sh "gs://${BUCKET_NAME}/*"
echo
echo "🔗 Bucket URL: https://console.cloud.google.com/storage/browser/${BUCKET_NAME}"
