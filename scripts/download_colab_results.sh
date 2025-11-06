#!/usr/bin/env bash
#
# Download Colab training results from GCS
#
# This script downloads models, artifacts, and logs from GCS medallion architecture
# to local colab/results directory

set -e

BUCKET="gs://nih-xrays"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/colab/results"

echo "📥 Downloading Colab Training Results from GCS"
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

# Create destination directory
mkdir -p "${DEST_DIR}"

# Download models
echo "📦 Downloading models from 40_models/..."
mkdir -p "${DEST_DIR}/models"
if gsutil ls "${BUCKET}/40_models/nih-cxr/" &> /dev/null; then
    gsutil -m rsync -r "${BUCKET}/40_models/nih-cxr/" "${DEST_DIR}/models/"
    echo "✓ Models downloaded to ${DEST_DIR}/models/"
else
    echo "⚠️  No models found at ${BUCKET}/40_models/nih-cxr/"
fi

echo

# Download artifacts
echo "📊 Downloading artifacts from 50_artifacts/..."
mkdir -p "${DEST_DIR}/artifacts"
if gsutil ls "${BUCKET}/50_artifacts/nih-cxr/" &> /dev/null; then
    gsutil -m rsync -r "${BUCKET}/50_artifacts/nih-cxr/" "${DEST_DIR}/artifacts/"
    echo "✓ Artifacts downloaded to ${DEST_DIR}/artifacts/"
else
    echo "⚠️  No artifacts found at ${BUCKET}/50_artifacts/nih-cxr/"
fi

echo

# Download logs
echo "📝 Downloading logs from 60_logs/..."
mkdir -p "${DEST_DIR}/logs"
if gsutil ls "${BUCKET}/60_logs/nih-cxr/" &> /dev/null; then
    gsutil -m rsync -r "${BUCKET}/60_logs/nih-cxr/" "${DEST_DIR}/logs/"
    echo "✓ Logs downloaded to ${DEST_DIR}/logs/"
else
    echo "⚠️  No logs found at ${BUCKET}/60_logs/nih-cxr/"
fi

echo
echo "=============================================="
echo "✅ Download Complete!"
echo "=============================================="
echo

# Show summary
echo "📊 Downloaded files:"
if [ -d "${DEST_DIR}" ]; then
    find "${DEST_DIR}" -type f | wc -l | xargs echo "  Total files:"
    du -sh "${DEST_DIR}" | awk '{print "  Total size:", $1}'
    echo

    if [ -d "${DEST_DIR}/models" ]; then
        echo "  Models:"
        find "${DEST_DIR}/models" -name "*.keras" -o -name "*.h5" | while read -r f; do
            size=$(du -h "$f" | awk '{print $1}')
            basename=$(basename "$f")
            echo "    - $basename ($size)"
        done
    fi

    echo

    if [ -d "${DEST_DIR}/artifacts" ]; then
        echo "  Artifacts:"
        find "${DEST_DIR}/artifacts" -name "*.csv" -o -name "*.json" -o -name "*.png" | wc -l | xargs echo "    Files:"
    fi
fi

echo
echo "📁 Results saved to: ${DEST_DIR}"
echo
echo "🔧 Next steps:"
echo "   1. Review results in ${DEST_DIR}"
echo "   2. Run: scripts/import_colab_mlflow.sh"
echo "   3. View in MLflow: http://localhost:5001"
