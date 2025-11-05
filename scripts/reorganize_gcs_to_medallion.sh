#!/usr/bin/env bash
#
# Reorganize existing GCS uploads to Medallion architecture
#
# This script moves files you've already uploaded into the proper
# medallion structure without re-uploading.

set -e

BUCKET="gs://nih-xrays"

echo "🔄 Reorganizing GCS bucket to Medallion architecture"
echo "====================================================="
echo

# Check what's currently in the bucket
echo "📊 Current bucket contents:"
gsutil ls "${BUCKET}/" || {
    echo "❌ Cannot access bucket. Authenticate first:"
    echo "   gcloud auth application-default login"
    exit 1
}

echo
echo "🔍 Analyzing current structure..."
echo

# Get list of current top-level items
CURRENT_ITEMS=$(gsutil ls "${BUCKET}/" | sed "s|${BUCKET}/||" | sed 's|/$||')

echo "Found:"
echo "${CURRENT_ITEMS}"
echo

# Move data_splits/ to 10_bronze/nih-cxr/manifests/
if echo "${CURRENT_ITEMS}" | grep -q "data_splits"; then
    echo "📦 Moving data_splits/ → 10_bronze/nih-cxr/manifests/"
    gsutil -m mv "${BUCKET}/data_splits/*" "${BUCKET}/10_bronze/nih-cxr/manifests/" || true
    gsutil rm "${BUCKET}/data_splits/.gitkeep" 2>/dev/null || true
    echo "✓ Manifests moved"
else
    echo "⏭️  No data_splits/ to move"
fi

echo

# Move images/ to 00_raw/nih-cxr/images/
if echo "${CURRENT_ITEMS}" | grep -q "images"; then
    echo "📦 Moving images/ → 00_raw/nih-cxr/images/"
    echo "   This may take a few minutes..."
    gsutil -m mv "${BUCKET}/images/*" "${BUCKET}/00_raw/nih-cxr/images/" || true
    gsutil rm "${BUCKET}/images/.gitkeep" 2>/dev/null || true
    echo "✓ Images moved"
else
    echo "⏭️  No images/ to move"
fi

echo

# Create empty .gitkeep files in other layers
echo "📁 Creating remaining layer directories..."

cat > /tmp/.gitkeep << 'EOF'
# Placeholder to maintain directory structure
EOF

for layer in 20_silver 30_gold 40_models 50_artifacts 60_logs 70_cfg 80_docs 90_tmp; do
    if ! echo "${CURRENT_ITEMS}" | grep -q "^${layer}$"; then
        echo "  Creating ${layer}/"
        gsutil cp /tmp/.gitkeep "${BUCKET}/${layer}/.gitkeep" 2>/dev/null || true
    else
        echo "  ${layer}/ already exists"
    fi
done

rm /tmp/.gitkeep
echo "✓ Layer structure complete"

echo

# Set lifecycle rules for 90_tmp/
echo "🔧 Setting lifecycle rules (90_tmp/ auto-deletes after 7 days)..."

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
echo "✓ Lifecycle rules set"

echo

# Final structure
echo "=========================================="
echo "✅ Reorganization Complete!"
echo "=========================================="
echo

echo "📊 New bucket structure:"
gsutil ls "${BUCKET}/"

echo
echo "💾 Storage by layer:"
gsutil du -sh "${BUCKET}/"*

echo
echo "🔗 View in console:"
echo "   https://console.cloud.google.com/storage/browser/nih-xrays"

echo
echo "📝 Verify:"
echo "   gsutil ls -r ${BUCKET}/00_raw/"
echo "   gsutil ls -r ${BUCKET}/10_bronze/"
