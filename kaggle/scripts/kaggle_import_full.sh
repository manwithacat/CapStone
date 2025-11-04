#!/bin/bash
# Full Kaggle Training Import Pipeline
#
# Downloads Kaggle kernel output and imports to MLflow with:
# - Correct run duration from Kaggle execution time
# - Model registration for versioning
# - Dataset tracking
#
# Usage:
#   ./scripts/kaggle_import_full.sh <kernel-slug>
#
# Example:
#   ./scripts/kaggle_import_full.sh manwithacat/cnn-optimized-training

set -e

if [ $# -eq 0 ]; then
    echo "❌ Error: No kernel slug specified"
    echo ""
    echo "Usage:"
    echo "  ./scripts/kaggle_import_full.sh <kernel-slug>"
    echo ""
    echo "Example:"
    echo "  ./scripts/kaggle_import_full.sh manwithacat/cnn-optimized-training"
    echo ""
    exit 1
fi

KERNEL_SLUG="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/kaggle_import_${TIMESTAMP}"

echo "================================================================"
echo "Kaggle Training Results → MLflow Import Pipeline"
echo "================================================================"
echo ""
echo "Kernel: $KERNEL_SLUG"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Check kernel status
echo "Step 1/5: Checking kernel status..."
KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status "$KERNEL_SLUG"

STATUS=$(KAGGLE_CONFIG_DIR=".kaggle" kaggle kernels status "$KERNEL_SLUG" | grep -o "KernelWorkerStatus\.[A-Z]*")

if [[ "$STATUS" != "KernelWorkerStatus.COMPLETE" ]]; then
    echo "❌ Error: Kernel is not complete (status: $STATUS)"
    exit 1
fi

echo "✓ Kernel completed successfully"
echo ""

# Step 2: Download kernel output
echo "Step 2/5: Downloading kernel output..."
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

KAGGLE_CONFIG_DIR="/Users/james/CodeInstitute/CapStone/.kaggle" kaggle kernels output "$KERNEL_SLUG"

echo "✓ Downloaded output files"
ls -lh
echo ""

# Step 3: Verify required files exist
echo "Step 3/5: Verifying required files..."

REQUIRED_FILES=(
    "outputs/reports/*training_history.csv"
    "outputs/reports/*training_summary.json"
    "models/*.keras"
)

for pattern in "${REQUIRED_FILES[@]}"; do
    if ! ls $pattern 1> /dev/null 2>&1; then
        echo "❌ Error: Required file not found: $pattern"
        exit 1
    fi
done

# Find the actual files
CSV_FILE=$(ls outputs/reports/*training_history.csv | head -1)
JSON_FILE=$(ls outputs/reports/*training_summary.json | head -1)
MODEL_FILE=$(ls models/*.keras | head -1)
LOG_FILE=$(ls *.log | head -1)

echo "✓ Found required files:"
echo "  CSV:   $CSV_FILE"
echo "  JSON:  $JSON_FILE"
echo "  Model: $MODEL_FILE"
echo "  Log:   $LOG_FILE"
echo ""

# Step 4: Import to MLflow with full metadata
echo "Step 4/5: Importing to MLflow..."
cd /Users/james/CodeInstitute/CapStone

python3 scripts/kaggle_import_enhanced.py \
    "$OUTPUT_DIR/$CSV_FILE" \
    "$OUTPUT_DIR/$JSON_FILE" \
    "$OUTPUT_DIR/$MODEL_FILE" \
    "$OUTPUT_DIR/$LOG_FILE" \
    --experiment "cnn-custom" \
    --kernel-slug "$KERNEL_SLUG"

echo ""

# Step 5: Cleanup
echo "Step 5/5: Cleanup..."
echo "Keep temporary files? [y/N]"
read -t 10 -n 1 -r CLEANUP || CLEANUP="n"
echo

if [[ ! $CLEANUP =~ ^[Yy]$ ]]; then
    rm -rf "$OUTPUT_DIR"
    echo "✓ Cleaned up temporary files"
else
    echo "✓ Keeping files at: $OUTPUT_DIR"
fi

echo ""
echo "================================================================"
echo "✅ Kaggle Import Complete!"
echo "================================================================"
echo ""
echo "💡 View in MLflow UI:"
echo "   make mlflow-ui"
echo "   Filter by: tags.platform = 'kaggle'"
echo ""
