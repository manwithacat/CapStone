#!/bin/bash
# Download results from Kaggle kernel execution (OPTIMIZED)

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

KAGGLE_USERNAME="${KAGGLE_USERNAME:-yourusername}"
KERNEL_SLUG="cnn-optimized-training"

# Set Kaggle config directory FIRST (before any checks)
export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"

echo "📥 Kaggle Results Downloader"
echo "============================"
echo ""

# Check prerequisites
if [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "❌ Error: kaggle.json not found at $KAGGLE_CONFIG_DIR/kaggle.json"
    exit 1
fi

KERNEL_URL="https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_SLUG"

# Check kernel status
echo "🔍 Checking kernel status..."
STATUS=$(kaggle kernels status "$KAGGLE_USERNAME/$KERNEL_SLUG" 2>/dev/null || echo "error")

echo "   Status: $STATUS"
echo "   URL: $KERNEL_URL"
echo ""

case "$STATUS" in
    *"complete"*)
        echo "✅ Kernel execution complete - ready to download"
        ;;
    *"running"*)
        echo "⏳ Kernel is still running"
        echo ""
        read -p "Download partial results anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled. Check back later when training is complete."
            exit 0
        fi
        ;;
    *"error"*|*"failed"*|*"cancelAcknowledged"*)
        echo "❌ Kernel failed or was cancelled"
        echo "   Check the kernel at: $KERNEL_URL"
        exit 1
        ;;
    *)
        echo "⚠️  Unknown status: $STATUS"
        echo "   Attempting to download anyway..."
        ;;
esac

# Download results
echo ""
echo "📥 Downloading results..."

# Download to temporary location
TEMP_DIR=$(mktemp -d)
kaggle kernels output "$KAGGLE_USERNAME/$KERNEL_SLUG" -p "$TEMP_DIR"

echo "✓ Downloaded to: $TEMP_DIR"
echo ""

# Check what was downloaded
if [ ! -d "$TEMP_DIR/download" ]; then
    echo "⚠️  No 'download' directory found in output"
    echo "   Contents of output:"
    ls -lh "$TEMP_DIR"
    echo ""
    echo "   Manually check: $TEMP_DIR"
    exit 1
fi

# Copy to correct locations in repo
echo "📂 Copying files to repository..."

# Model (check both old and new names)
if [ -f "$TEMP_DIR/download/cnn_optimized_best.keras" ]; then
    cp "$TEMP_DIR/download/cnn_optimized_best.keras" models/saved_models/
    echo "   ✓ Model: models/saved_models/cnn_optimized_best.keras"
elif [ -f "$TEMP_DIR/download/cnn_custom_best.keras" ]; then
    cp "$TEMP_DIR/download/cnn_custom_best.keras" models/saved_models/
    echo "   ✓ Model: models/saved_models/cnn_custom_best.keras (legacy)"
fi

# Reports
if [ -d "$TEMP_DIR/download/reports" ]; then
    cp "$TEMP_DIR/download/reports"/*.json outputs/reports/ 2>/dev/null || true
    cp "$TEMP_DIR/download/reports"/*.csv outputs/reports/ 2>/dev/null || true

    # Keep separate Kaggle versions (check both old and new filenames)
    if [ -f outputs/reports/06c_optimized_training_history.csv ]; then
        cp outputs/reports/06c_optimized_training_history.csv outputs/reports/06c_optimized_training_history_kaggle.csv
        echo "   ✓ Reports: outputs/reports/06c_optimized_*.{json,csv}"
    elif [ -f outputs/reports/06_cnn_training_history.csv ]; then
        cp outputs/reports/06_cnn_training_history.csv outputs/reports/06_cnn_training_history_kaggle.csv
        echo "   ✓ Reports: outputs/reports/06_cnn_*.{json,csv} (legacy)"
    fi
fi

# Figures
if [ -d "$TEMP_DIR/download/figures" ]; then
    cp "$TEMP_DIR/download/figures"/*.png outputs/figures/ 2>/dev/null || true
    echo "   ✓ Figures: outputs/figures/06_cnn_*.png"
fi

# Also keep raw download for reference
DOWNLOAD_ARCHIVE="$HOME/Downloads/results_$(date +%Y%m%d_%H%M%S)"
cp -r "$TEMP_DIR/download" "$DOWNLOAD_ARCHIVE"
echo "   ✓ Raw archive: $DOWNLOAD_ARCHIVE"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Download complete!"
echo ""

# Import MLflow tracking
echo "═══════════════════════════════════════════════════════════"
echo "Importing MLflow Experiment Tracking"
echo "═══════════════════════════════════════════════════════════"
echo ""

MLFLOW_IMPORTED=false

# Method 1: Import from mlruns directory (if MLflow was available on Kaggle)
if [ -d "$DOWNLOAD_ARCHIVE/mlruns" ]; then
    echo "📦 Found mlruns directory (MLflow ran on Kaggle)"
    if [ -f "./scripts/import_kaggle_mlflow.sh" ]; then
        if ./scripts/import_kaggle_mlflow.sh "$DOWNLOAD_ARCHIVE/mlruns"; then
            echo ""
            echo "✅ MLflow runs imported from mlruns/"
            MLFLOW_IMPORTED=true
        else
            echo "⚠️  MLflow import from mlruns failed"
        fi
    fi
fi

# Method 2: Import from CSV/JSON (if MLflow was NOT available on Kaggle)
if [ "$MLFLOW_IMPORTED" = false ]; then
    echo "📊 No mlruns directory - checking for CSV/JSON..."

    # Look for training history CSV and summary JSON
    CSV_FILE=$(find "$DOWNLOAD_ARCHIVE" -name "*training_history*.csv" | head -n 1)
    JSON_FILE=$(find "$DOWNLOAD_ARCHIVE" -name "*training_summary*.json" | head -n 1)

    if [ -n "$CSV_FILE" ] && [ -n "$JSON_FILE" ]; then
        echo "   Found CSV: $(basename "$CSV_FILE")"
        echo "   Found JSON: $(basename "$JSON_FILE")"
        echo ""

        if [ -f "./scripts/csv_to_mlflow.py" ]; then
            echo "🔄 Converting CSV to MLflow run..."
            if python3 scripts/csv_to_mlflow.py "$CSV_FILE" "$JSON_FILE"; then
                echo ""
                echo "✅ MLflow run created from CSV/JSON!"
                MLFLOW_IMPORTED=true
            else
                echo "⚠️  CSV to MLflow conversion failed"
                echo "   You can try manually:"
                echo "   python3 scripts/csv_to_mlflow.py $CSV_FILE $JSON_FILE"
            fi
        else
            echo "💡 CSV to MLflow converter not found"
            echo "   Install and run:"
            echo "   python3 scripts/csv_to_mlflow.py $CSV_FILE $JSON_FILE"
        fi
    else
        echo "⚠️  No training results found (CSV/JSON missing)"
    fi
fi

if [ "$MLFLOW_IMPORTED" = true ]; then
    echo ""
    echo "✅ MLflow tracking data imported!"
    echo "   View in UI: make mlflow-ui"
    echo "   Filter by: tags.platform = 'kaggle'"
else
    echo ""
    echo "⚠️  MLflow import skipped or failed"
    echo "   You can import manually later if needed"
fi

echo ""

echo "📊 Files ready:"
echo "   - Model: models/saved_models/cnn_optimized_best.keras"
echo "   - Reports: outputs/reports/"
echo "   - Figures: outputs/figures/"
echo "   - Archive: $DOWNLOAD_ARCHIVE"
if [ "$MLFLOW_IMPORTED" = true ]; then
    echo "   - MLflow: ✅ Imported to ./mlruns (view with 'make mlflow-ui')"
else
    echo "   - MLflow: ⚠️  Not imported (see warnings above)"
fi
echo ""
echo "🎯 Next steps:"
if [ "$MLFLOW_IMPORTED" = true ]; then
    echo "   - View training in MLflow: make mlflow-ui"
    echo "   - Filter Kaggle runs: tags.platform = 'kaggle'"
fi
echo "   - Compare Kaggle vs local training"
echo "   - Evaluate model performance (notebook 08)"
echo ""
