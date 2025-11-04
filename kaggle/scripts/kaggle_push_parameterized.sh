#!/bin/bash
# Kaggle Kernel Push with Papermill Parameters
# Usage: ./scripts/kaggle_push_parameterized.sh <params_json_file>

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <params_json_file>"
    echo "Example: $0 config/test_params/test_params_v8_kaggle.json"
    exit 1
fi

PARAMS_FILE="$1"

# Support both absolute and relative paths
if [ ! -f "$PARAMS_FILE" ]; then
    # Try relative to project root
    PARAMS_FILE="$PROJECT_DIR/$PARAMS_FILE"
fi

# Validate params file exists
if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Parameter file not found: $1"
    echo "Tried: $1 and $PARAMS_FILE"
    exit 1
fi

# Paths
SOURCE_NOTEBOOK="$PROJECT_DIR/jupyter_notebooks/06c_cnn_optimized.ipynb"
TEMP_NOTEBOOK="/tmp/kaggle_parameterized.ipynb"
KAGGLE_NOTEBOOK="$PROJECT_DIR/kaggle_kernel/notebook.ipynb"

echo "========================================"
echo "Kaggle Parameterized Kernel Push"
echo "========================================"
echo "Source: $SOURCE_NOTEBOOK"
echo "Params: $PARAMS_FILE"
echo ""

# Step 1: Apply parameters with papermill
echo "[1/4] Applying parameters with papermill..."
papermill \
    "$SOURCE_NOTEBOOK" \
    "$TEMP_NOTEBOOK" \
    -f "$PARAMS_FILE" \
    --prepare-only

if [ $? -ne 0 ]; then
    echo "Error: Failed to parameterize notebook"
    exit 1
fi

echo "✅ Parameters applied successfully"

# Step 2: Copy to kaggle_kernel
echo ""
echo "[2/4] Copying to kaggle_kernel/..."
cp "$TEMP_NOTEBOOK" "$KAGGLE_NOTEBOOK"
echo "✅ Notebook copied"

# Step 3: Show what parameters were applied
echo ""
echo "[3/4] Applied parameters:"
python3 -c "
import json
with open('$PARAMS_FILE') as f:
    params = json.load(f)
print(f\"  RUN_NAME: {params.get('RUN_NAME', 'N/A')}\")
print(f\"  WARM_START: {params.get('WARM_START', 'N/A')}\")
if params.get('WARM_START'):
    print(f\"  PREVIOUS_MODEL_PATH: {params.get('PREVIOUS_MODEL_PATH', 'N/A')}\")
print(f\"  USE_SAMPLE: {params.get('USE_SAMPLE', 'N/A')}\")
if 'CONFIG' in params:
    print(f\"  Epochs: {params['CONFIG'].get('epochs', 'N/A')}\")
    print(f\"  Learning Rate: {params['CONFIG'].get('learning_rate', 'N/A')}\")
    print(f\"  Dropout: {params['CONFIG'].get('dropout_rate', 'N/A')}\")
    print(f\"  L2 Reg: {params['CONFIG'].get('l2_reg', 'N/A')}\")
"

# Step 4: Push to Kaggle
echo ""
echo "[4/4] Pushing to Kaggle..."
cd "$PROJECT_DIR/kaggle_kernel"
export KAGGLE_CONFIG_DIR="../.kaggle"
kaggle kernels push

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Kernel pushed successfully!"
    echo "Monitor at: https://www.kaggle.com/code/manwithacat/cnn-optimized-training"
else
    echo "❌ Failed to push kernel"
    exit 1
fi

# Cleanup
rm -f "$TEMP_NOTEBOOK"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
