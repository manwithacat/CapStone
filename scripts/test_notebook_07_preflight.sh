#!/bin/bash
# Quick pre-flight test for Notebook 07 before Kaggle push
# Tests both local and Kaggle notebook versions

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Notebook 07 Pre-Flight Test"
echo "=========================================="
echo ""

# Test 1: Local notebook
echo "[1/2] Testing local notebook..."
cd "$PROJECT_ROOT"
timeout 120 papermill \
    jupyter_notebooks/07_transfer_learning.ipynb \
    /tmp/07_test_local.ipynb \
    -p MODELS_TO_TRAIN "['densenet121']" \
    -p RETRAIN_MODELS True \
    > /tmp/07_test_local.log 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Local notebook passed"
else
    echo "  ❌ Local notebook failed"
    tail -20 /tmp/07_test_local.log
    exit 1
fi

# Test 2: Kaggle notebook
echo "[2/2] Testing Kaggle notebook..."
timeout 120 papermill \
    kaggle/kernels/07_transfer_learning/notebook.ipynb \
    /tmp/07_test_kaggle.ipynb \
    -p MODELS_TO_TRAIN "['densenet121']" \
    -p RETRAIN_MODELS True \
    > /tmp/07_test_kaggle.log 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Kaggle notebook passed"
else
    echo "  ❌ Kaggle notebook failed"
    tail -20 /tmp/07_test_kaggle.log
    exit 1
fi

# Check for errors in output notebooks
echo ""
echo "Checking for execution errors..."
python3 << 'EOF'
import json
import sys

errors = []

for nb_path in ['/tmp/07_test_local.ipynb', '/tmp/07_test_kaggle.ipynb']:
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for out in cell['outputs']:
                    if out.get('output_type') == 'error':
                        errors.append((nb_path, idx, out.get('ename'), out.get('evalue')))
    except:
        pass

if errors:
    print(f"❌ Found {len(errors)} errors:")
    for path, cell, ename, eval in errors:
        print(f"  {path} Cell {cell}: {ename}: {eval}")
    sys.exit(1)
else:
    print("✓ No execution errors found")

EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "Safe to push to Kaggle:"
echo "  cd kaggle/kernels/07_transfer_learning"
echo "  KAGGLE_CONFIG_DIR=../../../.kaggle kaggle kernels push"
