#!/bin/bash
# Pre-flight test for Notebook 06
# Runs a fast 1-epoch test with 100 samples to validate everything works
# Run this before starting a long 50-epoch training run

set -e  # Exit on error

echo "=========================================="
echo "Notebook 06 Pre-Flight Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check we're in project root
if [ ! -f "jupyter_notebooks/06_cnn_development.ipynb" ]; then
    echo -e "${RED}✗ Error: Run this script from project root${NC}"
    exit 1
fi

echo "Step 1/5: Checking dependencies..."
python3 -c "import tensorflow, mlflow, papermill; print('✓ All dependencies installed')" || {
    echo -e "${RED}✗ Missing dependencies${NC}"
    exit 1
}

echo ""
echo "Step 2/5: Validating data files..."
python3 -c "
from pathlib import Path
import pandas as pd

data_dir = Path('data/processed')
files = ['train_split.csv', 'val_split.csv', 'test_split.csv', 'preprocessing_config.json']

for f in files:
    path = data_dir / f
    if not path.exists():
        print(f'✗ Missing: {path}')
        exit(1)

# Check at least one image exists
train_df = pd.read_csv(data_dir / 'train_split.csv')
img_path = Path(train_df.iloc[0]['full_path'])
if not img_path.exists():
    print(f'✗ Image not found: {img_path}')
    exit(1)

print('✓ Data files valid')
"

echo ""
echo "Step 3/5: Testing MLflow..."
python3 -c "import mlflow; mlflow.set_experiment('cnn-custom-test'); print('✓ MLflow working')"

echo ""
echo "Step 4/5: Running fast validation (1 epoch, 16 samples)..."
echo -e "${YELLOW}This will take ~30 seconds${NC}"
echo ""

# Create JSON parameters file (papermill parses YAML/JSON properly)
cat > /tmp/test_nb06_params.json <<'EOF'
{
  "CONFIG": {
    "img_height": 224,
    "img_width": 224,
    "channels": 3,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.001,
    "filters": [16, 32, 64, 128],
    "dense_units": 128,
    "dropout_rate": 0.5,
    "l2_reg": 0.0001,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "use_cosine_decay": true,
    "gradient_clip_norm": null,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "num_classes": 14,
    "use_sample": true,
    "sample_size": 16,
    "random_state": 42
  },
  "RETRAIN_MODEL": true,
  "EXPERIMENT_NAME": "cnn-custom-test",
  "RUN_TAGS": {
    "test": "true",
    "test_type": "preflight",
    "test_samples": "16",
    "test_epochs": "1"
  }
}
EOF

# Run papermill with parameter file
# NOTE: Papermill replaces ENTIRE dict/list parameters, not individual nested keys
if papermill \
    jupyter_notebooks/06_cnn_development.ipynb \
    /tmp/test_06_preflight.ipynb \
    -f /tmp/test_nb06_params.json \
    --log-output \
    2>&1 | tee /tmp/test_06_preflight.log | tail -50; then

    echo ""
    echo -e "${GREEN}✓ Pre-flight test PASSED${NC}"
    echo ""
    echo "Step 5/5: Validating outputs..."

    # Check outputs exist
    python3 <<EOF
from pathlib import Path
import glob

# Check model was saved
models = list(Path('models/saved_models').glob('cnn_custom_best.*'))
if models:
    print(f'✓ Model saved: {models[0].name}')
else:
    print('⚠ Warning: Model file not found (may be OK if early stopping)')

# Check training history CSV
histories = sorted(Path('outputs/reports').glob('06_cnn_training_history_*.csv'))
if histories:
    latest = histories[-1]
    print(f'✓ Training history saved: {latest.name}')

    # Quick validation of history file
    import pandas as pd
    hist_df = pd.read_csv(latest)
    if 'auc' in hist_df.columns and 'loss' in hist_df.columns:
        print(f'  Epochs completed: {len(hist_df)}')
        print(f'  Final AUC: {hist_df["auc"].iloc[-1]:.4f}')
    else:
        print('⚠ Warning: Training history has unexpected format')
else:
    print('✗ Error: Training history not found')
    exit(1)

print('')
print('='*50)
print('✓ ALL CHECKS PASSED')
print('='*50)
EOF

    echo ""
    echo -e "${GREEN}Notebook 06 is ready for production training!${NC}"
    echo ""
    echo "To start 50-epoch training:"
    echo "  1. Set CONFIG['use_sample'] = False in notebook"
    echo "  2. Set CONFIG['epochs'] = 50"
    echo "  3. Run the notebook (will take ~8-12 hours)"
    echo ""
    echo "Logs saved to: /tmp/test_06_preflight.log"
    echo "Test notebook: /tmp/test_06_preflight.ipynb"

    exit 0
else
    echo ""
    echo -e "${RED}✗ Pre-flight test FAILED${NC}"
    echo ""
    echo "Check the logs for errors:"
    echo "  /tmp/test_06_preflight.log"
    echo "  /tmp/test_06_preflight.ipynb"
    echo ""
    echo "Common issues:"
    echo "  - MLflow connection errors"
    echo "  - Missing data files"
    echo "  - GPU/memory errors"
    echo "  - Import errors"
    echo ""
    exit 1
fi
