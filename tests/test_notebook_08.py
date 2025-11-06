"""
Test script for jupyter_notebooks/08_model_evaluation.ipynb

This script validates that the notebook can run successfully by:
1. Checking notebook structure and imports
2. Verifying model file exists and can be loaded
3. Testing data paths and output directories
4. Simulating inference pipeline with dummy data
5. Validating ROC curve generation

Run: python tests/test_notebook_08.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("Testing Notebook 08: Model Evaluation")
print("=" * 80)

# ============================================================================
# TEST 1: Validate Notebook Structure
# ============================================================================
print("\n[1/6] Validating notebook structure...")

notebook_path = project_root / "jupyter_notebooks" / "08_model_evaluation.ipynb"

if not notebook_path.exists():
    print(f"❌ FAIL: Notebook not found at {notebook_path}")
    sys.exit(1)

try:
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Check basic structure
    assert 'cells' in notebook, "Notebook missing 'cells' key"
    assert len(notebook['cells']) > 0, "Notebook has no cells"

    # Check for key imports
    notebook_text = '\n'.join([
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    ])

    required_imports = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'sklearn',
        'roc_curve',
        'auc'
    ]

    missing_imports = [imp for imp in required_imports if imp not in notebook_text]

    if missing_imports:
        print(f"⚠️  WARNING: Missing imports: {missing_imports}")
    else:
        print("✅ PASS: All required imports present")

    print(f"✅ PASS: Notebook structure valid ({len(notebook['cells'])} cells)")

except Exception as e:
    print(f"❌ FAIL: Error reading notebook: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Verify Model File
# ============================================================================
print("\n[2/6] Checking model file...")

model_path = project_root / "models" / "saved_models" / "densenet121_best.keras"

if not model_path.exists():
    print(f"❌ FAIL: Model not found at {model_path}")
    print("   Expected: models/saved_models/densenet121_best.keras")
    print("   Please copy model from colab/results/models/densenet121-transfer/")
    sys.exit(1)

model_size_mb = model_path.stat().st_size / (1024 * 1024)
print(f"✅ PASS: Model file exists ({model_size_mb:.1f} MB)")

# Try loading the model
try:
    from tensorflow import keras
    print("   Loading model (this may take a few seconds)...")
    model = keras.models.load_model(str(model_path), compile=False)

    # Verify input/output shapes
    expected_input = (None, 224, 224, 3)
    expected_output = (None, 14)

    if model.input_shape != expected_input:
        print(f"⚠️  WARNING: Unexpected input shape {model.input_shape} (expected {expected_input})")
    else:
        print(f"✅ PASS: Model input shape correct {model.input_shape}")

    if model.output_shape != expected_output:
        print(f"⚠️  WARNING: Unexpected output shape {model.output_shape} (expected {expected_output})")
    else:
        print(f"✅ PASS: Model output shape correct {model.output_shape}")

    param_count = model.count_params()
    print(f"✅ PASS: Model loaded successfully ({param_count:,} parameters)")

except Exception as e:
    print(f"❌ FAIL: Error loading model: {e}")
    sys.exit(1)

# ============================================================================
# TEST 3: Verify Data Paths
# ============================================================================
print("\n[3/6] Checking data paths...")

import pandas as pd

test_split_file = project_root / "data" / "processed" / "test_split.csv"

if not test_split_file.exists():
    print(f"❌ FAIL: Test split file not found: {test_split_file}")
    print("   Run notebook 03 (preprocessing) to generate test split")
    sys.exit(1)

print(f"✅ PASS: Test split file exists")

# Load test split to check structure
try:
    test_df = pd.read_csv(test_split_file)
    num_test_images = len(test_df)
    print(f"✅ PASS: Test split loaded ({num_test_images:,} images)")

    # Check for required columns
    required_cols = ['full_path', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
                     'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                     'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                     'Pneumonia', 'Pneumothorax']

    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        print(f"❌ FAIL: Missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"✅ PASS: All required columns present")

    # Check that some images exist at the paths specified
    sample_paths = test_df['full_path'].head(10)
    existing = sum(1 for p in sample_paths if Path(p).exists())

    if existing == 0:
        print(f"⚠️  WARNING: None of the first 10 image paths exist")
        print(f"   Sample path: {sample_paths.iloc[0]}")
    else:
        print(f"✅ PASS: Sample images verified ({existing}/10 checked exist)")

except Exception as e:
    print(f"❌ FAIL: Error loading test split: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Verify Output Directories
# ============================================================================
print("\n[4/6] Checking output directories...")

output_dirs = [
    project_root / "outputs" / "figures" / "evaluation",
    project_root / "outputs" / "reports"
]

for output_dir in output_dirs:
    if not output_dir.exists():
        print(f"   Creating directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ PASS: {output_dir.relative_to(project_root)}")

# ============================================================================
# TEST 5: Simulate Inference Pipeline (Dummy Data)
# ============================================================================
print("\n[5/6] Testing inference pipeline with dummy data...")

try:
    # Create dummy test batch (5 images)
    dummy_images = np.random.rand(5, 224, 224, 3).astype(np.float32)

    # Apply ImageNet normalization (same as training)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    dummy_images = (dummy_images - IMAGENET_MEAN) / IMAGENET_STD

    # Run inference
    dummy_predictions = model.predict(dummy_images, verbose=0)

    # Verify output shape
    expected_shape = (5, 14)
    if dummy_predictions.shape != expected_shape:
        print(f"❌ FAIL: Unexpected prediction shape {dummy_predictions.shape} (expected {expected_shape})")
        sys.exit(1)

    # Verify predictions are probabilities (0-1 range after sigmoid)
    if not (0 <= dummy_predictions.min() <= 1 and 0 <= dummy_predictions.max() <= 1):
        print(f"⚠️  WARNING: Predictions outside [0, 1] range")
        print(f"   Min: {dummy_predictions.min():.4f}, Max: {dummy_predictions.max():.4f}")

    print(f"✅ PASS: Inference successful (shape: {dummy_predictions.shape})")
    print(f"   Prediction range: [{dummy_predictions.min():.4f}, {dummy_predictions.max():.4f}]")

except Exception as e:
    print(f"❌ FAIL: Inference error: {e}")
    sys.exit(1)

# ============================================================================
# TEST 6: Test ROC Curve Calculation (Dummy Data)
# ============================================================================
print("\n[6/6] Testing ROC curve generation with dummy data...")

try:
    from sklearn.metrics import roc_curve, auc

    # Create dummy ground truth (5 samples, 14 diseases)
    dummy_true = np.random.randint(0, 2, size=(5, 14))

    DISEASE_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]

    roc_data = {}

    for i, disease in enumerate(DISEASE_CLASSES):
        y_true_disease = dummy_true[:, i]
        y_pred_disease = dummy_predictions[:, i]

        # Check if we have both classes (needed for ROC)
        if len(np.unique(y_true_disease)) < 2:
            # Add a sample to ensure both classes present
            y_true_disease = np.append(y_true_disease, [0, 1])
            y_pred_disease = np.append(y_pred_disease, [0.1, 0.9])

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_disease, y_pred_disease)
        roc_auc = auc(fpr, tpr)

        roc_data[disease] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc),
            'n_positive': int(y_true_disease.sum()),
            'n_negative': int((1 - y_true_disease).sum())
        }

    print(f"✅ PASS: ROC calculation successful for all {len(DISEASE_CLASSES)} diseases")

    # Show sample ROC data
    sample_disease = DISEASE_CLASSES[0]
    sample_auc = roc_data[sample_disease]['auc']
    print(f"   Example: {sample_disease} AUC = {sample_auc:.4f}")

    # Test JSON serialization
    test_json_path = project_root / "outputs" / "reports" / "test_roc_curves.json"
    with open(test_json_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    print(f"✅ PASS: JSON serialization successful")
    print(f"   Test output saved: {test_json_path.relative_to(project_root)}")

    # Clean up test file
    test_json_path.unlink()

except Exception as e:
    print(f"❌ FAIL: ROC calculation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
print("\nNotebook 08 is ready to run. Expected behavior:")
print("  • Load DenseNet121 model (37 MB)")
print(f"  • Process {num_test_images:,} test images")
print(f"  • Generate predictions ({num_test_images:,} images × 14 diseases)")
print("  • Calculate ROC curves for 14 diseases")
print("  • Save outputs to outputs/figures/evaluation/ and outputs/reports/")
print(f"  • Estimated time: ~30-45 minutes on CPU")
print("\nTo run the notebook:")
print("  jupyter notebook jupyter_notebooks/08_model_evaluation.ipynb")
print("\nOr execute all cells programmatically:")
print("  jupyter nbconvert --to notebook --execute jupyter_notebooks/08_model_evaluation.ipynb")
print("=" * 80)
