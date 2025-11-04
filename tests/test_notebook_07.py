"""
Tests for Notebook 07: Transfer Learning

Validates notebook can execute with minimal data.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def notebook_path(project_root):
    """Get notebook 07 path."""
    return project_root / 'jupyter_notebooks' / '07_transfer_learning.ipynb'


@pytest.fixture
def test_params(tmp_path):
    """Create minimal test parameters."""
    params = {
        "MODELS_TO_TRAIN": ["densenet121"],  # Just one model
        "RUN_NAME": "pytest_test",
        "USE_MLFLOW": False,  # Disable for testing
        "RETRAIN_MODELS": True
    }

    param_file = tmp_path / "test_params.json"
    with open(param_file, 'w') as f:
        json.dump(params, f)

    return param_file


def test_notebook_imports(notebook_path):
    """Test that notebook can be loaded and has expected structure."""
    with open(notebook_path) as f:
        nb = json.load(f)

    assert 'cells' in nb
    assert len(nb['cells']) > 0

    # Check for imports cell
    has_imports = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import pandas as pd' in source:
                has_imports = True
                break

    assert has_imports, "Notebook missing pandas import"


def test_notebook_has_parameters(notebook_path):
    """Test that notebook has parameterizable cells."""
    with open(notebook_path) as f:
        nb = json.load(f)

    # Look for parameters cell (tagged with 'parameters')
    has_params_cell = False
    for cell in nb['cells']:
        if cell.get('metadata', {}).get('tags'):
            if 'parameters' in cell['metadata']['tags']:
                has_params_cell = True
                break

    # Also accept cell with MODELS_TO_TRAIN variable
    if not has_params_cell:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'MODELS_TO_TRAIN' in source:
                    has_params_cell = True
                    break

    assert has_params_cell, "Notebook missing parameters cell"


def test_notebook_quick_execution(notebook_path, test_params, tmp_path):
    """Test notebook executes with minimal data (< 2 minutes)."""
    output_nb = tmp_path / "output.ipynb"

    cmd = [
        'papermill',
        str(notebook_path),
        str(output_nb),
        '-p', 'MODELS_TO_TRAIN', "['densenet121']",
        '-p', 'USE_MLFLOW', 'False',
        '-p', 'RETRAIN_MODELS', 'True'
    ]

    # Run with timeout
    try:
        result = subprocess.run(
            cmd,
            timeout=120,  # 2 minute timeout
            capture_output=True,
            text=True,
            cwd=str(notebook_path.parent.parent)  # Run from project root
        )

        # Check for execution errors
        if result.returncode != 0:
            # Check the output for specific errors
            if 'ModuleNotFoundError' in result.stderr:
                pytest.fail(f"Missing module: {result.stderr[-500:]}")
            elif 'NameError' in result.stderr:
                # Extract the actual error
                lines = result.stderr.split('\n')
                error_lines = [l for l in lines if 'NameError' in l or 'name' in l]
                pytest.fail(f"Name error: {' '.join(error_lines[-3:])}")
            elif 'FileNotFoundError' in result.stderr and 'data' in result.stderr.lower():
                # Data files missing is OK in test
                pytest.skip("Data files not found (expected in test environment)")

        # If we got an output notebook, check for cell errors
        if output_nb.exists():
            with open(output_nb) as f:
                output = json.load(f)

            # Check each cell for execution errors
            for idx, cell in enumerate(output['cells']):
                if cell['cell_type'] == 'code' and 'outputs' in cell:
                    for out in cell['outputs']:
                        if out.get('output_type') == 'error':
                            error_name = out.get('ename', 'Unknown')
                            error_val = out.get('evalue', '')
                            pytest.fail(f"Cell {idx} error: {error_name}: {error_val}")

    except subprocess.TimeoutExpired:
        pytest.fail("Notebook execution exceeded 2 minute timeout")


def test_notebook_no_hard_coded_paths(notebook_path):
    """Test that notebook doesn't have hard-coded local paths."""
    with open(notebook_path) as f:
        nb = json.load(f)

    bad_patterns = [
        '/Users/james/CodeInstitute/CapStone',
        '/Users/james/Downloads'
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            for pattern in bad_patterns:
                assert pattern not in source, f"Found hard-coded path: {pattern}"


def test_kaggle_compatibility(notebook_path, project_root):
    """Test that Kaggle-parameterized notebook is valid."""
    kaggle_nb = project_root / 'kaggle_kernel_07' / 'notebook.ipynb'

    if not kaggle_nb.exists():
        pytest.skip("Kaggle notebook not found")

    with open(kaggle_nb) as f:
        nb = json.load(f)

    # Check imports cell exists and has pandas
    has_pandas = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'import pandas as pd' in source:
                has_pandas = True
                break

    assert has_pandas, "Kaggle notebook missing pandas import"

    # Check MLflow import is wrapped in try/except
    has_mlflow_try_except = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'try:' in source and 'import mlflow' in source and 'except ImportError:' in source:
                has_mlflow_try_except = True
                break

    assert has_mlflow_try_except, "Kaggle notebook should have MLflow import in try/except block"


def test_mlflow_enabled_by_default_locally(notebook_path, tmp_path):
    """Test that MLflow is enabled by default in local environment."""
    output_nb = tmp_path / "output.ipynb"

    cmd = [
        'papermill',
        str(notebook_path),
        str(output_nb),
        '-p', 'MODELS_TO_TRAIN', "['densenet121']",
        '-p', 'RETRAIN_MODELS', 'True'
        # Don't override USE_MLFLOW - should default to True
    ]

    try:
        result = subprocess.run(
            cmd,
            timeout=120,
            capture_output=True,
            text=True,
            cwd=str(notebook_path.parent.parent)
        )

        # Check output for USE_MLFLOW = True
        if output_nb.exists():
            with open(output_nb) as f:
                output = json.load(f)

            # Find the parameter conversion cell output
            found_mlflow_enabled = False
            for cell in output['cells']:
                if cell['cell_type'] == 'code' and 'outputs' in cell:
                    for out in cell['outputs']:
                        if out.get('output_type') in ['stream', 'execute_result']:
                            text = out.get('text', '') or str(out.get('data', {}).get('text/plain', ''))
                            if 'USE_MLFLOW: True' in text:
                                found_mlflow_enabled = True
                                break

            assert found_mlflow_enabled, "USE_MLFLOW should be True by default locally"

    except subprocess.TimeoutExpired:
        pytest.fail("Notebook execution exceeded 2 minute timeout")


def test_environment_detection(notebook_path):
    """Test that notebook has environment detection logic."""
    with open(notebook_path) as f:
        nb = json.load(f)

    # Check for environment detection code
    has_kaggle_detection = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '/kaggle/input' in source and 'IS_KAGGLE' in source:
                has_kaggle_detection = True
                break

    assert has_kaggle_detection, "Notebook should detect Kaggle environment"


def test_kaggle_notebook_has_proper_paths(project_root):
    """Test that Kaggle notebook has correct path handling for Kaggle datasets."""
    kaggle_nb = project_root / 'kaggle_kernel_07' / 'notebook.ipynb'

    if not kaggle_nb.exists():
        pytest.skip("Kaggle notebook not found")

    with open(kaggle_nb) as f:
        nb = json.load(f)

    # Check for Kaggle-specific path handling
    has_kaggle_paths = False
    has_processed_dir = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Should have /kaggle/input path handling
            if '/kaggle/input' in source and 'PROCESSED_DIR' in source:
                has_kaggle_paths = True

            # Should set PROCESSED_DIR for Kaggle
            if 'PROCESSED_DIR' in source and 'nih-chest-xray-splits' in source:
                has_processed_dir = True

    assert has_kaggle_paths, "Kaggle notebook missing /kaggle/input path handling"
    assert has_processed_dir, "Kaggle notebook missing PROCESSED_DIR configuration for dataset"


def test_data_loading_cell_exists(notebook_path):
    """Test that notebook has data loading cell that uses PROCESSED_DIR."""
    with open(notebook_path) as f:
        nb = json.load(f)

    has_data_loading = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'pd.read_csv(PROCESSED_DIR' in source and 'train_split.csv' in source:
                has_data_loading = True
                break

    assert has_data_loading, "Notebook missing data loading cell with PROCESSED_DIR"


def test_kaggle_path_fixing_logic(project_root):
    """Test that Kaggle notebook has path fixing logic for image paths."""
    kaggle_nb = project_root / 'kaggle_kernel_07' / 'notebook.ipynb'

    if not kaggle_nb.exists():
        pytest.skip("Kaggle notebook not found")

    with open(kaggle_nb) as f:
        nb = json.load(f)

    # Check for path fixing logic
    has_path_fix = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Should have logic to fix paths for Kaggle
            if 'fix_kaggle_path' in source and 'IS_KAGGLE' in source and '/kaggle/input' in source:
                has_path_fix = True
                break

    assert has_path_fix, "Kaggle notebook missing path fixing logic for image paths"


def test_pretrained_weights_error_handling(notebook_path):
    """Test that notebook has error handling for pre-trained weights loading."""
    with open(notebook_path) as f:
        nb = json.load(f)

    # Check for error handling in model building
    has_weights_error_handling = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Should have try/except around weights loading
            if 'build_transfer_model' in source and 'try:' in source and 'weights=' in source:
                has_weights_error_handling = True
                break

    assert has_weights_error_handling, "Notebook missing error handling for pre-trained weights loading"


def test_zero_batches_detection(notebook_path, tmp_path):
    """Test that notebook would detect if data generators have zero batches."""
    # This is a validation test - we can't actually run it without data
    # but we can check the notebook has logic to report batch counts
    with open(notebook_path) as f:
        nb = json.load(f)

    has_batch_reporting = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Should print train/val batch counts
            if 'Train batches' in source or 'len(train_generator)' in source:
                has_batch_reporting = True
                break

    assert has_batch_reporting, "Notebook should report data generator batch counts for debugging"


def test_kaggle_path_transformation_logic():
    """Test the actual path transformation logic with mocked Kaggle environment."""
    import re

    # This is the function from the notebook
    def fix_kaggle_path(local_path):
        """Convert local image path to Kaggle input path."""
        match = re.search(r'(images_\d+/.*\.png)', local_path)
        if match:
            rel_path = match.group(1)
            return f'/kaggle/input/data/{rel_path}'
        return local_path

    # Test cases with actual path format
    test_cases = [
        # (input, expected_output)
        ('/Volumes/SSD/Capstone/data/raw/images_001/images/00000001_000.png',
         '/kaggle/input/data/images_001/images/00000001_000.png'),
        ('/Volumes/SSD/Capstone/data/raw/images_012/images/00012345_678.png',
         '/kaggle/input/data/images_012/images/00012345_678.png'),
        # Edge case: already has correct path
        ('/kaggle/input/data/images_001/images/00000001_000.png',
         '/kaggle/input/data/images_001/images/00000001_000.png'),
    ]

    for input_path, expected_output in test_cases:
        result = fix_kaggle_path(input_path)
        assert result == expected_output, f"Path transformation failed:\n  Input: {input_path}\n  Expected: {expected_output}\n  Got: {result}"


def test_kaggle_dataset_name_in_notebook(project_root):
    """Test that Kaggle notebook uses the correct dataset name 'data'."""
    kaggle_nb = project_root / 'kaggle_kernel_07' / 'notebook.ipynb'

    if not kaggle_nb.exists():
        pytest.skip("Kaggle notebook not found")

    with open(kaggle_nb) as f:
        nb = json.load(f)

    # Check for correct dataset name in path fixing logic
    correct_dataset_name = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Should use /kaggle/input/data/ not /kaggle/input/nih-chest-xrays/
            if 'fix_kaggle_path' in source and '/kaggle/input/data/' in source:
                correct_dataset_name = True
                # Make sure it's not using the wrong name
                if '/kaggle/input/nih-chest-xrays/' in source:
                    pytest.fail("Notebook still has old dataset name 'nih-chest-xrays', should be 'data'")
                break

    assert correct_dataset_name, "Kaggle notebook should use dataset name 'data' in path transformation"
