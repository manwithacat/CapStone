"""
Centralized path management for the project.

This module provides a single source of truth for all project paths,
automatically detecting the project root regardless of where scripts/notebooks are run from.
"""

from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by looking for marker files.

    Args:
        start_path: Starting directory for search (defaults to current file location)

    Returns:
        Path object pointing to project root

    Raises:
        RuntimeError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path

    # Look for marker files that indicate project root
    markers = ['setup.py', 'README.md', '.git', 'CLAUDE.md']

    # Search up to 5 levels
    for _ in range(5):
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    raise RuntimeError(
        f"Could not find project root from {start_path}. "
        f"Looking for one of: {markers}"
    )


# ===== PROJECT PATHS =====

PROJECT_ROOT = find_project_root()

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
REPORTS_DIR = OUTPUTS_DIR / 'reports'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'

# Config directories
CONFIG_DIR = PROJECT_ROOT / 'config'
TEST_PARAMS_DIR = CONFIG_DIR / 'test_params'

# Notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / 'jupyter_notebooks'

# Scripts directory
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

# MLflow
MLFLOW_DB_PATH = PROJECT_ROOT / 'mlflow.db'
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"


def ensure_dirs_exist():
    """Create all necessary directories if they don't exist."""
    dirs_to_create = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR,
        MODELS_DIR, SAVED_MODELS_DIR,
        CONFIG_DIR, TEST_PARAMS_DIR,
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_image_path(image_index: str, check_exists: bool = False) -> Optional[Path]:
    """
    Find the full path to an image given its index.

    Searches through all image subdirectories in data/raw/.

    Args:
        image_index: Image filename (e.g., "00000001_000.png")
        check_exists: If True, verify file exists before returning

    Returns:
        Path to image, or None if not found (when check_exists=True)
    """
    # Images are in subdirectories like images_001/images/, images_002/images/, etc.
    image_dirs = sorted(RAW_DATA_DIR.glob('images_*/images'))

    for img_dir in image_dirs:
        img_path = img_dir / image_index
        if not check_exists or img_path.exists():
            return img_path

    return None if check_exists else RAW_DATA_DIR / 'images_001' / 'images' / image_index


# ===== CONVENIENCE FUNCTIONS =====

def get_split_path(split: str, validated: bool = False) -> Path:
    """
    Get path to train/val/test split CSV file.

    Args:
        split: One of 'train', 'val', 'test'
        validated: If True, return path to validated split (if exists)

    Returns:
        Path to split CSV file
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    if validated and split == 'train':
        validated_path = PROCESSED_DATA_DIR / 'train_split_validated.csv'
        if validated_path.exists():
            return validated_path

    return PROCESSED_DATA_DIR / f'{split}_split.csv'


def get_preprocessing_config() -> Path:
    """Get path to preprocessing configuration JSON."""
    return PROCESSED_DATA_DIR / 'preprocessing_config.json'


def get_class_weights() -> Path:
    """Get path to class weights JSON."""
    return PROCESSED_DATA_DIR / 'class_weights.json'


def get_test_params(param_name: str) -> Path:
    """
    Get path to test parameters JSON file.

    Args:
        param_name: Name of parameter file (with or without .json extension)

    Returns:
        Path to parameter file
    """
    if not param_name.endswith('.json'):
        param_name = f'{param_name}.json'

    return TEST_PARAMS_DIR / param_name


if __name__ == '__main__':
    # Print all paths for debugging
    print("="*60)
    print("PROJECT PATHS")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Raw Data: {RAW_DATA_DIR}")
    print(f"Processed Data: {PROCESSED_DATA_DIR}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Models: {SAVED_MODELS_DIR}")
    print(f"Config: {CONFIG_DIR}")
    print(f"Test Params: {TEST_PARAMS_DIR}")
    print(f"MLflow DB: {MLFLOW_DB_PATH}")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print("\nAll paths exist:", all([
        PROJECT_ROOT.exists(),
        DATA_DIR.exists(),
        RAW_DATA_DIR.exists(),
        PROCESSED_DATA_DIR.exists(),
        OUTPUTS_DIR.exists(),
        MODELS_DIR.exists(),
    ]))
