#!/usr/bin/env python3
"""
Prepare Jupyter notebooks for automated testing with pytest-nbmake.

This script:
1. Adds deterministic random seeds to notebooks that need them
2. Tags notebooks appropriately (slow, data_download, etc.)
3. Ensures notebooks can run in CI/CD
"""

import json
from pathlib import Path

# Notebook configurations
NOTEBOOK_CONFIG = {
    "01_data_collection_and_setup.ipynb": {
        "tags": ["slow", "data_download"],
        "skip_in_ci": True,  # Downloads ~47GB of data
        "needs_seed": False,  # No randomness
        "description": "Downloads NIH dataset (47GB) - skip in CI"
    },
    "02_exploratory_data_analysis.ipynb": {
        "tags": [],
        "skip_in_ci": False,
        "needs_seed": True,  # Uses sampling/visualization
        "description": "EDA and statistical analysis"
    },
    "03_image_preprocessing.ipynb": {
        "tags": ["slow"],  # Processes images
        "skip_in_ci": False,
        "needs_seed": True,  # Already has seeds
        "description": "Image preprocessing pipeline"
    },
    "04_hypothesis_testing.ipynb": {
        "tags": [],
        "skip_in_ci": False,
        "needs_seed": True,  # Statistical tests
        "description": "Statistical hypothesis testing"
    }
}

SEED_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "tags": ["setup", "deterministic"]
    },
    "outputs": [],
    "source": [
        "# Set deterministic random seeds for reproducible results\n",
        "import os\n",
        "import random\n",
        "import numpy as np\n",
        "\n",
        "RANDOM_SEED = 42\n",
        "\n",
        "# Python random\n",
        "random.seed(RANDOM_SEED)\n",
        "\n",
        "# NumPy random\n",
        "np.random.seed(RANDOM_SEED)\n",
        "\n",
        "# Python hash seed (for reproducibility)\n",
        "os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)\n",
        "\n",
        "# TensorFlow random (if imported later)\n",
        "try:\n",
        "    import tensorflow as tf\n",
        "    tf.random.set_seed(RANDOM_SEED)\n",
        "except ImportError:\n",
        "    pass\n",
        "\n",
        "print(f\"✓ Random seeds set to {RANDOM_SEED} for reproducibility\")"
    ]
}


def add_notebook_metadata(notebook_path: Path, config: dict):
    """Add testing metadata to notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Add tags to notebook metadata
    if 'metadata' not in nb:
        nb['metadata'] = {}

    if config.get('tags'):
        if 'tags' not in nb['metadata']:
            nb['metadata']['tags'] = []
        nb['metadata']['tags'] = list(set(nb['metadata']['tags'] + config['tags']))

    # Add skip marker if needed
    if config.get('skip_in_ci'):
        if 'tags' not in nb['metadata']:
            nb['metadata']['tags'] = []
        if 'skip_ci' not in nb['metadata']['tags']:
            nb['metadata']['tags'].append('skip_ci')

    return nb


def check_has_seed(nb: dict) -> bool:
    """Check if notebook already has random seed setup."""
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if ('random.seed' in source or
                'tf.random.set_seed' in source or
                'np.random.seed' in source):
                return True
    return False


def add_seed_cell(nb: dict) -> dict:
    """Add deterministic seed cell after imports."""
    # Find the first code cell with imports
    import_cell_idx = None
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'import' in source:
                import_cell_idx = i
                break

    if import_cell_idx is not None:
        # Insert seed cell after imports
        nb['cells'].insert(import_cell_idx + 1, SEED_CELL.copy())
        print(f"  → Added seed cell after cell {import_cell_idx}")
    else:
        # Insert at beginning if no import cell found
        nb['cells'].insert(0, SEED_CELL.copy())
        print(f"  → Added seed cell at beginning")

    return nb


def main():
    notebook_dir = Path(__file__).parent.parent / "jupyter_notebooks"

    print("=" * 70)
    print("  PREPARING NOTEBOOKS FOR AUTOMATED TESTING")
    print("=" * 70)
    print()

    for notebook_name, config in NOTEBOOK_CONFIG.items():
        notebook_path = notebook_dir / notebook_name

        if not notebook_path.exists():
            print(f"⚠️  {notebook_name} - NOT FOUND (skipping)")
            continue

        print(f"📓 {notebook_name}")
        print(f"   Description: {config['description']}")
        print(f"   Tags: {', '.join(config['tags']) if config['tags'] else 'none'}")
        print(f"   Skip in CI: {config['skip_in_ci']}")

        # Load notebook
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        # Add metadata/tags
        nb = add_notebook_metadata(notebook_path, config)

        # Check if needs seed
        has_seed = check_has_seed(nb)

        if config['needs_seed'] and not has_seed:
            print(f"   Adding deterministic seed cell...")
            nb = add_seed_cell(nb)
        elif has_seed:
            print(f"   ✓ Already has random seed")

        # Save notebook
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f"   ✓ Updated")
        print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print("Notebooks ready for testing!")
    print()
    print("To run tests:")
    print("  make test           # Run all notebooks (excluding slow)")
    print("  make test-all       # Run all notebooks (including slow)")
    print("  make test-fast      # Run only fast notebooks")
    print()
    print("CI Configuration:")
    print("  - Notebook 01 will be skipped (downloads 47GB)")
    print("  - All notebooks have deterministic seeds")
    print("  - Slow notebooks are marked for selective testing")
    print()


if __name__ == '__main__':
    main()
