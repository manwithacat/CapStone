#!/usr/bin/env python3
"""
Fix notebook 07 structure:
1. Add RETRAIN_MODELS to parameters cell
2. Fix cell-11 to be code (not markdown)
3. Remove duplicate cell e124c15b
4. Reorder cells so SKIP_TRAINING is defined before use
"""

import json
from pathlib import Path

# Load notebook
notebook_path = Path(__file__).parent.parent / 'jupyter_notebooks' / '07_transfer_learning.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Add RETRAIN_MODELS to parameters cell (thx4jublr6)
for cell in nb['cells']:
    if cell.get('id') == 'thx4jublr6':
        # Update source to include RETRAIN_MODELS
        new_source = [
            "# Parameters (can be overridden by papermill)\n",
            "# This cell is tagged as \"parameters\" for papermill\n",
            "MODELS_TO_TRAIN = ['resnet50', 'densenet121', 'efficientnetb3']  # Which models to train\n",
            "RUN_NAME = None  # Custom run name for MLflow (auto-generated if None)\n",
            "USE_MLFLOW = True  # Enable MLflow tracking\n",
            "RETRAIN_MODELS = False  # Set to True to retrain even if models exist\n",
            "\n",
            "print(f\"Models to train: {MODELS_TO_TRAIN}\")\n",
            "print(f\"Run name: {RUN_NAME if RUN_NAME else 'Auto-generated'}\")\n",
            "print(f\"MLflow tracking: {'Enabled' if USE_MLFLOW else 'Disabled'}\")\n",
            "print(f\"Retrain models: {RETRAIN_MODELS}\")"
        ]
        cell['source'] = new_source
        print("✓ Added RETRAIN_MODELS to parameters cell")
        break

# 2. Remove standalone RETRAIN_MODELS cell (bebe5ceb)
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') != 'bebe5ceb']
print("✓ Removed standalone RETRAIN_MODELS cell")

# 3. Fix cell-11 to be code (not markdown)
for cell in nb['cells']:
    if cell.get('id') == 'cell-11':
        cell['cell_type'] = 'code'
        cell['execution_count'] = None
        if 'outputs' not in cell:
            cell['outputs'] = []
        print("✓ Changed cell-11 from markdown to code")
        break

# 4. Remove duplicate cell e124c15b
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') != 'e124c15b']
print("✓ Removed duplicate cell e124c15b")

# 5. Reorder cells: need proper order before training starts
# Order should be:
#   cell-9: Data Generators header
#   cell-11: SKIP_TRAINING logic (check if models exist)
#   cell-12: build_transfer_model function definition
#   cell-10: First training block (ResNet50) - uses both SKIP_TRAINING and build_transfer_model

# Find all relevant cell indices
cell_indices = {}
for idx, cell in enumerate(nb['cells']):
    cell_id = cell.get('id')
    if cell_id in ['cell-9', 'cell-10', 'cell-11', 'cell-12']:
        cell_indices[cell_id] = idx

if len(cell_indices) == 4:
    # Remove cells 11 and 12 from their current positions (in reverse order to preserve indices)
    indices_to_remove = sorted([cell_indices['cell-11'], cell_indices['cell-12']], reverse=True)
    cell_11 = nb['cells'].pop(cell_indices['cell-11'])
    # Recalculate cell-12 index after removing cell-11
    if cell_indices['cell-12'] > cell_indices['cell-11']:
        cell_12 = nb['cells'].pop(cell_indices['cell-12'] - 1)
    else:
        cell_12 = nb['cells'].pop(cell_indices['cell-12'])

    # Find cell-9 index again after removals
    cell_9_idx = None
    for idx, cell in enumerate(nb['cells']):
        if cell.get('id') == 'cell-9':
            cell_9_idx = idx
            break

    # Insert in correct order: cell-9, then cell-11, then cell-12, then cell-10
    if cell_9_idx is not None:
        nb['cells'].insert(cell_9_idx + 1, cell_11)
        nb['cells'].insert(cell_9_idx + 2, cell_12)
        print("✓ Reordered cells: cell-11 and cell-12 now before cell-10")

# Save notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Fixed notebook saved to {notebook_path}")
print("\nChanges made:")
print("  1. Added RETRAIN_MODELS parameter to parameters cell")
print("  2. Changed cell-11 from markdown to code")
print("  3. Removed duplicate cells")
print("  4. Reordered cells so SKIP_TRAINING is defined before use")
