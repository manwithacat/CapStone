# Jupytext Workflow Guide

## Overview

This project uses **jupytext** to maintain Jupyter notebooks in two formats:
- `.ipynb` files for human readability and interactive execution
- `.py` files for efficient editing, version control, and Claude Code interaction

## Why Jupytext?

### Benefits for Development
- **Token Efficiency**: `.py` files are 50-60% smaller than `.ipynb` (no outputs, metadata, JSON structure)
- **Easier Editing**: Clean Python code vs JSON manipulation
- **Better Git Diffs**: See exactly what code changed
- **Code Reuse**: Import functions from `.py` files for testing
- **Professional Practice**: Industry standard for notebook version control

### For This Project
- 9 notebooks (5-7 still need major work)
- Complex debugging needed (like image loading issues)
- MLflow integration (script-friendly)
- Assessment advantage (shows professional workflow)

## Setup

### Install Jupytext
```bash
pip install jupytext
```

### Convert Existing Notebook
```bash
# Convert to paired format (creates .py alongside .ipynb)
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

### Verify Pairing
```bash
# Check notebook metadata
jupytext --show-formats notebook.ipynb
# Should output: ipynb,py:percent
```

## Workflow

### 1. Edit the .py File
Work directly in the clean Python file:

```bash
# Example: Edit notebook 06
vim jupyter_notebooks/06_cnn_development.py

# Or use Claude Code to edit the .py file
```

The `.py` file uses **percent format** with cell markers:
```python
# %% [markdown]
# # Section Header
# Markdown content

# %%
# Code cell
import pandas as pd
df = pd.read_csv('data.csv')
```

### 2. Sync Changes Back to .ipynb
```bash
# Manually sync (if needed)
jupytext --sync jupyter_notebooks/06_cnn_development.py

# Or use auto-sync (Jupyter will sync on save)
```

### 3. Run in Jupyter
Open the `.ipynb` file in Jupyter/VS Code and run as normal. All outputs save to `.ipynb` only.

## Key Patterns

### Cell Types in .py Format

**Markdown Cell:**
```python
# %% [markdown]
# ## Data Loading
# Load and preprocess the dataset...
```

**Code Cell:**
```python
# %%
import pandas as pd
df = pd.read_csv('data.csv')
print(df.shape)
```

**Named Cell (for debugging):**
```python
# %% [markdown] tags=["explanation"]
# Detailed explanation...

# %% tags=["debug"]
# Debug code here
```

### What Gets Synced

**Synced:**
- Code
- Markdown
- Cell structure
- Cell metadata (minimal)

**NOT Synced (stays in .ipynb only):**
- Cell outputs
- Execution counts
- Plots/images
- Error tracebacks

## Common Commands

```bash
# Convert notebook to .py only (no pairing)
jupytext --to py:percent notebook.ipynb

# Convert .py back to .ipynb
jupytext --to ipynb notebook.py

# Pair existing files
jupytext --set-formats ipynb,py:percent notebook.ipynb

# Sync paired files
jupytext --sync notebook.py
# or
jupytext --sync notebook.ipynb

# Show current formats
jupytext --show-formats notebook.ipynb
```

## Git Workflow

### What to Commit

**Both formats for key notebooks:**
```bash
git add jupyter_notebooks/06_cnn_development.py
git add jupyter_notebooks/06_cnn_development.ipynb
```

### .gitignore Configuration

```gitignore
# Notebook checkpoints
**/.ipynb_checkpoints/

# Optional: Ignore .ipynb if you only want .py in version control
# jupyter_notebooks/*.ipynb

# Keep both for this project (explainability for assessment)
```

## Debugging Workflow

### Example: Image Loading Error

**Before (with .ipynb):**
1. Read 29k token notebook file
2. Parse JSON structure
3. Find relevant cells
4. Edit via JSON manipulation
5. Hard to add logging/debugging

**After (with .py):**
1. Read 5k token Python file
2. Find relevant code directly
3. Edit clean Python code
4. Add logging easily
5. Sync back to .ipynb

### Actual Fix Applied

**Problem:** TensorFlow UnidentifiedImageError when loading grayscale images

**Root Cause:** Missing `color_mode='rgb'` in flow_from_dataframe

**Fix in .py file (line 378):**
```python
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='full_path',
    y_col=disease_classes,
    target_size=(CONFIG['img_height'], CONFIG['img_width']),
    batch_size=CONFIG['batch_size'],
    class_mode='raw',
    color_mode='rgb',  # ← Added: Convert grayscale to RGB
    shuffle=True,
    seed=CONFIG['random_state']
)
```

**Synced to .ipynb:** `jupytext --sync jupyter_notebooks/06_cnn_development.py`

## Best Practices

### For Claude Code
1. Always work on `.py` files for efficiency
2. Use `Read` tool on `.py` (much faster)
3. Edit `.py` with `Edit` or `Write` tools
4. Sync after changes: `jupytext --sync file.py`

### For Human Review
1. Open `.ipynb` in Jupyter/VS Code
2. Run cells interactively
3. Outputs save to `.ipynb` only
4. Don't edit .ipynb directly (edit .py instead)

### For Assessment
1. Commit both `.py` and `.ipynb` to git
2. `.py` shows clean code structure
3. `.ipynb` shows execution results
4. Demonstrates professional workflow

## Troubleshooting

### Conflict Warning
```
Warning: notebook is not a paired notebook
```
**Fix:** Run `jupytext --set-formats ipynb,py:percent notebook.ipynb`

### Sync Not Working
```
# Force sync from .py to .ipynb
jupytext --sync --to ipynb notebook.py

# Force sync from .ipynb to .py
jupytext --sync --to py:percent notebook.ipynb
```

### Lost Outputs
Outputs only exist in `.ipynb`. If you delete `.ipynb`, outputs are lost (but code is safe in `.py`).

### Merge Conflicts
If both `.py` and `.ipynb` are edited:
1. Resolve conflict in `.py` file (easier to read)
2. Sync to regenerate `.ipynb`: `jupytext --sync notebook.py`

## Project Status

### Notebooks with Jupytext
- ✅ `06_cnn_development.ipynb` (paired with `.py`)

### To Be Converted
- [ ] `07_transfer_learning.ipynb`
- [ ] `08_model_evaluation.ipynb`
- [ ] `09_model_interpretation.ipynb`

### Decision for Notebooks 01-05
Keep as `.ipynb` only (already complete, mostly data exploration).

## References

- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [Percent Format Specification](https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format)
- [Version Control Best Practices](https://jupytext.readthedocs.io/en/latest/using-cli.html)
