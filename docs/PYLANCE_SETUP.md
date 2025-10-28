# Pylance Setup Guide

This guide helps resolve import errors in VS Code when working with Jupyter notebooks and the custom `preprocessing` module.

## Quick Fix (Most Common)

If you're seeing red squiggly lines under `from preprocessing import ...` in notebooks:

1. **Reload VS Code window**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type: "Developer: Reload Window"
   - Press Enter

2. **Verify Python interpreter**
   - Click the Python version in the bottom-right corner of VS Code
   - Ensure it matches your project's Python (should show `.pyenv` or your venv)

3. **If still having issues, run diagnostic:**
   ```bash
   make check-pylance
   ```

## Why Import Errors Occur

VS Code's Pylance language server analyzes code **statically** (without running it). When notebooks use:

```python
sys.path.append(str(Path.cwd().parent / 'src'))
from preprocessing import ...
```

Pylance doesn't execute `sys.path.append()`, so it can't find the module.

## Our Solution

We've configured the project in multiple ways to ensure imports work:

### 1. **Editable Package Installation** ✅ BEST
```bash
pip install -e .
```

This installs the `src/` directory as a Python package. Now you can import directly:
```python
from preprocessing import ChestXRayPreprocessingPipeline
```

No `sys.path` manipulation needed!

### 2. **VS Code Configuration** (`.vscode/settings.json`)
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}/src"],
  "python.envFile": "${workspaceFolder}/.env",
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

Tells Pylance where to find the `preprocessing` module.

### 3. **Pyright Configuration** (`pyrightconfig.json`)
```json
{
  "extraPaths": ["src"]
}
```

Configures the type checker (Pylance's backend).

### 4. **Environment File** (`.env`)
```bash
PYTHONPATH=${PYTHONPATH}:${workspaceFolder}/src
```

Adds `src/` to Python path when terminals are opened.

## Diagnostic Commands

### Check if everything is configured correctly:
```bash
make check-pylance
```

This will verify:
- ✓ Python version and executable
- ✓ Source directory exists
- ✓ Python path includes src/
- ✓ Module imports successfully
- ✓ Package is installed
- ✓ All configuration files exist

### Run type checker (same engine as Pylance):
```bash
make typecheck
```

Should show: `0 errors, 0 warnings, 0 informations`

## Common Issues

### Issue 1: "Cannot find module 'preprocessing'"

**Solution:**
```bash
pip install -e .
# Then reload VS Code window
```

### Issue 2: Imports work when running code, but Pylance shows errors

**Solution:**
1. Check Python interpreter matches the one where package is installed
2. Run: `make check-pylance` to see which Python is being used
3. Select correct interpreter in VS Code

### Issue 3: Multiple Python installations (pyenv, conda, system)

**Solution:**
1. Find where package is installed:
   ```bash
   pip show chest-xray-detection
   ```
2. Check Python executable:
   ```bash
   which python
   ```
3. In VS Code: `Cmd+Shift+P` → "Python: Select Interpreter"
4. Choose the Python that matches the installation location

### Issue 4: Works in some notebooks but not others

**Solution:**
- Ensure all notebooks use the same Python kernel
- Check kernel: Click kernel name in top-right of notebook
- Select correct kernel that matches your Python installation

## Notebook Import Best Practices

### ✅ RECOMMENDED (with package installed):
```python
# No sys.path manipulation needed!
from preprocessing import (
    ChestXRayPreprocessingPipeline,
    create_train_pipeline,
    create_inference_pipeline
)
```

### ⚠️ FALLBACK (if package not installed):
```python
import sys
from pathlib import Path

# Add src to path (only if package not installed)
sys.path.append(str(Path.cwd().parent / 'src'))

from preprocessing import ...
```

### ❌ AVOID:
```python
# Don't use relative imports in notebooks
from ..src.preprocessing import ...  # ❌ Won't work
```

## Verifying Setup

Run this in a notebook cell or terminal:

```python
# Test direct import (should work with our setup)
from preprocessing import ChestXRayPreprocessingPipeline
print("✓ Import successful!")

# Check where module is loaded from
import preprocessing
print(f"Module location: {preprocessing.__file__}")
```

Expected output:
```
✓ Import successful!
Module location: /Users/james/CodeInstitute/CapStone/src/preprocessing/__init__.py
```

## Still Having Issues?

1. **Restart VS Code completely** (not just reload window)
2. **Check VS Code Output panel**:
   - View → Output
   - Select "Python" from dropdown
   - Look for errors
3. **Run full diagnostic**:
   ```bash
   make check-pylance
   ```
4. **Verify installations**:
   ```bash
   pip list | grep chest-xray-detection
   pip show chest-xray-detection
   ```

## Additional Resources

- [VS Code Python Environments](https://code.visualstudio.com/docs/python/environments)
- [Pylance Documentation](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Python Editable Installs](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)
