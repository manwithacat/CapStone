# Colab Notebook Testing

## Quick Start

Before pushing any notebook to Colab, run:

```bash
./scripts/test_colab_preflight.sh
```

This catches common issues that would cause failures in Colab.

## What Gets Tested

### 1. Data Split Files (7 tests)

**Catches:**
- Missing CSV files
- Wrong column names (e.g., using `full_path` instead of `Image Index`)
- Invalid disease labels
- Oversized CSV files

**Example caught error:**
```
KeyError: 'full_path'
```
→ **Fix**: Colab CSVs use `Image Index` (built at runtime), not `full_path`

### 2. Notebook Structure (7 tests)

**Catches:**
- Path building cell running AFTER data generators
- Hardcoded local paths (`/Volumes/SSD/`, `/Users/`)
- Using Colab's built-in auth (shows "third party" warning)
- Missing cache check for persistent disk

**Example caught error:**
```
ValueError: The PyDataset has length 0
```
→ **Fix**: Build `full_path` column BEFORE creating ImageDataGenerator

### 3. Data Integrity (3 tests)

**Catches:**
- Patient overlap between train/val/test
- Empty or too-small splits
- Invalid filenames

### 4. OAuth Setup (3 tests)

**Catches:**
- Missing `client_secrets.json.example`
- Invalid JSON format
- Accidentally committing actual credentials

## Common Issues Caught

### Issue 1: KeyError: 'full_path'

**Symptom:**
```python
KeyError: 'full_path'
  at pandas/core/indexes/base.py in get_loc
```

**Root Cause:**
- Colab CSV files don't have `full_path` column
- It's built dynamically from `Image Index`
- Data generator ran before path building

**Caught By:**
- `test_csv_has_image_index_column` - Ensures CSV has correct columns
- `test_path_building_before_generators` - Ensures correct cell order

**Fix:**
Make sure Cell 15 (builds `full_path`) runs before Cell 17 (creates generators)

### Issue 2: ValueError: The PyDataset has length 0

**Symptom:**
```python
ValueError: The PyDataset has length 0
  at keras/src/trainers/data_adapters/py_dataset_adapter.py
```

**Root Cause:**
- `ImageDataGenerator` can't find images
- Paths are invalid

**Caught By:**
- `test_uses_image_index_not_full_path_for_loading` - Verifies notebook builds paths correctly

**Fix:**
Ensure path building function maps filenames to actual image locations

### Issue 3: "This notebook is written by a third party"

**Symptom:**
Google shows warning about third-party access

**Root Cause:**
- Using `auth.authenticate_user()` instead of own OAuth

**Caught By:**
- `test_no_colab_builtin_auth` - Flags use of Colab's built-in auth

**Fix:**
Upload your own `client_secrets.json` and use PyDrive2's `LocalWebserverAuth()`

## Test Categories

```bash
# Run all tests
./scripts/test_colab_preflight.sh

# Run specific category
python3 -m pytest tests/test_colab_notebook.py::TestColabDataSplits -c /dev/null -v

# Run single test
python3 -m pytest tests/test_colab_notebook.py::TestColabDataSplits::test_csv_has_image_index_column -c /dev/null -v
```

## Adding New Tests

When you encounter a Colab error:

1. **Document the error** in this README
2. **Create a test** that would have caught it
3. **Add to test suite**

Example:

```python
class TestNewIssue:
    """Test for issue #XYZ."""

    def test_something_specific(self):
        """Describe what this catches."""
        # Test code here
        assert condition, "Error message explaining the issue"
```

## CI/CD Integration

Add to your workflow:

```bash
# Before pushing to Colab
./scripts/test_colab_preflight.sh || {
    echo "❌ Pre-flight checks failed!"
    exit 1
}

# Upload notebook to Colab
# ...
```

## Test Output

### Success
```
==================================
🧪 Running Colab Pre-Flight Checks
==================================

test_splits_directory_exists PASSED
test_required_files_exist PASSED
...
======================== 21 passed in 0.88s ========================

✅ All pre-flight checks passed!
📤 Safe to push to Colab
```

### Failure
```
FAILED test_csv_has_image_index_column - AssertionError: Missing 'Image Index'
```

## Maintenance

**Update tests when:**
- Adding new disease classes
- Changing CSV format
- Modifying notebook structure
- Encountering new Colab-specific errors

**Test files:**
- `tests/test_colab_notebook.py` - Main test suite
- `scripts/test_colab_preflight.sh` - Runner script
- This file - Documentation

## References

- [pytest documentation](https://docs.pytest.org/)
- [nbformat](https://nbformat.readthedocs.io/) - Notebook testing
- Project-specific: `colab/SETUP_INSTRUCTIONS.md`
