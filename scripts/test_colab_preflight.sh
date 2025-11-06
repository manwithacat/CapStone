#!/usr/bin/env bash
#
# Pre-flight checks for Colab notebooks
# Run this before pushing notebooks to catch common issues

set -e

echo "🧪 Running Colab Pre-Flight Checks"
echo "===================================="
echo

# Run tests without pytest.ini (which has nbmake plugins)
python3 -m pytest tests/test_colab_notebook.py -c /dev/null -v --tb=short

echo
echo "✅ All pre-flight checks passed!"
echo "📤 Safe to push to Colab"
