#!/usr/bin/env python3
"""
Diagnostic script to check Pylance/Python import configuration.

Run this to verify that VS Code Pylance can find the preprocessing module.
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("  PYLANCE/PYTHON IMPORT DIAGNOSTIC")
    print("=" * 70)
    print()

    # Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ Python executable: {sys.executable}")
    print()

    # Check if src is in path
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    print(f"Source dir exists: {src_dir.exists()}")
    print()

    # Check Python path
    print("Python path includes:")
    for i, path in enumerate(sys.path[:10], 1):
        marker = " ← src directory" if "src" in path and "CapStone" in path else ""
        print(f"  {i}. {path}{marker}")
    if len(sys.path) > 10:
        print(f"  ... and {len(sys.path) - 10} more")
    print()

    # Try importing the preprocessing module
    print("Attempting to import preprocessing module...")
    try:
        from preprocessing import (
            ChestXRayPreprocessingPipeline,
            create_train_pipeline,
            create_inference_pipeline
        )
        print("✅ SUCCESS: preprocessing module imported successfully!")
        print()
        print("Available components:")
        print(f"  - ChestXRayPreprocessingPipeline")
        print(f"  - create_train_pipeline")
        print(f"  - create_inference_pipeline")
    except ImportError as e:
        print(f"❌ FAILED: Could not import preprocessing module")
        print(f"   Error: {e}")
        print()
        print("TROUBLESHOOTING STEPS:")
        print("  1. Install package in editable mode:")
        print("     pip install -e .")
        print()
        print("  2. Reload VS Code window:")
        print("     Press Cmd+Shift+P → 'Developer: Reload Window'")
        print()
        print("  3. Select correct Python interpreter:")
        print("     Press Cmd+Shift+P → 'Python: Select Interpreter'")
        print(f"     Choose: {sys.executable}")
        return 1

    print()

    # Check if package is installed
    print("Checking package installation...")
    try:
        import pkg_resources
        pkg = pkg_resources.get_distribution('chest-xray-detection')
        print(f"✅ Package 'chest-xray-detection' is installed")
        print(f"   Version: {pkg.version}")
        print(f"   Location: {pkg.location}")
    except pkg_resources.DistributionNotFound:
        print("⚠️  Package not found in pip list")
        print("   Run: pip install -e .")

    print()
    print("=" * 70)
    print("  CONFIGURATION FILES")
    print("=" * 70)
    print()

    # Check configuration files
    config_files = [
        ('.vscode/settings.json', 'VS Code settings'),
        ('pyrightconfig.json', 'Pyright configuration'),
        ('setup.py', 'Package setup file'),
        ('.env', 'Environment variables'),
    ]

    for file_path, description in config_files:
        full_path = project_root / file_path
        status = "✓" if full_path.exists() else "✗"
        print(f"{status} {description}: {file_path}")

    print()
    print("=" * 70)
    print("If Pylance still shows errors in VS Code:")
    print("  1. Reload window: Cmd+Shift+P → 'Developer: Reload Window'")
    print("  2. Restart VS Code completely")
    print("  3. Check Python interpreter matches this script's Python")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
