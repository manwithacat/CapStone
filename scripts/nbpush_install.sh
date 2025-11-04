#!/bin/bash
# Install nbpush CLI tool

set -e

echo "=========================================="
echo "Installing nbpush CLI"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q typer rich questionary

# Create executable wrapper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_PATH="$PROJECT_ROOT/nbpush"

cat > "$BIN_PATH" << 'EOF'
#!/usr/bin/env python3
"""
nbpush - Notebook Cloud Push CLI
"""
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from nbpush.cli import main

if __name__ == "__main__":
    main()
EOF

chmod +x "$BIN_PATH"

echo "✓ Dependencies installed"
echo "✓ Created executable: $BIN_PATH"
echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  ./nbpush list                          # List notebooks"
echo "  ./nbpush push <notebook> --service kaggle   # Push to Kaggle"
echo "  ./nbpush activity                      # Show activity log"
echo ""
echo "Add to PATH (optional):"
echo "  export PATH=\"$PROJECT_ROOT:\$PATH\""
echo ""
