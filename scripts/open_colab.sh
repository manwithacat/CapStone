#!/bin/bash
# Quick script to open notebook in Google Colab

NOTEBOOK="jupyter_notebooks/07_transfer_learning_colab.ipynb"

echo "=========================================="
echo "Opening Notebook 07 in Google Colab"
echo "=========================================="
echo ""
echo "Notebook: $NOTEBOOK"
echo ""

# Option 1: Use colab-cli (requires setup)
if command -v colab-cli &> /dev/null; then
    echo "📱 Opening via colab-cli..."
    colab-cli open-nb "$NOTEBOOK"
else
    echo "⚠️  colab-cli not configured"
    echo ""
    echo "📋 Manual upload instructions:"
    echo "1. Go to: https://colab.research.google.com/"
    echo "2. File → Upload notebook"
    echo "3. Select: $NOTEBOOK"
    echo "4. Runtime → Change runtime type → T4 GPU + High RAM"
    echo ""
    echo "Opening Colab in browser..."
    open "https://colab.research.google.com/"
fi

echo ""
echo "✓ Notebook location: $(pwd)/$NOTEBOOK"
