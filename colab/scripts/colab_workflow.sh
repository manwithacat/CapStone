#!/bin/bash
# Simplified Colab workflow - edit in place

NOTEBOOK_URL="https://colab.research.google.com/drive/1FJdto9vlXuvtofDpjIN9Vb5DzyPAlLEH"

echo "=========================================="
echo "Colab Workflow - Edit in Place"
echo "=========================================="
echo ""
echo "📝 Your stable notebook URL:"
echo "   $NOTEBOOK_URL"
echo ""
echo "🎯 Recommended workflow:"
echo "   1. Open the notebook (opening now...)"
echo "   2. Make all edits directly in Colab browser"
echo "   3. Run cells interactively"
echo "   4. Colab auto-saves to your Drive"
echo ""
echo "📥 To pull changes back to local (optional):"
echo "   colab-cli pull-nb jupyter_notebooks/07_transfer_learning_colab.ipynb"
echo ""

# Open the stable URL
open "$NOTEBOOK_URL"

echo "✓ Notebook opened in browser"
echo ""
echo "💡 Pro tip: Bookmark this URL - it's your permanent notebook!"
