#!/bin/bash
# Setup Kaggle credentials for project

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
KAGGLE_DIR="$PROJECT_DIR/.kaggle"
KAGGLE_FILE="$KAGGLE_DIR/kaggle.json"
HOME_KAGGLE="$HOME/.kaggle/kaggle.json"

echo "🔧 Setting up Kaggle credentials for project"
echo ""

# Create .kaggle directory if it doesn't exist
mkdir -p "$KAGGLE_DIR"
echo "✓ Created directory: $KAGGLE_DIR"

# Check if credentials exist in home directory
if [ -f "$HOME_KAGGLE" ]; then
    echo "✓ Found credentials in home directory"
    echo "  Copying to project..."
    cp "$HOME_KAGGLE" "$KAGGLE_FILE"
    chmod 600 "$KAGGLE_FILE"
    echo "✓ Copied to: $KAGGLE_FILE"
    echo "✓ Set permissions to 600"
    echo ""
    echo "🎉 Setup complete! Kaggle credentials ready in project directory."
else
    echo "⚠️  No credentials found in $HOME_KAGGLE"
    echo ""
    echo "Setup instructions:"
    echo "1. Download kaggle.json from: https://www.kaggle.com/settings/account"
    echo "2. Save it to: $KAGGLE_FILE"
    echo "3. Run: chmod 600 $KAGGLE_FILE"
    echo ""
    echo "Or run this command if you have kaggle.json elsewhere:"
    echo "  cp /path/to/kaggle.json $KAGGLE_FILE && chmod 600 $KAGGLE_FILE"
fi
