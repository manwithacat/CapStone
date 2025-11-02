#!/bin/bash
# Delete existing Kaggle kernel to resolve conflicts
#
# Usage:
#   ./scripts/kaggle_delete_kernel.sh [kernel-slug]
#
# Default kernel slug: cnn-optimized-training

set -e

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

export KAGGLE_CONFIG_DIR="$(pwd)/.kaggle"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-yourusername}"
KERNEL_SLUG="${1:-cnn-optimized-training}"

echo "🗑️  Kaggle Kernel Deletion Tool"
echo "================================"
echo ""
echo "Username: $KAGGLE_USERNAME"
echo "Kernel:   $KERNEL_SLUG"
echo "Full ID:  $KAGGLE_USERNAME/$KERNEL_SLUG"
echo ""

# Check if kernel exists
echo "🔍 Checking if kernel exists..."
if kaggle kernels list --mine 2>&1 | grep -q "$KERNEL_SLUG"; then
    echo "✓ Found kernel: $KERNEL_SLUG"
else
    echo "⚠️  Kernel not found in your kernels list"
    echo "   This might be okay - the conflict may be resolved."
    echo ""
    read -p "Continue anyway to ensure clean state? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

echo ""
echo "⚠️  WARNING: This will DELETE the kernel and all its versions!"
echo "   URL: https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_SLUG"
echo ""
read -p "Are you sure you want to delete this kernel? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 0
fi

echo ""
echo "🗑️  Deleting kernel..."

# Try to delete via API
# Note: Kaggle API doesn't have a direct delete command for kernels
# We need to use their web interface or wait for kernel to be overwritten

echo "⚠️  Note: Kaggle API doesn't support direct kernel deletion"
echo ""
echo "To delete manually:"
echo "1. Go to: https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_SLUG"
echo "2. Click the '...' menu (top right)"
echo "3. Select 'Delete'"
echo ""
echo "OR: Just re-run the training script - it will overwrite the existing kernel"
echo ""
echo "💡 Alternative: The script has been updated to use a matching title."
echo "   Try running again: ./scripts/kaggle_train_headless.sh"
echo ""
