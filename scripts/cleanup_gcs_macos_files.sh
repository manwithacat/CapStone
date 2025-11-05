#!/usr/bin/env bash
#
# Remove macOS metadata files from GCS bucket
#
# Removes:
#   - .DS_Store
#   - ._* (AppleDouble files)
#   - .Spotlight-V100
#   - .Trashes
#   - .fseventsd

set -e

BUCKET="gs://nih-xrays"

echo "🧹 Cleaning macOS metadata files from GCS"
echo "=========================================="
echo

# Find and list .DS_Store files
echo "🔍 Finding .DS_Store files..."
DS_STORE_FILES=$(gsutil ls -r "${BUCKET}/**/.DS_Store" 2>/dev/null || true)

if [ -n "$DS_STORE_FILES" ]; then
    echo "Found .DS_Store files:"
    echo "$DS_STORE_FILES"
    echo

    # Count them
    COUNT=$(echo "$DS_STORE_FILES" | wc -l | tr -d ' ')
    echo "Total: $COUNT files"
    echo

    read -p "Delete these files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$DS_STORE_FILES" | xargs -I {} gsutil rm {}
        echo "✓ Deleted $COUNT .DS_Store files"
    else
        echo "Skipped deletion"
    fi
else
    echo "✓ No .DS_Store files found"
fi

echo

# Find and list AppleDouble files (._*)
echo "🔍 Finding AppleDouble files (._*)..."
APPLEDOUBLE_FILES=$(gsutil ls -r "${BUCKET}/**/._*" 2>/dev/null || true)

if [ -n "$APPLEDOUBLE_FILES" ]; then
    echo "Found AppleDouble files:"
    echo "$APPLEDOUBLE_FILES"
    echo

    COUNT=$(echo "$APPLEDOUBLE_FILES" | wc -l | tr -d ' ')
    echo "Total: $COUNT files"
    echo

    read -p "Delete these files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$APPLEDOUBLE_FILES" | xargs -I {} gsutil rm {}
        echo "✓ Deleted $COUNT AppleDouble files"
    else
        echo "Skipped deletion"
    fi
else
    echo "✓ No AppleDouble files found"
fi

echo

# Find other macOS metadata
echo "🔍 Finding other macOS metadata..."
OTHER_FILES=$(gsutil ls -r "${BUCKET}/**/.Spotlight-V100" "${BUCKET}/**/.Trashes" "${BUCKET}/**/.fseventsd" 2>/dev/null || true)

if [ -n "$OTHER_FILES" ]; then
    echo "Found:"
    echo "$OTHER_FILES"
    echo

    read -p "Delete these? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$OTHER_FILES" | xargs -I {} gsutil rm -r {}
        echo "✓ Deleted"
    fi
else
    echo "✓ No other macOS metadata found"
fi

echo
echo "=========================================="
echo "✅ Cleanup complete!"
echo "=========================================="
