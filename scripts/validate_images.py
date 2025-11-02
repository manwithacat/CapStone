#!/usr/bin/env python3
"""
Validate all images in the dataset and identify corrupted files.

Usage:
    python scripts/validate_images.py data/processed/train_split.csv
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def validate_images(csv_path, output_path=None):
    """
    Validate all images listed in a CSV file.

    Args:
        csv_path: Path to CSV with 'full_path' column
        output_path: Optional path to save valid images CSV

    Returns:
        Tuple of (valid_df, corrupted_files)
    """
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'full_path' not in df.columns:
        print("ERROR: CSV must have 'full_path' column")
        sys.exit(1)

    print(f"Validating {len(df)} images...")

    valid_indices = []
    corrupted_files = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['full_path']

        try:
            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image

            # Reopen and try to load it (verify() makes the file object unusable)
            with Image.open(img_path) as img:
                img.load()  # Actually load image data

            valid_indices.append(idx)

        except Exception as e:
            corrupted_files.append({
                'path': img_path,
                'index': idx,
                'error': str(e)
            })

    # Create DataFrame with only valid images
    valid_df = df.loc[valid_indices].reset_index(drop=True)

    # Report results
    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"{'='*60}")
    print(f"✅ Valid images:     {len(valid_df):,}")
    print(f"❌ Corrupted images: {len(corrupted_files):,}")

    if corrupted_files:
        print(f"\nCorrupted files:")
        for item in corrupted_files:
            print(f"  - {item['path']}")
            print(f"    Error: {item['error']}")

    # Save valid images to new CSV
    if output_path:
        valid_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved valid images to: {output_path}")

    return valid_df, corrupted_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_images.py <csv_path> [output_path]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not output_path:
        # Auto-generate output path
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_validated.csv"

    valid_df, corrupted = validate_images(csv_path, output_path)

    if corrupted:
        print(f"\n⚠️  Found {len(corrupted)} corrupted images. They have been excluded.")
        print(f"   Valid images saved to: {output_path}")
    else:
        print(f"\n✅ All images are valid!")
