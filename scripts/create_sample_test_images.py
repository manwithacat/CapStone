"""
Create a representative sample of test images for Streamlit Disease Detector.

This script selects ~100 test images covering all 14 diseases, resizes them to
224x224, and saves them for the "Pick Random X-Ray" feature.

Target: ~10MB total (100 images × ~100KB each)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TEST_SPLIT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'test_split.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'sample_test_images'
IMG_SIZE = (224, 224)
TARGET_SIZE_MB = 10
MAX_IMAGES = 100  # ~10MB at ~100KB per image

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def main():
    print("=" * 70)
    print("Creating Sample Test Images for Disease Detector")
    print("=" * 70)

    # Load test split
    print(f"\n1. Loading test split from {TEST_SPLIT_CSV.name}...")
    test_df = pd.read_csv(TEST_SPLIT_CSV)
    print(f"   Total test images: {len(test_df):,}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Output directory: {OUTPUT_DIR}")

    # Strategy: Sample images to ensure all diseases are represented
    # Prioritize rare diseases, then fill with diverse samples

    selected_indices = []

    # First, get at least 2 images per disease (especially rare ones)
    print("\n3. Selecting representative images per disease...")
    for disease in DISEASE_CLASSES:
        disease_df = test_df[test_df[disease] == 1]
        n_available = len(disease_df)
        n_select = min(3, n_available)  # At least 3 per disease if available

        if n_available > 0:
            sampled = disease_df.sample(n=n_select, random_state=42)
            selected_indices.extend(sampled.index.tolist())
            print(f"   {disease:20}: {n_select:2d} images (from {n_available:4d} available)")
        else:
            print(f"   {disease:20}:  0 images (NONE IN TEST SET)")

    # Remove duplicates (some images may have multiple diseases)
    selected_indices = list(set(selected_indices))
    print(f"\n   Initial selection: {len(selected_indices)} unique images")

    # Fill remaining slots with diverse samples
    remaining_slots = MAX_IMAGES - len(selected_indices)
    if remaining_slots > 0:
        # Get images not yet selected
        remaining_df = test_df.drop(selected_indices)

        # Sample randomly
        additional = remaining_df.sample(n=min(remaining_slots, len(remaining_df)), random_state=42)
        selected_indices.extend(additional.index.tolist())
        print(f"   Added {len(additional)} additional diverse samples")

    # Get final selection
    selected_df = test_df.loc[selected_indices].copy()
    print(f"\n   Final selection: {len(selected_df)} images")

    # Process and save images
    print("\n4. Processing and saving images...")

    processed_count = 0
    total_size_bytes = 0
    metadata_records = []

    for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="   Processing"):
        source_path = Path(row['full_path'])

        if not source_path.exists():
            print(f"\n   Warning: {source_path.name} not found, skipping")
            continue

        try:
            # Load and resize image
            img = Image.open(source_path).convert('RGB')
            img_resized = img.resize(IMG_SIZE, Image.LANCZOS)

            # Save to output directory
            output_filename = source_path.name
            output_path = OUTPUT_DIR / output_filename
            img_resized.save(output_path, 'PNG', optimize=True)

            # Track size
            file_size = output_path.stat().st_size
            total_size_bytes += file_size
            processed_count += 1

            # Store metadata
            metadata_records.append({
                'filename': output_filename,
                'Image Index': row['Image Index'],
                'size_bytes': file_size,
                **{disease: int(row[disease]) for disease in DISEASE_CLASSES}
            })

            # Stop if we exceed target size
            if total_size_bytes > TARGET_SIZE_MB * 1024 * 1024:
                print(f"\n   Reached target size ({TARGET_SIZE_MB}MB), stopping")
                break

        except Exception as e:
            print(f"\n   Error processing {source_path.name}: {e}")

    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = OUTPUT_DIR / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)

    # Summary
    total_size_mb = total_size_bytes / (1024 * 1024)
    avg_size_kb = (total_size_bytes / processed_count) / 1024 if processed_count > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Images processed: {processed_count}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Average image size: {avg_size_kb:.1f} KB")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata saved: {metadata_path.name}")

    # Disease coverage
    print(f"\nDisease Coverage:")
    for disease in DISEASE_CLASSES:
        count = metadata_df[disease].sum()
        pct = (count / len(metadata_df)) * 100
        print(f"  {disease:20}: {count:3d} images ({pct:5.1f}%)")

    print("\n✅ Sample test images created successfully!")
    print(f"\nNext step: Use these images in Disease Detector tab")
    print(f"  Location: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
