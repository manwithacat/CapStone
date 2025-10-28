"""
Extract Individual Sample X-Ray Images

This script extracts the 14 sample X-ray images (one per disease class)
from the NIH dataset and saves them as individual files with metadata.

Creates both full-resolution and thumbnail versions for use in the Streamlit app.
"""

import json
import pathlib
from PIL import Image
import pandas as pd
import numpy as np

# Paths
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "outputs" / "figures"
SAMPLES_DIR = FIGURES_DIR / "sample_xrays"
SAMPLES_DIR.mkdir(exist_ok=True)

# Create subdirectories for full and thumbnail versions
FULL_DIR = SAMPLES_DIR / "full"
THUMB_DIR = SAMPLES_DIR / "thumbnails"
FULL_DIR.mkdir(exist_ok=True)
THUMB_DIR.mkdir(exist_ok=True)

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(DATA_DIR / "Data_Entry_2017.csv")

# Disease classes (14 pathology labels)
disease_classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# Sample one image per disease class
print("Sampling one image per disease class...")
sample_images = []
sample_metadata = []

for disease in disease_classes:
    # Find images with this disease
    disease_images = metadata[metadata['Finding Labels'].str.contains(disease, na=False)]

    if len(disease_images) > 0:
        # Sample one image
        sample = disease_images.sample(n=1, random_state=42).iloc[0]
        sample_images.append(sample['Image Index'])
        sample_metadata.append({
            'disease': disease,
            'image_index': str(sample['Image Index']),
            'patient_id': int(sample['Patient ID']),
            'patient_age': int(sample['Patient Age']),
            'patient_gender': str(sample['Patient Gender']),
            'finding_labels': str(sample['Finding Labels']),
            'view_position': str(sample['View Position'])
        })
        print(f"  {disease}: {sample['Image Index']}")

# Load and save individual images
print("\nExtracting and saving individual images...")
saved_count = 0

for i, (img_name, meta) in enumerate(zip(sample_images, sample_metadata)):
    # Find image in subdirectories (nested under images/ folder)
    img_path = None
    for subdir in DATA_DIR.glob("images_*"):
        potential_path = subdir / "images" / img_name
        if potential_path.exists():
            img_path = potential_path
            break

    if img_path is None:
        print(f"  Warning: Image not found: {img_name}")
        continue

    # Load image
    img = Image.open(img_path)

    # Save full resolution
    disease_name = meta['disease'].lower().replace('_', '-')
    full_path = FULL_DIR / f"{disease_name}.png"
    img.save(full_path)
    print(f"  Saved full: {full_path.name}")

    # Create and save thumbnail (512x512)
    thumbnail = img.copy()
    thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
    thumb_path = THUMB_DIR / f"{disease_name}.png"
    thumbnail.save(thumb_path)
    print(f"  Saved thumbnail: {thumb_path.name}")

    saved_count += 1

# Save metadata JSON
metadata_path = SAMPLES_DIR / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(sample_metadata, f, indent=2)

print(f"\n✅ Successfully extracted {saved_count}/{len(disease_classes)} images")
print(f"   Full images: {FULL_DIR}")
print(f"   Thumbnails: {THUMB_DIR}")
print(f"   Metadata: {metadata_path}")
