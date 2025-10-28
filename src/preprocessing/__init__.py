"""
Image preprocessing pipelines for NIH Chest X-Ray dataset.

## Why Not feature_engine?

feature_engine is an excellent library for **tabular data preprocessing**
(handling missing values, categorical encoding, outlier treatment, etc.).
However, it is NOT designed for image data preprocessing.

For images, we need specialized operations:
- Image loading and format conversion
- Resizing and augmentation (rotation, flipping, noise)
- Pixel normalization for deep learning

## Our Approach

We follow the same **design patterns** as feature_engine and sklearn:
- Custom transformers with fit()/transform() methods
- Composable Pipeline architecture
- Reusable, testable, maintainable code
- Easy serialization and deployment

This demonstrates good software engineering while using the right tools
for computer vision tasks.

## Pipeline Architecture

```
Input: Image file paths
    ↓
┌─────────────────────────┐
│  1. ImageLoader         │  Load images from disk
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  2. GrayscaleToRGB      │  Convert to 3-channel RGB
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  3. ImageResizer        │  Resize to 224x224
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  4. ImageAugmenter      │  (Training only) Flip, rotate, noise
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  5. ImageNormalizer     │  ImageNet mean/std normalization
└─────────────────────────┘
    ↓
Output: Preprocessed numpy arrays ready for model training
```
"""

from .image_transformers import (
    ImageLoader,
    ImageResizer,
    GrayscaleToRGB,
    ImageNormalizer,
    ImageAugmenter
)

from .pipeline import (
    ChestXRayPreprocessingPipeline,
    create_train_pipeline,
    create_inference_pipeline
)

__all__ = [
    'ImageLoader',
    'ImageResizer',
    'GrayscaleToRGB',
    'ImageNormalizer',
    'ImageAugmenter',
    'ChestXRayPreprocessingPipeline',
    'create_train_pipeline',
    'create_inference_pipeline'
]
