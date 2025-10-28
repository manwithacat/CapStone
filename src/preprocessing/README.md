# Image Preprocessing Pipeline

## Overview

This module provides reusable, sklearn-style transformers for preprocessing chest X-ray images, following the design patterns of `feature_engine` and `sklearn`.

## Why Not feature_engine?

`feature_engine` is designed for **tabular data** preprocessing (missing values, categorical encoding, outliers). For **image data**, we need specialized operations like:

- Image loading and format conversion
- Resizing and augmentation
- Pixel normalization for neural networks

## Architecture

Our pipeline follows **sklearn/feature_engine design patterns**:

```python
# Each transformer has fit/transform methods
transformer = ImageLoader()
transformer.fit(X_train)
X_transformed = transformer.transform(X_test)

# Or use fit_transform
X_transformed = transformer.fit_transform(X_train)
```

## Pipeline Components

### 1. ImageLoader
Loads images from file paths.

```python
from preprocessing import ImageLoader

loader = ImageLoader()
images = loader.fit_transform(image_paths)
```

### 2. GrayscaleToRGB
Converts grayscale images to RGB (needed for transfer learning).

```python
from preprocessing import GrayscaleToRGB

converter = GrayscaleToRGB()
rgb_images = converter.fit_transform(images)
```

### 3. ImageResizer
Resizes images to target dimensions using high-quality LANCZOS resampling.

```python
from preprocessing import ImageResizer

resizer = ImageResizer(target_size=(224, 224))
resized = resizer.fit_transform(images)
```

### 4. ImageAugmenter
Applies data augmentation (rotation, flip, noise, brightness) using Albumentations.

```python
from preprocessing import ImageAugmenter

# Training augmentation
train_augmenter = ImageAugmenter(
    apply_augmentation=True,
    horizontal_flip=True,
    rotate=True,
    brightness_contrast=True,
    noise=True
)

# Validation (no augmentation)
val_augmenter = ImageAugmenter(apply_augmentation=False)
```

### 5. ImageNormalizer
Normalizes pixel values (ImageNet mean/std or 0-1 scaling).

```python
from preprocessing import ImageNormalizer

# ImageNet normalization (for transfer learning)
normalizer = ImageNormalizer(method='imagenet')

# Or simple 0-1 scaling
normalizer = ImageNormalizer(method='0-1')
```

## Complete Pipeline

### Option 1: Use Pre-configured Pipelines

```python
from preprocessing import create_train_pipeline, create_inference_pipeline

# Training pipeline (with augmentation)
train_pipeline = create_train_pipeline(target_size=(224, 224))
train_images = train_pipeline.fit_transform(train_paths)

# Inference pipeline (no augmentation)
val_pipeline = create_inference_pipeline(target_size=(224, 224))
val_images = val_pipeline.fit_transform(val_paths)
```

### Option 2: Custom Pipeline

```python
from preprocessing import ChestXRayPreprocessingPipeline

# Custom configuration
pipeline = ChestXRayPreprocessingPipeline(
    target_size=(224, 224),
    normalize_method='imagenet',
    apply_augmentation=True,
    augmentation_config={
        'horizontal_flip': True,
        'rotate': False,  # Disable rotation
        'brightness_contrast': True,
        'noise': False  # Disable noise
    }
)

# Fit and transform
pipeline.fit(train_paths)
processed_images = pipeline.transform(test_paths)
```

## Serialization

Save and load pipelines for production deployment:

```python
# Save
pipeline.save('models/preprocessing_pipeline.pkl')

# Load
from preprocessing import ChestXRayPreprocessingPipeline
pipeline = ChestXRayPreprocessingPipeline.load('models/preprocessing_pipeline.pkl')
```

## Benefits

- ✅ **Modular**: Each transformer is independent and testable
- ✅ **Composable**: Build custom pipelines easily
- ✅ **Reusable**: Use across notebooks and production
- ✅ **Consistent**: Follows sklearn/feature_engine patterns
- ✅ **Maintainable**: Clear separation of concerns
- ✅ **Serializable**: Easy deployment

## Example: End-to-End Usage

```python
from preprocessing import create_train_pipeline, create_inference_pipeline
import numpy as np

# Load your data
train_paths = ['path/to/image1.png', 'path/to/image2.png', ...]
val_paths = ['path/to/val1.png', 'path/to/val2.png', ...]

# Create pipelines
train_pipeline = create_train_pipeline(target_size=(224, 224))
val_pipeline = create_inference_pipeline(target_size=(224, 224))

# Preprocess
train_images = train_pipeline.fit_transform(train_paths)
val_images = val_pipeline.transform(val_paths)  # Use same fitted pipeline

# Now train_images and val_images are ready for model training
print(train_images.shape)  # (N, 224, 224, 3)
print(val_images.shape)    # (M, 224, 224, 3)
```

## Testing

Each transformer can be tested independently:

```python
from preprocessing import ImageLoader
from pathlib import Path

# Test ImageLoader
loader = ImageLoader()
test_paths = ['test_image.png']
images = loader.fit_transform(test_paths)
assert len(images) == 1
assert images[0] is not None
```

## Integration with TensorFlow/Keras

```python
from tensorflow import keras
from preprocessing import create_train_pipeline

# Create dataset
train_pipeline = create_train_pipeline()
train_images = train_pipeline.fit_transform(train_paths)
train_labels = np.array(labels)

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Train model
model.fit(train_dataset, epochs=10)
```

## References

- sklearn Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- feature_engine: https://feature-engine.readthedocs.io/
- Albumentations: https://albumentations.ai/
