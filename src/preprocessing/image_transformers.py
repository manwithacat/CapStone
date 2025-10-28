"""
sklearn-style image transformers for preprocessing pipeline.

These transformers follow the fit/transform pattern similar to feature_engine,
making them composable and reusable.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import albumentations as A


class ImageLoader:
    """
    Load images from file paths.

    sklearn-style transformer that converts file paths to image arrays.
    """

    def __init__(self):
        """Initialize the ImageLoader."""
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for ImageLoader).

        Args:
            X: Input data (not used, for sklearn compatibility)
            y: Target data (not used)

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[list, np.ndarray]) -> np.ndarray:
        """
        Load images from file paths.

        Args:
            X: Array-like of file paths

        Returns:
            Array of PIL Image objects
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")

        images = []
        for path in X:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                raise ValueError(f"Error loading image {path}: {str(e)}")

        return np.array(images, dtype=object)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class GrayscaleToRGB:
    """
    Convert grayscale images to RGB format.

    Needed for transfer learning with models pretrained on ImageNet.
    """

    def __init__(self):
        """Initialize the converter."""
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Args:
            X: Input images
            y: Target data (not used)

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert images to RGB if not already.

        Args:
            X: Array of PIL Image objects

        Returns:
            Array of RGB PIL Images
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")

        rgb_images = []
        for img in X:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            rgb_images.append(img)

        return np.array(rgb_images, dtype=object)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class ImageResizer:
    """
    Resize images to target dimensions.

    Uses high-quality LANCZOS resampling.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the resizer.

        Args:
            target_size: Target (height, width)
        """
        self.target_size = target_size
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Args:
            X: Input images
            y: Target data (not used)

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Resize images to target size.

        Args:
            X: Array of PIL Image objects

        Returns:
            Array of resized PIL Images
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")

        resized_images = []
        for img in X:
            # Use Image.Resampling.LANCZOS for Pillow 10+ compatibility
            # Falls back to Image.LANCZOS for older versions
            if hasattr(Image, 'Resampling'):
                resample_filter = Image.Resampling.LANCZOS
            else:
                resample_filter = Image.LANCZOS  # type: ignore
            img_resized = img.resize(self.target_size, resample_filter)
            resized_images.append(img_resized)

        return np.array(resized_images, dtype=object)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class ImageNormalizer:
    """
    Normalize image pixel values.

    Supports ImageNet normalization or simple 0-1 scaling.
    """

    def __init__(self, method: str = 'imagenet'):
        """
        Initialize the normalizer.

        Args:
            method: Normalization method ('imagenet' or '0-1')
        """
        if method not in ['imagenet', '0-1']:
            raise ValueError("method must be 'imagenet' or '0-1'")

        self.method = method
        self.is_fitted_ = False

        # ImageNet mean and std
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Args:
            X: Input images
            y: Target data (not used)

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize images.

        Args:
            X: Array of PIL Image objects

        Returns:
            Array of normalized float arrays
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")

        normalized_images = []
        for img in X:
            # Convert to float array
            img_array = np.array(img, dtype=np.float32)

            # Normalize
            if self.method == 'imagenet':
                img_array /= 255.0
                img_array -= self.mean
                img_array /= self.std
            elif self.method == '0-1':
                img_array /= 255.0

            normalized_images.append(img_array)

        return np.array(normalized_images)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class ImageAugmenter:
    """
    Apply data augmentation to images using Albumentations.

    Can be enabled/disabled for training vs validation.
    """

    def __init__(
        self,
        apply_augmentation: bool = True,
        horizontal_flip: bool = True,
        rotate: bool = True,
        brightness_contrast: bool = True,
        noise: bool = True
    ):
        """
        Initialize the augmenter.

        Args:
            apply_augmentation: Whether to apply augmentation
            horizontal_flip: Enable horizontal flipping
            rotate: Enable rotation/shift/scale
            brightness_contrast: Enable brightness/contrast adjustment
            noise: Enable Gaussian noise
        """
        self.apply_augmentation = apply_augmentation
        self.is_fitted_ = False

        # Build augmentation pipeline
        transforms = []

        if apply_augmentation:
            if horizontal_flip:
                transforms.append(A.HorizontalFlip(p=0.5))
            if rotate:
                transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5
                    )
                )
            if brightness_contrast:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    )
                )
            if noise:
                # GaussNoise uses std_range in albumentations 1.3+
                # std_range normalized values (0-1 range)
                transforms.append(A.GaussNoise(std_range=(0.01, 0.05), p=0.3))

        self.transform_pipeline = A.Compose(transforms) if transforms else None

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Args:
            X: Input images
            y: Target data (not used)

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to images.

        Args:
            X: Array of numpy arrays (H, W, C)

        Returns:
            Array of augmented images
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")

        if not self.apply_augmentation or self.transform_pipeline is None:
            return X

        augmented_images = []
        for img_array in X:
            # Apply augmentation
            augmented = self.transform_pipeline(image=img_array)['image']
            augmented_images.append(augmented)

        return np.array(augmented_images)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
