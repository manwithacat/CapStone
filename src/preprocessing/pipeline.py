"""
Composable preprocessing pipeline for chest X-ray images.

This pipeline follows sklearn/feature_engine patterns, making it:
- Reusable across different notebooks
- Easy to serialize/deserialize
- Testable and maintainable
"""

import numpy as np
from typing import List, Optional
import pickle
from pathlib import Path

from .image_transformers import (
    ImageLoader,
    GrayscaleToRGB,
    ImageResizer,
    ImageNormalizer,
    ImageAugmenter
)


class ChestXRayPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for chest X-ray images.

    Follows the sklearn Pipeline pattern for composability.

    Example:
        >>> pipeline = ChestXRayPreprocessingPipeline(
        ...     target_size=(224, 224),
        ...     apply_augmentation=True
        ... )
        >>> pipeline.fit(train_paths)
        >>> processed_images = pipeline.transform(train_paths)
    """

    def __init__(
        self,
        target_size: tuple = (224, 224),
        normalize_method: str = 'imagenet',
        apply_augmentation: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            target_size: Target image dimensions (height, width)
            normalize_method: Normalization method ('imagenet' or '0-1')
            apply_augmentation: Whether to apply augmentation
            augmentation_config: Custom augmentation settings
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.apply_augmentation = apply_augmentation
        self.augmentation_config = augmentation_config or {}

        # Build pipeline steps
        self.steps = self._build_steps()
        self.is_fitted_ = False

    def _build_steps(self) -> List:
        """
        Build the sequence of transformation steps.

        Returns:
            List of (name, transformer) tuples
        """
        steps = [
            ('loader', ImageLoader()),
            ('rgb_converter', GrayscaleToRGB()),
            ('resizer', ImageResizer(target_size=self.target_size)),
        ]

        # Add augmentation before normalization
        if self.apply_augmentation:
            steps.append((
                'augmenter',
                ImageAugmenter(
                    apply_augmentation=True,
                    **self.augmentation_config
                )
            ))

        # Normalize last
        steps.append((
            'normalizer',
            ImageNormalizer(method=self.normalize_method)
        ))

        return steps

    def fit(self, X, y=None):
        """
        Fit all transformers in the pipeline.

        Args:
            X: Input data (file paths)
            y: Target labels (optional)

        Returns:
            self
        """
        # Fit each transformer
        for name, transformer in self.steps:
            transformer.fit(X, y)

        self.is_fitted_ = True
        return self

    def transform(self, X) -> np.ndarray:
        """
        Apply all transformations sequentially.

        Args:
            X: Input data (file paths)

        Returns:
            Processed image arrays
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before transform")

        # Apply transformations sequentially
        X_transformed = X
        for name, transformer in self.steps:
            X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Args:
            X: Input data (file paths)
            y: Target labels (optional)

        Returns:
            Processed image arrays
        """
        return self.fit(X, y).transform(X)

    def get_params(self) -> dict:
        """
        Get pipeline parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            'target_size': self.target_size,
            'normalize_method': self.normalize_method,
            'apply_augmentation': self.apply_augmentation,
            'augmentation_config': self.augmentation_config
        }

    def save(self, filepath: Path):
        """
        Save the pipeline to disk.

        Args:
            filepath: Path to save the pipeline
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path):
        """
        Load a pipeline from disk.

        Args:
            filepath: Path to the saved pipeline

        Returns:
            Loaded pipeline instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        """String representation of the pipeline."""
        steps_repr = '\n'.join([f"  {i+1}. {name}" for i, (name, _) in enumerate(self.steps)])
        return (
            f"ChestXRayPreprocessingPipeline(\n"
            f"  target_size={self.target_size},\n"
            f"  normalize_method='{self.normalize_method}',\n"
            f"  apply_augmentation={self.apply_augmentation},\n"
            f"  steps=[\n{steps_repr}\n  ]\n"
            f")"
        )


def create_train_pipeline(target_size: tuple = (224, 224)) -> ChestXRayPreprocessingPipeline:
    """
    Create a training pipeline with augmentation enabled.

    Args:
        target_size: Target image dimensions

    Returns:
        Configured training pipeline
    """
    return ChestXRayPreprocessingPipeline(
        target_size=target_size,
        normalize_method='imagenet',
        apply_augmentation=True,
        augmentation_config={
            'horizontal_flip': True,
            'rotate': True,
            'brightness_contrast': True,
            'noise': True
        }
    )


def create_inference_pipeline(target_size: tuple = (224, 224)) -> ChestXRayPreprocessingPipeline:
    """
    Create an inference pipeline without augmentation.

    Args:
        target_size: Target image dimensions

    Returns:
        Configured inference pipeline
    """
    return ChestXRayPreprocessingPipeline(
        target_size=target_size,
        normalize_method='imagenet',
        apply_augmentation=False
    )
