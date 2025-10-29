"""
HOG (Histogram of Oriented Gradients) feature extractor.
"""
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog

from .io import load_grayscale_image


class HOGFeatures(BaseEstimator, TransformerMixin):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.

    HOG captures edge patterns and shape information by computing
    gradient orientations in localized portions of an image.

    Parameters:
        img_size: Target image size (width, height) for resizing
        orientations: Number of orientation bins
        pixels_per_cell: Size (in pixels) of a cell
        cells_per_block: Number of cells in each block
        block_norm: Block normalization method ('L1', 'L1-sqrt', 'L2', 'L2-Hys')
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (128, 128),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = 'L2-Hys'
    ):
        self.img_size = img_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        """
        Fit does nothing; this is a stateless transformer.

        Args:
            X: Array-like of image paths
            y: Ignored

        Returns:
            self
        """
        return self

    def transform(self, X) -> np.ndarray:
        """
        Extract HOG features from images.

        Args:
            X: Array-like of image paths (1D)

        Returns:
            Feature matrix of shape (n_samples, n_hog_features)

        Raises:
            ValueError: If any image fails to load
        """
        features_list = []

        for i, path in enumerate(X):
            try:
                # Load image
                img = load_grayscale_image(path, self.img_size)

                # Extract HOG features
                features = hog(
                    img,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm,
                    visualize=False,
                    feature_vector=True
                )

                features_list.append(features)

            except Exception as e:
                raise ValueError(
                    f"Failed to extract HOG features from image {i} ({path}): {e}"
                )

        feature_matrix = np.array(features_list)

        # Validate no NaN/Inf
        if not np.isfinite(feature_matrix).all():
            raise ValueError("HOG features contain NaN or Inf values")

        return feature_matrix
