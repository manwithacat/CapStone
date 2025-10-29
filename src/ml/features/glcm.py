"""
GLCM (Gray-Level Co-occurrence Matrix) texture feature extractor.
"""
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import graycomatrix, graycoprops

from .io import load_grayscale_image


class GLCMTexture(BaseEstimator, TransformerMixin):
    """
    Extract texture features using GLCM (Gray-Level Co-occurrence Matrix).

    GLCM captures texture patterns by measuring spatial relationships
    between pixel intensities.

    Parameters:
        img_size: Target image size (width, height) for resizing
        levels: Number of gray levels to quantize to (reduces computation)
        distances: List of pixel pair distance offsets
        angles: List of angles in radians
        properties: List of texture properties to compute
            ('contrast', 'dissimilarity', 'homogeneity', 'energy',
             'correlation', 'ASM')
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (128, 128),
        levels: int = 8,
        distances: tuple[int, ...] = (1,),
        angles: tuple[float, ...] = (0, np.pi/4, np.pi/2, 3*np.pi/4),
        properties: tuple[str, ...] = ('contrast', 'homogeneity', 'energy', 'correlation')
    ):
        self.img_size = img_size
        self.levels = levels
        self.distances = distances
        self.angles = angles
        self.properties = properties

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
        Extract GLCM texture features from images.

        Args:
            X: Array-like of image paths (1D)

        Returns:
            Feature matrix of shape (n_samples, n_texture_features)
            where n_texture_features = len(properties) * len(angles) * len(distances)

        Raises:
            ValueError: If any image fails to load
        """
        features_list = []

        for i, path in enumerate(X):
            try:
                # Load image
                img = load_grayscale_image(path, self.img_size)

                # Quantize to reduce gray levels (speeds up GLCM computation)
                img_quantized = (img // (256 // self.levels)).astype(np.uint8)

                # Compute GLCM
                glcm = graycomatrix(
                    img_quantized,
                    distances=list(self.distances),
                    angles=list(self.angles),
                    levels=self.levels,
                    symmetric=True,
                    normed=True
                )

                # Extract texture properties
                property_features = []
                for prop in self.properties:
                    prop_values = graycoprops(glcm, prop).flatten()
                    property_features.extend(prop_values)

                features_list.append(property_features)

            except Exception as e:
                raise ValueError(
                    f"Failed to extract GLCM features from image {i} ({path}): {e}"
                )

        feature_matrix = np.array(features_list)

        # Validate no NaN/Inf
        if not np.isfinite(feature_matrix).all():
            raise ValueError("GLCM features contain NaN or Inf values")

        return feature_matrix
