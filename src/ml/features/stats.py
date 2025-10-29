"""
Statistical pixel feature extractor.
"""
from typing import Union

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .io import load_grayscale_image


class StatisticalPixels(BaseEstimator, TransformerMixin):
    """
    Extract basic statistical features from pixel intensities.

    Computes global statistics like mean intensity, standard deviation,
    and edge density using Canny edge detection.

    Parameters:
        img_size: Target image size (width, height) for resizing
        canny_threshold1: First threshold for Canny edge detector
        canny_threshold2: Second threshold for Canny edge detector
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (128, 128),
        canny_threshold1: int = 50,
        canny_threshold2: int = 150
    ):
        self.img_size = img_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

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
        Extract statistical features from images.

        Args:
            X: Array-like of image paths (1D)

        Returns:
            Feature matrix of shape (n_samples, 3)
            Features: [mean_intensity, std_intensity, edge_density]

        Raises:
            ValueError: If any image fails to load
        """
        features_list = []

        for i, path in enumerate(X):
            try:
                # Load image
                img = load_grayscale_image(path, self.img_size)

                # Statistical features
                mean_intensity = float(np.mean(img))
                std_intensity = float(np.std(img))

                # Edge density using Canny edge detection
                edges = cv2.Canny(img, self.canny_threshold1, self.canny_threshold2)
                edge_density = float(np.sum(edges > 0) / edges.size)

                features = [mean_intensity, std_intensity, edge_density]
                features_list.append(features)

            except Exception as e:
                raise ValueError(
                    f"Failed to extract statistical features from image {i} ({path}): {e}"
                )

        feature_matrix = np.array(features_list)

        # Validate no NaN/Inf
        if not np.isfinite(feature_matrix).all():
            raise ValueError("Statistical features contain NaN or Inf values")

        return feature_matrix
