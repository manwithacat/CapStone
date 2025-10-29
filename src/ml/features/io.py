"""
Shared image loading utilities for feature extractors.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_grayscale_image(
    path: Union[str, Path],
    img_size: tuple[int, int]
) -> np.ndarray:
    """
    Load and preprocess a grayscale image for feature extraction.

    Args:
        path: Path to image file
        img_size: Target size (width, height) to resize to

    Returns:
        Grayscale image as numpy array

    Raises:
        ValueError: If image cannot be loaded
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Image path does not exist: {path}")

    # Load as grayscale
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # Resize to target size
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    return img
