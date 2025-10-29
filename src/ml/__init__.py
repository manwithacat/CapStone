"""
ML pipeline package for image-based disease classification.

This package provides a scikit-learn compatible pipeline for:
1. Feature extraction from medical images (HOG, GLCM, statistical)
2. Feature scaling
3. Multi-label classification with XGBoost

Key components:
- features: Feature extraction transformers
- pipeline: Pipeline factory
- persist: Save/load utilities
- config: Configuration loading
"""
from .config import load_config, validate_config
from .persist import get_manifest, load_pipeline, save_pipeline
from .pipeline import build_pipeline

__version__ = "0.1.0"

__all__ = [
    'build_pipeline',
    'save_pipeline',
    'load_pipeline',
    'get_manifest',
    'load_config',
    'validate_config',
]
