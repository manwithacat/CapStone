"""
Model loader and prediction utilities for DenseNet121 chest X-ray classifier.

This module provides functions for loading the trained DenseNet121 model,
preprocessing images, and making predictions on chest X-ray images.
"""

import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras


# Disease classes (14 conditions in NIH Chest X-Ray dataset)
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

# ImageNet normalization parameters (used during training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@st.cache_resource
def load_densenet_model(model_path: str) -> keras.Model:
    """
    Load the trained DenseNet121 model from disk.

    Uses Streamlit's cache_resource to load the model only once
    and reuse it across sessions.

    Args:
        model_path: Path to the saved .keras model file

    Returns:
        Loaded Keras model ready for inference

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted or invalid
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please ensure the DenseNet121 model is in models/saved_models/"
        )

    try:
        # Load model (TensorFlow/Keras format)
        model = keras.models.load_model(model_path, compile=False)

        # Verify model structure
        expected_input_shape = (None, 224, 224, 3)
        if model.input_shape != expected_input_shape:
            raise ValueError(
                f"Unexpected model input shape: {model.input_shape}\n"
                f"Expected: {expected_input_shape}"
            )

        expected_output_classes = 14
        if model.output_shape[1] != expected_output_classes:
            raise ValueError(
                f"Unexpected model output shape: {model.output_shape}\n"
                f"Expected: (None, {expected_output_classes})"
            )

        return model

    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess a chest X-ray image for model inference.

    Steps:
    1. Resize to target size (224x224)
    2. Convert to RGB (3 channels)
    3. Normalize to [0, 1] range
    4. Apply ImageNet normalization (mean/std)
    5. Add batch dimension

    Args:
        image: PIL Image object (grayscale or RGB)
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed numpy array ready for model input
        Shape: (1, 224, 224, 3)
    """
    # Resize to target size
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)

    # Convert to RGB if grayscale (NIH X-rays are grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Apply ImageNet normalization (same as training)
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_diseases(
    model: keras.Model,
    image_array: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Predict disease probabilities for a chest X-ray image.

    Args:
        model: Loaded Keras model
        image_array: Preprocessed image array from preprocess_image()
        threshold: Probability threshold for positive predictions (default: 0.5)

    Returns:
        Dictionary mapping disease names to probabilities
        Example: {"Pneumonia": 0.85, "Effusion": 0.62, ...}
    """
    # Run inference
    predictions = model.predict(image_array, verbose=0)

    # Extract probabilities (remove batch dimension)
    probabilities = predictions[0]

    # Create dictionary mapping disease names to probabilities
    results = {
        disease: float(prob)
        for disease, prob in zip(DISEASE_CLASSES, probabilities)
    }

    return results


def get_disease_classes() -> List[str]:
    """
    Get the list of disease classes.

    Returns:
        List of 14 disease class names
    """
    return DISEASE_CLASSES.copy()


def get_top_predictions(
    predictions: Dict[str, float],
    top_k: int = 3,
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Get the top K disease predictions sorted by probability.

    Args:
        predictions: Dictionary of disease predictions from predict_diseases()
        top_k: Number of top predictions to return
        threshold: Minimum probability threshold (default: 0.0)

    Returns:
        List of (disease_name, probability) tuples sorted by probability
        Example: [("Pneumonia", 0.85), ("Effusion", 0.62), ...]
    """
    # Filter by threshold
    filtered = {
        disease: prob
        for disease, prob in predictions.items()
        if prob >= threshold
    }

    # Sort by probability (descending) and return top K
    sorted_predictions = sorted(
        filtered.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_predictions[:top_k]


def format_prediction_text(
    disease: str,
    probability: float,
    threshold: float = 0.5
) -> str:
    """
    Format a prediction for display.

    Args:
        disease: Disease name
        probability: Predicted probability
        threshold: Threshold for positive/negative classification

    Returns:
        Formatted string with disease name, probability, and classification
        Example: "Pneumonia: 85.2% (POSITIVE)"
    """
    percentage = probability * 100
    classification = "POSITIVE" if probability >= threshold else "NEGATIVE"

    return f"**{disease}**: {percentage:.1f}% ({classification})"
