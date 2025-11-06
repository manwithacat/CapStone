"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization utility.

Provides functions to generate heatmap visualizations showing which regions
of an X-ray image influenced the model's disease predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a specific prediction.

    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: Keras model
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Index of prediction to visualize (None = max prediction)

    Returns:
        heatmap: Numpy array (224, 224) with values in [0, 1]
    """
    # Create a model that maps input to activations + predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # If pred_index not specified, use the max prediction
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Get the prediction for the specified class
        class_channel = predictions[:, pred_index]

    # Compute gradients of the prediction with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map channels by their importance
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Create the heatmap (channel-wise mean)
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        img: Original PIL Image or numpy array
        heatmap: Grad-CAM heatmap (H, W) in [0, 1]
        alpha: Transparency of overlay (0=invisible, 1=opaque)
        colormap: OpenCV colormap for heatmap

    Returns:
        PIL Image with heatmap overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Resize heatmap to match image size
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR (OpenCV) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    overlay = (heatmap_colored * alpha + img * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(overlay)


def generate_gradcam_for_disease(model, img_array, disease_index, disease_name=None):
    """
    Generate Grad-CAM visualization for a specific disease prediction.

    Args:
        model: Trained Keras model (DenseNet121)
        img_array: Preprocessed image (1, 224, 224, 3)
        disease_index: Index of disease in output (0-13)
        disease_name: Optional name of disease for logging

    Returns:
        tuple: (heatmap, overlay_image)
            - heatmap: Raw heatmap array (224, 224)
            - overlay_image: PIL Image with overlay
    """
    # For DenseNet121, the last conv layer is typically 'conv5_block16_concat'
    # or you can find it with: [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    # Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'filters'):  # Convolutional layer
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer in model")

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer,
        pred_index=disease_index
    )

    # Get original image for overlay (denormalize from ImageNet preprocessing)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    img_denorm = img_array[0] * IMAGENET_STD + IMAGENET_MEAN
    img_denorm = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)

    # Create overlay
    overlay = overlay_heatmap_on_image(img_denorm, heatmap, alpha=0.4)

    return heatmap, overlay


def get_top_gradcam_predictions(model, img_array, predictions, top_k=3):
    """
    Generate Grad-CAM visualizations for top K predictions.

    Args:
        model: Trained model
        img_array: Preprocessed image
        predictions: Model predictions (14 probabilities)
        top_k: Number of top predictions to visualize

    Returns:
        list of dicts with keys: disease_index, disease_name, probability, heatmap, overlay
    """
    # Get top K disease indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        try:
            heatmap, overlay = generate_gradcam_for_disease(
                model,
                img_array,
                disease_index=int(idx)
            )

            results.append({
                'disease_index': int(idx),
                'probability': float(predictions[idx]),
                'heatmap': heatmap,
                'overlay': overlay
            })
        except Exception as e:
            print(f"Error generating Grad-CAM for index {idx}: {e}")

    return results
