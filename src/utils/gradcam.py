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
        last_conv_layer_name: Name/object of last conv layer, or tuple (base_model_name, layer_name) for nested
        pred_index: Index of prediction to visualize (None = max prediction)

    Returns:
        heatmap: Numpy array (224, 224) with values in [0, 1]
    """
    # Handle different input types
    if isinstance(last_conv_layer_name, tuple):
        # Nested model: (base_model_name, layer_name)
        base_model_name, layer_name = last_conv_layer_name

        # For nested models, use a manual approach with GradientTape
        # to extract intermediate activations
        base_model = model.get_layer(base_model_name)

        def find_layer_in_model(model, layer_name):
            """Recursively search for a layer by name."""
            for layer in model.layers:
                if layer.name == layer_name:
                    return layer
                if isinstance(layer, keras.Model):
                    found = find_layer_in_model(layer, layer_name)
                    if found is not None:
                        return found
            return None

        conv_layer = find_layer_in_model(base_model, layer_name)
        if conv_layer is None:
            raise ValueError(f"Could not find layer '{layer_name}' in base model")

        # Manual computation with GradientTape watching intermediate outputs
        with tf.GradientTape() as tape:
            # Convert to tensor and watch it
            img_tensor = tf.convert_to_tensor(img_array)
            tape.watch(img_tensor)

            # Forward pass through base model, storing intermediate activations
            # We'll capture the conv layer output during the forward pass
            conv_output_val = None

            # Create a custom forward pass that captures the conv layer output
            # by monkey-patching the layer's call method temporarily
            original_call = conv_layer.call

            def capturing_call(inputs, *args, **kwargs):
                nonlocal conv_output_val
                output = original_call(inputs, *args, **kwargs)
                conv_output_val = output
                return output

            # Temporarily replace the call method
            conv_layer.call = capturing_call

            try:
                # Run the full model
                predictions = model(img_tensor, training=False)

                # If pred_index not specified, use the max prediction
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])

                # Get the prediction for the specified class
                class_channel = predictions[:, pred_index]

            finally:
                # Restore original call method
                conv_layer.call = original_call

            # Compute gradients of the prediction with respect to conv layer output
            try:
                grads = tape.gradient(class_channel, conv_output_val)
            except Exception as e:
                raise ValueError(f"Gradient computation failed: {str(e)}")

        # If gradients or conv_output_val are None, fall back to simple approach
        if grads is None or conv_output_val is None:
            raise ValueError("Could not compute gradients for nested model. Layer may not be in gradient path.")

        conv_outputs = conv_output_val
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    elif isinstance(last_conv_layer_name, str):
        # Simple case: layer name at model level
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
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
        try:
            grads = tape.gradient(class_channel, conv_outputs)
        except Exception as e:
            raise ValueError(f"Gradient computation failed: {str(e)}")

        if grads is None:
            raise ValueError("Gradients are None - layer may not be in gradient path")

        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    else:
        # Layer object directly (only works for top-level layers)
        last_conv_layer = last_conv_layer_name
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
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
        try:
            grads = tape.gradient(class_channel, conv_outputs)
        except Exception as e:
            raise ValueError(f"Gradient computation failed: {str(e)}")

        if grads is None:
            raise ValueError("Gradients are None - layer may not be in gradient path")

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
    # For models with nested structure (like our DenseNet121 wrapper), we need to look inside
    last_conv_layer = None

    # Check if model has a densenet121 base model (common when using transfer learning)
    densenet_base = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower() or isinstance(layer, keras.Model):
            # This might be our base model
            try:
                densenet_base = model.get_layer(layer.name)
                break
            except:
                continue

    # If we found a nested base model, look for conv layers inside it
    if densenet_base is not None:
        from tensorflow.keras.layers import Conv2D
        # Get all conv layers from the base model
        conv_layers = [l for l in densenet_base.layers if isinstance(l, Conv2D)]
        if not conv_layers:
            # Try by name if isinstance doesn't work
            conv_layers = [l for l in densenet_base.layers if 'conv' in l.name.lower()]

        if conv_layers:
            # Use the last convolutional layer from the base model
            # Pass as tuple (base_model_name, layer_name) for nested models
            last_conv_layer = (densenet_base.name, conv_layers[-1].name)
        else:
            # Fall back to using a known DenseNet121 layer name
            last_conv_layer = (densenet_base.name, 'conv5_block16_2_conv')

    # If we still don't have a layer, try the direct approach (for simpler architectures)
    if last_conv_layer is None:
        from tensorflow.keras.layers import Conv2D
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                last_conv_layer = layer
                break

        # Try by name if isinstance doesn't work
        if last_conv_layer is None:
            conv_layer_names = [layer for layer in model.layers if 'conv' in layer.name.lower()]
            if conv_layer_names:
                last_conv_layer = conv_layer_names[-1]

    if last_conv_layer is None:
        raise ValueError(f"Could not find convolutional layer in model. Model may not support Grad-CAM. Model layers: {[l.name for l in model.layers[:10]]}")

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
    errors = []

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
            # Silently skip errors - don't pollute logs
            # Grad-CAM failures are non-critical (predictions still work)
            errors.append(str(e))
            continue

    # Only log if ALL Grad-CAMs failed (indicates a real problem)
    if len(errors) == top_k and len(results) == 0:
        print(f"Warning: All Grad-CAM visualizations failed. Predictions still available but explanations unavailable.")

    return results
