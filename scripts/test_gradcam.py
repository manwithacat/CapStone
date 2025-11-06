"""
Test Grad-CAM generation to debug visualization issues.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_loader import load_densenet_model, preprocess_image, get_disease_classes
from src.utils.gradcam import generate_gradcam_for_disease

def test_gradcam():
    print("=" * 70)
    print("Testing Grad-CAM Generation")
    print("=" * 70)

    # 1. Load model
    print("\n1. Loading DenseNet121 model...")
    model_path = PROJECT_ROOT / "models/saved_models/densenet121_best.keras"

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return

    try:
        model = load_densenet_model(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Check model layers for convolutional layers
    print("\n2. Checking model architecture for convolutional layers...")
    from tensorflow.keras.layers import Conv2D
    import tensorflow as tf

    # Method 1: Check by type (top-level only)
    conv_layers_by_type = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]
    print(f"   Method 1 (top-level isinstance Conv2D): {len(conv_layers_by_type)} layers")

    # Method 2: Check by name (top-level only)
    conv_layers_by_name = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
    print(f"   Method 2 (top-level name contains 'conv'): {len(conv_layers_by_name)} layers")

    # List all top-level layers
    print("\n   Top-level model layers:")
    for i, layer in enumerate(model.layers):
        print(f"      {i}: {layer.name} ({type(layer).__name__})")

    # Method 3: Check for nested DenseNet121 base model
    print("\n   Method 3: Looking inside nested models...")
    densenet_base = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower():
            densenet_base = layer
            print(f"   ✅ Found nested DenseNet121: {layer.name} ({type(layer).__name__})")
            break
        elif isinstance(layer, tf.keras.Model) and layer != model:
            print(f"   Found nested model: {layer.name} ({type(layer).__name__})")
            densenet_base = layer
            break

    # If we found a nested base model, inspect its layers
    conv_layers = []
    if densenet_base is not None:
        print(f"\n   Inspecting {densenet_base.name} internal structure:")
        print(f"   Total layers in base model: {len(densenet_base.layers)}")

        # Show first 20 and last 10 layers
        print(f"\n   First 20 layers inside {densenet_base.name}:")
        for i, layer in enumerate(densenet_base.layers[:20]):
            is_conv = isinstance(layer, Conv2D)
            conv_marker = " <- Conv2D" if is_conv else ""
            print(f"      {i}: {layer.name} ({type(layer).__name__}){conv_marker}")

        print(f"\n   Last 10 layers inside {densenet_base.name}:")
        for i, layer in enumerate(densenet_base.layers[-10:], start=len(densenet_base.layers)-10):
            is_conv = isinstance(layer, Conv2D)
            conv_marker = " <- Conv2D" if is_conv else ""
            print(f"      {i}: {layer.name} ({type(layer).__name__}){conv_marker}")

        # Count convolutional layers
        conv_layers = [l.name for l in densenet_base.layers if isinstance(l, Conv2D)]
        print(f"\n   ✅ Found {len(conv_layers)} Conv2D layers inside {densenet_base.name}")

        if conv_layers:
            print(f"   First conv layer: {conv_layers[0]}")
            print(f"   Last conv layer: {conv_layers[-1]}")
            print(f"   Full path would be: {densenet_base.name}/{conv_layers[-1]}")
    else:
        print("   ❌ No nested DenseNet121 model found!")
        return

    # 3. Load a test image
    print("\n3. Loading test image...")
    sample_images_dir = PROJECT_ROOT / "data/sample_test_images"
    metadata_path = sample_images_dir / "metadata.csv"

    if not metadata_path.exists():
        print(f"❌ Sample images not found at {sample_images_dir}")
        return

    import pandas as pd
    metadata_df = pd.read_csv(metadata_path)
    test_image_filename = metadata_df.iloc[0]['filename']
    test_image_path = sample_images_dir / test_image_filename

    print(f"   Loading: {test_image_filename}")

    try:
        image = Image.open(test_image_path)
        print(f"   ✅ Image loaded: {image.size}, mode: {image.mode}")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        return

    # 4. Preprocess image
    print("\n4. Preprocessing image...")
    try:
        processed_image = preprocess_image(image)
        print(f"   ✅ Preprocessed shape: {processed_image.shape}")
        print(f"   ✅ Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Run prediction
    print("\n5. Running prediction...")
    try:
        predictions = model.predict(processed_image, verbose=0)
        print(f"   ✅ Predictions shape: {predictions.shape}")
        print(f"   Top 3 predictions:")
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        disease_classes = get_disease_classes()
        for idx in top_3_indices:
            print(f"      {disease_classes[idx]}: {predictions[0][idx]:.3f}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Generate Grad-CAM
    print("\n6. Generating Grad-CAM for top prediction...")
    top_prediction_idx = top_3_indices[0]
    print(f"   Disease: {disease_classes[top_prediction_idx]}")
    print(f"   Probability: {predictions[0][top_prediction_idx]:.3f}")

    try:
        heatmap, overlay = generate_gradcam_for_disease(
            model,
            processed_image,
            disease_index=int(top_prediction_idx)
        )
        print(f"   ✅ Grad-CAM generated successfully!")
        print(f"   Heatmap shape: {heatmap.shape}")
        print(f"   Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   Overlay size: {overlay.size}")
    except Exception as e:
        print(f"   ❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("✅ All tests passed! Grad-CAM is working correctly.")
    print("=" * 70)

if __name__ == "__main__":
    test_gradcam()
