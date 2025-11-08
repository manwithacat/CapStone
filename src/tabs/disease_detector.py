"""
Disease Detector Tab - Interactive Prediction Interface

This module renders the Disease Detector tab allowing users to upload
chest X-ray images and receive AI-powered disease predictions with
Grad-CAM visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import random
import time
from src.utils.model_loader import (
    load_densenet_model,
    preprocess_image,
    predict_diseases,
    get_disease_classes,
    get_top_predictions,
    format_prediction_text
)
from src.utils.gradcam import (
    generate_gradcam_for_disease,
    get_top_gradcam_predictions
)


def render_disease_detector_tab(df, disease_colors=None):
    """
    Render the Disease Detector tab with image upload and prediction interface.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata
        disease_colors (dict, optional): Color mapping for disease visualization
    """
    # Initialize rate limiting in session state
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = 0

    # Memory management: Clear old cached predictions (keep only last 5)
    prediction_keys = [k for k in st.session_state.keys() if k.startswith('predictions_')]
    if len(prediction_keys) > 5:
        # Sort by creation time (not available, so just keep last 5 alphabetically)
        prediction_keys.sort()
        for old_key in prediction_keys[:-5]:
            del st.session_state[old_key]

    st.header("🔍 AI-Powered Disease Detection")

    st.markdown("""
    Upload a chest X-ray image or pick a random test image to receive AI-powered disease predictions with
    visual explanations showing which regions of the image influenced the model's decisions.
    """)

    # Load DenseNet121 model (cache it to avoid reloading on every rerun)
    model_path = "models/saved_models/densenet121_best.keras"

    @st.cache_resource
    def get_model():
        """Load and cache the model. Use clear_cache to reload on errors."""
        try:
            return load_densenet_model(model_path)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None

    model = get_model()
    if model is None:
        st.stop()

    # -------------------------------------------------------------------------
    # IMAGE SOURCE SELECTION
    # -------------------------------------------------------------------------

    # Create tabs for upload vs random selection (random first)
    random_tab, upload_tab = st.tabs(["🎲 Pick Random X-Ray", "📤 Upload Image"])

    # Store the image for prediction and ground truth
    image_to_predict = None
    ground_truth_labels = None
    image_source = None

    with upload_tab:
        st.markdown("""
        Upload a chest X-ray image for AI-powered disease prediction.

        **Requirements:**
        - File format: PNG, JPG, JPEG
        - Maximum file size: 10 MB
        - Image type: Frontal-view chest X-ray
        - Recommended: Grayscale X-ray images
        - Will be automatically resized to 224x224 for the model
        """)

        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image file...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image (PNG, JPG, or JPEG). Maximum file size: 10 MB",
        )

        if uploaded_file is not None:
            # Check file size (10 MB limit)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum file size is 10 MB.")
            else:
                try:
                    image_to_predict = Image.open(uploaded_file)
                    image_source = uploaded_file.name
                    st.success(f"✅ Image uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
                except Exception as e:
                    st.error(f"❌ Failed to load image: {e}")

    with random_tab:
        st.markdown("""
        Pick a random X-ray from our test set (100 representative images covering all 14 diseases).

        **Benefits:**
        - See model performance on real test data
        - Compare predictions with actual ground truth labels
        - Explore diverse disease presentations
        """)

        # Load sample test images metadata
        sample_images_dir = Path("data/sample_test_images")
        metadata_path = sample_images_dir / "metadata.csv"

        if not metadata_path.exists():
            st.error(f"❌ Sample images not found. Please run: `python scripts/create_sample_test_images.py`")
        else:
            # Button to pick random image with cooldown
            current_time = time.time()
            time_since_last = current_time - st.session_state.last_prediction_time
            cooldown_seconds = 10.0

            # Check if in cooldown period
            in_cooldown = time_since_last < cooldown_seconds

            if in_cooldown:
                remaining = int(cooldown_seconds - time_since_last) + 1
                button_label = f"Next prediction available in: {remaining} seconds"
                button_disabled = True
            else:
                button_label = "🎲 Pick Random X-Ray"
                button_disabled = False

            if st.button(button_label, type="primary", width='stretch', disabled=button_disabled):
                # Load metadata
                metadata_df = pd.read_csv(metadata_path)

                # Pick random image
                random_row = metadata_df.sample(n=1, random_state=None).iloc[0]
                image_filename = random_row['filename']
                image_path = sample_images_dir / image_filename

                if image_path.exists():
                    try:
                        image_to_predict = Image.open(image_path)
                        image_source = image_filename

                        # Get ground truth labels
                        disease_classes = get_disease_classes()
                        ground_truth_labels = {
                            disease: int(random_row[disease])
                            for disease in disease_classes
                        }

                        # Store in session state for persistent display
                        st.session_state.random_image = image_to_predict
                        st.session_state.random_filename = image_filename
                        st.session_state.random_ground_truth = ground_truth_labels

                    except Exception as e:
                        st.error(f"❌ Failed to load random image: {e}")
                else:
                    st.error(f"❌ Image file not found: {image_path}")

            # Display area with equal columns (50% smaller than before)
            col_image, col_info = st.columns([1, 1])

            with col_image:
                if 'random_image' in st.session_state:
                    # Show the selected image
                    st.image(
                        st.session_state.random_image,
                        caption=f"Selected: {st.session_state.random_filename}",
                        width="stretch"
                    )
                    # Set for prediction
                    image_to_predict = st.session_state.random_image
                    image_source = st.session_state.random_filename
                    ground_truth_labels = st.session_state.random_ground_truth
                else:
                    # Show placeholder
                    st.markdown("""
                    <div style='
                        border: 2px dashed #ccc;
                        border-radius: 10px;
                        padding: 80px 20px;
                        text-align: center;
                        background-color: #f8f9fa;
                        color: #6c757d;
                    '>
                        <p style='font-size: 3rem; margin: 0;'>🖼️</p>
                        <p style='font-size: 1.1rem; margin-top: 10px; margin-bottom: 5px;'>X-Ray will appear here</p>
                        <p style='font-size: 0.9rem; color: #999;'>Click the button above to pick a random X-ray</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_info:
                if 'random_ground_truth' in st.session_state:
                    st.markdown("**📋 Ground Truth Labels**")

                    # Count positive findings
                    positive_findings = [
                        disease for disease, label in st.session_state.random_ground_truth.items()
                        if label == 1
                    ]

                    st.metric("Positive Findings", len(positive_findings))

                    if positive_findings:
                        st.markdown("**Diseases Present:**")
                        for disease in positive_findings:
                            st.markdown(f"- ✅ {disease}")
                    else:
                        st.success("✅ No Finding (Normal)")

                    st.markdown("---")
                    st.caption(f"📁 {st.session_state.random_filename}")
                else:
                    st.markdown("**💡 How to use:**")
                    st.markdown("""
                    1. Click the button to select a random X-ray
                    2. View the image and ground truth labels
                    3. Scroll down to see AI predictions
                    4. Compare predictions with actual diagnoses
                    """)

                    st.markdown("---")
                    st.caption("🎲 100 test images available")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # PREDICTION RESULTS
    # -------------------------------------------------------------------------
    st.subheader("3️⃣ Prediction Results")

    if image_to_predict is not None:
        # Create a unique key for this image to cache results
        if isinstance(image_to_predict, Image.Image):
            import hashlib
            image_key = hashlib.md5(image_to_predict.tobytes()).hexdigest()
        else:
            image_key = str(id(image_to_predict))

        # Check if we've already computed predictions for this image
        cache_key = f"predictions_{image_key}"
        is_loading = cache_key not in st.session_state

        if is_loading:
            # Update timestamp when prediction starts (starts cooldown)
            st.session_state.last_prediction_time = time.time()

            # Set loading flag to disable all interactive elements
            st.session_state['is_loading'] = True

            # Run prediction (only once per image)
            with st.spinner("🔮 Running AI prediction..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image_to_predict)

                    # Get predictions
                    predictions = predict_diseases(model, processed_image)

                    # Get top 3 predictions
                    top_3 = get_top_predictions(predictions, top_k=3, threshold=0.0)

                    # Generate Grad-CAM for top predictions
                    # Note: disease_classes is defined outside this block for reuse
                    predictions_array = np.array([predictions[disease] for disease in get_disease_classes()])

                    gradcam_results = get_top_gradcam_predictions(
                        model,
                        processed_image,
                        predictions_array,
                        top_k=3
                    )

                    # Cache results in session state (convert to native Python types)
                    # Important: Don't store TensorFlow tensors - causes MPS issues on Apple Silicon
                    st.session_state[cache_key] = {
                        'predictions': {k: float(v) for k, v in predictions.items()},
                        'top_3': [(disease, float(prob)) for disease, prob in top_3],
                        'gradcam_results': gradcam_results  # PIL Images are safe to cache
                    }

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Prediction failed: {error_msg}")

                    # If it's a TensorFlow Metal error, suggest clearing cache
                    if "mps_placeholder" in error_msg or "Metal" in error_msg or "feed tensor" in error_msg:
                        st.warning("""
                        🔧 **TensorFlow Metal Error Detected**

                        This is a known issue with Apple Silicon GPU acceleration. Try:
                        1. Refresh the page (F5 or Cmd+R)
                        2. If issue persists, the model cache may be corrupted

                        Technical details: TensorFlow's Metal backend can crash when handling rapid concurrent requests.
                        The rate limiting should prevent this, but if you see this error, please refresh.
                        """)

                    st.session_state[cache_key] = {
                        'predictions': None,
                        'top_3': [],
                        'gradcam_results': []
                    }
                finally:
                    # Clear loading flag
                    st.session_state['is_loading'] = False

        # Retrieve cached results
        cached_data = st.session_state.get(cache_key, {})
        predictions = cached_data.get('predictions')
        top_3 = cached_data.get('top_3', [])
        gradcam_results = cached_data.get('gradcam_results', [])

        # Check if currently loading
        is_currently_loading = st.session_state.get('is_loading', False)

        # Get disease classes list (needed for rendering, always available)
        disease_classes = get_disease_classes()

        if predictions is not None and not is_currently_loading:
            # Show threshold slider only after predictions are ready and not loading
            # Slider is completely hidden during load events to prevent any interaction
            threshold = st.slider(
                "🎯 Prediction Threshold (for binary classification):",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                help="""
                Probabilities above this threshold are classified as POSITIVE.

                **Default: 0.35** (optimized for screening/diagnostic support)
                - Lower thresholds (0.3-0.4): Higher sensitivity, fewer missed diseases (screening)
                - Higher thresholds (0.6-0.7): Higher specificity, fewer false alarms (confirmation)

                For medical screening tools, literature recommends lower thresholds to minimize false negatives.
                """
            )

            st.markdown("---")


            # Display top predictions with Grad-CAM
            st.markdown("### 🏆 Top 3 Most Likely Diseases with Grad-CAM Visualization")

            # Display top 3 predictions with indicators
            st.markdown("**Top 3 Predictions:**")
            pred_cols = st.columns(3)
            for i, (disease, prob) in enumerate(top_3):
                with pred_cols[i]:
                    # Color based on probability
                    if prob >= 0.5:
                        color = "🔴"  # High probability
                        bgcolor = "#ffe6e6"
                    elif prob >= 0.3:
                        color = "🟡"  # Medium probability
                        bgcolor = "#fff9e6"
                    else:
                        color = "🟢"  # Low probability
                        bgcolor = "#e6ffe6"

                    st.markdown(f"""
                    <div style='
                        background-color: {bgcolor};
                        padding: 15px;
                        border-radius: 10px;
                        text-align: center;
                        border: 2px solid #ddd;
                    '>
                        <p style='font-size: 1.5rem; margin: 0;'>{color}</p>
                        <p style='font-size: 1.1rem; font-weight: bold; margin: 5px 0;'>{disease}</p>
                        <p style='font-size: 1.3rem; font-weight: bold; color: #333; margin: 0;'>{prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Display Grad-CAM heatmaps for comparison
            st.markdown("**🔥 Grad-CAM Heatmap Comparison**")
            st.caption("Compare which regions influenced each disease prediction (red/yellow = high influence)")

            if len(gradcam_results) > 0:
                # Display available Grad-CAMs (may be less than 3 if generation failed for some)
                num_cols = min(len(gradcam_results), 3)
                gradcam_cols = st.columns(num_cols)

                for i, result in enumerate(gradcam_results[:3]):
                    with gradcam_cols[i]:
                        # Get disease name from top_3 predictions
                        disease_name = top_3[i][0] if i < len(top_3) else f"Disease {i+1}"
                        st.markdown(f"**{disease_name}**")
                        gradcam_overlay = result['overlay']
                        st.image(gradcam_overlay, caption=f"Focus areas for {disease_name}", width="stretch")

                # Show info if some Grad-CAMs failed to generate
                if len(gradcam_results) < 3:
                    st.info(f"ℹ️ Generated {len(gradcam_results)} of 3 Grad-CAM visualizations")
            else:
                st.warning("⚠️ Grad-CAM visualization unavailable. This may occur due to model architecture constraints.")

            st.markdown("---")

            # Ground truth comparison (if available)
            if ground_truth_labels is not None:
                st.markdown("### ✅ Ground Truth Comparison")
                st.markdown("**How well did the model perform on this test image?**")

                # Create comprehensive comparison table (using threshold from above)
                comparison_data = []

                for disease in disease_classes:
                    pred_prob = predictions[disease]
                    pred_binary = 1 if pred_prob >= threshold else 0
                    actual = ground_truth_labels[disease]

                    # Determine if prediction matches ground truth
                    match = pred_binary == actual
                    match_icon = "✅" if match else "❌"

                    # Prediction label
                    pred_label = "✅ POS" if pred_binary == 1 else "❌ NEG"
                    actual_label = "POSITIVE" if actual == 1 else "NEGATIVE"

                    comparison_data.append({
                        "Disease": disease,
                        "Confidence (%)": f"{pred_prob*100:.1f}%",
                        "Binary Prediction": pred_label,
                        "Ground Truth": actual_label,
                        "Match": match_icon,
                        "Correct": match
                    })

                comparison_df = pd.DataFrame(comparison_data)

                # Calculate accuracy on this image
                correct_count = comparison_df['Correct'].sum()
                total_count = len(comparison_df)
                accuracy = (correct_count / total_count) * 100

                st.metric(
                    label="Prediction Accuracy on This Image",
                    value=f"{accuracy:.1f}%",
                    delta=f"{correct_count}/{total_count} diseases correct"
                )

                # Column explanations
                st.caption("""
                **Column Guide:**
                • **Confidence (%)** = Model's certainty (0-100%)
                • **Binary Prediction** = Classification using threshold (POS if ≥ threshold, NEG if < threshold)
                • **Ground Truth** = Actual diagnosis from radiologist
                • **Match** = Does binary prediction match ground truth?
                """)

                # Display comparison table (all 14 diseases in full without scrolling)
                st.dataframe(
                    comparison_df[["Disease", "Confidence (%)", "Binary Prediction", "Ground Truth", "Match"]],
                    width="stretch",
                    height=525,  # Enough to show all 14 diseases without scrolling
                    hide_index=True
                )

                # Summary
                if accuracy == 100:
                    st.success("🎉 Perfect predictions! All diseases correctly identified.")
                elif accuracy >= 80:
                    st.info(f"👍 Good performance! {correct_count}/{total_count} diseases correct.")
                else:
                    st.warning(f"⚠️ Room for improvement. {correct_count}/{total_count} diseases correct.")

    else:
        st.info("👆 Please select an X-ray image to get started (upload or pick random)")
        predictions = None

    # -------------------------------------------------------------------------
    # DISCLAIMER & TECHNICAL DETAILS
    # -------------------------------------------------------------------------
    st.markdown("---")

    st.error("""
    ⚠️ **IMPORTANT DISCLAIMER**

    This tool is for **research and educational purposes ONLY**. It is NOT approved for clinical use.

    - Do NOT use for actual medical diagnosis
    - Do NOT upload real patient data without proper authorization
    - All predictions must be reviewed by qualified radiologists
    - False positives and false negatives are possible
    - This tool does NOT replace professional medical judgment
    """)
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        **Prediction Pipeline:**

        1. **Image Preprocessing**
           - Resize to 224x224 pixels
           - Convert to RGB (3 channels)
           - Apply ImageNet normalization (mean/std)

        2. **Model Inference**
           - Forward pass through DenseNet121
           - Multi-label binary classification (14 outputs)
           - Sigmoid activation for independent disease probabilities

        3. **Post-processing**
           - Apply threshold (adjustable slider) for binary predictions
           - Rank diseases by probability
           - Generate Grad-CAM visualizations for top predictions

        **Model Architecture: DenseNet121**
        - Pre-trained on ImageNet, fine-tuned on NIH X-rays
        - 7.6M parameters (parameter efficient)
        - Two-stage training (feature extraction + fine-tuning)
        - Test AUC: 0.7529 (best performing model)
        - Trained on NVIDIA A100 GPU (80GB)

        **Performance:**
        - Inference time: ~2-3 seconds per image (CPU)
        - Model size: 37 MB
        - Memory footprint: ~500 MB
        - Suitable for Streamlit Cloud deployment
        """)

    with st.expander("🔥 What is Grad-CAM?"):
        st.markdown("""
        **Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainable AI technique that helps us understand
        *what* the model is "looking at" when making predictions.

        **How it works:**

        1. **Identify Important Regions**
           - The model processes the X-ray through multiple layers
           - Grad-CAM analyzes which parts of the image activate the final prediction
           - It computes gradients flowing back to the last convolutional layer

        2. **Generate Heatmap**
           - Red/yellow regions = areas that strongly influenced the prediction
           - Blue/purple regions = areas with less influence
           - The brighter the color, the more important that region

        3. **Clinical Interpretation**
           - Radiologists can verify if the model is focusing on clinically relevant areas
           - Helps identify if the model has learned correct diagnostic patterns
           - Reveals potential biases or spurious correlations

        **Why it matters for medical AI:**

        - **Trust & Transparency**: Radiologists can see the model's "reasoning"
        - **Error Detection**: Spot cases where the model focuses on wrong regions
        - **Educational Value**: Learn which image features correlate with diseases
        - **Regulatory Compliance**: Explainability is often required for clinical AI

        **Example:**
        - For pneumonia, Grad-CAM should highlight areas of lung consolidation
        - For cardiomegaly, it should focus on the heart's borders and size
        - If it highlights medical devices instead of pathology, that's a red flag!

        **Reference:**
        - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
        """)

    st.markdown("---")

    st.info("""
    💡 **Tip:** Try uploading different chest X-ray images to see how the model performs!
    The model was trained on 78,566 images from 21,563 patients and tested on 16,491 images.
    """)
