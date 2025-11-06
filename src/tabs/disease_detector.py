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
import plotly.graph_objects as go
import random
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
    st.header("🔍 AI-Powered Disease Detection")

    st.markdown("""
    Upload a chest X-ray image or pick a random test image to receive AI-powered disease predictions with
    visual explanations showing which regions of the image influenced the model's decisions.
    """)

    # Load DenseNet121 model
    model_path = "models/saved_models/densenet121_best.keras"

    try:
        model = load_densenet_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
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
            # Button to pick random image
            if st.button("🎲 Pick Random X-Ray", type="primary"):
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

                        st.success(f"✅ Randomly selected: {image_filename}")
                        st.info(f"📊 Ground truth labels loaded for comparison")

                    except Exception as e:
                        st.error(f"❌ Failed to load random image: {e}")
                else:
                    st.error(f"❌ Image file not found: {image_path}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # PREDICTION RESULTS
    # -------------------------------------------------------------------------
    st.subheader("3️⃣ Prediction Results")

    if image_to_predict is not None:
        # Run prediction
        with st.spinner("🔮 Running AI prediction..."):
            try:
                # Preprocess image
                processed_image = preprocess_image(image_to_predict)

                # Get predictions
                predictions = predict_diseases(model, processed_image)

                # Get top 3 predictions
                top_3 = get_top_predictions(predictions, top_k=3, threshold=0.0)

                # Generate Grad-CAM for top predictions
                disease_classes = get_disease_classes()
                predictions_array = np.array([predictions[disease] for disease in disease_classes])

                gradcam_results = get_top_gradcam_predictions(
                    model,
                    processed_image,
                    predictions_array,
                    top_k=3
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                predictions = None
                gradcam_results = []

        if predictions is not None:
            # Display top predictions with Grad-CAM
            st.markdown("### 🏆 Top 3 Most Likely Diseases with Grad-CAM")

            for i, (disease, prob) in enumerate(top_3, 1):
                # Color based on probability
                if prob >= 0.5:
                    color = "🔴"  # High probability
                elif prob >= 0.3:
                    color = "🟡"  # Medium probability
                else:
                    color = "🟢"  # Low probability

                # Display disease prediction
                st.markdown(f"{color} **{i}. {disease}**: {prob*100:.1f}%")

                # Display original and Grad-CAM side by side
                if i <= len(gradcam_results):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**📷 Original X-Ray**")
                        st.image(image_to_predict, caption=f"{image_source or 'X-ray'}", width="stretch")

                    with col2:
                        st.markdown(f"**🔥 Grad-CAM Heatmap for {disease}**")
                        gradcam_overlay = gradcam_results[i-1]['overlay']
                        st.image(gradcam_overlay, caption="Regions influencing prediction", width="stretch")

                st.markdown("---")

            # Ground truth comparison (if available)
            if ground_truth_labels is not None:
                st.markdown("### ✅ Ground Truth Comparison")
                st.markdown("**How well did the model perform on this test image?**")

                # Create comparison table
                comparison_data = []
                threshold = 0.5  # Binary threshold

                for disease in disease_classes:
                    pred_prob = predictions[disease]
                    pred_binary = 1 if pred_prob >= threshold else 0
                    actual = ground_truth_labels[disease]

                    # Determine if prediction matches ground truth
                    match = pred_binary == actual
                    match_icon = "✅" if match else "❌"

                    # Prediction label
                    pred_label = "POSITIVE" if pred_binary == 1 else "NEGATIVE"
                    actual_label = "POSITIVE" if actual == 1 else "NEGATIVE"

                    comparison_data.append({
                        "Disease": disease,
                        "Prediction": f"{pred_label} ({pred_prob*100:.1f}%)",
                        "Actual": actual_label,
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

                # Display comparison table
                st.dataframe(
                    comparison_df[["Disease", "Prediction", "Actual", "Match"]],
                    width="stretch",
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

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DETAILED PREDICTIONS TABLE
    # -------------------------------------------------------------------------
    st.subheader("4️⃣ Detailed Disease Predictions")

    if predictions is not None:
        # Create a comprehensive predictions dataframe
        threshold = st.slider(
            "Prediction Threshold (for binary classification):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probabilities above this threshold are classified as POSITIVE"
        )

        pred_data = []
        for disease in get_disease_classes():
            prob = predictions[disease]
            binary_pred = "✅ POSITIVE" if prob >= threshold else "❌ NEGATIVE"

            pred_data.append({
                "Disease": disease,
                "Probability": f"{prob*100:.2f}%",
                "Prediction": binary_pred,
                "Confidence": f"{max(prob, 1-prob)*100:.1f}%"
            })

        pred_df = pd.DataFrame(pred_data)

        # Display with formatting
        st.dataframe(pred_df, width="stretch", hide_index=True)

        # Probability bar chart
        st.markdown("### 📊 Probability Distribution")

        fig = go.Figure([
            go.Bar(
                x=list(predictions.values()),
                y=list(predictions.keys()),
                orientation='h',
                marker=dict(
                    color=list(predictions.values()),
                    colorscale='RdYlGn_r',  # Red for high, green for low
                    showscale=True,
                    colorbar=dict(title="Probability")
                ),
                text=[f"{v*100:.1f}%" for v in predictions.values()],
                textposition='auto',
            )
        ])

        fig.update_layout(
            xaxis_title="Probability",
            yaxis_title="Disease",
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload an image to see detailed predictions")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # CLINICAL INTERPRETATION
    # -------------------------------------------------------------------------
    st.subheader("5️⃣ Clinical Interpretation")

    if predictions is not None:
        # Generate simple interpretation
        positive_findings = [
            disease for disease, prob in predictions.items()
            if prob >= threshold
        ]

        if positive_findings:
            st.warning(f"""
            **⚠️ {len(positive_findings)} Positive Finding(s) Detected**

            The model predicts the following conditions may be present:
            {', '.join(positive_findings)}

            **Important:**
            - These are AI-generated predictions and may contain errors
            - All findings must be validated by a qualified radiologist
            - False positives and false negatives are possible
            - Do not make clinical decisions based solely on these predictions
            """)
        else:
            st.success("""
            **✅ No Positive Findings Above Threshold**

            The model did not detect any conditions with probability above the selected threshold.
            This does NOT guarantee the absence of disease - radiologist review is still required.
            """)
    else:
        st.info("Upload an image to see clinical interpretation")

    st.error("""
    ⚠️ **IMPORTANT DISCLAIMER**

    This tool is for **research and educational purposes ONLY**. It is NOT approved for clinical use.

    - Do NOT use for actual medical diagnosis
    - Do NOT upload real patient data without proper authorization
    - All predictions must be reviewed by qualified radiologists
    - False positives and false negatives are possible
    - This tool does NOT replace professional medical judgment
    """)

    # -------------------------------------------------------------------------
    # TECHNICAL DETAILS
    # -------------------------------------------------------------------------
    st.markdown("---")
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
           - Generate clinical interpretation

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

    st.markdown("---")

    st.info("""
    💡 **Tip:** Try uploading different chest X-ray images to see how the model performs!
    The model was trained on 78,566 images from 21,563 patients and tested on 16,491 images.
    """)
