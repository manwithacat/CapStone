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
from src.utils.model_loader import (
    load_densenet_model,
    preprocess_image,
    predict_diseases,
    get_disease_classes,
    get_top_predictions,
    format_prediction_text
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
    Upload a chest X-ray image to receive AI-powered disease predictions with
    visual explanations showing which regions of the image influenced the model's decisions.
    """)

    # Disclaimer
    st.error("""
    ⚠️ **IMPORTANT DISCLAIMER**

    This tool is for **research and educational purposes ONLY**. It is NOT approved for clinical use.

    - Do NOT use for actual medical diagnosis
    - Do NOT upload real patient data without proper authorization
    - All predictions must be reviewed by qualified radiologists
    - False positives and false negatives are possible
    - This tool does NOT replace professional medical judgment
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------------------------------
    st.subheader("1️⃣ Select Model")

    st.success("""
    **✅ Model Ready**

    DenseNet121 model is now available for predictions!
    - Test AUC: 0.7529 (best performing model)
    - Parameters: 7.6M (parameter efficient)
    - Trained on NVIDIA A100 GPU
    - 14 disease classes supported
    """)

    # Load DenseNet121 model
    model_path = "models/saved_models/densenet121_best.keras"

    try:
        model = load_densenet_model(model_path)
        st.info("✅ DenseNet121 model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    st.markdown("---")

    # -------------------------------------------------------------------------
    # IMAGE UPLOAD
    # -------------------------------------------------------------------------
    st.subheader("2️⃣ Upload Chest X-Ray Image")

    st.markdown("""
    Upload a chest X-ray image for AI-powered disease prediction.

    **Requirements:**
    - File format: PNG, JPG, JPEG
    - Image type: Frontal-view chest X-ray
    - Recommended: Grayscale X-ray images
    - Will be automatically resized to 224x224 for the model
    """)

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image file...",
        type=['png', 'jpg', 'jpeg'],
    )

    # Store the image for prediction
    image_to_predict = None

    if uploaded_file is not None:
        try:
            image_to_predict = Image.open(uploaded_file)
            st.success(f"✅ Image uploaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # PREDICTION RESULTS
    # -------------------------------------------------------------------------
    st.subheader("3️⃣ Prediction Results")

    if image_to_predict is not None:
        # Display original image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📷 Original Image**")
            st.image(image_to_predict, caption="Uploaded X-ray", width="stretch")

        with col2:
            st.markdown("**🔥 Grad-CAM Heatmap**")
            st.info("🚧 Grad-CAM visualization coming soon! For now, see predictions below.")

        st.markdown("---")

        # Run prediction
        with st.spinner("🔮 Running AI prediction..."):
            try:
                # Preprocess image
                processed_image = preprocess_image(image_to_predict)

                # Get predictions
                predictions = predict_diseases(model, processed_image)

                # Get top 3 predictions
                top_3 = get_top_predictions(predictions, top_k=3, threshold=0.0)

                # Display top predictions
                st.markdown("### 🏆 Top 3 Most Likely Diseases")

                for i, (disease, prob) in enumerate(top_3, 1):
                    # Color based on probability
                    if prob >= 0.5:
                        color = "🔴"  # High probability
                    elif prob >= 0.3:
                        color = "🟡"  # Medium probability
                    else:
                        color = "🟢"  # Low probability

                    st.markdown(f"{color} **{i}. {disease}**: {prob*100:.1f}%")

                st.markdown("---")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                predictions = None

    else:
        st.info("👆 Please upload a chest X-ray image to get started")
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

    st.warning("""
    **⚠️ Disclaimer:** All AI predictions must be validated by qualified healthcare professionals.
    This tool is designed to support, not replace, clinical decision-making.
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
