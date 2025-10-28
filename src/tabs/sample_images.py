"""
Sample Images Tab - Display X-ray sample images from each disease class

This module renders the Sample Images tab showing clickable X-ray thumbnails
for all 14 disease classes in the NIH dataset.
"""

import streamlit as st
from src.utils.image_modal import render_xray_grid


def render_sample_images_tab(df=None, disease_colors=None):
    """
    Render the Sample Images tab with clickable X-ray thumbnails.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata (optional)
        disease_colors (dict, optional): Color mapping for disease visualization
    """
    st.header("Sample Chest X-Ray Images")

    st.markdown("""
    Sample X-ray images from the dataset showing various pathological conditions.
    **Click on any image to view the full-resolution version with patient details.**
    """)

    # Define disease classes (starting with normal/healthy for comparison)
    disease_classes = [
        "No Finding",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]

    # Render clickable grid of X-ray thumbnails (3 columns on desktop, responsive)
    render_xray_grid(disease_classes, use_thumbnails=True)

    st.markdown("---")

    st.info("""
    💡 **About These Images:** These sample images represent the 14 thoracic pathology
    classes in the NIH Chest X-Ray dataset. Each image was selected to clearly demonstrate
    the characteristic radiological features of its respective condition.
    """)
