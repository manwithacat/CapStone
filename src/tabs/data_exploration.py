"""
Data Exploration Tab - Dataset Statistics and Demographics

This module renders the Data Exploration tab showing dataset overview,
patient demographics, disease distribution, and sample X-ray visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def render_data_exploration_tab(df, eda_report, disease_colors=None):
    """
    Render the Data Exploration tab with dataset statistics and EDA results.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata
        eda_report (dict): EDA report from Notebook 02
        disease_colors (dict, optional): Color mapping for disease visualization
    """
    st.header("Dataset Overview & Exploratory Data Analysis")

    st.markdown("""
    Comprehensive analysis of the NIH Chest X-Ray dataset including patient demographics,
    disease distribution, and co-occurrence patterns.
    """)

    # -------------------------------------------------------------------------
    # KEY METRICS ROW
    # -------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", f"{len(df):,}")

    with col2:
        if 'Patient ID' in df.columns:
            n_patients = df['Patient ID'].nunique()
            st.metric("Unique Patients", f"{n_patients:,}")
        else:
            st.metric("Unique Patients", "N/A")

    with col3:
        if 'Finding Labels' in df.columns:
            # Count disease classes
            all_labels = set()
            for labels in df['Finding Labels'].dropna():
                all_labels.update(labels.split('|'))
            st.metric("Disease Classes", len(all_labels))
        else:
            st.metric("Disease Classes", "N/A")

    with col4:
        if eda_report and 'image_dimensions' in eda_report:
            st.metric("Image Size", eda_report['image_dimensions'])
        else:
            st.metric("Image Size", "1024 x 1024")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DEMOGRAPHICS SECTION
    # -------------------------------------------------------------------------
    st.subheader("📊 Patient Demographics")

    # Load demographic visualizations from outputs/figures/
    root = Path(__file__).resolve().parent.parent.parent
    figures_dir = root / "outputs" / "figures"

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        age_fig_path = figures_dir / "02_age_distribution.png"
        if age_fig_path.exists():
            st.image(str(age_fig_path), caption="Age Distribution", use_container_width=True)
        else:
            st.info("Age distribution figure will appear here after running Notebook 02")

    with col2:
        # Gender distribution
        gender_fig_path = figures_dir / "02_gender_distribution.png"
        if gender_fig_path.exists():
            st.image(str(gender_fig_path), caption="Gender Distribution", use_container_width=True)
        else:
            st.info("Gender distribution figure will appear here after running Notebook 02")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DISEASE DISTRIBUTION
    # -------------------------------------------------------------------------
    st.subheader("🦠 Disease Distribution")

    # Disease prevalence chart
    disease_dist_path = figures_dir / "02_disease_distribution.png"
    if disease_dist_path.exists():
        st.image(str(disease_dist_path), caption="Disease Prevalence Across Dataset", use_container_width=True)
    else:
        st.info("Disease distribution figure will appear here after running Notebook 02")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DISEASE CO-OCCURRENCE
    # -------------------------------------------------------------------------
    st.subheader("🔗 Disease Co-occurrence Patterns")

    st.markdown("""
    Many patients have multiple conditions simultaneously. This heatmap shows which
    diseases commonly occur together.
    """)

    cooccurrence_path = figures_dir / "02_disease_cooccurrence.png"
    if cooccurrence_path.exists():
        st.image(str(cooccurrence_path), caption="Disease Co-occurrence Heatmap", use_container_width=True)
    else:
        st.info("Disease co-occurrence heatmap will appear here after running Notebook 02")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SAMPLE X-RAY IMAGES
    # -------------------------------------------------------------------------
    st.subheader("🖼️ Sample Chest X-Ray Images")

    st.markdown("""
    Sample X-ray images from the dataset showing various pathological conditions.
    **Hover over images to see disease labels and patient information.**
    """)

    # Load interactive Plotly HTML version for hover tooltips
    sample_xrays_interactive = figures_dir / "02_sample_xrays_interactive.html"
    sample_xrays_static = figures_dir / "02_sample_xrays.png"

    if sample_xrays_interactive.exists():
        # Display interactive Plotly figure with hover tooltips
        with open(sample_xrays_interactive, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1200, scrolling=True)
    elif sample_xrays_static.exists():
        # Fallback to static PNG if HTML not available
        st.image(str(sample_xrays_static), caption="Sample X-Ray Images with Labels", use_container_width=True)
        st.info("💡 Interactive version with hover tooltips available in Notebook 02")
    else:
        st.info("Sample X-ray images will appear here after running Notebook 02")

    # -------------------------------------------------------------------------
    # RAW DATA PREVIEW
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Raw Data Preview")

    # Display first 100 rows with key columns
    display_cols = [col for col in df.columns if col in [
        'Image Index', 'Finding Labels', 'Patient ID', 'Patient Age',
        'Patient Gender', 'View Position', 'OriginalImage[Width', 'OriginalImage[Height'
    ]]

    if display_cols:
        st.dataframe(df[display_cols].head(100), use_container_width=True)
    else:
        st.dataframe(df.head(100), use_container_width=True)

    # -------------------------------------------------------------------------
    # DATASET STATISTICS FROM EDA REPORT
    # -------------------------------------------------------------------------
    if eda_report:
        st.markdown("---")
        st.subheader("📈 Dataset Statistics")

        with st.expander("View Detailed EDA Report"):
            st.json(eda_report)
