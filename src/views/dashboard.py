"""
Dashboard View - Main Application Dashboard

This module renders the main dashboard view with 6 tabs:
1. Data Exploration
2. Sample Images
3. Hypothesis Validation
4. Model Performance
5. Disease Detector
6. Clinical Insights
"""

import streamlit as st
from src.tabs import (
    render_data_exploration_tab,
    render_sample_images_tab,
    render_hypothesis_validation_tab,
    render_model_performance_tab,
    render_disease_detector_tab,
    render_clinical_insights_tab,
)


def render_dashboard(df, eda_report, hypothesis_report, disease_colors):
    """
    Render the main dashboard view with tabs and sidebar.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata
        eda_report (dict): EDA report from Notebook 02
        hypothesis_report (dict): Hypothesis test results from Notebook 04
        disease_colors (dict): Color mapping for disease visualization
    """
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown('<div class="main-header">🏥 NIH Chest X-Ray Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Diagnostic Support System for Thoracic Pathologies</div>', unsafe_allow_html=True)

    # Disclaimer banner
    st.error("""
    ⚠️ **RESEARCH & EDUCATIONAL USE ONLY**
    This tool is designed for research and educational purposes. It is NOT approved for clinical use
    and should NOT be used for medical diagnosis without professional radiologist review.
    """)

    # ========================================================================
    # SIDEBAR - Dataset information and navigation
    # ========================================================================

    # Render navigation button first for immediate interactivity
    if st.sidebar.button("📖 Radiology Guide", width='stretch', type="primary"):
        # Store current view before navigating away
        st.session_state.previous_view = st.session_state.get('view', 'dashboard')
        st.session_state.view = 'radiology_guide'
        st.query_params['view'] = 'radiology_guide'
        st.rerun()

    st.sidebar.markdown("---")

    # Dataset information (rendered after button for faster page load)
    st.sidebar.header("Dataset Information")

    # Display key metrics
    st.sidebar.metric("Total Images", f"{len(df):,}")
    if 'Patient ID' in df.columns:
        st.sidebar.metric("Unique Patients", f"{df['Patient ID'].nunique():,}")
    if 'Finding Labels' in df.columns:
        # Count disease classes
        all_labels = set()
        for labels in df['Finding Labels'].dropna():
            all_labels.update(labels.split('|'))
        st.sidebar.metric("Disease Classes", len(all_labels))

    st.sidebar.markdown("---")

    # Optional filters (to be implemented based on specific needs)
    st.sidebar.header("Filters")
    st.sidebar.info("Filters will be populated as tabs are developed")

    st.sidebar.markdown("---")

    # References section (de-emphasized)
    st.sidebar.markdown('<p style="color: #888; font-size: 0.9rem; margin-bottom: 0.5rem;">References</p>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    - [Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
    - [Original Paper (CVPR 2017)](https://arxiv.org/abs/1705.02315)
    - [GitHub Repository](https://github.com/manwithacat/CapStone)
    """)

    # ========================================================================
    # MAIN CONTENT - Organized into tabs
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Exploration",
        "🖼️ Sample Images",
        "📈 Hypothesis Validation",
        "📏 Model Performance",
        "🔍 Disease Detector",
        "💡 Clinical Insights"
    ])

    # ========================================================================
    # TAB 1: DATA EXPLORATION
    # ========================================================================
    with tab1:
        render_data_exploration_tab(df, eda_report, disease_colors=disease_colors)

    # ========================================================================
    # TAB 2: SAMPLE IMAGES
    # ========================================================================
    with tab2:
        render_sample_images_tab(df, disease_colors=disease_colors)

    # ========================================================================
    # TAB 3: HYPOTHESIS VALIDATION
    # ========================================================================
    with tab3:
        render_hypothesis_validation_tab(df, hypothesis_report, disease_colors=disease_colors)

    # ========================================================================
    # TAB 4: MODEL PERFORMANCE
    # ========================================================================
    with tab4:
        render_model_performance_tab(df, disease_colors=disease_colors)

    # ========================================================================
    # TAB 5: DISEASE DETECTOR
    # ========================================================================
    with tab5:
        render_disease_detector_tab(df, disease_colors=disease_colors)

    # ========================================================================
    # TAB 6: CLINICAL INSIGHTS
    # ========================================================================
    with tab6:
        render_clinical_insights_tab(df, disease_colors=disease_colors)

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem 0;'>
        <p><strong>Code Institute Data Analytics & AI Bootcamp - Capstone Project</strong></p>
        <p>NIH Chest X-Ray Disease Detection | November 2025</p>
        <p>🔬 For research and educational purposes only</p>
        <p><a href="https://github.com/manwithacat/CapStone" target="_blank" style="color: #3498db; text-decoration: none;">📂 View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
