"""
NIH Chest X-Ray Disease Detection Dashboard

Interactive Streamlit dashboard for exploring chest X-ray data,
hypothesis testing results, and model performance metrics.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pathlib

import pandas as pd
import streamlit as st

# Tab modules: Modular rendering functions for each dashboard tab
from src.tabs import (
    render_data_exploration_tab,
    render_hypothesis_validation_tab,
    render_model_performance_tab,
    render_disease_detector_tab,
    render_clinical_insights_tab,
)

# ============================================================================
# COLOR SCHEME CONFIGURATION
# ============================================================================
# Medical-grade color scheme for pathology visualization
DISEASE_COLORS = {
    "No Finding": "#2ecc71",      # Green - healthy/normal
    "Pathology": "#e74c3c",       # Red - disease detected
    "Atelectasis": "#3498db",     # Blue
    "Cardiomegaly": "#e67e22",    # Orange
    "Effusion": "#9b59b6",        # Purple
    "Infiltration": "#f39c12",    # Yellow-orange
    "Mass": "#e74c3c",            # Red
    "Nodule": "#c0392b",          # Dark red
    "Pneumonia": "#d35400",       # Burnt orange
    "Pneumothorax": "#2c3e50",    # Navy
}

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
@st.cache_data
def load_data():
    """
    Load processed chest X-ray metadata and reports.

    This function loads the exploratory data analysis results and
    processed metadata from the data pipeline. If files don't exist,
    it provides graceful fallbacks and informative messages.

    Returns:
        tuple: (main_df, eda_report, hypothesis_report)
            - main_df: DataFrame with patient/image metadata
            - eda_report: Dict with EDA statistics
            - hypothesis_report: Dict with hypothesis test results
    """
    root = pathlib.Path(__file__).resolve().parent

    # Load main metadata
    data_entry_path = root / "data" / "raw" / "Data_Entry_2017.csv"
    main_df = pd.DataFrame()

    if data_entry_path.exists():
        main_df = pd.read_csv(data_entry_path)
    else:
        st.warning("⚠️ Data not found. Please run Notebook 01 to download the dataset.")

    # Load EDA report
    eda_report_path = root / "outputs" / "reports" / "02_eda_report.json"
    eda_report = {}
    if eda_report_path.exists():
        import json
        with open(eda_report_path, 'r') as f:
            eda_report = json.load(f)

    # Load hypothesis testing report
    hypothesis_report_path = root / "outputs" / "reports" / "04_hypothesis_testing_results.json"
    hypothesis_report = {}
    if hypothesis_report_path.exists():
        import json
        with open(hypothesis_report_path, 'r') as f:
            hypothesis_report = json.load(f)

    return main_df, eda_report, hypothesis_report


# ============================================================================
# PAGE CONFIGURATION - This MUST be the first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
df, eda_report, hypothesis_report = load_data()

# Check if DataFrame is empty
if df.empty:
    st.error("⚠️ No data found. Please run Notebook 01 (data collection) to download the NIH Chest X-Ray dataset.")
    st.stop()

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<div class="main-header">🏥 NIH Chest X-Ray Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Diagnostic Support System for Thoracic Pathologies</div>', unsafe_allow_html=True)

# Disclaimer banner
st.error("""
⚠️ **RESEARCH & EDUCATIONAL USE ONLY**
This tool is designed for research and educational purposes. It is NOT approved for clinical use
and should NOT be used for medical diagnosis without professional radiologist review.
""")

# ============================================================================
# SIDEBAR - Dataset information and filters
# ============================================================================
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
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Original Paper (CVPR 2017)](https://arxiv.org/abs/1705.02315)
- [GitHub Repository](https://github.com/manwithacat/CapStone)
""")

# ============================================================================
# MAIN CONTENT - Organized into tabs
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration",
    "📈 Hypothesis Validation",
    "📏 Model Performance",
    "🔍 Disease Detector",
    "💡 Clinical Insights"
])

# ============================================================================
# TAB 1: DATA EXPLORATION
# ============================================================================
with tab1:
    render_data_exploration_tab(df, eda_report, disease_colors=DISEASE_COLORS)

# ============================================================================
# TAB 2: HYPOTHESIS VALIDATION
# ============================================================================
with tab2:
    render_hypothesis_validation_tab(df, hypothesis_report, disease_colors=DISEASE_COLORS)

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================
with tab3:
    render_model_performance_tab(df, disease_colors=DISEASE_COLORS)

# ============================================================================
# TAB 4: DISEASE DETECTOR
# ============================================================================
with tab4:
    render_disease_detector_tab(df, disease_colors=DISEASE_COLORS)

# ============================================================================
# TAB 5: CLINICAL INSIGHTS
# ============================================================================
with tab5:
    render_clinical_insights_tab(df, disease_colors=DISEASE_COLORS)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem 0;'>
    <p><strong>Code Institute Data Analytics & AI Bootcamp - Capstone Project</strong></p>
    <p>NIH Chest X-Ray Disease Detection | November 2025</p>
    <p>🔬 For research and educational purposes only</p>
    <p><a href="https://github.com/manwithacat/CapStone" target="_blank" style="color: #3498db; text-decoration: none;">📂 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
