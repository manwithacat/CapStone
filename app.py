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

# View modules: Standalone page views (separate from dashboard tabs)
from src.views import render_radiology_guide_tab, render_dashboard

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

    This function loads the preprocessed metadata from Notebook 02 (parquet)
    which includes optimal labels (3-tier hierarchy) and expert label integration.
    If the parquet doesn't exist, falls back to raw CSV.

    Returns:
        tuple: (main_df, eda_report, hypothesis_report)
            - main_df: DataFrame with patient/image metadata and optimal labels
            - eda_report: Dict with EDA statistics
            - hypothesis_report: Dict with hypothesis test results
    """
    root = pathlib.Path(__file__).resolve().parent

    # Try loading preprocessed parquet first (preferred - has optimal labels)
    parquet_path = root / "data" / "processed" / "metadata_with_optimal_labels.parquet"
    main_df = pd.DataFrame()

    if parquet_path.exists():
        main_df = pd.read_parquet(parquet_path)
        st.sidebar.success("✅ Using preprocessed data with optimal labels")
    else:
        # Fallback to raw CSV
        data_entry_path = root / "data" / "raw" / "Data_Entry_2017.csv"
        if data_entry_path.exists():
            main_df = pd.read_csv(data_entry_path)
            st.sidebar.warning("⚠️ Using raw CSV data. Run Notebook 02 to generate optimized parquet.")
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
    /* Reset any scrollbar styling that might interfere */
    [data-testid="column"] .stElementContainer::-webkit-scrollbar,
    [data-testid="column"] img::-webkit-scrollbar {
        width: unset !important;
        height: unset !important;
    }

    /* Fix container width for X-ray images in columns */
    [data-testid="column"] .stElementContainer,
    [data-testid="column"] .stImage,
    [data-testid="column"] [data-testid="stImage"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: block !important;
    }

    /* X-ray images in columns should fill container width - override inline styles */
    [data-testid="column"] img[src],
    [data-testid="column"] img {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        height: auto !important;
        display: block !important;
        object-fit: contain !important;
        background-color: #000;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.2s ease;
    }

    /* Even more specific to override Streamlit's inline styles */
    [data-testid="column"] div > img,
    [data-testid="column"] div div > img {
        width: 100% !important;
        height: auto !important;
    }

    /* Hover effect for clickable X-ray images */
    [data-testid="column"] img:hover {
        opacity: 0.85;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE - View Navigation with Query Parameters
# ============================================================================
# Initialize session state first
if 'view' not in st.session_state:
    # On first load, check if URL has a view parameter
    view_param = st.query_params.get('view', 'dashboard')
    st.session_state.view = view_param

# Sync with query params only when they differ (browser back/forward was used)
# This check is deferred to avoid blocking initial render
if 'view' in st.query_params:
    url_view = st.query_params['view']
    if url_view != st.session_state.view:
        st.session_state.view = url_view

# ============================================================================
# LOAD DATA
# ============================================================================
df, eda_report, hypothesis_report = load_data()

# Check if DataFrame is empty
if df.empty:
    st.error("⚠️ No data found. Please run Notebook 01 (data collection) to download the NIH Chest X-Ray dataset.")
    st.stop()

# ============================================================================
# RADIOLOGY GUIDE VIEW
# ============================================================================
if st.session_state.view == 'radiology_guide':
    render_radiology_guide_tab(df, disease_colors=DISEASE_COLORS)
    st.stop()  # Don't render dashboard

# ============================================================================
# DASHBOARD VIEW (default)
# ============================================================================
render_dashboard(df, eda_report, hypothesis_report, DISEASE_COLORS)
