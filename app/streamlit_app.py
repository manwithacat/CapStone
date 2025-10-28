"""
NIH Chest X-Ray Disease Detection Dashboard
Main Streamlit Application

This is the entry point for the interactive web dashboard.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade interface
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
    .stAlert {
        background-color: #fff3cd;
        border-left: 4px solid #ff6b6b;
    }
    </style>
""", unsafe_allow_html=True)

# Main page header
st.markdown('<div class="main-header">🏥 Chest X-Ray Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Diagnostic Support Tool for Thoracic Pathologies</div>', unsafe_allow_html=True)

# Disclaimer banner
st.error("""
⚠️ **RESEARCH & EDUCATIONAL USE ONLY**
This tool is designed for research and educational purposes. It is NOT approved for clinical use and should NOT be used for medical diagnosis without professional radiologist review.
""")

# Welcome section
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 **Dataset**  \n112,120 Chest X-Rays  \n30,805 Patients  \n15 Disease Classes")

with col2:
    st.info("🧠 **AI Models**  \nDeep Learning CNNs  \nTransfer Learning  \nGrad-CAM Interpretation")

with col3:
    st.info("🎯 **Purpose**  \nDiagnostic Support  \nDisease Triage  \nClinical Research")

# Project overview
st.markdown("## Project Overview")

st.write("""
This dashboard presents an AI-powered system for detecting and classifying thoracic diseases from chest X-ray images.
Using the NIH Chest X-Ray dataset, we've developed deep learning models capable of identifying 14 different pathological
conditions plus normal (healthy) cases.

**Key Features:**
- **Multi-label Classification**: Detects multiple co-existing diseases in a single image
- **Transfer Learning**: Leverages state-of-the-art CNN architectures (ResNet, DenseNet, EfficientNet)
- **Visual Explanations**: Grad-CAM heatmaps show which image regions influenced predictions
- **Statistical Analysis**: Comprehensive hypothesis testing and demographic insights
- **Clinical Context**: Disease descriptions and prevalence statistics
""")

# Navigation guide
st.markdown("## 📖 Navigation Guide")

st.write("""
Use the sidebar to navigate between different sections of the dashboard:

1. **📊 Data Exploration**: Patient demographics, disease distribution, and dataset statistics
2. **📈 Hypothesis Validation**: Statistical analysis and hypothesis testing results
3. **📏 Model Performance**: Model comparison, metrics, and evaluation results
4. **🔍 Disease Detector**: Upload X-ray images for interactive disease prediction
5. **💡 Clinical Insights**: Key findings, recommendations, and ethical considerations
""")

# Dataset information
with st.expander("ℹ️ About the NIH Chest X-Ray Dataset"):
    st.write("""
    **Source:** National Institutes of Health Clinical Center

    **Publication:** Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray
    Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thoracic Diseases.
    IEEE CVPR 2017

    **Dataset Details:**
    - 112,120 frontal-view X-ray images (1024 x 1024 pixels)
    - 30,805 unique patients
    - 15 disease classes: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
      Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia, and No Finding
    - Multi-label annotations (images can have multiple diseases)
    - Class imbalance: "No Finding" accounts for >50% of images

    **License:** CC0: Public Domain
    """)

# Technical specifications
with st.expander("🛠️ Technical Specifications"):
    st.write("""
    **Technologies Used:**
    - **Deep Learning:** TensorFlow 2.x, Keras
    - **Computer Vision:** OpenCV, Pillow, scikit-image
    - **Machine Learning:** scikit-learn, XGBoost
    - **Data Processing:** pandas, numpy
    - **Visualization:** matplotlib, seaborn, plotly
    - **Dashboard:** Streamlit
    - **Interpretation:** Grad-CAM, saliency maps

    **Model Architectures:**
    - Custom CNN
    - ResNet50 (transfer learning)
    - DenseNet121 (medical imaging standard)
    - EfficientNetB3 (efficient scaling)

    **Evaluation Metrics:**
    - AUC-ROC, AUC-PR
    - Sensitivity, Specificity
    - F1-Score, Precision, Recall
    - Multi-label metrics (Hamming loss, subset accuracy)
    """)

# Ethical considerations
with st.expander("⚖️ Ethical Considerations"):
    st.warning("""
    **Important Ethical and Legal Considerations:**

    **1. Not for Clinical Use**
    - This system has NOT been validated for clinical deployment
    - NOT FDA approved or CE marked
    - Requires professional radiologist review for all diagnoses

    **2. Dataset Limitations**
    - Training data may not represent all populations
    - Potential biases in disease prevalence and demographics
    - Geographic and temporal limitations

    **3. Privacy & Security**
    - No patient data is stored by this dashboard
    - All predictions are performed locally
    - Upload features should NOT be used with real patient data without HIPAA compliance

    **4. False Positives/Negatives**
    - No AI system is 100% accurate
    - False negatives could delay critical diagnoses
    - False positives may cause unnecessary procedures

    **5. Human-in-the-Loop Required**
    - AI should augment, not replace, clinical expertise
    - Final diagnostic decisions must be made by qualified healthcare professionals

    **6. Regulatory Requirements**
    - Clinical deployment requires regulatory approval (FDA, local authorities)
    - Must comply with medical device regulations
    - Requires ongoing validation and monitoring
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem 0;'>
    <p><strong>Code Institute Data Analytics & AI Bootcamp - Capstone Project</strong></p>
    <p>NIH Chest X-Ray Disease Detection | January 2025</p>
    <p>🔬 For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png", width=200)

    st.markdown("### Quick Stats")
    st.metric("Total Images", "112,120")
    st.metric("Patients", "30,805")
    st.metric("Disease Classes", "15")

    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [GitHub Repository](#)
    - [Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
    - [Original Paper (CVPR 2017)](https://arxiv.org/abs/1705.02315)
    """)

    st.markdown("---")
    st.info("""
    💡 **Getting Started**

    1. Explore data statistics
    2. Review hypothesis validation
    3. Compare model performance
    4. Try the disease detector
    5. Read clinical insights
    """)
