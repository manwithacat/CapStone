"""
Hypothesis Validation Tab - Statistical Testing Results

This module renders the Hypothesis Validation tab showing results from
statistical hypothesis tests conducted in Notebook 04.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


def render_hypothesis_validation_tab(df, hypothesis_report, disease_colors=None):
    """
    Render the Hypothesis Validation tab with statistical test results.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata
        hypothesis_report (dict): Hypothesis testing results from Notebook 04
        disease_colors (dict, optional): Color mapping for disease visualization
    """
    st.header("Statistical Hypothesis Testing")

    st.markdown("""
    This section presents the results of statistical hypothesis tests exploring relationships
    between patient demographics, disease prevalence, and clinical patterns.
    """)

    # -------------------------------------------------------------------------
    # HYPOTHESIS OVERVIEW
    # -------------------------------------------------------------------------
    st.subheader("📋 Research Hypotheses")

    st.markdown("""
    The following hypotheses were tested using appropriate statistical methods:

    **H1: Age & Disease Relationship**
    - Null hypothesis: Patient age has no relationship with disease prevalence
    - Statistical test: Chi-square test, ANOVA
    - Purpose: Identify age-related disease patterns

    **H2: Gender & Disease Relationship**
    - Null hypothesis: Disease prevalence is independent of patient gender
    - Statistical test: Chi-square test, proportion z-test
    - Purpose: Detect gender-specific disease patterns

    **H3: Multi-label Co-occurrence**
    - Null hypothesis: Diseases occur independently (no significant co-occurrence)
    - Statistical test: Correlation analysis, chi-square
    - Purpose: Identify commonly co-occurring conditions

    **H4: Image Quality & Diagnostic Accuracy**
    - Null hypothesis: Image quality metrics don't correlate with diagnostic certainty
    - Statistical test: Correlation analysis (to be implemented)
    - Purpose: Quality control for imaging data

    **H5: Transfer Learning Performance**
    - Null hypothesis: Transfer learning models don't outperform baseline models
    - Statistical test: Paired t-test, Wilcoxon signed-rank (to be implemented)
    - Purpose: Validate deep learning approach
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 1: AGE & DISEASE
    # -------------------------------------------------------------------------
    st.subheader("📊 Hypothesis 1: Age & Disease Relationship")

    root = Path(__file__).resolve().parent.parent.parent
    figures_dir = root / "outputs" / "figures"

    age_disease_path = figures_dir / "04_age_disease_distributions.png"
    if age_disease_path.exists():
        st.image(str(age_disease_path), caption="Age Distribution by Disease", width="stretch")

        if hypothesis_report and 'hypothesis_1_age_disease' in hypothesis_report:
            h1_results = hypothesis_report['hypothesis_1_age_disease']
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Chi-square Statistic", f"{h1_results.get('chi2', 'N/A'):.2f}")
            with col2:
                p_value = h1_results.get('p_value', 0)
                st.metric("P-value", f"{p_value:.4f}")
            with col3:
                significant = "Yes ✅" if p_value < 0.05 else "No ❌"
                st.metric("Significant?", significant)

            if p_value < 0.05:
                st.success("✅ **Result**: Reject null hypothesis. Age significantly correlates with disease prevalence.")
            else:
                st.info("ℹ️ **Result**: Fail to reject null hypothesis. No significant age-disease relationship detected.")
    else:
        st.info("Hypothesis 1 results will appear here after running Notebook 04")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 2: GENDER & DISEASE
    # -------------------------------------------------------------------------
    st.subheader("📊 Hypothesis 2: Gender & Disease Relationship")

    gender_disease_path = figures_dir / "04_gender_disease_prevalence.png"
    if gender_disease_path.exists():
        st.image(str(gender_disease_path), caption="Disease Prevalence by Gender", width="stretch")

        if hypothesis_report and 'hypothesis_2_gender_disease' in hypothesis_report:
            h2_results = hypothesis_report['hypothesis_2_gender_disease']

            # Display summary statistics
            st.markdown("**Statistical Test Results:**")
            st.dataframe(pd.DataFrame(h2_results.get('test_results', {})), width="stretch")

            # Interpretation
            significant_diseases = [d for d, p in h2_results.get('test_results', {}).items() if p < 0.05]
            if significant_diseases:
                st.success(f"✅ **Result**: Significant gender differences detected in {len(significant_diseases)} diseases")
                st.markdown(f"**Diseases with significant gender differences:** {', '.join(significant_diseases)}")
            else:
                st.info("ℹ️ **Result**: No significant gender differences in disease prevalence")
    else:
        st.info("Hypothesis 2 results will appear here after running Notebook 04")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 3: DISEASE CO-OCCURRENCE
    # -------------------------------------------------------------------------
    st.subheader("📊 Hypothesis 3: Disease Co-occurrence Patterns")

    disease_corr_path = figures_dir / "04_disease_correlation_heatmap.png"
    if disease_corr_path.exists():
        st.image(str(disease_corr_path), caption="Disease Co-occurrence Correlation Matrix", width="stretch")

        st.markdown("""
        **Interpretation:**
        - Darker colors indicate stronger positive correlations (diseases that occur together)
        - Light colors indicate weak or no correlation
        - Red indicates negative correlation (diseases that rarely occur together)
        """)

        if hypothesis_report and 'hypothesis_3_cooccurrence' in hypothesis_report:
            h3_results = hypothesis_report['hypothesis_3_cooccurrence']
            st.markdown("**Top Co-occurring Disease Pairs:**")
            st.json(h3_results.get('top_pairs', {}))
    else:
        st.info("Hypothesis 3 results will appear here after running Notebook 04")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # DISEASE COUNT DISTRIBUTION
    # -------------------------------------------------------------------------
    st.subheader("📊 Multi-label Disease Distribution")

    disease_count_path = figures_dir / "04_disease_count_distribution.png"
    if disease_count_path.exists():
        st.image(str(disease_count_path), caption="Number of Diseases per Patient", width="stretch")

        st.markdown("""
        This chart shows how many diseases are present in each X-ray image. Many patients
        have multiple co-existing conditions, which makes this a challenging multi-label
        classification problem.
        """)
    else:
        st.info("Disease count distribution will appear here after running Notebook 04")

    # -------------------------------------------------------------------------
    # DETAILED HYPOTHESIS REPORT
    # -------------------------------------------------------------------------
    if hypothesis_report:
        st.markdown("---")
        st.subheader("📄 Detailed Statistical Report")

        with st.expander("View Complete Hypothesis Testing Report"):
            st.json(hypothesis_report)

    # -------------------------------------------------------------------------
    # STATISTICAL SIGNIFICANCE GUIDE
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("ℹ️ Understanding Statistical Significance"):
        st.markdown("""
        **P-value Interpretation:**
        - **p < 0.05**: Statistically significant (reject null hypothesis)
        - **p < 0.01**: Highly significant (strong evidence)
        - **p < 0.001**: Very highly significant (very strong evidence)
        - **p ≥ 0.05**: Not statistically significant (fail to reject null hypothesis)

        **Effect Size:**
        - Small effect: Still statistically meaningful but may not be clinically significant
        - Large effect: Both statistically and practically significant

        **Note:** Statistical significance does not always imply clinical significance.
        Results should be interpreted in the context of medical knowledge and practice.
        """)
