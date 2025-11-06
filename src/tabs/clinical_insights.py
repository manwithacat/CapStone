"""
Clinical Insights Tab - Key Findings and Recommendations

This module renders the Clinical Insights tab presenting key findings,
clinical recommendations, ethical considerations, and future directions.
"""

import streamlit as st


def render_clinical_insights_tab(df, disease_colors=None):
    """
    Render the Clinical Insights tab with key findings and recommendations.

    Args:
        df (pd.DataFrame): Main dataset with patient/image metadata
        disease_colors (dict, optional): Color mapping for disease visualization
    """
    st.header("💡 Clinical Insights & Key Findings")

    st.markdown("""
    This section synthesizes findings from data exploration, statistical analysis,
    and model evaluation to provide actionable clinical insights and recommendations.
    """)

    # -------------------------------------------------------------------------
    # KEY FINDINGS
    # -------------------------------------------------------------------------
    st.subheader("🔍 Key Findings")

    st.markdown("### 1. Dataset Characteristics")
    st.write("""
    **Demographics:**
    - Large-scale dataset: 112,120 chest X-rays from 30,805 patients
    - Age range: Pediatric to elderly (diverse age distribution)
    - Gender distribution: Relatively balanced male/female representation
    - Geographic source: NIH Clinical Center (United States)

    **Disease Prevalence:**
    - "No Finding" (healthy): Majority class (~60% of images)
    - Most common pathologies: Infiltration, Effusion, Atelectasis
    - Rare conditions: Hernia, Pneumothorax (class imbalance challenges)
    - Multi-label complexity: Many patients have 2-3 co-existing conditions

    **Data Quality:**
    - High-resolution images (1024x1024 pixels)
    - Consistent frontal-view positioning
    - Labels extracted via NLP from radiology reports
    - Expert validation available for subset (5,184 images)
    """)

    st.markdown("### 2. Statistical Insights")
    st.write("""
    **Age-Disease Relationships (Hypothesis 1):**
    - Certain diseases show strong age correlation
    - Pneumonia and infiltration more common in elderly patients
    - Mass and nodule detection increases with age
    - Pediatric cases predominantly "No Finding"

    **Gender-Disease Patterns (Hypothesis 2):**
    - Some diseases show gender-specific prevalence
    - Cardiomegaly may differ by gender (cardiovascular risk factors)
    - Most pulmonary conditions show no significant gender bias
    - Important for developing unbiased AI models

    **Co-occurrence Patterns (Hypothesis 3):**
    - Strong correlations between certain disease pairs
    - Infiltration often co-occurs with Effusion
    - Atelectasis commonly appears with Pneumonia
    - Implications for differential diagnosis and model architecture
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # CLINICAL RECOMMENDATIONS
    # -------------------------------------------------------------------------
    st.subheader("🏥 Clinical Recommendations")

    st.markdown("### For Healthcare Providers")
    st.write("""
    **1. AI-Assisted Triage**
    - Use AI predictions for initial screening and prioritization
    - Flag high-risk cases for urgent radiologist review
    - Reduce radiologist workload for routine "No Finding" cases
    - Maintain human oversight for all final diagnoses

    **2. Multi-Disease Awareness**
    - Be alert to co-occurring conditions identified by statistical analysis
    - When one condition is detected, check for commonly associated diseases
    - Use AI predictions as a "second opinion" to reduce oversight errors

    **3. Age and Gender Considerations**
    - Account for age-specific disease patterns in diagnosis
    - Consider gender-specific prevalence when interpreting borderline cases
    - Use demographic information to inform prior probabilities

    **4. Quality Control**
    - Ensure X-ray images meet minimum quality standards
    - Frontal-view positioning is critical for accurate AI predictions
    - Poor image quality may lead to false negatives
    """)

    st.markdown("### For Healthcare Administrators")
    st.write("""
    **1. Implementation Planning**
    - Pilot program in low-risk screening scenarios first
    - Integrate with existing PACS (Picture Archiving and Communication Systems)
    - Establish clear protocols for AI-flagged cases
    - Provide training for radiologists on AI tool interpretation

    **2. Resource Allocation**
    - AI can help optimize radiologist schedules based on case complexity
    - Prioritize expert review for AI-uncertain predictions
    - Track time savings and diagnostic accuracy improvements
    - Plan for ongoing model updates and retraining

    **3. Compliance and Documentation**
    - Ensure HIPAA compliance for all patient data
    - Document AI predictions in patient records
    - Maintain audit trails for regulatory requirements
    - Prepare for FDA approval process if pursuing clinical deployment
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # ETHICAL CONSIDERATIONS
    # -------------------------------------------------------------------------
    st.subheader("⚖️ Ethical Considerations & Limitations")

    with st.expander("1. Bias and Fairness"):
        st.warning("""
        **Potential Biases:**
        - **Geographic Bias**: Dataset from single U.S. institution (NIH)
        - **Demographic Bias**: May not represent global population diversity
        - **Disease Prevalence Bias**: Reflects specific patient population
        - **Labeling Bias**: NLP-extracted labels may contain errors

        **Mitigation Strategies:**
        - Validate on diverse external datasets
        - Monitor performance across demographic subgroups
        - Use expert-validated labels for critical evaluation
        - Implement fairness metrics in model evaluation
        - Be transparent about training data limitations
        """)

    with st.expander("2. Privacy and Security"):
        st.warning("""
        **Privacy Concerns:**
        - Patient data must be de-identified (HIPAA compliance)
        - X-ray images may contain metadata requiring removal
        - Cloud deployment raises data sovereignty questions
        - Risk of re-identification through image characteristics

        **Security Measures:**
        - Encrypt data in transit and at rest
        - Implement access controls and audit logging
        - Regular security audits and penetration testing
        - Incident response plan for data breaches
        - Compliance with local healthcare data regulations
        """)

    with st.expander("3. Clinical Responsibility"):
        st.warning("""
        **Liability and Accountability:**
        - Who is responsible for AI errors? (Doctor, institution, vendor?)
        - Need clear protocols for AI-assisted vs. AI-overridden decisions
        - Documentation requirements for legal protection
        - Insurance and malpractice considerations

        **Best Practices:**
        - AI as decision support, not decision maker
        - Always require radiologist confirmation
        - Clearly communicate AI limitations to patients
        - Maintain human expertise and clinical judgment
        - Regular audits of AI-assisted diagnoses
        """)

    with st.expander("4. Model Limitations"):
        st.warning("""
        **Technical Limitations:**
        - **No Perfect Accuracy**: False positives and false negatives will occur
        - **Class Imbalance**: Performance varies significantly across diseases
        - **Out-of-Distribution Cases**: Rare diseases may not be detected
        - **Image Quality Dependency**: Poor images lead to poor predictions
        - **Temporal Drift**: Model performance may degrade over time

        **Clinical Limitations:**
        - **Cannot Replace Radiologists**: Lacks clinical context and patient history
        - **Limited to 14 Diseases**: Other conditions not in training data will be missed
        - **No Severity Assessment**: Only presence/absence, not stage or severity
        - **No Follow-up Recommendations**: Cannot suggest treatment or next steps
        - **No Explanation of Uncertainty**: Difficult to convey confidence to clinicians
        """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # FUTURE DIRECTIONS
    # -------------------------------------------------------------------------
    st.subheader("🚀 Future Directions")

    st.markdown("### Research Opportunities")
    st.write("""
    **1. Model Improvements**
    - Incorporate patient metadata (age, gender, history) for better predictions
    - Multi-modal learning (combine X-rays with clinical notes, lab results)
    - Temporal modeling (analyze changes across multiple time points)
    - Uncertainty quantification (Bayesian deep learning, ensembles)
    - Explainability enhancements (better Grad-CAM, attention visualization)

    **2. Clinical Validation**
    - Prospective studies comparing AI vs. radiologist performance
    - Multi-center trials across diverse institutions
    - Real-world effectiveness studies (impact on patient outcomes)
    - Cost-effectiveness analysis for healthcare systems
    - Randomized controlled trials for clinical decision support

    **3. Dataset Expansion**
    - Collect more diverse patient populations
    - Include rare diseases and edge cases
    - Longitudinal data for disease progression tracking
    - Multi-view X-rays (lateral, oblique) for 3D understanding
    - Integration with other imaging modalities (CT, MRI)
    """)

    st.markdown("### Deployment Roadmap")
    st.write("""
    **Phase 1: Pilot Testing (Months 1-6)**
    - Select low-risk screening scenarios
    - Deploy in shadow mode (AI predictions not shown to clinicians)
    - Collect performance data and user feedback
    - Refine model based on real-world patterns

    **Phase 2: Clinical Decision Support (Months 7-12)**
    - Enable AI predictions for radiologist review
    - Implement alert system for high-risk cases
    - Monitor impact on workflow and accuracy
    - Gather evidence for regulatory submission

    **Phase 3: Regulatory Approval (Months 13-24)**
    - Compile clinical evidence for FDA 510(k) or De Novo submission
    - Conduct additional validation studies if required
    - Prepare technical documentation and risk analysis
    - Obtain regulatory clearance for clinical use

    **Phase 4: Widespread Deployment (Months 25+)**
    - Roll out to multiple institutions
    - Continuous monitoring and model updates
    - Integration with EHR systems
    - Ongoing post-market surveillance
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # CONCLUSIONS
    # -------------------------------------------------------------------------
    st.subheader("📝 Conclusions")

    st.success("""
    **Summary:**

    This project demonstrates the potential of AI-powered chest X-ray analysis for
    supporting radiologists in disease detection and triage. Key accomplishments include:

    ✅ Comprehensive exploratory data analysis of large-scale chest X-ray dataset
    ✅ Statistical validation of age-disease and gender-disease relationships
    ✅ Identification of disease co-occurrence patterns for differential diagnosis
    ✅ Framework for multi-label deep learning classification
    ✅ Emphasis on clinical interpretability (Grad-CAM) and ethical considerations

    **Critical Takeaways:**

    1. **AI is a tool, not a replacement**: Human expertise remains essential
    2. **Bias awareness is crucial**: Models reflect training data limitations
    3. **Continuous validation is necessary**: Performance monitoring never stops
    4. **Ethical deployment is paramount**: Patient safety and privacy first
    5. **Collaboration is key**: Clinicians, ML engineers, and ethicists must work together

    **Next Steps:**

    - Complete model training and evaluation (Notebooks 05-09)
    - Validate on external datasets for generalization assessment
    - Engage with radiologists for clinical feedback
    - Prepare for pilot deployment in controlled setting
    - Pursue regulatory pathway for clinical approval
    """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    with st.expander("📚 References & Further Reading"):
        st.markdown("""
        **Original Dataset:**
        - Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray
          Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thoracic Diseases.
          *IEEE CVPR 2017*. [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)

        **Expert Labels:**
        - Majkowska A, Mittal S, Steiner DF, et al. Chest Radiograph Interpretation with Deep Learning Models:
          Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation.
          *Radiology* 2020; 294(2):421-431. [doi:10.1148/radiol.2019191293](https://doi.org/10.1148/radiol.2019191293)

        - Nabulsi Z, Sellergren A, Jamshy S, et al. Deep learning for distinguishing normal versus abnormal
          chest radiographs and generalization to two unseen diseases tuberculosis and COVID-19.
          *Scientific Reports* 2021; 11:15523. [doi:10.1038/s41598-021-93967-2](https://doi.org/10.1038/s41598-021-93967-2)

        **Grad-CAM:**
        - Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual Explanations from
          Deep Networks via Gradient-based Localization. *IEEE ICCV 2017*.

        **Medical AI Ethics:**
        - Topol EJ. High-performance medicine: the convergence of human and artificial intelligence.
          *Nature Medicine* 2019; 25(1):44-56.

        - Char DS, Shah NH, Magnus D. Implementing Machine Learning in Health Care - Addressing Ethical Challenges.
          *NEJM* 2018; 378(11):981-983.
        """)
