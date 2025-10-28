"""
Radiology Guide Tab - Educational resource for chest X-ray interpretation

This tab provides a comprehensive guide to reading chest X-rays, understanding
normal anatomy, recognizing pathologies, and interpreting AI predictions.
"""

import streamlit as st
from typing import Dict, List


def render_radiology_guide_tab(df=None, disease_colors: Dict[str, str] | None = None):
    """
    Render the Radiology Guide tab with hierarchical navigation.

    Args:
        df: Main dataset (optional, for statistics)
        disease_colors: Color mapping for diseases (optional)
    """
    # Custom CSS for layout and smooth scrolling
    st.markdown("""
    <style>
    /* Sticky navigation sidebar */
    .nav-sidebar {
        position: sticky;
        top: 80px;
        max-height: calc(100vh - 100px);
        overflow-y: auto;
        padding-right: 1rem;
    }

    /* Navigation links */
    .nav-link {
        display: block;
        padding: 0.5rem 0.75rem;
        color: #2c3e50;
        text-decoration: none;
        border-left: 3px solid transparent;
        transition: all 0.2s;
        cursor: pointer;
    }

    .nav-link:hover {
        background-color: #f8f9fa;
        border-left-color: #3498db;
        padding-left: 1rem;
    }

    .nav-link.active {
        color: #3498db;
        font-weight: 600;
        border-left-color: #3498db;
    }

    /* Nested navigation items */
    .nav-link-sub {
        font-size: 0.9rem;
        padding-left: 1.5rem;
    }

    .nav-link-subsub {
        font-size: 0.85rem;
        padding-left: 2.5rem;
        color: #5a6c7d;
    }

    /* Content section styling */
    .content-section {
        scroll-margin-top: 100px;
    }

    /* Medical term highlighting */
    .medical-term {
        font-weight: 600;
        color: #2c3e50;
    }

    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }

    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("# 📚 Radiology Guide: Chest X-Ray Interpretation")
    st.markdown("""
    <div class="info-box">
    <strong>Educational Resource for Healthcare Professionals and Students</strong><br>
    This guide provides foundational knowledge for interpreting chest X-rays and understanding
    the AI-powered disease detection system. It is intended for educational purposes only and
    should not replace formal radiology training or clinical judgment.
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout
    col_nav, col_content = st.columns([1, 3])

    with col_nav:
        render_navigation()

    with col_content:
        render_content(df, disease_colors)


def render_navigation():
    """Render the sticky navigation sidebar with hierarchical topics."""
    st.markdown('<div class="nav-sidebar">', unsafe_allow_html=True)
    st.markdown("### Navigation")

    # Define navigation structure
    nav_structure = {
        "Introduction": [],
        "Normal Anatomy": [
            "Heart & Mediastinum",
            "Lungs & Airways",
            "Bones & Soft Tissue"
        ],
        "Pathology Guide": [
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
            "Pleural Thickening",
            "Pneumonia",
            "Pneumothorax"
        ],
        "Reading Strategy": [],
        "AI in Radiology": []
    }

    # Render navigation links
    for section, subsections in nav_structure.items():
        section_id = section.lower().replace(" ", "-").replace("&", "and")
        st.markdown(
            f'<a href="#{section_id}" class="nav-link">{section}</a>',
            unsafe_allow_html=True
        )

        for subsection in subsections:
            subsection_id = f"{section_id}-{subsection.lower().replace(' ', '-').replace('&', 'and')}"
            st.markdown(
                f'<a href="#{subsection_id}" class="nav-link nav-link-sub">• {subsection}</a>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)


def render_content(df=None, disease_colors: Dict[str, str] | None = None):
    """Render the main content area with all sections."""

    # Import content modules
    from src.content.radiology.intro import get_intro_content
    from src.content.radiology.anatomy import get_anatomy_content
    from src.content.radiology.pathologies import get_pathology_content
    from src.content.radiology.ai_radiology import get_ai_content

    # Section 1: Introduction
    st.markdown('<div id="introduction" class="content-section">', unsafe_allow_html=True)
    with st.expander("📖 Introduction to Chest X-Rays", expanded=True):
        st.markdown(get_intro_content())
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Normal Anatomy
    st.markdown('<div id="normal-anatomy" class="content-section">', unsafe_allow_html=True)
    with st.expander("🫀 Normal Anatomy", expanded=False):
        anatomy_content = get_anatomy_content()

        # Heart & Mediastinum
        st.markdown('<div id="normal-anatomy-heart-and-mediastinum">', unsafe_allow_html=True)
        st.markdown("### Heart & Mediastinum")
        st.markdown(anatomy_content["heart_mediastinum"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Lungs & Airways
        st.markdown('<div id="normal-anatomy-lungs-and-airways">', unsafe_allow_html=True)
        st.markdown("### Lungs & Airways")
        st.markdown(anatomy_content["lungs_airways"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Bones & Soft Tissue
        st.markdown('<div id="normal-anatomy-bones-and-soft-tissue">', unsafe_allow_html=True)
        st.markdown("### Bones & Soft Tissue")
        st.markdown(anatomy_content["bones_soft_tissue"])
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Pathology Guide
    st.markdown('<div id="pathology-guide" class="content-section">', unsafe_allow_html=True)
    with st.expander("🔬 Pathology Guide (14 Conditions)", expanded=False):
        pathologies = get_pathology_content()

        for disease_name, content in pathologies.items():
            disease_id = f"pathology-guide-{disease_name.lower()}"
            st.markdown(f'<div id="{disease_id}">', unsafe_allow_html=True)

            st.markdown(f"### {disease_name}")
            st.markdown(f"**Definition:** {content['definition']}")
            st.markdown(f"**X-Ray Appearance:** {content['appearance']}")
            st.markdown(f"**Key Diagnostic Features:** {content['features']}")
            st.markdown(f"**Clinical Significance:** {content['significance']}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Reading Strategy
    st.markdown('<div id="reading-strategy" class="content-section">', unsafe_allow_html=True)
    with st.expander("📋 Systematic Reading Approach", expanded=False):
        st.markdown("""
        ### The ABCDE Method

        A systematic approach to chest X-ray interpretation helps ensure nothing is missed:

        #### **A - Airway**
        - Trachea: Should be midline
        - Carina: Visible at T4-T5 level
        - Main bronchi: Check for foreign bodies or masses

        #### **B - Breathing (Lungs)**
        - Lung fields: Symmetrical, clear
        - Hilum: Size, density, contour
        - Pleura: Check for effusions, pneumothorax
        - Diaphragm: Position, contour, costophrenic angles

        #### **C - Circulation (Heart & Vessels)**
        - Cardiac size: Cardiothoracic ratio <0.5 on PA
        - Cardiac borders: Sharp and clear
        - Aorta: Size and contour
        - Pulmonary vessels: Distribution and size

        #### **D - Disability (Bones)**
        - Ribs: Fractures, lesions
        - Clavicles: Symmetry
        - Spine: Alignment, fractures
        - Shoulders: Dislocation, arthritis

        #### **E - Everything Else**
        - Soft tissues: Subcutaneous emphysema, masses
        - Foreign bodies: Lines, tubes, devices
        - Review areas: Behind heart, below diaphragm, lung apices

        ### Common Pitfalls

        <div class="warning-box">
        <strong>⚠️ Areas Often Missed:</strong><br>
        • Lung apices (pneumothorax, Pancoast tumor)<br>
        • Behind the heart (lower lobe consolidation)<br>
        • Costophrenic angles (small effusions)<br>
        • Hilar regions (lymphadenopathy, masses)<br>
        • Bones (fractures, metastases)
        </div>

        ### When to Seek Expert Review

        - **Urgent findings**: Large pneumothorax, tension pneumothorax, massive effusion
        - **Unclear diagnoses**: Subtle abnormalities, equivocal findings
        - **Complex cases**: Multiple pathologies, post-surgical changes
        - **Discrepancy with clinical picture**: X-ray findings don't match symptoms
        - **Legal/medicolegal concerns**: Potential missed diagnoses

        <div class="info-box">
        <strong>💡 Remember:</strong> Always correlate radiological findings with clinical history,
        physical examination, and laboratory results. Imaging is just one piece of the diagnostic puzzle.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: AI in Radiology
    st.markdown('<div id="ai-in-radiology" class="content-section">', unsafe_allow_html=True)
    with st.expander("🤖 AI in Radiology", expanded=False):
        st.markdown(get_ai_content())
    st.markdown('</div>', unsafe_allow_html=True)

    # Smooth scroll JavaScript
    st.components.v1.html("""
    <script>
    // Smooth scroll to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    </script>
    """, height=0)
