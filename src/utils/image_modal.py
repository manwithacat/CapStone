"""
Image Modal Utility

Provides functionality to display clickable thumbnail images that open
in a full-screen modal/dialog with metadata.
"""

import streamlit as st
import pathlib
import json
from typing import Dict, Optional


def load_sample_metadata() -> Dict:
    """Load metadata for sample X-ray images."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    metadata_path = root / "outputs" / "figures" / "sample_xrays" / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        # Convert list to dict keyed by disease name
        return {item['disease']: item for item in metadata_list}
    return {}


def get_disease_context(disease: str) -> Optional[Dict]:
    """
    Get layman's context about what to look for in the X-ray and what the condition means.

    Returns dict with 'what_to_see' and 'about_condition' keys.
    """
    contexts = {
        "No Finding": {
            "what_to_see": "This is what healthy lungs look like! Notice the **lung fields appear uniformly dark** (black) on both sides, indicating they're filled with air. The **trachea (windpipe) runs straight down the center** of the chest. The **heart silhouette is normal-sized**, taking up less than 50% of the chest width. The **ribs are clearly visible** as curved white lines. Most importantly, look at the **sharp, pointed angles** where the diaphragm meets the ribs at the bottom corners (costophrenic angles)—these should be crisp and well-defined. The **blood vessel patterns branch gradually** from the center outward, becoming fainter toward the edges.",
            "about_condition": "**A normal chest X-ray** shows healthy lungs with no signs of disease. The lungs are fully expanded with air, the heart is normal-sized, there's no fluid or abnormal masses, and all structures are in their proper positions. This is the baseline we compare against when looking for pathology. When your doctor says your chest X-ray is 'clear' or shows 'no acute findings,' this is what they mean—everything looks as it should. Regular chest X-rays showing no findings are reassuring and indicate your lungs and heart are functioning well."
        },
        "Atelectasis": {
            "what_to_see": "Look for **areas that appear whiter** (more opaque) than normal lung tissue, often at the **base of the lungs**. The affected area looks denser because the lung has partially collapsed, similar to a deflated balloon. You may also notice the nearby structures shifting slightly toward the collapsed area.",
            "about_condition": "**Atelectasis** is a partial or complete collapse of lung tissue. This commonly happens after surgery, when lying in bed for long periods, or when airways get blocked by mucus. The collapsed area can't exchange oxygen properly, but it's usually reversible with deep breathing exercises and coughing. Think of it like a sponge that's been compressed—once you expand it again, it returns to normal."
        },
        "Cardiomegaly": {
            "what_to_see": "Notice the **large white shadow in the center-left** of the chest—that's the heart. In this image, the heart appears **wider than it should be**, taking up more than half the width of the chest cavity. A normal heart should be less than 50% of the chest width.",
            "about_condition": "**Cardiomegaly** means an enlarged heart. Your heart is a muscle that grows bigger when it has to work harder—like from high blood pressure, valve problems, or heart failure. Unlike building muscle at the gym, an enlarged heart is less efficient and can lead to serious complications if not treated. It's a sign your heart is struggling and needs medical attention."
        },
        "Consolidation": {
            "what_to_see": "Look for **dense white patches** in the lung fields where you should see dark (air-filled) lung. You might notice **dark lines within the white areas**—these are air-filled airways (bronchi) standing out against the fluid-filled lung tissue, called 'air bronchograms.' This is a characteristic sign of consolidation.",
            "about_condition": "**Consolidation** means the air spaces in your lungs are filled with liquid instead of air—usually from pneumonia, fluid from heart failure, or blood. Imagine a sponge full of water instead of air. The affected area can't exchange oxygen, making breathing difficult. Once the underlying cause is treated (like antibiotics for pneumonia), the fluid clears and the lung returns to normal."
        },
        "Edema": {
            "what_to_see": "Notice the **hazy, white appearance spreading from the center** of both lungs, often in a 'butterfly' or 'bat-wing' pattern. The normally sharp borders of blood vessels appear **fuzzy and indistinct**. You may also see fine horizontal white lines at the lung bases (Kerley B lines).",
            "about_condition": "**Pulmonary edema** is fluid accumulation in the lungs—essentially, your lungs are filling with water. This most commonly happens when your heart isn't pumping well (heart failure), causing blood to back up and force fluid into the lung tissue. It can also occur from lung injuries or infections. This is a medical emergency as it severely impairs breathing. People often describe feeling like they're drowning."
        },
        "Effusion": {
            "what_to_see": "Look at the **bottom corners of the lungs** where they meet the ribs and diaphragm. Normally these corners are sharp angles. Here, you'll see **blunting or a white layer** along the bottom and sides, representing fluid pooling in the space around the lungs. The fluid appears as a curved white line that tracks up the chest wall.",
            "about_condition": "**Pleural effusion** is fluid collecting in the space between your lung and chest wall. Normally this space contains just a tiny amount of lubricating fluid. Excess fluid accumulates from heart failure, infections, cancer, liver disease, or inflammation. The fluid compresses the lung from outside, making it harder to take deep breaths. Doctors can drain large effusions with a needle to relieve symptoms."
        },
        "Emphysema": {
            "what_to_see": "Notice the lungs appear **darker than normal** (almost black) and look **overinflated and larger**. The diaphragm (the curved line at the bottom) is **flattened instead of dome-shaped**. Blood vessel markings are **sparse or absent**, especially at the lung edges. You might see large **dark circular areas** (bullae) where lung tissue has been destroyed.",
            "about_condition": "**Emphysema** is permanent destruction of the air sacs in your lungs, almost always caused by smoking. The walls between air sacs break down, creating large, useless air pockets instead of millions of tiny efficient ones. Your lungs lose elasticity and trap stale air, making it hard to breathe out. It's progressive and irreversible—the damage can't be undone. Quitting smoking stops further damage but can't repair what's already destroyed."
        },
        "Fibrosis": {
            "what_to_see": "Look for a **reticular (net-like) pattern** of white lines throughout the lungs, especially at the **bases and outer edges**. In advanced cases, you may see **honeycombing**—small clustered cystic spaces that look like a honeycomb. The lungs may appear **smaller than normal** due to scarring and contraction.",
            "about_condition": "**Pulmonary fibrosis** is scarring of lung tissue. Just like a scar on your skin is tough and inflexible, scarred lung tissue is stiff and can't expand or exchange oxygen efficiently. It can be caused by medications, radiation, autoimmune diseases, or environmental exposures—but often the cause is never found (idiopathic). Unlike inflammation, scar tissue is permanent. This is a serious, progressive disease that gradually makes breathing more difficult."
        },
        "Hernia": {
            "what_to_see": "Look **behind the heart** or along the diaphragm for an **unusual air bubble or rounded mass**. You might see an **air-fluid level** (horizontal line between air and fluid) in an unexpected location. This represents part of the stomach that has pushed up through the diaphragm into the chest cavity.",
            "about_condition": "**Hiatal hernia** occurs when part of your stomach pushes up through the diaphragm into your chest. The diaphragm is the breathing muscle that separates your chest from your abdomen—it has an opening where your esophagus passes through. When this opening enlarges, the stomach can slide up. Most people don't have symptoms, but it can cause heartburn and acid reflux. Large hernias are usually repaired surgically."
        },
        "Infiltration": {
            "what_to_see": "Notice **hazy, cloudy areas** in the lungs that are whiter than normal but **not as dense as consolidation**. The edges are **fuzzy and ill-defined** rather than sharp. Think of it as looking through a foggy window—you can still make out structures beneath, but everything appears hazy.",
            "about_condition": "**Infiltration** is a general term meaning 'something that shouldn't be there' is filling the spaces in your lungs—could be fluid, inflammatory cells, or infection. It's non-specific, so doctors use your symptoms and other tests to determine the exact cause. It might be early pneumonia, lung inflammation, drug reactions, or various other conditions. Treatment depends entirely on identifying and addressing the underlying cause."
        },
        "Mass": {
            "what_to_see": "Look for a **round or irregular white area** (opacity) within the dark lung tissue that measures **larger than 3cm** (about 1.2 inches). Pay attention to the **edges**—irregular, spiky borders are more concerning than smooth, well-defined ones. The mass might be anywhere in the lung fields.",
            "about_condition": "**A lung mass** is a localized area of abnormal tissue larger than 3cm. While the word 'mass' often raises concerns about cancer, it's not always malignant—it could be a large infection (abscess), inflammatory tissue, or benign tumor. However, lung masses do require immediate investigation with CT scans and often biopsies to determine what they are. Early detection and diagnosis are crucial for the best outcomes."
        },
        "Nodule": {
            "what_to_see": "Look for a **small, round white spot** within the dark lung tissue, **smaller than 3cm** (about 1.2 inches). Nodules can appear anywhere in the lungs. Note whether the edges are **smooth** (often benign) or **spiky/irregular** (more concerning). Some nodules have **bright white calcium deposits** inside, which usually indicates they're old scars.",
            "about_condition": "**A pulmonary nodule** is a small spot in the lung. They're incredibly common—found in about 1 in 4 chest CTs—and most are benign (non-cancerous). They're often scars from old infections, small benign tumors, or inflammatory tissue. However, some can be early lung cancer, especially in smokers. Size, appearance, and risk factors determine next steps—small nodules in low-risk people are usually just monitored, while larger or suspicious ones need further testing."
        },
        "Pleural_Thickening": {
            "what_to_see": "Look along the **outer edges of the lungs** where they meet the chest wall. You'll see a **white line or sheet** following the contour of the ribs, thicker than normal. This represents a thickened, scarred pleura (the membrane lining the lungs). Sometimes you'll see **bright white calcifications** within the thickened areas.",
            "about_condition": "**Pleural thickening** means the thin membrane lining your lungs has become thick and scarred. The most important cause to recognize is asbestos exposure, which causes characteristic calcified pleural plaques. Other causes include previous infections (tuberculosis, pneumonia), previous fluid collections, or inflammatory conditions. Most pleural thickening doesn't cause symptoms or need treatment, but it marks someone at higher risk for certain complications if caused by asbestos."
        },
        "Pneumonia": {
            "what_to_see": "Look for **white, cloudy patches** in the lung fields, often with **air bronchograms** (dark air-filled airways visible against the white infected lung). Bacterial pneumonia often creates **dense, well-defined white areas** filling entire lobes. Viral pneumonia typically causes **patchy, scattered infiltrates**. Check for **fluid at the lung bases** as well.",
            "about_condition": "**Pneumonia** is an infection of the lung tissue itself. Bacteria, viruses, or fungi invade your air sacs, causing them to fill with fluid and inflammatory cells. This prevents oxygen from reaching your bloodstream efficiently. Symptoms include cough, fever, chest pain, and difficulty breathing. Bacterial pneumonia is treated with antibiotics, while viral pneumonia usually requires supportive care. It's one of the most common reasons for hospitalization and can be prevented with vaccines."
        },
        "Pneumothorax": {
            "what_to_see": "Look for a **thin white line** outlining the edge of the collapsed lung, with **excessive blackness beyond it** (that's the air that shouldn't be there). The lung appears **pulled away from the chest wall**, especially visible at the **top (apex) of the lung** where air rises naturally. In severe cases, the entire lung is collapsed into a small ball, and the heart and windpipe may be pushed to the opposite side.",
            "about_condition": "**Pneumothorax** means air has leaked into the space between your lung and chest wall, breaking the vacuum seal that keeps your lung inflated. The lung collapses like a deflated balloon. It can happen spontaneously (especially in tall, thin young men), from trauma, or from lung disease. Small ones may heal on their own, but larger pneumothoraces require a chest tube to remove the air and re-expand the lung. A tension pneumothorax (where pressure builds up) is a life-threatening emergency."
        }
    }

    return contexts.get(disease)


@st.dialog("X-Ray Image Viewer", width="large")
def show_xray_modal(disease: str, metadata: Optional[Dict] = None):
    """
    Display X-ray image in a modal dialog with metadata and contextual information.

    Args:
        disease: Disease name (e.g., "Atelectasis")
        metadata: Optional metadata dict for this image
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    disease_slug = disease.lower().replace('_', '-').replace(' ', '-')
    full_img_path = root / "outputs" / "figures" / "sample_xrays" / "full" / f"{disease_slug}.png"

    # Display title
    st.markdown(f"### {disease}")

    # Add CSS to ensure modal images display at full size
    st.markdown("""
    <style>
    /* Override any conflicting styles for modal images */
    [data-testid="stDialog"] img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        aspect-ratio: unset !important;
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display image - width='stretch' fills the container
    if full_img_path.exists():
        st.image(str(full_img_path), width='stretch')
    else:
        st.error(f"Image not found: {full_img_path}")

    # Get disease context
    disease_context = get_disease_context(disease)

    # Display educational context
    if disease_context:
        st.markdown("---")
        st.markdown("#### 📖 What You're Looking At")
        st.markdown(disease_context['what_to_see'])

        st.markdown("#### 🩺 About This Condition")
        st.markdown(disease_context['about_condition'])

    # Display radiologist note if available (before patient information)
    if metadata and 'radiologist_note' in metadata:
        st.markdown("---")
        st.markdown("**Radiologist's Note:**")
        st.info(metadata['radiologist_note'])

    # Display metadata if available
    if metadata:
        st.markdown("---")
        st.markdown("#### 📋 Patient Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient ID", metadata.get('patient_id', 'N/A'))
        with col2:
            st.metric("Age", f"{metadata.get('patient_age', 'N/A')} years")
        with col3:
            st.metric("Gender", metadata.get('patient_gender', 'N/A'))

        st.markdown("**All Findings:** " + metadata.get('finding_labels', 'N/A'))
        st.caption(f"View Position: {metadata.get('view_position', 'N/A')} | Image: {metadata.get('image_index', 'N/A')}")

        # Display attribution if available
        if 'attribution' in metadata:
            st.markdown("---")
            st.markdown("**Image Attribution:**")
            st.markdown(metadata['attribution'], unsafe_allow_html=True)


def render_clickable_xray(disease: str, use_thumbnail: bool = True, caption: Optional[str] = None, width: Optional[int] = None):
    """
    Render a clickable X-ray image that opens in a modal when clicked.

    Args:
        disease: Disease name (e.g., "Atelectasis")
        use_thumbnail: If True, use thumbnail version; otherwise use full version
        caption: Optional caption for the image
        width: Optional width in pixels for the image
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent

    # Determine image path
    disease_slug = disease.lower().replace('_', '-').replace(' ', '-')
    if use_thumbnail:
        img_path = root / "outputs" / "figures" / "sample_xrays" / "thumbnails" / f"{disease_slug}.png"
    else:
        img_path = root / "outputs" / "figures" / "sample_xrays" / "full" / f"{disease_slug}.png"

    # Load metadata
    metadata = load_sample_metadata().get(disease)

    if not img_path.exists():
        st.warning(f"Image not found for {disease}")
        return

    # Display clickable image
    if caption is None:
        caption = disease

    # Create container for clickable image
    with st.container():
        # Create a button that looks like it's part of the image using custom styling
        # We use st.button but style it to appear as a clickable image overlay
        button_label = f"**{caption}**\n\n_Click to view details_"

        # Display image with consistent sizing
        # Don't specify width parameter - let CSS handle it
        st.image(str(img_path))

        # Clickable button integrated with the image
        if st.button(button_label, key=f"modal_{disease_slug}", use_container_width=True, type="secondary"):
            show_xray_modal(disease, metadata)


def render_xray_grid(diseases: list, use_thumbnails: bool = True):
    """
    Render a responsive grid of clickable X-ray thumbnails.

    Grid automatically adjusts:
    - Desktop (>1024px): 3 columns
    - Tablet (769-1024px): 2 columns
    - Mobile (<768px): 1 column

    Args:
        diseases: List of disease names
        use_thumbnails: Whether to use thumbnail or full images
    """
    # Create responsive grid using Streamlit columns
    # We'll create rows of 3 items each
    num_cols = 3

    for i in range(0, len(diseases), num_cols):
        cols = st.columns(num_cols)
        row_diseases = diseases[i:i+num_cols]

        for col_idx, disease in enumerate(row_diseases):
            with cols[col_idx]:
                render_clickable_xray(disease, use_thumbnail=use_thumbnails)
