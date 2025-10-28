"""
Introduction content for the Radiology Guide.
"""


def get_intro_content() -> str:
    """Return introduction content for chest X-ray interpretation."""
    return """
    ### What is a Chest X-Ray?

    A **chest X-ray** (CXR) is one of the most common diagnostic imaging tests, using a small amount
    of ionizing radiation to create pictures of the structures inside the chest. It's often the first
    imaging test ordered to evaluate symptoms like cough, shortness of breath, chest pain, or fever.

    **Common Uses:**
    - Diagnosing lung conditions (pneumonia, tuberculosis, lung cancer)
    - Detecting heart problems (cardiomegaly, heart failure)
    - Evaluating chest injuries (rib fractures, pneumothorax)
    - Monitoring chronic conditions (COPD, cystic fibrosis)
    - Pre-operative screening

    ---

    ### Standard X-Ray Views

    #### **PA (Posteroanterior) View**
    - Most common view for chest X-rays
    - X-ray beam passes from **back to front** (posterior to anterior)
    - Patient stands facing the detector, arms raised
    - Taken at full inspiration (deep breath in)
    - **Best for**: Evaluating heart size and lung fields

    <div class="info-box">
    <strong>💡 Why PA?</strong> The PA view minimizes magnification of the heart because the heart
    is closer to the detector (front of chest). This gives a more accurate assessment of cardiac size.
    </div>

    #### **AP (Anteroposterior) View**
    - X-ray beam passes from **front to back**
    - Often taken when patient cannot stand (bedside, ICU)
    - Heart appears larger due to magnification
    - Less preferred but necessary in certain situations

    #### **Lateral View**
    - Side view of the chest
    - Patient stands sideways with arms raised
    - **Best for**: Localizing abnormalities (anterior vs posterior)
    - Helps identify lesions hidden behind heart on PA view

    ---

    ### How to Orient a Chest X-Ray

    When viewing a PA chest X-ray:

    1. **Patient's Right is on Your Left** - Like shaking hands with the patient
    2. **Look for markers** - "R" or "L" marker confirms orientation
    3. **Identify key landmarks**:
       - Trachea (dark air column down the midline)
       - Carina (where trachea splits into bronchi)
       - Heart shadow (left side larger than right)
       - Diaphragms (right usually higher than left)
       - Ribs (count posteriorly for accuracy)

    <div class="warning-box">
    <strong>⚠️ Common Mistake:</strong> Viewing the X-ray with patient's right on your right.
    Always imagine you're facing the patient!
    </div>

    ---

    ### Key Radiological Principles

    #### **Radiodensity**
    Different tissues absorb X-rays differently, creating contrast:

    | Appearance | Density | Examples |
    |-----------|---------|----------|
    | **Black (Radiolucent)** | Low | Air (lungs, trachea, pneumothorax) |
    | **Dark Grey** | Low-Medium | Fat, soft tissue |
    | **Light Grey** | Medium | Fluid, blood, muscle, heart |
    | **White (Radiopaque)** | High | Bone, calcification, metal |

    #### **Silhouette Sign**
    When two structures of the same density are adjacent, their borders disappear:
    - Loss of right heart border → Right middle lobe pneumonia
    - Loss of left heart border → Lingular pneumonia
    - Loss of diaphragm → Lower lobe consolidation

    #### **Air Bronchogram Sign**
    Air-filled bronchi visible against consolidated (fluid-filled) lung:
    - Indicates **alveolar** disease (not interstitial)
    - Common in pneumonia, pulmonary edema, ARDS

    ---

    ### Reading Order Matters

    A systematic approach prevents missing subtle findings. Most radiologists use:

    1. **Patient details**: Name, date, compare to priors
    2. **Technical quality**: Rotation, inspiration, penetration
    3. **ABCDE method**: Airway → Breathing → Circulation → Disability → Everything else
    4. **Review areas**: Don't miss lung apices, angles, behind heart
    5. **Clinical correlation**: Does it match the history?

    <div class="info-box">
    <strong>📚 Learning Tip:</strong> Practice reading normal X-rays first! You need to know what
    "normal" looks like before you can identify abnormalities. Look at 100 normal chest X-rays
    before diving into pathology.
    </div>

    ---

    ### Quality Assessment

    Before interpreting, ensure adequate technical quality:

    #### **Rotation**
    - Spinous processes should be equidistant from clavicular heads
    - Rotated X-rays can mimic mediastinal widening

    #### **Inspiration**
    - Should see 5-6 anterior ribs or 9-10 posterior ribs
    - Poor inspiration can mimic cardiomegaly or basal consolidation

    #### **Penetration**
    - Vertebrae barely visible through heart
    - Overexposed (too black) → Miss subtle opacities
    - Underexposed (too white) → Miss pneumothorax

    ---

    ### Resources for Further Learning

    - **Radiopaedia**: Comprehensive online radiology resource
    - **STATdx**: Clinical decision support tool
    - **RSNA (Radiological Society of North America)**: Educational materials
    - **Felson's Principles of Chest Roentgenology**: Classic textbook

    """
