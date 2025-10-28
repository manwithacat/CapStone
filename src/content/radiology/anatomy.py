"""
Normal anatomy content for the Radiology Guide.
"""


def get_anatomy_content() -> dict:
    """
    Return content for normal chest anatomy.

    Returns:
        dict: Dictionary with keys for each anatomical region
    """
    return {
        "heart_mediastinum": """
### Heart & Mediastinum

The **mediastinum** is the central compartment of the thoracic cavity, containing the heart, great vessels, trachea, esophagus, thymus, and lymph nodes.

#### **Cardiac Silhouette**

**Normal Size:**
- **Cardiothoracic Ratio (CTR)**: Heart width / Chest width
- Normal CTR: **< 0.5** (50%) on PA view
- CTR > 0.5 suggests **cardiomegaly** (enlarged heart)
- Note: AP films exaggerate heart size (magnification)

**Cardiac Borders (PA View):**

**Right Heart Border** (top to bottom):
1. **Superior Vena Cava (SVC)**: Straight vertical line
2. **Right Atrium (RA)**: Convex bulge
3. **Inferior Vena Cava (IVC)**: Continues below diaphragm

**Left Heart Border** (top to bottom):
1. **Aortic Knob**: Rounded bump (aortic arch)
2. **Pulmonary Artery**: Below aortic knob
3. **Left Atrial Appendage**: Small bump (often not visible)
4. **Left Ventricle (LV)**: Main convex border

<div class="info-box">
<strong>💡 Clinical Tip:</strong> Specific cardiac chambers enlarge in different conditions:
<ul>
<li><strong>Left atrium enlargement:</strong> Mitral stenosis, atrial fibrillation (double density sign, splayed carina)</li>
<li><strong>Left ventricle enlargement:</strong> Hypertension, aortic valve disease (downward, leftward cardiac apex)</li>
<li><strong>Right atrium enlargement:</strong> Tricuspid regurgitation (right border bulges)</li>
<li><strong>Right ventricle enlargement:</strong> Pulmonary hypertension (lifts apex, fills retrosternal space on lateral)</li>
</ul>
</div>

#### **Mediastinal Contours**

**Aorta:**
- **Aortic knob**: Visible on left at ~T4 level
- **Descending aorta**: Parallels left spine
- Width: < 3 cm at knob level
- Widening: Aneurysm, dissection, unfolding (elderly)

**Trachea:**
- Midline structure, 2-3 cm wide
- Air-filled (appears black)
- Deviation: Mass, tension pneumothorax, goiter

**Hila:**
- Left hilum typically **1-2 cm higher** than right
- Contains pulmonary arteries and veins
- Normal: Concave lateral border, uniform density
- Abnormal: Enlarged (lymphadenopathy, masses), asymmetric

#### **Key Anatomical Relationships**

**Right Paratracheal Stripe:**
- Thin line right of trachea (< 4 mm)
- Widened: Lymphadenopathy, mediastinal mass

**Azygoesophageal Recess:**
- Interface between right lung and esophagus/azygos vein
- Convex bulge: Mediastinal mass, enlarged nodes, dilated azygos

**Aortopulmonary Window:**
- Space between aortic arch and pulmonary artery
- Should be concave
- Convex: Lymphadenopathy (e.g., sarcoidosis, lymphoma)

<div class="warning-box">
<strong>⚠️ Red Flags:</strong>
<ul>
<li><strong>Widened mediastinum (>8 cm):</strong> Aortic dissection, hemorrhage, lymphoma</li>
<li><strong>Anterior mediastinal mass:</strong> "4 Ts" - Thyroid, Thymus, Teratoma, Terrible lymphoma</li>
<li><strong>Posterior mediastinal mass:</strong> Neurogenic tumors, esophageal lesions</li>
</ul>
</div>

---
        """,

        "lungs_airways": """
### Lungs & Airways

The lungs are paired organs responsible for gas exchange. Understanding normal lung appearance is critical for identifying pathology.

#### **Lung Fields**

**Normal Appearance:**
- **Radiolucent (black)**: Air-filled alveoli
- **Vascular markings**: Branching vessels extending to periphery
- **Symmetry**: Left and right lungs should appear similar
- **Clarity**: No focal opacities or masses

**Lung Zones (for description):**
- **Upper zones**: Above 2nd anterior rib
- **Mid zones**: 2nd to 4th anterior rib
- **Lower zones**: Below 4th anterior rib

**Lobes:**
- **Right lung**: Upper, middle, lower (3 lobes)
- **Left lung**: Upper (+ lingula), lower (2 lobes)
- Lobes separated by **fissures** (visible as thin white lines)

#### **Fissures**

**Horizontal (Minor) Fissure:**
- Separates right upper and middle lobes
- Visible on PA at ~4th rib anteriorly
- Horizontal line extending from hilum to lateral chest

**Oblique (Major) Fissures:**
- Separate upper from lower lobes (both sides)
- Runs from T4-5 posteriorly to 6th rib anteriorly
- Best seen on lateral view

<div class="info-box">
<strong>💡 Why Fissures Matter:</strong> Fissures help localize pathology to specific lobes:
<ul>
<li><strong>Pneumonia above horizontal fissure:</strong> Right upper lobe</li>
<li><strong>Pneumonia below horizontal fissure:</strong> Right middle or lower lobe</li>
<li><strong>Fluid tracking along fissure:</strong> Pleural effusion</li>
</ul>
</div>

#### **Bronchi & Airways**

**Trachea:**
- Midline, air-filled tube (~2-3 cm diameter)
- Extends from larynx (C6) to carina (T4-5)
- Should be straight and central

**Carina:**
- Bifurcation of trachea into left and right main bronchi
- Located at T4-5 level
- Angle: 60-90 degrees (widened in left atrial enlargement)

**Main Bronchi:**
- **Right main bronchus**: Shorter, wider, more vertical (aspirations more common)
- **Left main bronchus**: Longer, narrower, more horizontal

**Bronchial Markings:**
- Visible as branching white lines (bronchial walls)
- Air bronchograms: Dark bronchi visible in consolidated lung

#### **Pulmonary Vasculature**

**Arteries & Veins:**
- **Upper lobes**: Vessels smaller, less prominent
- **Lower lobes**: Vessels larger, more prominent (gravity effect)
- **Redistribution**: Upper lobe vessels same size as lower → pulmonary venous hypertension, heart failure

**Pulmonary Blood Flow:**
- Normally balanced between upper and lower zones
- **Cephalization** (upper lobe diversion): Heart failure, mitral stenosis
- **Oligemia** (decreased vessels): Emphysema, pneumothorax
- **Hyperemia** (increased vessels): Left-to-right shunt (ASD, VSD)

<div class="warning-box">
<strong>⚠️ Abnormal Lung Patterns:</strong>
<ul>
<li><strong>Hyperlucent lung:</strong> Pneumothorax, emphysema, mastectomy</li>
<li><strong>Opaque lung:</strong> Consolidation, collapse, pleural effusion</li>
<li><strong>Reticular pattern:</strong> Interstitial lung disease, fibrosis</li>
<li><strong>Nodular pattern:</strong> Metastases, miliary TB, sarcoidosis</li>
</ul>
</div>

#### **Pleura**

**Pleural Spaces:**
- **Visceral pleura**: Covers lung surface
- **Parietal pleura**: Lines chest wall
- **Pleural space**: Potential space (normally not visible)

**Costophrenic Angles:**
- Sharp, acute angles where diaphragm meets chest wall
- **Blunting**: First sign of pleural effusion (~200 mL)
- Bilateral: Heart failure, hypoalbuminemia
- Unilateral: Infection, malignancy, trauma

**Hemidiaphragms:**
- **Right hemidiaphragm**: Usually 1-2 cm higher than left (liver)
- Smooth, dome-shaped contours
- **Flattening**: COPD, hyperinflation, asthma
- **Elevated**: Phrenic nerve palsy, subphrenic abscess

---
        """,

        "bones_soft_tissue": """
### Bones & Soft Tissue

While chest X-rays focus on lungs and heart, the bones and soft tissues provide important diagnostic clues and can harbor significant pathology.

#### **Ribs**

**Anatomy:**
- **12 pairs** of ribs (7 true, 3 false, 2 floating)
- Count ribs **posteriorly** (more accurate than anterior)
- Anterior ribs: More horizontal, harder to count

**Normal Features:**
- Smooth, continuous cortex
- Symmetrical spacing
- Gradual tapering from back to front

**Rib Fractures:**
- **Most common**: Ribs 4-9 (middle ribs)
- **Most serious**: Ribs 1-3 (high force trauma), ribs 10-12 (splenic/liver injury)
- **Flail chest**: Segmental fractures (≥3 consecutive ribs) → respiratory compromise
- **Stress fractures**: Overuse (rowers, golfers), chronic cough

<div class="warning-box">
<strong>⚠️ Clinical Significance:</strong>
<ul>
<li><strong>Multiple rib fractures:</strong> Consider non-accidental injury (child abuse), metastases</li>
<li><strong>Healing fractures:</strong> Callus formation appears as bumpy cortex</li>
<li><strong>Pathological fractures:</strong> Metastases (breast, lung, prostate, kidney, thyroid)</li>
</ul>
</div>

**Rib Abnormalities:**
- **Notching**: Coarctation of aorta (collateral vessels), neurofibromatosis
- **Cervical rib**: Extra rib from C7 vertebra (can compress neurovascular structures)
- **Bifid rib**: Split rib (developmental variant, usually benign)

#### **Clavicles**

**Normal Appearance:**
- S-shaped bones connecting sternum to scapula
- Medial 2/3 convex anteriorly, lateral 1/3 concave anteriorly
- Symmetrical on PA view

**Clavicle Fractures:**
- **Most common**: Middle third (80%)
- Often from fall on outstretched hand or direct blow
- Easily visible on X-ray (step-off, angulation)

**Clavicle Abnormalities:**
- **Acromioclavicular (AC) joint separation**: Widened joint space, elevated clavicle
- **Sternoclavicular dislocation**: Rare, high-force injury
- **Bone lesions**: Metastases, myeloma, Paget's disease

#### **Scapulae**

**Anatomy:**
- Large, flat triangular bones on posterior chest wall
- Usually symmetric and flat against ribs

**Scapular Fractures:**
- Rare (high-force trauma needed)
- Associated injuries: Rib fractures, pneumothorax, aortic injury
- Body, neck, glenoid, acromion, or coracoid process

#### **Spine**

**Visible on Chest X-Ray:**
- **Thoracic spine**: T1-T12 visible behind heart and mediastinum
- **Cervical spine**: C7 (sometimes visible at top of film)
- **Alignment**: Should be straight on PA view

**Spinal Abnormalities:**
- **Scoliosis**: Lateral curvature (rotation of vertebrae)
- **Kyphosis**: Excessive forward curvature (hunchback)
- **Compression fractures**: Wedge-shaped vertebrae (osteoporosis, trauma, metastases)
- **Degenerative changes**: Osteophytes (bone spurs), disc space narrowing

<div class="info-box">
<strong>💡 Hidden Pathology:</strong> Don't forget to look at the spine on every chest X-ray!
Lytic (dark) or sclerotic (white) lesions can indicate metastatic disease, myeloma, or infection.
</div>

**Vertebral Lesions:**
- **Lytic (dark)**: Metastases, myeloma, infection
- **Sclerotic (white)**: Prostate cancer metastases, Paget's disease, osteoblastic metastases
- **Compression deformity**: Osteoporosis, pathological fracture

#### **Soft Tissues**

**Chest Wall:**
- **Breast shadows**: Visible as soft tissue density over lower lungs (women)
- **Pectoralis muscles**: Visible along lateral chest wall
- **Mastectomy**: Unilateral hyperlucent hemithorax (more black)

**Subcutaneous Emphysema:**
- Air in soft tissues (appears as black streaks)
- Causes: Pneumothorax, trauma, recent surgery, pneumomediastinum
- Can track into neck and face (crepitus on exam)

**Calcifications:**
- **Vascular calcifications**: Aorta, coronary arteries (atherosclerosis)
- **Costal cartilage**: Normal in older adults
- **Breast calcifications**: Benign (often), require mammogram if suspicious
- **Pleural plaques**: Asbestos exposure (calcified, bilateral)

**Lines and Tubes (Iatrogenic):**
- **Endotracheal tube (ETT)**: Tip should be 2-4 cm above carina
- **Central venous catheter**: Tip in SVC/right atrium
- **Nasogastric tube (NGT)**: Should extend below diaphragm into stomach
- **Chest tubes**: Positioned in pleural space (pneumothorax, effusion)
- **Pacemaker/ICD**: Leads in right atrium and right ventricle

<div class="warning-box">
<strong>⚠️ Line Misplacements:</strong>
<ul>
<li><strong>ETT too high:</strong> Risk of extubation, ventilating right main bronchus only</li>
<li><strong>ETT too low:</strong> Right main bronchus intubation (left lung collapse)</li>
<li><strong>CVC in wrong position:</strong> Can perforate vessel, cause arrhythmias</li>
<li><strong>NGT in lung:</strong> Life-threatening if feeds administered</li>
</ul>
</div>

#### **Surgical Changes**

**Post-Operative Findings:**
- **Sternotomy wires**: Vertical midline wires (cardiac surgery)
- **Thoracotomy clips**: Lateral clips (lung resection)
- **Prosthetic valves**: Metallic heart valves (visible as bright white)
- **CABG clips**: Vascular clips along coronary arteries

**Resections:**
- **Lobectomy**: Missing lobe (volume loss, mediastinal shift)
- **Pneumonectomy**: Entire lung removed (opaque hemithorax, mediastinal shift)
- **Mastectomy**: Unilateral hyperlucency, missing breast shadow

---
        """
    }
