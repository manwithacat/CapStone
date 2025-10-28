"""
Pathology content for the Radiology Guide (14 NIH dataset conditions).
"""


def get_pathology_content() -> dict:
    """
    Return content for each of the 14 pathologies in the NIH dataset.

    Returns:
        dict: Dictionary with disease names as keys, each containing:
            - definition: What the condition is
            - appearance: How it looks on X-ray
            - features: Key diagnostic features
            - significance: Clinical importance
    """
    return {
        "Atelectasis": {
            "definition": "Collapse of part or all of a lung due to loss of air in the alveoli. Can be complete (entire lung) or partial (segment/lobe).",
            "appearance": "Increased opacity (whiteness) with volume loss. Displaced fissures, elevated hemidiaphragm, mediastinal shift toward the affected side, crowded ribs.",
            "features": (
                "**Direct signs:** Displaced fissures (major or minor), crowding of vessels and bronchi. "
                "**Indirect signs:** Elevation of hemidiaphragm, shift of mediastinum toward collapse, compensatory hyperinflation of adjacent lung, "
                "crowded ribs on affected side. **Air bronchogram:** Usually absent (distinguishes from consolidation)."
            ),
            "significance": (
                "Common post-operative complication (especially after abdominal or thoracic surgery). Can be caused by mucus plugging, "
                "tumor obstruction, pleural effusion, or pneumothorax. Leads to reduced oxygenation and increased infection risk. "
                "Treatment depends on cause: chest physiotherapy, bronchoscopy, or treating underlying obstruction."
            )
        },
        "Cardiomegaly": {
            "definition": "Enlargement of the heart, typically defined as cardiothoracic ratio (CTR) > 0.5 on a PA chest X-ray. Can involve one or more cardiac chambers.",
            "appearance": "Cardiac silhouette wider than half the chest width. May see specific chamber enlargement patterns (e.g., boot-shaped heart in tetralogy of Fallot, flask-shaped in pericardial effusion).",
            "features": (
                "**CTR > 0.5** (heart width / chest width) on PA film (note: AP films magnify the heart and can overestimate size). "
                "**Specific patterns:** Left ventricular enlargement (downward, leftward apex), right ventricular enlargement (lifts apex), "
                "left atrial enlargement (double density sign, splayed carina), right atrial enlargement (right border prominence)."
            ),
            "significance": (
                "Reflects chronic cardiac conditions: heart failure, valvular disease (aortic/mitral regurgitation or stenosis), "
                "hypertension, cardiomyopathy (dilated, hypertrophic, restrictive), congenital heart disease, or pericardial effusion. "
                "Associated with increased mortality and morbidity. Requires echocardiography for detailed assessment and underlying cause determination."
            )
        },
        "Consolidation": {
            "definition": "Replacement of air in the alveoli by fluid, pus, blood, cells, or other material. The alveoli are filled but bronchi remain patent (open).",
            "appearance": "Focal or patchy airspace opacity (white area) with air bronchograms. Margins may be ill-defined or follow lobar/segmental distribution. No volume loss (distinguishes from atelectasis).",
            "features": (
                "**Air bronchograms:** Dark bronchi visible against white (consolidated) lung - key diagnostic feature. "
                "**Silhouette sign:** Loss of normal borders (e.g., right heart border in RML pneumonia, left heart border in lingular pneumonia). "
                "**No volume loss:** Helps distinguish from atelectasis. May be lobar (follows anatomic boundaries) or patchy (scattered)."
            ),
            "significance": (
                "Most commonly due to **pneumonia** (bacterial, viral, fungal), but also seen in pulmonary edema, hemorrhage, aspiration, "
                "ARDS, or alveolar cell carcinoma. Clinical context essential for diagnosis: fever and cough suggest infection, "
                "heart failure suggests edema, trauma suggests hemorrhage. Treatment targets underlying cause (antibiotics for pneumonia, "
                "diuretics for edema)."
            )
        },
        "Edema": {
            "definition": "Accumulation of fluid in the pulmonary interstitium and alveoli, typically due to elevated hydrostatic pressure (cardiogenic) or increased capillary permeability (non-cardiogenic).",
            "appearance": "**Cardiogenic:** Bilateral, symmetric perihilar (bat wing) opacities, Kerley B lines (short horizontal lines at periphery), cephalization (upper lobe vessel enlargement), cardiomegaly, pleural effusions. **Non-cardiogenic (ARDS):** Bilateral diffuse opacities, normal heart size.",
            "features": (
                "**Cardiogenic (heart failure):** Enlarged heart (CTR > 0.5), cephalization (upper lobe blood diversion), "
                "Kerley B lines (septal thickening), perihilar opacities (bat wing pattern), bilateral pleural effusions. "
                "**Non-cardiogenic (ARDS, sepsis, drowning):** Normal heart size, diffuse bilateral opacities, no cephalization, "
                "often more peripheral distribution."
            ),
            "significance": (
                "**Cardiogenic:** Acute decompensated heart failure, acute MI, valvular disease (mitral stenosis/regurgitation), "
                "fluid overload. Life-threatening if severe (respiratory failure, hypoxemia). Treatment: diuretics, oxygen, afterload reduction. "
                "**Non-cardiogenic (ARDS):** Sepsis, pneumonia, aspiration, trauma, pancreatitis, transfusion reaction. High mortality (30-40%). "
                "Requires mechanical ventilation and ICU care."
            )
        },
        "Effusion": {
            "definition": "Abnormal accumulation of fluid in the pleural space (between visceral and parietal pleura). Can be transudative (hydrostatic imbalance) or exudative (inflammation/infection).",
            "appearance": "**Upright:** Blunting of costophrenic angle (earliest sign, ~200 mL), meniscus sign (fluid tracking up lateral chest wall), complete opacification (massive effusion >2L). **Supine:** Diffuse haziness, apical cap. **Loculated:** Lentiform (lens-shaped) opacity that doesn't layer.",
            "features": (
                "**Small (200-500 mL):** Blunted costophrenic angle. **Moderate (500-1000 mL):** Opacification up to 4th rib anteriorly, "
                "meniscus sign (concave upper border). **Large (>1000 mL):** Opacification above 4th rib, possible mediastinal shift away. "
                "**Massive:** Complete opacification, mediastinal shift (unless lung collapse or fixed mediastinum). "
                "**Lateral decubitus view:** Fluid layers out (helpful to distinguish from consolidation)."
            ),
            "significance": (
                "**Transudative (bilateral):** Heart failure, cirrhosis, nephrotic syndrome, hypoalbuminemia. "
                "**Exudative (often unilateral):** Pneumonia (parapneumonic), TB, malignancy, pulmonary embolism, pancreatitis, trauma (hemothorax). "
                "**Hemothorax:** Trauma, aortic dissection, malignancy. **Empyema:** Infected effusion (pneumonia complication). "
                "Large effusions compromise breathing and require drainage (thoracentesis or chest tube)."
            )
        },
        "Emphysema": {
            "definition": "Permanent enlargement of airspaces distal to terminal bronchioles with destruction of alveolar walls. Part of COPD spectrum, usually due to smoking or alpha-1 antitrypsin deficiency.",
            "appearance": "**Hyperinflation:** Flattened diaphragms, increased lung volumes, increased AP diameter (barrel chest). **Hyperlucency:** Darker lungs with decreased vascular markings. **Bullae:** Large air-filled spaces (>1 cm) without visible walls.",
            "features": (
                "**Hyperinflation signs:** Flattened hemidiaphragms (<1.5 cm curvature), diaphragms below 10th posterior rib, "
                "increased retrosternal airspace (>2.5 cm on lateral), saber-sheath trachea. "
                "**Vascular changes:** Decreased peripheral vascular markings (pruning), prominent central pulmonary arteries (pulmonary hypertension). "
                "**Bullae:** Large air-filled spaces, often upper lobes. **Vertical heart:** Narrow cardiac silhouette (dropped heart)."
            ),
            "significance": (
                "Chronic, irreversible lung disease causing progressive dyspnea (shortness of breath). **Risk factors:** Smoking (90% of cases), "
                "alpha-1 antitrypsin deficiency (younger patients, lower lobes). **Complications:** Pneumothorax (ruptured bullae), "
                "respiratory failure, pulmonary hypertension, right heart failure (cor pulmonale). "
                "Management: smoking cessation, bronchodilators, oxygen therapy, pulmonary rehabilitation. Advanced cases may need lung volume reduction surgery or transplant."
            )
        },
        "Fibrosis": {
            "definition": "Scarring and thickening of lung tissue due to chronic inflammation or injury. Part of interstitial lung disease (ILD) spectrum. Can be idiopathic or secondary to many causes.",
            "appearance": "**Reticular pattern:** Fine or coarse lines (thickened interlobular septa). **Honeycombing:** Cystic spaces with thick walls (end-stage fibrosis), typically subpleural and lower lobes. **Volume loss:** Elevated diaphragms, crowded ribs. **Traction bronchiectasis:** Dilated airways due to scarring.",
            "features": (
                "**Early:** Reticular (linear) opacities, ground-glass opacities. **Advanced:** Honeycombing (cystic spaces, 3-10 mm), "
                "traction bronchiectasis (irregular dilated bronchi), volume loss, architectural distortion. "
                "**Distribution:** Often lower lobes and peripheral (subpleural). Nodules may be present (sarcoidosis, hypersensitivity pneumonitis). "
                "**HRCT:** Superior to chest X-ray for detecting early fibrosis and determining pattern."
            ),
            "significance": (
                "**Causes:** Idiopathic pulmonary fibrosis (IPF), connective tissue disease (scleroderma, RA, lupus), drugs (bleomycin, amiodarone, methotrexate), "
                "radiation therapy, occupational exposures (asbestos, silica, coal dust), hypersensitivity pneumonitis (chronic). "
                "**Prognosis:** IPF has poor prognosis (median survival 3-5 years). Progressive dyspnea, dry cough, hypoxemia. "
                "Treatment: antifibrotics (nintedanib, pirfenidone), oxygen, lung transplant in severe cases."
            )
        },
        "Hernia": {
            "definition": "Protrusion of abdominal contents (usually stomach) into the thoracic cavity through the diaphragm. Most commonly **hiatal hernia** (through esophageal hiatus) or traumatic diaphragmatic hernia.",
            "appearance": "**Hiatal hernia:** Retrocardiac air-fluid level or soft tissue mass behind the heart. **Diaphragmatic hernia:** Bowel loops, stomach, or solid organs (liver, spleen) visible in chest. Loss of normal diaphragm contour.",
            "features": (
                "**Hiatal hernia:** Air-fluid level behind heart (gastric fundus in chest), can mimic mediastinal mass. "
                "**Congenital diaphragmatic hernia (Bochdalek):** Usually posterior, left-sided (90%), detected in infants. "
                "**Traumatic diaphragmatic rupture:** Bowel loops in chest, loss of diaphragm contour, mediastinal shift, "
                "nasogastric tube curls up into chest. More common on left (liver protects right). "
                "**Morgagni hernia:** Anterior (rare), right-sided, often asymptomatic."
            ),
            "significance": (
                "**Hiatal hernia:** Common in elderly, obesity, increased abdominal pressure. Usually asymptomatic or causes GERD symptoms. "
                "Rarely causes complications (volvulus, strangulation). "
                "**Traumatic diaphragmatic hernia:** Blunt abdominal/chest trauma (MVA, falls). Can be life-threatening if bowel strangulation occurs. "
                "Requires surgical repair. May present late (months to years) after trauma. "
                "**Congenital:** Requires urgent surgical repair in neonates (respiratory compromise)."
            )
        },
        "Infiltration": {
            "definition": "Diffuse or patchy process with cellular material, fluid, or abnormal substances filling the pulmonary interstitium or alveoli. Non-specific term describing various pathologic processes.",
            "appearance": "Ill-defined opacities that can be patchy, diffuse, or ground-glass. May be focal (one area) or multifocal (multiple areas). Lacks specific features like air bronchograms or volume loss.",
            "features": (
                "**Non-specific findings:** Hazy increased opacity, indistinct borders, may be unilateral or bilateral. "
                "**Differs from consolidation:** Less dense, no clear air bronchograms, less defined margins. "
                "**Differs from interstitial patterns:** More alveolar-like but not as dense as frank consolidation. "
                "Often represents early or subtle disease process not fitting classic patterns."
            ),
            "significance": (
                "**Broad differential diagnosis:** Early pneumonia, atypical infections (viral, mycoplasma), pulmonary edema, "
                "hypersensitivity pneumonitis, drug reaction, aspiration, organizing pneumonia, alveolar hemorrhage, early ARDS. "
                "**Clinical correlation essential:** Fever suggests infection, heart failure suggests edema, travel history suggests endemic fungi or TB. "
                "**Further imaging:** HRCT often needed to characterize infiltrates. Bronchoscopy or biopsy may be required for diagnosis."
            )
        },
        "Mass": {
            "definition": "Focal area of increased opacity >3 cm in diameter. Well-defined or ill-defined margins. Differential includes malignancy (primary lung cancer, metastasis) or benign lesions.",
            "appearance": "Round, oval, or irregular opacity >3 cm. May be solitary or multiple. Can be well-circumscribed (smooth margins) or spiculated (irregular, star-like margins suggesting malignancy).",
            "features": (
                "**Size:** >3 cm (by definition; <3 cm is nodule). **Margins:** Smooth (benign more likely), spiculated/irregular (malignant more likely). "
                "**Density:** Solid, ground-glass, or mixed. **Calcification:** Central, diffuse, or popcorn pattern (usually benign); eccentric or absent (suspicious). "
                "**Cavitation:** Thick irregular walls suggest malignancy or abscess; thin smooth walls suggest benign. "
                "**Associated findings:** Lymphadenopathy (hilar/mediastinal), pleural effusion, chest wall invasion, bone destruction."
            ),
            "significance": (
                "**Malignant causes:** Primary lung cancer (adenocarcinoma, squamous cell, large cell, small cell), metastases (breast, colon, kidney, sarcoma, melanoma), "
                "lymphoma, carcinoid tumor. **Benign causes:** Hamartoma, granuloma (TB, histoplasmosis), abscess, fungal infection, organizing pneumonia, arteriovenous malformation. "
                "**Workup:** CT chest with contrast, PET scan, tissue diagnosis (CT-guided biopsy, bronchoscopy, or surgical resection). "
                "**Prognosis depends on etiology:** Lung cancer has variable prognosis based on stage and type (5-year survival 15-60%)."
            )
        },
        "Nodule": {
            "definition": "Focal area of increased opacity <3 cm in diameter. Described as 'pulmonary nodule' or 'coin lesion'. Can be solitary or multiple.",
            "appearance": "Round or oval opacity <3 cm, well-defined or ill-defined. May be solid, ground-glass, or part-solid. Single or multiple nodules (miliary pattern if very numerous and tiny).",
            "features": (
                "**Size:** <3 cm (>3 cm is mass). **Margins:** Smooth, lobulated, or spiculated. **Density:** Solid (soft tissue), ground-glass (hazy), part-solid (mixed). "
                "**Calcification:** Benign patterns (central, diffuse, popcorn, laminated) vs eccentric or no calcification (suspicious). "
                "**Growth:** Comparison to prior imaging crucial - stability >2 years suggests benign; growth indicates malignancy or active infection. "
                "**Fleischner Society Guidelines:** Risk stratification based on size, density, number, and patient risk factors (smoking, age)."
            ),
            "significance": (
                "**Malignant (10-70% depending on risk factors):** Primary lung cancer (adenocarcinoma most common), metastases, carcinoid tumor. "
                "**Benign:** Granuloma (TB, fungal), hamartoma (benign cartilaginous tumor), infection (pneumonia, abscess, fungal ball), "
                "infarct (pulmonary embolism), arteriovenous malformation, rheumatoid nodule. "
                "**Workup depends on size and risk:** Small low-risk nodules (<6 mm) → surveillance CT. Larger or higher-risk → PET, biopsy, or resection. "
                "**Incidental finding:** Often discovered on imaging for other reasons (chest X-ray, CT abdomen)."
            )
        },
        "Pleural Thickening": {
            "definition": "Abnormal thickening of the visceral or parietal pleura, either focal or diffuse. Can be due to inflammation, infection, asbestos exposure, or malignancy.",
            "appearance": "Smooth or irregular increased density along chest wall or fissures. May be focal (localized to one area) or diffuse (entire hemithorax). Can be unilateral or bilateral.",
            "features": (
                "**Focal:** Localized area of pleural thickening, often result of prior infection (empyema, TB) or trauma. "
                "**Diffuse:** Entire hemithorax involved, causes include asbestos exposure (pleural plaques), mesothelioma, chronic empyema, hemothorax. "
                "**Plaques:** Well-defined, often calcified, bilateral, diaphragmatic and lower chest wall (asbestos). "
                "**Peel:** Thick layer encasing lung (fibrothorax), restricts lung expansion, causes dyspnea. May require decortication surgery."
            ),
            "significance": (
                "**Asbestos-related:** Pleural plaques (benign marker of exposure, no treatment needed), diffuse pleural thickening, mesothelioma (malignant, poor prognosis). "
                "**Post-infectious:** Sequelae of empyema, TB, pneumonia. Usually stable and benign. "
                "**Malignant:** Mesothelioma (unilateral, nodular, associated effusion), metastatic disease (lung, breast, lymphoma). "
                "**Hemothorax:** Organized blood after trauma can cause thick pleural peel. "
                "**Diagnosis:** CT better than X-ray for characterizing pleural disease. Biopsy needed if malignancy suspected."
            )
        },
        "Pneumonia": {
            "definition": "Infection of the lung parenchyma (alveoli) by bacteria, viruses, fungi, or other pathogens. Classified as community-acquired (CAP), hospital-acquired (HAP), or ventilator-associated (VAP).",
            "appearance": "Focal or patchy airspace opacity (consolidation) with air bronchograms. Can be lobar (follows anatomical boundaries), bronchopneumonia (patchy, multifocal), or interstitial (reticular, hazy). May have associated pleural effusion.",
            "features": (
                "**Lobar pneumonia:** Dense homogeneous opacity filling entire lobe, air bronchograms, sharp borders at fissures. "
                "Common organisms: Streptococcus pneumoniae, Klebsiella (upper lobe, bulging fissure). "
                "**Bronchopneumonia:** Patchy, multifocal opacities in multiple lobes. Common organisms: Staphylococcus, Haemophilus, Pseudomonas. "
                "**Interstitial pneumonia:** Reticular, hazy opacities. Common organisms: Viruses (COVID-19, influenza), Mycoplasma, Pneumocystis (PCP in HIV). "
                "**Complications:** Pleural effusion (parapneumonic or empyema), abscess formation (cavitation), sepsis, ARDS."
            ),
            "significance": (
                "Leading cause of infection-related death worldwide. **Risk factors:** Age >65, chronic lung disease (COPD, asthma), immunosuppression, "
                "smoking, alcoholism, aspiration, hospitalization. "
                "**CAP:** Usually bacterial (S. pneumoniae most common), treated outpatient if mild (antibiotics). "
                "**HAP/VAP:** More resistant organisms (MRSA, Pseudomonas), higher mortality, requires broad-spectrum antibiotics. "
                "**Severe pneumonia:** Sepsis, respiratory failure, ICU admission, mechanical ventilation. Mortality 5-30% depending on severity and organism. "
                "**Follow-up X-ray:** Recommended 6-8 weeks after treatment to ensure resolution and exclude underlying malignancy (especially in smokers >50)."
            )
        },
        "Pneumothorax": {
            "definition": "Air in the pleural space causing partial or complete lung collapse. Can be spontaneous (primary or secondary), traumatic, or iatrogenic (procedure-related).",
            "appearance": "Visceral pleural line (thin white line) visible separated from chest wall, with no lung markings peripheral to it. Hyperlucent (very dark) area between lung edge and chest wall. Flattened or inverted hemidiaphragm if tension pneumothorax.",
            "features": (
                "**Visceral pleural line:** Key finding - thin line representing edge of collapsed lung, parallel to chest wall. "
                "**Deep sulcus sign (supine film):** Abnormally deep, lucent costophrenic angle. "
                "**Tension pneumothorax (emergency):** Mediastinal shift away from pneumothorax, flattened hemidiaphragm, hypotension, tracheal deviation. "
                "**Size estimation:** Small (<2 cm at apex), moderate (2-3 cm), large (>3 cm or complete collapse). "
                "**Expiratory film:** Accentuates small pneumothorax by reducing lung volume."
            ),
            "significance": (
                "**Primary spontaneous:** Tall, thin young males (20-30 years), often smokers, ruptured subpleural bleb. Usually benign, recurrence rate 30-50%. "
                "**Secondary spontaneous:** Underlying lung disease (COPD, asthma, CF, ILD, PCP). Higher morbidity and mortality. "
                "**Traumatic:** Rib fractures, penetrating injury, blunt chest trauma. Associated with hemothorax, pulmonary contusion. "
                "**Iatrogenic:** Central line placement, thoracentesis, lung biopsy, mechanical ventilation (barotrauma). "
                "**Treatment:** Small asymptomatic → observation and oxygen. Large or symptomatic → needle aspiration or chest tube. "
                "**Tension pneumothorax:** Life-threatening, immediate needle decompression (2nd intercostal space, midclavicular line) before chest tube."
            )
        }
    }
