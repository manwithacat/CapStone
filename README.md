# Chest X-Ray Disease Detection and Classification

![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Project Overview

This project leverages deep learning and computer vision techniques to detect and classify thoracic diseases from chest X-ray images. Using the NIH Chest X-Ray dataset, the analysis develops automated diagnostic support tools to assist radiologists and healthcare providers in identifying multiple pathological conditions, improving diagnostic accuracy and patient care efficiency.

**Dataset**: NIH Chest X-Rays (112,000+ images from 30,000+ patients)
**Domain**: Healthcare - Medical Imaging and Diagnostic Support
**Target Audience**: Radiologists, healthcare administrators, medical AI researchers, hospital decision-makers

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Business Requirements](#business-requirements)
4. [Project Hypothesis](#project-hypothesis)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Dashboard Design](#dashboard-design)
7. [Project Structure](#project-structure)
8. [Technologies Used](#technologies-used)
9. [Installation and Setup](#installation-and-setup)
10. [Development Process](#development-process)
11. [Testing and Validation](#testing-and-validation)
12. [Deployment](#deployment)
13. [Future Enhancements](#future-enhancements)
14. [Credits and Acknowledgments](#credits-and-acknowledgments)

---

## Dataset Description

### Source
- **Platform**: Kaggle / NIH Clinical Center
- **Dataset**: NIH Chest X-Ray Dataset
- **Images**: 112,120 frontal-view X-ray images
- **Patients**: 30,805 unique patients
- **Resolution**: 1024 x 1024 pixels
- **Format**: PNG grayscale images

### Disease Labels (Multi-Label Classification)
The dataset includes 14 thoracic pathology labels plus "No Finding":
1. **Atelectasis**: Partial lung collapse
2. **Cardiomegaly**: Enlarged heart
3. **Effusion**: Fluid around lungs
4. **Infiltration**: Abnormal substances in lungs
5. **Mass**: Abnormal tissue growth
6. **Nodule**: Small rounded growth
7. **Pneumonia**: Lung infection
8. **Pneumothorax**: Collapsed lung
9. **Consolidation**: Lung tissue solidification
10. **Edema**: Fluid buildup in lungs
11. **Emphysema**: Damaged air sacs
12. **Fibrosis**: Lung scarring
13. **Pleural Thickening**: Thickened lung lining
14. **Hernia**: Organ displacement
15. **No Finding**: Healthy/normal

### Key Characteristics
- **Multi-label**: Images can have multiple disease labels (co-morbidities)
- **Class Imbalance**: "No Finding" (60,361 images) vs rare diseases
- **Metadata**: Patient age, gender, view position, image dimensions
- **Clinical Annotations**: Text-mined from radiological reports (90%+ accuracy)

### Enhanced Expert Labels (Google Cloud Healthcare)
**Important**: The original NIH labels were automatically extracted from radiological reports using NLP, which can introduce labeling noise. To improve model quality, this project incorporates **enhanced expert labels** provided by Google Cloud Healthcare:

- **Source**: [Google Cloud Public Dataset - NIH Chest X-Ray Labels](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels)
- **Quality**: Expert radiologist annotations via adjudicated review process
- **Coverage**: Two label sets provided:

  **1. Four Findings Expert Labels** (4,374 images)
  - **Publication**: Majkowska et al., *Radiology*, 2019
  - **Paper**: [Chest Radiograph Interpretation with Deep Learning Models](https://pubs.rsna.org/doi/10.1148/radiol.2019191293)
  - **Findings**: Airspace opacity, pneumothorax, nodule/mass, fracture
  - **Process**: Adjudicated review by 3 radiologists from cohort of 11+ board-certified radiologists
  - **Sets**: Validation (2,412 images) + Test (1,962 images)

  **2. All Findings Expert Labels** (810 images)
  - **Publication**: Nabulsi et al., *Nature Scientific Reports*, 2021
  - **Paper**: [Deep Learning for Distinguishing Normal vs Abnormal Chest Radiographs](https://arxiv.org/abs/2010.11375)
  - **Findings**: All 14 pathologies + normal/abnormal classification
  - **Process**: Independent review by 5 board-certified radiologists, majority vote for ground truth
  - **Set**: Test set only (PA views)

- **Format**: CSV files with image IDs and adjudicated labels per finding
- **Provider**: Google LLC / Google Health AI
- **License**: Same as NIH dataset (CC0: Public Domain)

**Why Expert Labels Matter**: The original NIH labels have known accuracy limitations (~90%) due to automated extraction. Expert-validated labels provide ground truth for:
- Training more accurate models on high-quality annotations
- Validating model performance against radiologist consensus
- Reducing false positive/negative rates in critical diagnoses
- Benchmarking against published results from *Radiology* and *Scientific Reports*

---

## Business Requirements

### Primary Objectives
1. **Automated Disease Detection**: Develop AI-powered models to accurately identify thoracic pathologies from chest X-rays
2. **Multi-Disease Classification**: Build systems capable of detecting multiple co-existing conditions in single images
3. **Diagnostic Support Tool**: Create an interactive application to assist radiologists in preliminary screening and triage
4. **Clinical Decision Support**: Provide confidence scores and visual explanations to support clinical decision-making
5. **Healthcare Efficiency**: Reduce diagnostic time and improve early detection rates for critical conditions

### Success Criteria
- Achieve model AUC-ROC >0.80 for primary disease categories
- Successfully detect multi-label cases (images with 2+ diseases)
- Deliver interpretable predictions with visual heatmaps (Grad-CAM)
- Create user-friendly dashboard for both technical (radiologists) and non-technical (administrators) audiences
- Demonstrate statistical significance in disease pattern analysis
- Address ethical considerations (bias, privacy, clinical validation)

### Clinical Impact
- **Early Detection**: Identify subtle disease patterns humans might miss
- **Triage Support**: Prioritize urgent cases in high-volume settings
- **Second Opinion**: Provide supplementary diagnostic confirmation
- **Resource Optimization**: Allocate radiologist time to complex cases
- **Rural Healthcare**: Support under-resourced medical facilities with limited specialists

---

## Project Hypothesis

### Hypothesis 1: Age-Disease Correlation
**Statement**: Patient age significantly correlates with the prevalence of specific thoracic diseases, with conditions like cardiomegaly and emphysema more common in older patients, while pneumonia shows more uniform age distribution.

**Validation Method**:
- Age distribution analysis across disease categories
- Statistical significance testing (ANOVA, t-tests)
- Correlation coefficients between age and disease presence
- Visualization: Box plots and violin plots of age vs disease

### Hypothesis 2: Multi-Label Disease Patterns
**Statement**: Certain thoracic pathologies co-occur more frequently than random chance (e.g., effusion with cardiomegaly, infiltration with pneumonia), indicating underlying clinical relationships.

**Validation Method**:
- Co-occurrence matrix analysis
- Chi-square test for independence between disease pairs
- Association rule mining (support, confidence, lift metrics)
- Heatmap visualization of disease co-occurrence rates

### Hypothesis 3: Class Imbalance Impact
**Statement**: Deep learning models trained on the imbalanced dataset will show significantly better performance on common conditions (e.g., "No Finding", Infiltration) compared to rare diseases (e.g., Hernia, Pneumothorax) without specialized balancing techniques.

**Validation Method**:
- Per-class AUC-ROC and F1-score comparison
- Baseline model vs balanced model (SMOTE, class weights) performance
- Precision-recall curves for rare vs common diseases
- Statistical significance of performance differences

### Hypothesis 4: Transfer Learning Superiority
**Statement**: Pre-trained convolutional neural networks (ResNet, DenseNet, EfficientNet) fine-tuned on chest X-rays will significantly outperform models trained from scratch, achieving higher AUC-ROC scores with fewer training epochs.

**Validation Method**:
- Compare baseline CNN vs transfer learning models
- Training efficiency: epochs to convergence, training time
- Performance metrics: AUC-ROC, sensitivity, specificity per disease
- Feature visualization: t-SNE plots of learned representations

### Hypothesis 5: Gender Differences in Disease Prevalence
**Statement**: Certain thoracic diseases show statistically significant gender differences in prevalence rates within the dataset.

**Validation Method**:
- Stratified disease prevalence analysis by gender
- Chi-square tests for gender-disease associations
- Odds ratios with confidence intervals
- Visualization: Grouped bar charts of disease rates by gender

---

## Machine Learning Pipeline

### 1. Data Collection and Understanding
- Download NIH Chest X-Ray dataset from Kaggle
- Load image metadata (patient ID, age, gender, disease labels)
- Explore dataset structure and label distribution
- Sample image visualization and quality assessment
- Analyze class imbalance and multi-label statistics

### 2. Data Preprocessing and Augmentation
- **Image Loading**: Read PNG files, convert to arrays
- **Resizing**: Standardize to 224x224 or 512x512 pixels
- **Normalization**: Scale pixel values to [0,1] or standardize
- **Label Encoding**: Convert multi-label disease annotations to binary vectors
- **Train/Validation/Test Split**: 70/15/15 stratified split
- **Data Augmentation** (training only):
  - Random rotation (±15 degrees)
  - Horizontal flip (chest X-rays are symmetric)
  - Brightness/contrast adjustment
  - Zoom and crop variations
  - Gaussian noise addition

### 3. Exploratory Data Analysis
- **Metadata Analysis**: Age distribution, gender ratio, view positions
- **Label Distribution**: Disease frequency analysis, class imbalance quantification
- **Co-occurrence Analysis**: Disease correlation heatmap
- **Image Statistics**: Pixel intensity distributions, contrast patterns
- **Statistical Hypothesis Testing**: Age-disease correlation, gender differences
- **Visualization**: Sample images per disease category

### 4. Feature Extraction and Engineering
- **Traditional Features** (for baseline models):
  - Histogram of Oriented Gradients (HOG)
  - Edge detection features
  - Texture descriptors (GLCM)
- **Deep Learning Features**:
  - Pre-trained CNN feature extraction (ImageNet weights)
  - Transfer learning embeddings (ResNet, DenseNet, EfficientNet)
- **Metadata Features**:
  - Age bins (categorical)
  - Gender encoding
  - Image quality metrics

### 5. Model Development

#### Baseline Models (Traditional ML)
- **Logistic Regression**: On extracted HOG/texture features
- **Random Forest**: Multi-output classifier for multi-label prediction
- **XGBoost**: Gradient boosting with class weight balancing

#### Deep Learning Models (Primary Focus)

**5a. Custom CNN Architecture**
- Convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Dense layers for multi-label classification
- Sigmoid activation (multi-label output)

**5b. Transfer Learning Models**
- **ResNet50**: Deep residual network pre-trained on ImageNet
- **DenseNet121**: Densely connected architecture (commonly used for medical imaging)
- **EfficientNetB3**: Efficient scaling of CNN architecture
- Fine-tuning strategy: Freeze early layers, train final layers

**5c. Ensemble Methods**
- Weighted average of multiple model predictions
- Stacking classifier combining CNN outputs

#### Multi-Label Classification Strategies
- **Binary Relevance**: Separate binary classifier per disease
- **Classifier Chains**: Sequential classifiers capturing label dependencies
- **Problem Transformation**: Multi-label to multi-class conversion

### 6. Handling Class Imbalance
- **Class Weights**: Assign higher weights to rare diseases
- **Focal Loss**: Focus on hard-to-classify examples
- **Oversampling**: SMOTE for minority classes (traditional ML)
- **Undersampling**: Reduce majority class samples
- **Threshold Optimization**: Adjust decision thresholds per class

### 7. Model Evaluation

#### Metrics (Per Disease + Overall)
- **AUC-ROC**: Area under ROC curve (primary metric)
- **AUC-PR**: Precision-recall AUC (for imbalanced classes)
- **Sensitivity/Recall**: True positive rate (critical for medical screening)
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **Hamming Loss**: Multi-label classification error
- **Subset Accuracy**: Exact match of all labels

#### Validation Strategies
- **K-Fold Cross-Validation**: 5-fold stratified CV
- **Patient-Level Split**: Ensure no patient data leakage between sets
- **Temporal Validation**: If timestamp data available

#### Confusion Matrix Analysis
- Per-disease confusion matrices
- Multi-label confusion visualization

### 8. Model Interpretation and Explainability
- **Grad-CAM** (Gradient-weighted Class Activation Mapping): Visual heatmaps showing which image regions influence predictions
- **Saliency Maps**: Highlight important pixels for classification
- **Feature Importance**: For traditional ML models
- **Error Analysis**: Study false positives and false negatives
- **Clinical Validation**: Compare predictions with radiologist annotations

### 9. Model Optimization
- **Hyperparameter Tuning**: Learning rate, batch size, dropout rate
- **Architecture Search**: Layer depth, filter sizes
- **Regularization**: L2 weight decay, dropout tuning
- **Early Stopping**: Prevent overfitting using validation loss
- **Learning Rate Scheduling**: Reduce LR on plateau

---

## Dashboard Design

### Page 1: Project Overview and Clinical Context
- **Healthcare Challenge**: Current radiologist workload and diagnostic accuracy
- **AI Solution**: Automated screening and triage support
- **Dataset Overview**: NIH Chest X-Ray statistics (112K images, 15 conditions)
- **Key Findings Summary**: Model performance highlights, clinical insights
- **Navigation Guide**: Dashboard structure and user instructions
- **Ethical Disclaimer**: Tool is for research/educational purposes, not clinical use

### Page 2: Dataset Exploration and Statistics
- **Patient Demographics**:
  - Age distribution histogram with disease overlays
  - Gender distribution pie chart
  - Interactive filters by age groups and gender
- **Disease Distribution**:
  - Bar chart: Disease frequency (logarithmic scale for imbalance)
  - Multi-label statistics: Co-occurrence heatmap
  - Pie chart: Single vs multi-disease cases
- **Sample Image Gallery**:
  - Grid display: One example per disease category
  - Image viewer with zoom capability
  - Healthy vs diseased comparison
- **Statistical Summaries**:
  - Dataset size, patient count, image resolution
  - Class imbalance metrics (Gini coefficient, imbalance ratio)

### Page 3: Hypothesis Validation and Clinical Insights
- **Hypothesis 1 - Age Correlation**:
  - Box plots: Age distribution per disease
  - Statistical test results (ANOVA p-values)
  - Interpretation: Age-specific disease patterns
- **Hypothesis 2 - Disease Co-occurrence**:
  - Interactive heatmap: Disease pair correlations
  - Association rules table (support, confidence, lift)
  - Clinical significance of findings
- **Hypothesis 3 - Class Imbalance**:
  - Performance comparison: Balanced vs imbalanced models
  - Per-class F1-score visualization
  - Impact analysis
- **Hypothesis 4 - Transfer Learning**:
  - Training curves: Accuracy and loss over epochs
  - Model comparison table (AUC-ROC scores)
  - Training time efficiency chart
- **Hypothesis 5 - Gender Differences**:
  - Grouped bar charts: Disease prevalence by gender
  - Chi-square test results
  - Odds ratios with confidence intervals

### Page 4: Model Performance and Evaluation
- **Overall Performance Metrics**:
  - Model comparison table (Logistic, Random Forest, CNN, Transfer Learning)
  - Best model highlight with key metrics
- **Per-Disease Performance**:
  - Interactive table: AUC-ROC, Sensitivity, Specificity, F1 per disease
  - Sort and filter capabilities
- **ROC Curves**:
  - Multi-class ROC plot (15 diseases)
  - Interactive legend to toggle disease curves
- **Precision-Recall Curves**:
  - Especially important for imbalanced classes
- **Confusion Matrices**:
  - Dropdown selector for disease category
  - Heatmap visualization
- **Training History**:
  - Loss and accuracy curves (train vs validation)
  - Early stopping indicator

### Page 5: Disease Detection Tool (Interactive Predictor)
- **Image Upload Interface**:
  - Drag-and-drop or file browser
  - Image preview display
- **Prediction Results**:
  - Top 5 predicted diseases with confidence scores
  - Probability bars for all 15 conditions
  - Multi-label predictions highlighted
- **Visual Explanation**:
  - Grad-CAM heatmap overlay on X-ray
  - Regions of interest highlighted
  - Toggle original vs heatmap view
- **Clinical Context**:
  - Brief description of detected conditions
  - Typical symptoms and severity indicators
- **Disclaimer**: Prominent note about non-clinical use

### Page 6: Clinical Insights and Recommendations
- **Key Disease Patterns**:
  - Most common conditions
  - Frequently co-occurring diseases
  - Age and gender risk factors
- **Model Strengths and Limitations**:
  - Diseases with highest accuracy
  - Challenges with rare conditions
  - Error analysis: Common misclassifications
- **Clinical Applications**:
  - Triage workflow integration
  - Second-opinion support
  - Rural/under-resourced healthcare settings
- **Future Improvements**:
  - Larger dataset requirements
  - External validation needs
  - Integration with PACS systems
- **Ethical Considerations**:
  - Bias in dataset (population representation)
  - Privacy and HIPAA compliance
  - Human-in-the-loop necessity
  - Regulatory approval requirements (FDA, CE marking)

### Design Principles
- **Medical-Grade Interface**: Clean, professional, clinical aesthetic
- **Color Scheme**: Healthcare-appropriate (blues, whites, minimal red for alerts)
- **Accessibility**: WCAG 2.1 AA compliance, screen reader support
- **Responsive Design**: Desktop focus (radiologist workstations), mobile-friendly
- **Clear Labels**: Medical terminology with tooltips for explanations
- **Performance**: Fast loading for large images (lazy loading, caching)
- **Privacy**: No data retention, local processing only

---

## Project Structure

```
CapStone/
│
├── .venv/                          # Virtual environment (not tracked by Git)
├── data/
│   ├── raw/                        # Original dataset from Kaggle
│   └── processed/                  # Cleaned and transformed data
│
├── jupyter_notebooks/
│   ├── 01_data_collection.ipynb    # Data import and initial exploration
│   ├── 02_data_cleaning.ipynb      # Data cleaning and preprocessing
│   ├── 03_eda.ipynb                # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb # Feature creation and selection
│   ├── 05_modeling.ipynb           # Model training and evaluation
│   └── 06_model_evaluation.ipynb   # Final model assessment
│
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading utilities
│   ├── preprocessing/
│   │   ├── cleaning.py             # Data cleaning functions
│   │   └── feature_engineering.py  # Feature engineering functions
│   ├── modeling/
│   │   ├── train.py                # Model training scripts
│   │   └── evaluate.py             # Model evaluation scripts
│   └── visualization/
│       └── plots.py                # Plotting functions
│
├── app/
│   ├── streamlit_app.py            # Main Streamlit dashboard
│   └── pages/
│       ├── 1_summary.py            # Project summary page
│       ├── 2_exploration.py        # Data exploration page
│       ├── 3_hypothesis.py         # Hypothesis validation page
│       ├── 4_prediction.py         # Churn prediction page
│       └── 5_insights.py           # Business insights page
│
├── models/
│   └── saved_models/               # Trained model artifacts
│
├── docs/
│   ├── Assessment_Handbook.md      # Project requirements
│   └── development_log.md          # Development progress notes
│
├── tests/
│   └── test_data_processing.py     # Unit tests
│
├── .gitignore
├── .python-version
├── requirements.txt
├── Makefile                        # Automation scripts
├── README.md
└── LICENSE

```

---

## Technologies Used

### Programming Language
- **Python 3.12.8**: Primary language for data analysis and application development

### Deep Learning Frameworks
- **TensorFlow 2.x**: Deep learning framework for CNN development
- **Keras**: High-level neural network API (integrated with TensorFlow)
- **PyTorch** (optional): Alternative deep learning framework

### Computer Vision and Image Processing
- **OpenCV (cv2)**: Image loading, preprocessing, and manipulation
- **Pillow (PIL)**: Image file handling
- **scikit-image**: Image processing algorithms
- **albumentations**: Advanced image augmentation library

### Machine Learning Libraries
- **scikit-learn**: Traditional ML algorithms (Logistic Regression, Random Forest) and evaluation metrics
- **xgboost**: Gradient boosting framework for baseline models
- **imbalanced-learn**: Handling class imbalance (SMOTE, class weights)
- **scipy**: Statistical functions and hypothesis testing

### Data Analysis
- **pandas**: Metadata manipulation and label management
- **numpy**: Numerical computing and array operations

### Data Visualization
- **matplotlib**: Static plotting library (training curves, distributions)
- **seaborn**: Statistical data visualization (heatmaps, box plots)
- **plotly**: Interactive visualizations for dashboard

### Model Interpretation and Explainability
- **tf-keras-vis**: Grad-CAM and visualization for TensorFlow/Keras
- **keras-gradcam**: Alternative Grad-CAM implementation
- **shap**: Model explainability (for traditional ML models)

### Dashboard and Web Application
- **Streamlit**: Interactive web dashboard framework
- **streamlit-extras**: Additional Streamlit components
- **streamlit-drawable-canvas**: Image annotation (if needed)

### Development and DevOps
- **jupyter**: Interactive notebook environment
- **nbstripout**: Strip output from Jupyter notebooks for version control
- **nbdime**: Diff and merge for notebooks
- **pytest**: Unit testing framework
- **black**: Code formatting
- **flake8**: Code linting

### Version Control and Collaboration
- **Git**: Version control system
- **GitHub**: Repository hosting and collaboration
- **GitHub Actions**: CI/CD automation

---

## Installation and Setup

### Prerequisites
- **Python 3.9+** (3.12.8 recommended)
- **Git** installed on your machine
- **Make** utility (see installation instructions below)
- **VS Code** (recommended) or other IDE
- **10+ GB disk space** for dataset

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd CapStone
```

### Step 2: Install Make (if not already installed)

**macOS**:
```bash
# Install Xcode Command Line Tools (includes make)
xcode-select --install

# Or via Homebrew
brew install make
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install build-essential
```

**Windows**:
```powershell
# Option 1: Install via Chocolatey (recommended)
choco install make

# Option 2: Install via winget
winget install GnuWin32.Make

# Option 3: Use WSL (Windows Subsystem for Linux)
# Then follow Linux instructions above
```

**Verify make installation**:
```bash
make --version
# Should show: GNU Make 4.x or similar
```

### Step 3: Create Virtual Environment
**In VS Code**:
1. Open Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
2. Type "Python: Create Environment"
3. Select "Venv"
4. Choose Python 3.12.8 (or 3.9+)
5. Do NOT select requirements.txt yet

**Or via terminal**:
```bash
python -m venv .venv
```

### Step 4: Activate Virtual Environment
**Windows**:
```bash
.venv\Scripts\activate
```

**Mac/Linux**:
```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies and Project Package
```bash
# Install all dependencies
make install

# OR manually:
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .  # Install project in editable mode
```

**Why `pip install -e .`?**
- Installs the `src/` directory as a Python package
- Enables clean imports: `from preprocessing import ...`
- No need for `sys.path` manipulation in notebooks
- Required for VS Code Pylance to recognize custom modules

### Step 6: Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# On Windows:
copy .env.example .env
```

**What's in .env?**
```bash
# Adds src/ to Python path for VS Code Pylance
PYTHONPATH=${PYTHONPATH}:${workspaceFolder}/src
```

### Step 7: Verify Installation
**Quick verification**:
```bash
python --version
jupyter --version
streamlit --version
make --version
```

**Comprehensive diagnostic**:
```bash
make check-pylance
```

This will verify:
- ✓ Python version and executable
- ✓ Source directory exists
- ✓ Python path includes src/
- ✓ Preprocessing module imports successfully
- ✓ Package is installed
- ✓ All configuration files exist

**Expected output**:
```
✅ SUCCESS: preprocessing module imported successfully!
✅ Package 'chest-xray-detection' is installed
✓ All configuration files exist
```

### Step 8: Run Development Tools

**View available commands**:
```bash
make help
```

**Output**:
```
Usage:
  make install       - install dev tools and project deps
  make lint          - run Ruff lint only
  make format        - run Black format only
  make typecheck     - run Pyright type checker
  make check-pylance - diagnose Pylance/import issues
  make pre-commit    - run lint and format
  make app           - run Streamlit dashboard
  make clean         - remove caches and temp files
```

**Run pre-commit checks** (before committing code):
```bash
make pre-commit
```

**Check for type errors**:
```bash
make typecheck
```

### Step 9: Run Jupyter Notebooks
```bash
jupyter notebook

# Or use VS Code's built-in notebook support (recommended)
# File → Open File → jupyter_notebooks/01_data_collection_and_setup.ipynb
```

### Step 10: Run Streamlit Dashboard
```bash
make app

# OR manually:
streamlit run app/streamlit_app.py
```

### Troubleshooting

**Issue**: VS Code shows "Cannot find module 'preprocessing'" in notebooks

**Solution**:
```bash
# 1. Ensure package is installed in editable mode
pip install -e .

# 2. Reload VS Code window
# Press Cmd+Shift+P → "Developer: Reload Window"

# 3. Run diagnostic
make check-pylance

# 4. See full guide
cat docs/PYLANCE_SETUP.md
```

**Issue**: `make: command not found`

**Solution**: Follow Step 2 to install make for your operating system

**Issue**: Import errors when running notebooks

**Solution**:
```bash
# Ensure you're in the correct directory
pwd  # Should show: .../CapStone

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Reinstall package
pip install -e .
```

**Issue**: Jupyter kernel not found

**Solution**:
```bash
# Install ipykernel
pip install ipykernel

# Add environment to Jupyter
python -m ipykernel install --user --name=capstone --display-name="Python (Capstone)"

# In Jupyter/VS Code: Select kernel → Python (Capstone)
```

For more detailed troubleshooting, see:
- **Import issues**: `docs/PYLANCE_SETUP.md`
- **Data download issues**: Check Notebook 01
- **Environment setup**: `docs/`

---

## Development Process

### Phase 1: Data Collection and Initial Exploration (Week 1)
- Download dataset from Kaggle
- Initial data inspection and quality assessment
- Document dataset characteristics
- Set up project structure and version control

### Phase 2: Data Cleaning and Preprocessing (Week 1-2)
- Handle missing values
- Remove duplicates and irrelevant features
- Outlier detection and treatment
- Create data quality report

### Phase 3: Exploratory Data Analysis (Week 2)
- Univariate, bivariate, and multivariate analysis
- Statistical hypothesis testing
- Initial insights documentation
- Visualization development

### Phase 4: Feature Engineering (Week 2-3)
- Create derived features
- Encoding categorical variables
- Feature scaling and normalization
- Feature selection and dimensionality reduction

### Phase 5: Model Development (Week 3-4)
- Baseline model creation (Logistic Regression)
- Advanced models (Random Forest, XGBoost)
- Regression models for customer value prediction
- Model comparison and selection

### Phase 6: Model Evaluation and Optimization (Week 4)
- K-fold cross-validation
- Hyperparameter tuning
- Model interpretation (SHAP values)
- Final model selection

### Phase 7: Dashboard Development (Week 4-5)
- Streamlit app structure creation
- Page-by-page implementation
- Interactive visualization integration
- UX/UI refinement

### Phase 8: Testing and Documentation (Week 5)
- Unit testing for data processing functions
- Integration testing for dashboard
- README and code documentation completion
- Peer review and feedback incorporation

### Phase 9: Deployment and Finalization (Week 5-6)
- Heroku deployment setup
- Final testing in production environment
- Documentation review
- Project submission preparation

---

## Testing and Validation

### Automated Notebook Testing 🆕

All Jupyter notebooks are automatically tested using `pytest` and `nbmake` to ensure:
- ✅ Notebooks execute without errors
- ✅ Deterministic results (fixed random seeds)
- ✅ No broken imports or dependencies
- ✅ Consistent execution in CI/CD

**Run tests locally:**

```bash
# Run fast tests (default - recommended)
make test

# Run only fast notebooks (2 & 4)
make test-fast

# Run ALL notebooks including slow ones
make test-all

# Prepare notebooks for testing
make test-notebooks
```

**Test categories:**
- **Fast tests**: Notebooks 02 (EDA) and 04 (Hypothesis Testing)
- **Slow tests**: Notebook 03 (Image Preprocessing)
- **Skipped in CI**: Notebook 01 (Data Download - 47GB)

**CI/CD Integration:**
- Notebooks automatically tested on every push/PR
- Tests run in parallel across Python 3.9-3.12
- Slow tests only run on main branch
- Fast tests complete in ~5 minutes

**Configuration:**
- `pytest.ini`: Test configuration and markers
- `.github/workflows/notebook-tests.yml`: CI/CD workflow
- `scripts/prepare_notebooks_for_testing.py`: Notebook preparation

### Data Quality Tests
- Missing value checks
- Duplicate detection
- Data type validation
- Range and constraint validation

### Model Validation
- **Cross-Validation**: 5-fold stratified k-fold for classification
- **Train-Test Split**: 70/15/15 split (train/val/test) with patient-level stratification
- **Baseline Comparison**: Compare against naive baselines
- **Overfitting Check**: Compare train vs validation metrics
- **Expert Label Validation**: Test on Google Cloud expert-validated labels

### Unit Tests (Coming Soon)
Located in `tests/` directory:
```bash
pytest tests/
```

### Dashboard Testing
- Page load functionality
- Interactive component responsiveness
- Data filter functionality
- Visualization rendering
- Error handling

### Code Quality
```bash
# Run all pre-commit checks
make pre-commit

# Format code
make format  # or: black .

# Lint code
make lint    # or: ruff check .

# Type checking
make typecheck  # or: pyright src/

# Check Pylance configuration
make check-pylance
```

### Test Coverage

**What's Tested:**
- ✅ Notebook execution (all cells run top-to-bottom)
- ✅ Import statements (no missing dependencies)
- ✅ Data loading and path resolution
- ✅ Preprocessing pipeline
- ✅ Statistical analysis reproducibility
- ✅ Visualization generation

**What's NOT Tested in CI:**
- ❌ Large data downloads (Notebook 01)
- ❌ Full image preprocessing (marked as slow)
- ❌ Deep learning model training (future notebooks)

### Deterministic Testing

All notebooks include deterministic random seeds:
```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# TensorFlow (if used)
tf.random.set_seed(RANDOM_SEED)
```

This ensures consistent results across test runs.

---

## Deployment

### Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Deployment Steps

1. **Log in to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Set Python Version**
   Ensure `.python-version` contains a [Heroku-22](https://devcenter.heroku.com/articles/python-support#supported-runtimes) supported version.

4. **Configure Buildpacks** (if needed)
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Deploy via Git**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push heroku main
   ```

6. **Alternative: GitHub Integration**
   - Go to Heroku Dashboard
   - Select your app
   - Navigate to **Deploy** tab
   - Choose **GitHub** as deployment method
   - Search and connect your repository
   - Enable automatic deploys (optional)
   - Click **Deploy Branch**

7. **Open Application**
   ```bash
   heroku open
   ```

8. **Monitor Logs**
   ```bash
   heroku logs --tail
   ```

#### Troubleshooting
- If slug size is too large, add large files to `.slugignore`
- Ensure all dependencies are in `requirements.txt`
- Check Procfile for correct Streamlit command

---

## Future Enhancements

### Short-term Improvements
1. **REST API Deployment**: Develop RESTful API for X-ray image upload and real-time disease prediction
2. **Bounding Box Detection**: Implement object detection to localize disease regions (using BBox annotations)
3. **Model Ensemble**: Combine predictions from multiple architectures for improved accuracy
4. **Additional Augmentation**: Experiment with CutMix, MixUp, and other advanced augmentation techniques
5. **Uncertainty Quantification**: Implement Monte Carlo Dropout or Bayesian networks to provide prediction confidence intervals

### Medium-term Enhancements
1. **Multi-View Integration**: Combine frontal and lateral X-ray views for improved diagnosis
2. **Temporal Analysis**: Track disease progression over time for individual patients with longitudinal data
3. **Attention Mechanisms**: Implement attention-based architectures (Vision Transformers) for better interpretability
4. **External Validation**: Test model on independent datasets (e.g., CheXpert, MIMIC-CXR, PadChest)
5. **Federated Learning**: Enable privacy-preserving model training across multiple hospitals
6. **Report Generation**: Automatic radiological report generation from X-ray images (image captioning)

### Long-term Vision
1. **Clinical Deployment**: Integration with hospital PACS (Picture Archiving and Communication Systems)
2. **FDA/CE Approval**: Pursue regulatory approval for clinical decision support tool
3. **Multi-Modal Fusion**: Combine X-rays with CT scans, MRI, and patient electronic health records (EHR)
4. **Real-Time Triage System**: Automated prioritization of urgent cases in emergency departments
5. **Mobile Diagnostic Tool**: Point-of-care diagnostic app for resource-limited settings
6. **Continuous Learning**: MLOps pipeline with automated retraining as new annotated data becomes available
7. **3D Reconstruction**: Generate 3D chest models from 2D X-rays using deep learning
8. **Treatment Recommendation**: Integrate with clinical guidelines to suggest treatment protocols

---

## Credits and Acknowledgments

### Dataset
- **Primary Dataset**: [NIH Chest X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Original Publication**: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thoracic Diseases. IEEE CVPR 2017
- **Citation**:
  ```
  @inproceedings{wang2017chestx,
    title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thoracic diseases},
    author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={2097--2106},
    year={2017}
  }
  ```
- **License**: CC0: Public Domain
- **Acknowledgment**: National Institutes of Health Clinical Center

- **Enhanced Expert Labels**: [Google Cloud Healthcare - NIH Chest X-Ray Additional Labels](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels)
- **Provider**: Google LLC / Google Health AI
- **Storage**: Google Cloud Storage public bucket `gs://gcs-public-data--healthcare-nih-chest-xray-labels`
- **Quality**: Expert radiologist annotations via adjudicated review, higher quality than text-mined labels
- **License**: Same as NIH dataset (CC0: Public Domain)
- **Access Method**: Google Cloud Storage (`gsutil`) or HTTP download

- **Four Findings Expert Labels Citation**:
  ```
  Majkowska A, Mittal S, Steiner DF, et al. Chest Radiograph Interpretation
  with Deep Learning Models: Assessment with Radiologist-adjudicated Reference
  Standards and Population-adjusted Evaluation. Radiology. 2020;294(2):421-431.
  doi:10.1148/radiol.2019191293
  ```

- **All Findings Expert Labels Citation**:
  ```
  Nabulsi Z, Sellergren A, Jamshy S, et al. Deep Learning for Distinguishing
  Normal versus Abnormal Chest Radiographs and Generalization to Two Unseen
  Diseases Tuberculosis and COVID-19. Sci Rep. 2021;11:15523.
  doi:10.1038/s41598-021-93967-2
  ```

- **Acknowledgment**: Google Cloud Healthcare team and radiologist co-authors for curating and providing expert-validated labels

### Learning Resources
- **Code Institute**: Data Analytics & AI Bootcamp curriculum and support
- **TensorFlow/Keras Documentation**: Deep learning implementation guidance
- **Scikit-learn Documentation**: Machine learning and metrics
- **Streamlit Documentation**: Dashboard development resources
- **Stanford CS231n**: Convolutional Neural Networks for Visual Recognition course materials
- **Papers with Code**: Medical imaging benchmarks and sota models

### Code References
All external code snippets and inspirations are documented inline with appropriate attribution:
- **Grad-CAM implementation**: Adapted from keras-vis and tf-keras-vis documentation
- **Data augmentation**: Based on albumentations library examples
- **Multi-label classification**: Scikit-learn multi-label approaches
- **Transfer learning**: Keras Applications pre-trained models
- **Medical imaging preprocessing**: Techniques from radiology AI literature

### Key Research Papers

**Dataset Papers:**
- Wang X, Peng Y, Lu L, et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thoracic Diseases. *IEEE CVPR*, 2017.
- Majkowska A, Mittal S, Steiner DF, et al. Chest Radiograph Interpretation with Deep Learning Models: Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation. *Radiology*, 2020;294(2):421-431.
- Nabulsi Z, Sellergren A, Jamshy S, et al. Deep Learning for Distinguishing Normal versus Abnormal Chest Radiographs and Generalization to Two Unseen Diseases Tuberculosis and COVID-19. *Sci Rep*, 2021;11:15523.

**Deep Learning Methods:**
- He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. *IEEE CVPR*, 2016.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely Connected Convolutional Networks. *IEEE CVPR*, 2017.
- Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*, 2017.

### Tools and Libraries
Grateful acknowledgment to the open-source community for the excellent tools used in this project (see [Technologies Used](#technologies-used) section).

### Feedback and Contributions
- Code Institute instructors and mentors
- Peer reviewers from cohort
- Stack Overflow and Kaggle community for troubleshooting support
- Medical imaging research community for best practices

---

## License

This project is created for educational purposes as part of the Code Institute Data Analytics & AI Bootcamp capstone project.

---

## Contact

For questions or feedback regarding this project, please open an issue in the GitHub repository or contact the project maintainer.

---

**Last Updated**: January 2025
