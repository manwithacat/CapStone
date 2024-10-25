# Jupyter Notebooks - NIH Chest X-Ray Disease Detection

This directory contains the complete ETL and ML pipeline for the project, organized in sequential numbered notebooks.

## Notebook Sequence

### 📥 Data Collection
**01_data_collection_and_setup.ipynb**
- Kaggle API setup and authentication
- Dataset download (NIH Chest X-Rays, 112K images)
- Initial data inspection
- Directory structure creation
- Metadata loading and basic statistics

### 📊 Exploratory Data Analysis
**02_exploratory_data_analysis.ipynb**
- Patient demographics analysis (age, gender)
- Disease distribution visualization
- Class imbalance quantification
- Multi-label statistics
- Sample image visualization
- Data quality assessment

### 🖼️ Image Processing
**03_image_preprocessing.ipynb**
- Image loading and inspection
- Resizing strategies (224x224 vs 512x512)
- Normalization techniques
- Data augmentation pipeline (rotation, flip, brightness, zoom)
- Train/validation/test split (70/15/15)
- Sample data creation for quick testing

### 📈 Statistical Analysis
**04_hypothesis_testing.ipynb**
- **Hypothesis 1**: Age-disease correlation (ANOVA, t-tests)
- **Hypothesis 2**: Disease co-occurrence patterns (chi-square, association rules)
- **Hypothesis 3**: Class imbalance impact analysis
- **Hypothesis 4**: Transfer learning performance comparison
- **Hypothesis 5**: Gender differences in disease prevalence
- Visualizations: box plots, heatmaps, statistical test results

### 🤖 Baseline Models
**05_baseline_models.ipynb**
- Feature extraction (HOG, texture descriptors, GLCM)
- Logistic Regression (multi-label)
- Random Forest Classifier
- XGBoost with class weights
- Performance evaluation (AUC-ROC, F1-score, precision, recall)
- Baseline benchmarks for deep learning comparison

### 🧠 Custom CNN Development
**06_cnn_development.ipynb**
- Custom CNN architecture design
- Convolutional layers with batch normalization
- Pooling and dropout regularization
- Multi-label classification head (sigmoid activation)
- Training with early stopping
- Learning curves and overfitting analysis

### 🚀 Transfer Learning
**07_transfer_learning.ipynb**
- **ResNet50**: Pre-trained on ImageNet, fine-tuned for chest X-rays
- **DenseNet121**: Popular for medical imaging
- **EfficientNetB3**: Efficient scaling architecture
- Fine-tuning strategies (freeze early layers, train final layers)
- Comparison of transfer learning vs training from scratch
- Training time and efficiency analysis

### 📏 Model Evaluation
**08_model_evaluation.ipynb**
- Performance comparison across all models
- Per-disease metrics (AUC-ROC, sensitivity, specificity, F1)
- ROC curves (15 disease classes)
- Precision-recall curves
- Confusion matrices
- Multi-label evaluation metrics (Hamming loss, subset accuracy)
- Model selection and best model identification

### 🔍 Model Interpretation
**09_model_interpretation.ipynb**
- **Grad-CAM**: Visual attention heatmaps
- Saliency maps for pixel-level importance
- Error analysis (false positives, false negatives)
- Disease-specific interpretation
- Clinical validation considerations
- Model explainability for stakeholders

---

## Execution Order

Run notebooks sequentially (01 → 09) as each depends on outputs from previous notebooks.

**Required:**
- Notebooks 01-03: Data preparation
- Notebook 04: Statistical analysis (for assessment LO1)
- At least one model notebook (05, 06, or 07)
- Notebook 08: Evaluation
- Notebook 09: Interpretation (for Grad-CAM visualizations in dashboard)

**Recommended:**
- Run all notebooks for comprehensive analysis
- Notebooks 05-07 can be run independently after 01-03

---

## Output Artifacts

### Data
- `data/raw/`: Original dataset (not committed to git)
- `data/processed/`: Train/val/test splits, preprocessed images
- `data/sample/`: Small subset for quick testing

### Models
- `models/saved_models/`: Trained model weights (.h5, .pkl files)
- `models/training_history/`: Training curves data (.json)

### Analysis
- `outputs/figures/`: Generated plots and visualizations
- `outputs/reports/`: Statistical analysis results (.json, .csv)
- `outputs/predictions/`: Model predictions on test set

---

## Best Practices

### Before Running
1. Ensure virtual environment is activated
2. Install all requirements: `pip install -r requirements.txt`
3. Set up Kaggle API credentials in `~/.kaggle/kaggle.json`
4. Check available disk space (~50 GB for raw data)

### During Execution
- **Notebook 01** takes 30-60 min (dataset download)
- **Notebooks 06-07** (deep learning) require GPU for reasonable training times
- Use `data/sample/` for quick iterations before full dataset training
- Save checkpoints frequently during long training runs

### After Execution
- Strip notebook outputs before committing: `nbstripout jupyter_notebooks/*.ipynb`
- Large model files should be in `.gitignore`
- Commit analysis reports and figures

---

## Troubleshooting

### Kaggle API Issues
```bash
# Verify credentials
cat ~/.kaggle/kaggle.json

# Fix permissions
chmod 600 ~/.kaggle/kaggle.json

# Accept dataset terms at:
https://www.kaggle.com/datasets/nih-chest-xrays/data
```

### Memory Issues
- Use `data/sample/` subset for development
- Reduce batch size in deep learning notebooks
- Use gradient accumulation for large models
- Clear Keras/TensorFlow session: `K.clear_session()`

### GPU/Training Issues
- Check GPU availability: `tf.config.list_physical_devices('GPU')`
- Reduce image resolution (224x224 instead of 512x512)
- Use mixed precision training: `tf.keras.mixed_precision`

---

## Assessment Alignment

| Notebook | Learning Outcomes |
|----------|-------------------|
| 01 | LO5 (data management), LO11 (tool adaptation) |
| 02 | LO1 (statistics), LO3 (real-world analysis) |
| 03 | LO2 (data manipulation), LO5 (preprocessing) |
| 04 | LO1 (hypothesis testing), LO3 (methodology) |
| 05-07 | LO2 (ML techniques), LO4 (AI integration), LO11 (adaptation) |
| 08 | LO2 (evaluation), LO3 (problem-solving) |
| 09 | LO4 (AI interpretation), LO8 (communication) |

All notebooks contribute to:
- **LO6**: Ethical considerations (medical data, privacy, bias)
- **LO7**: Research project design
- **LO9**: Healthcare domain application
- **LO10**: Implementation and maintenance planning

---

**Last Updated:** January 2025
