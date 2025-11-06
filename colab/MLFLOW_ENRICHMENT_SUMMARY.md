# MLflow Run Enrichment Summary

**Date**: November 6, 2025
**Experiment**: NIH-XRay-Transfer-Learning
**Models**: 3 (ResNet50, DenseNet121, EfficientNetB3)

## Enriched Metadata

### 1. Dataset Information
- **Dataset**: NIH Chest X-Ray Dataset (2017)
- **Source**: NIH Clinical Center
- **Total Images**: 112,120
- **Total Patients**: 30,805
- **Classes**: 14 diseases (multi-label)
- **Train/Val/Test Split**: 78,566 / 17,063 / 16,491 images
- **Split Strategy**: Patient-level (no patient leakage)
- **Data Collection**: 1992-2015
- **Citation**: Wang et al., IEEE CVPR 2017

**Disease Classes**:
- Atelectasis, Cardiomegaly, Effusion, Infiltration
- Mass, Nodule, Pneumonia, Pneumothorax
- Consolidation, Edema, Emphysema, Fibrosis
- Pleural Thickening, Hernia

### 2. Model Architecture Details

| Model | Total Params | Trainable | Non-Trainable | Size (MB) | Layers |
|-------|-------------|-----------|---------------|-----------|--------|
| **DenseNet121** | 7,569,486 | ~7.5M | ~0 | 37.2 | Multiple |
| **ResNet50** | 24,643,982 | ~24.6M | ~0 | 170.9 | 50+ |
| **EfficientNetB3** | 11,577,661 | ~11.6M | ~0 | 77.3 | Multiple |

**Architecture Components**:
- Base: Pre-trained on ImageNet
- Top layers: Dense(512) + Dropout(0.5) + Dense(14)
- Input shape: (224, 224, 3)
- Output shape: (14,) with sigmoid activation

### 3. Training Strategy
- **Strategy**: Two-stage transfer learning
- **Stage 1**: Feature extraction (base frozen, 5 epochs)
  - Learning rate: 0.001
  - Optimizer: Adam
- **Stage 2**: Fine-tuning (top 20 layers unfrozen, 10 epochs)
  - Learning rate: 0.0001
  - Optimizer: Adam
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, AUC
- **Data Augmentation**: Horizontal flip, rotation, zoom, shift

### 4. Training Environment
- **Platform**: Google Colab Pro
- **Hardware**: NVIDIA A100 GPU (80GB)
- **Location**: Google Cloud Platform
- **Framework**: TensorFlow/Keras v2.18.0
- **Python**: 3.11.13
- **Batch Size**: 1024 (optimized for A100)
- **Training Mode**: Parallel (3 models simultaneously)

### 5. Model Performance (Test Set)

| Model | Test AUC | Test Loss | Test Accuracy |
|-------|----------|-----------|---------------|
| **DenseNet121** 🏆 | **0.7529** | 0.1743 | 0.1799 |
| ResNet50 | 0.6810 | 0.1989 | 0.2364 |
| EfficientNetB3 | 0.5350 | 0.1990 | 0.1012 |

**Winner**: DenseNet121 with AUC = 0.7529

### 6. MLflow Model Registry

All models registered with versioning:

```
NIH-XRay-DENSENET121 (version 1)
├── Stage: Staging
├── Platform: Colab
└── Status: Ready for production evaluation

NIH-XRay-RESNET50 (version 1)
├── Stage: Staging
├── Platform: Colab
└── Status: Ready for production evaluation

NIH-XRay-EFFICIENTNETB3 (version 1)
├── Stage: Staging
├── Platform: Colab
└── Status: Ready for production evaluation
```

### 7. Additional Tags

**Project Metadata**:
- Project: NIH-Chest-XRay-Disease-Detection
- Task: Multi-label classification
- Domain: Medical imaging
- Use case: Chest X-ray disease detection
- Purpose: Research

**Data Provenance**:
- Data source: NIH Clinical Center
- Collection period: 1992-2015
- Publication year: 2017
- Label source: NLP extraction from radiology reports
- Label type: Multi-label (multiple diseases per image)

### 8. Artifacts Logged

Each run includes:
- ✅ Trained model (.keras file)
- ✅ Training configuration (config.json)
- ✅ Training history plot (PNG)
- ✅ Model evaluation metrics (CSV)
- ✅ Model architecture summary
- ✅ Complete parameter set

## Access MLflow

**MLflow UI**: http://localhost:5001

**Features Available**:
- Compare runs side-by-side
- View training curves
- Inspect model architectures
- Track parameter changes
- Download models for deployment
- Model versioning and registry

## Next Steps

### 1. Model Evaluation
- [ ] Generate per-class ROC curves
- [ ] Create confusion matrices
- [ ] Calculate per-disease metrics (sensitivity, specificity)
- [ ] Test on expert-labeled validation set

### 2. Model Deployment
- [ ] Transition best model to "Production" stage
- [ ] Create deployment package
- [ ] Set up model serving endpoint
- [ ] Implement monitoring and logging

### 3. Documentation
- [ ] Create model card
- [ ] Document limitations and biases
- [ ] Write deployment guide
- [ ] Prepare assessment documentation

## References

**Dataset**:
- Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR 2017

**Expert Labels**:
- Majkowska et al., "Chest Radiograph Interpretation with Deep Learning Models: Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation", Radiology 2020
- Nabulsi et al., "Deep learning-based diagnostic support system for detecting multiple pathologies in chest X-rays", Scientific Reports 2021

**Platform**:
- MLflow: https://mlflow.org/
- Google Colab: https://colab.research.google.com/
