# Streamlit DenseNet121 Integration Plan

**Branch**: `feature/streamlit-densenet-integration`
**Model**: DenseNet121 (Best performing - AUC: 0.7529)
**Date**: November 6, 2025

## Overview

Integrate the trained DenseNet121 transfer learning model into the Streamlit dashboard to enable:
1. Real-time chest X-ray disease prediction
2. Model performance visualization
3. Grad-CAM explainability
4. Comparison with other models

## Current App Structure

```
app.py (main entry point)
├── src/views/
│   ├── dashboard.py (main dashboard with 6 tabs)
│   └── radiology_guide.py
├── src/tabs/
│   ├── data_exploration.py
│   ├── sample_images.py
│   ├── hypothesis_validation.py
│   ├── model_performance.py      ← **UPDATE THIS**
│   ├── disease_detector.py        ← **UPDATE THIS**
│   └── clinical_insights.py
└── src/utils/
    └── (create model_loader.py)  ← **CREATE THIS**
```

## Implementation Tasks

### 1. Model Setup ✅ READY

**Source**: `colab/results/models/densenet121-transfer/runs/2025-11-05_030929/`
- `densenet121_transfer_best.keras` (37.2 MB)
- `config.json` (training parameters)

**Target**: `models/saved_models/densenet121_best.keras`

**Metadata** (from MLflow):
- Parameters: 7,569,486
- Test AUC: 0.7529
- Test Loss: 0.1743
- Test Accuracy: 0.1799
- Input: (224, 224, 3)
- Output: 14 classes (multi-label)

### 2. Model Loader Utility

**File**: `src/utils/model_loader.py`

**Functions**:
```python
def load_densenet_model(model_path: str) -> keras.Model
def preprocess_image(image: PIL.Image, target_size=(224, 224)) -> np.ndarray
def predict_diseases(model, image: np.ndarray) -> dict
def get_disease_classes() -> list
```

**Caching**: Use `@st.cache_resource` for model loading

### 3. Update Model Performance Tab

**File**: `src/tabs/model_performance.py`

**Add**:
- Model architecture comparison (DenseNet121, ResNet50, EfficientNetB3)
- Performance metrics table (AUC, Loss, Accuracy)
- Parameter count comparison
- Model size comparison
- Training history plots (from colab/results/artifacts/plots/)
- ROC curves (if available)

**Data Source**:
- MLflow database
- `colab/results/artifacts/metrics/2025-11-05_063105_model_results.csv`
- Training history plots

### 4. Update Disease Detector Tab

**File**: `src/tabs/disease_detector.py`

**Features**:
1. **Image Upload**:
   - File uploader (PNG, JPG, JPEG)
   - Image preprocessing (resize to 224x224, normalize)
   - Display uploaded image

2. **Prediction**:
   - Load DenseNet121 model
   - Run inference
   - Display predictions with confidence scores
   - Show top 3 most likely diseases

3. **Visualization**:
   - Bar chart of all 14 disease probabilities
   - Color coding (red: high probability, yellow: medium, green: low)
   - Threshold slider (default: 0.5)

4. **Grad-CAM (Future)**:
   - Heatmap overlay showing regions of interest
   - Explainability for predictions

### 5. Grad-CAM Implementation

**File**: `src/utils/gradcam.py`

**Functions**:
```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None)
def overlay_heatmap_on_image(img, heatmap, alpha=0.4)
def generate_gradcam_visualization(model, image, class_index)
```

**Integration**:
- Add checkbox in Disease Detector: "Show Explainability (Grad-CAM)"
- Display heatmap overlay when enabled
- Add interpretation guide

### 6. Dependencies

**Add to `requirements.txt`**:
```txt
# Already present:
tensorflow>=2.18.0
keras>=3.0.0
streamlit>=1.38.0
pandas>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# May need to add:
opencv-python>=4.8.0  # For Grad-CAM image processing
scikit-image>=0.21.0  # For image transformations
```

### 7. Model Deployment Considerations

**For Streamlit Cloud**:
- Model size: 37.2 MB (acceptable)
- Memory usage: ~500 MB loaded (within limits)
- CPU inference: ~2-3 seconds per image (acceptable)

**Optimization Options** (if needed):
- Model quantization (TFLite)
- Lazy loading (load on first prediction)
- Request caching for duplicate images

### 8. Testing Plan

**Local Testing**:
1. Load model successfully
2. Preprocess test images correctly
3. Generate predictions (verify shape: 14 probabilities)
4. Display results in UI
5. Test with various image formats/sizes
6. Verify Grad-CAM heatmaps

**Test Images** (from dataset):
- Normal (No Finding)
- Single disease (e.g., Pneumonia)
- Multiple diseases (e.g., Effusion + Infiltration)
- Edge cases (poor quality, rotated)

### 9. Documentation Updates

**Update**:
- `README.md` - Add Model Performance section
- `docs/DEPLOYMENT.md` - Add model deployment notes
- `docs/USER_GUIDE.md` - Add Disease Detector usage

**Create**:
- `docs/MODEL_CARD.md` - DenseNet121 model card
- `docs/EXPLAINABILITY.md` - Grad-CAM interpretation guide

## File Structure After Integration

```
models/
└── saved_models/
    └── densenet121_best.keras         ← NEW

src/utils/
├── model_loader.py                    ← NEW
└── gradcam.py                          ← NEW

src/tabs/
├── model_performance.py                ← UPDATED
└── disease_detector.py                 ← UPDATED

colab/results/                          ← REFERENCE ONLY
├── models/densenet121-transfer/...
└── artifacts/
    ├── metrics/model_results.csv
    └── plots/*.png

docs/
├── MODEL_CARD.md                       ← NEW
└── EXPLAINABILITY.md                   ← NEW
```

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model too large for Streamlit Cloud | High | Already acceptable (37 MB) |
| Slow inference on CPU | Medium | Cache predictions, lazy load model |
| Memory issues | Medium | Use `@st.cache_resource`, clear session |
| Grad-CAM slow | Low | Make it optional, cache results |
| TensorFlow version conflicts | Low | Pin versions in requirements.txt |

## Success Criteria

- ✅ Model loads without errors
- ✅ Predictions display correctly with confidence scores
- ✅ Upload and process user images
- ✅ Model Performance tab shows DenseNet results
- ✅ Disease Detector provides real-time predictions
- ✅ Grad-CAM visualizations work (optional)
- ✅ App deploys successfully to Streamlit Cloud
- ✅ Response time < 5 seconds for prediction

## Timeline

- **Phase 1**: Model setup & loader (30 min)
- **Phase 2**: Update Model Performance tab (30 min)
- **Phase 3**: Implement Disease Detector (1 hour)
- **Phase 4**: Grad-CAM (1 hour, optional)
- **Phase 5**: Testing & refinement (30 min)
- **Phase 6**: Documentation (30 min)

**Total**: ~3.5-4.5 hours

## Next Steps

1. ✅ Create feature branch
2. ✅ Create this plan document
3. ⏭️ Copy model to `models/saved_models/`
4. ⏭️ Implement model loader utility
5. ⏭️ Update Model Performance tab
6. ⏭️ Update Disease Detector tab
7. ⏭️ Test locally
8. ⏭️ Deploy to Streamlit Cloud
9. ⏭️ Merge to main

## References

- **Model Training**: `colab/07_transfer_learning_gcs.ipynb`
- **MLflow Tracking**: http://localhost:5001
- **Model Registry**: `NIH-XRay-DENSENET121 v1`
- **Performance**: `colab/MLFLOW_ENRICHMENT_SUMMARY.md`
