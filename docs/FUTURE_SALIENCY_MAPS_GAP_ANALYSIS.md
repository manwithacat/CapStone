# Saliency Maps Implementation Gap Analysis

**Current Status**: ❌ NOT IMPLEMENTED  
**Claimed in README**: ✅ Listed as planned feature  
**Actual Implementation**: Grad-CAM only

---

## What's Already Implemented ✅

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- **File**: `src/utils/gradcam.py` (339 lines)
- **Functions**:
  - `make_gradcam_heatmap()` - Generate heatmap from conv layer
  - `overlay_heatmap_on_image()` - Overlay heatmap on X-ray
  - `generate_gradcam_for_disease()` - Disease-specific visualization
  - `get_top_gradcam_predictions()` - Top-K predictions with heatmaps
- **Integration**: Dashboard Disease Detector tab (real-time)
- **Quality**: Production-ready, handles nested models

### Infrastructure Available
- ✅ TensorFlow/Keras model loaded
- ✅ Image preprocessing pipeline
- ✅ Gradient computation with `GradientTape`
- ✅ Visualization utilities (cv2, PIL)
- ✅ Dashboard integration pattern established

---

## What's Missing: Saliency Maps ❌

### 1. Vanilla Saliency Maps
**What**: Compute gradient of prediction with respect to INPUT pixels

**Delta**:
```python
def compute_saliency_map(model, img_array, pred_index=None):
    """
    Generate vanilla saliency map.
    
    Shows which pixels, if changed, would most affect the prediction.
    """
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor, training=False)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_output = predictions[:, pred_index]
    
    # Gradient of prediction w.r.t. input image
    grads = tape.gradient(class_output, img_tensor)
    
    # Take absolute value and normalize
    saliency = tf.abs(grads)
    saliency = tf.reduce_max(saliency, axis=-1)  # Across RGB channels
    saliency = saliency[0].numpy()
    
    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency
```

**Effort**: ~50 lines (function + tests)  
**Time**: 1-2 hours

---

### 2. Smoothed Saliency Maps (SmoothGrad)
**What**: Average saliency maps across noisy samples (reduces noise)

**Delta**:
```python
def compute_smoothed_saliency(model, img_array, pred_index=None, 
                              num_samples=50, noise_stddev=0.15):
    """
    SmoothGrad: Average saliency over noisy samples.
    
    More robust than vanilla saliency.
    """
    saliencies = []
    
    for _ in range(num_samples):
        # Add Gaussian noise
        noise = tf.random.normal(img_array.shape, stddev=noise_stddev)
        noisy_img = img_array + noise
        
        # Compute saliency for noisy image
        saliency = compute_saliency_map(model, noisy_img, pred_index)
        saliencies.append(saliency)
    
    # Average across samples
    smoothed_saliency = np.mean(saliencies, axis=0)
    
    return smoothed_saliency
```

**Effort**: ~30 lines (builds on vanilla saliency)  
**Time**: 1 hour

---

### 3. Integrated Gradients
**What**: Attribute prediction to input features by integrating gradients along path

**Delta**:
```python
def compute_integrated_gradients(model, img_array, pred_index=None, 
                                 baseline=None, steps=50):
    """
    Integrated Gradients: More faithful attribution than vanilla saliency.
    
    Integrates gradients along path from baseline to input.
    """
    if baseline is None:
        # Use black image as baseline
        baseline = np.zeros_like(img_array)
    
    # Generate interpolated images along path
    alphas = np.linspace(0, 1, steps)
    interpolated = baseline + alphas[:, None, None, None, None] * (img_array - baseline)
    
    gradients = []
    for i, interp_img in enumerate(interpolated):
        interp_tensor = tf.convert_to_tensor(interp_img[None, ...])
        
        with tf.GradientTape() as tape:
            tape.watch(interp_tensor)
            predictions = model(interp_tensor, training=False)
            
            if pred_index is None and i == len(interpolated) - 1:
                pred_index = tf.argmax(predictions[0])
            
            class_output = predictions[:, pred_index]
        
        grads = tape.gradient(class_output, interp_tensor)
        gradients.append(grads[0].numpy())
    
    # Integrate gradients using trapezoidal rule
    avg_gradients = np.mean(gradients, axis=0)
    integrated_grads = (img_array[0] - baseline[0]) * avg_gradients
    
    # Reduce to single channel and normalize
    integrated_grads = np.abs(integrated_grads).max(axis=-1)
    integrated_grads = (integrated_grads - integrated_grads.min()) / \
                      (integrated_grads.max() - integrated_grads.min() + 1e-8)
    
    return integrated_grads
```

**Effort**: ~80 lines  
**Time**: 2-3 hours

---

### 4. Visualization Utilities
**What**: Overlay saliency maps on X-rays (similar to Grad-CAM overlay)

**Delta**:
```python
def overlay_saliency_on_image(img, saliency, alpha=0.5, colormap=cv2.COLORMAP_HOT):
    """
    Overlay saliency map on original image.
    
    Similar to Grad-CAM overlay but for pixel-level saliency.
    """
    # Convert saliency to uint8
    saliency_uint8 = np.uint8(255 * saliency)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(saliency_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original image to uint8 if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = np.uint8(255 * img)
    else:
        img_uint8 = img
    
    # Overlay
    overlayed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlayed)
```

**Effort**: ~30 lines (can reuse Grad-CAM overlay logic)  
**Time**: 30 minutes

---

### 5. Dashboard Integration
**What**: Add saliency maps to Disease Detector tab

**Delta in** `src/tabs/disease_detector.py`:
```python
# Add toggle for visualization method
viz_method = st.radio(
    "Visualization Method",
    ["Grad-CAM (Region-based)", "Saliency Map (Pixel-based)", "Both"],
    horizontal=True
)

if viz_method in ["Saliency Map (Pixel-based)", "Both"]:
    # Import saliency utilities
    from src.utils.saliency import compute_saliency_map, overlay_saliency_on_image
    
    # Generate saliency maps for top predictions
    saliency_results = []
    for disease_idx, disease_name, prob in top_diseases:
        saliency = compute_saliency_map(model, preprocessed_img, disease_idx)
        saliency_overlay = overlay_saliency_on_image(original_img, saliency)
        saliency_results.append((disease_name, saliency_overlay, prob))
    
    # Display saliency maps
    st.subheader("🔍 Saliency Maps (Pixel Attribution)")
    for disease_name, saliency_img, prob in saliency_results:
        st.image(saliency_img, caption=f"{disease_name} ({prob:.1%})")
```

**Effort**: ~50 lines  
**Time**: 1 hour

---

### 6. Notebook Integration
**What**: Add saliency analysis to Notebook 09 (Model Interpretation)

**Delta**: Create new notebook or section
```python
# Generate saliency maps for test set sample
sample_images = test_df.sample(20)

for idx, row in sample_images.iterrows():
    img = load_and_preprocess_image(row['full_path'])
    
    # Get predictions
    pred = model.predict(img)
    top_disease_idx = np.argmax(pred[0])
    
    # Generate visualizations
    gradcam = make_gradcam_heatmap(img, model, 'conv5_block16_2_conv', top_disease_idx)
    saliency = compute_saliency_map(model, img, top_disease_idx)
    
    # Plot side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img[0], cmap='gray')
    axes[0].set_title('Original X-ray')
    
    axes[1].imshow(gradcam, cmap='jet')
    axes[1].set_title('Grad-CAM (Region)')
    
    axes[2].imshow(saliency, cmap='hot')
    axes[2].set_title('Saliency (Pixel)')
    
    plt.tight_layout()
    plt.savefig(f'outputs/figures/interpretation/sample_{idx}_comparison.png')
```

**Effort**: Full notebook section ~200 lines  
**Time**: 3-4 hours

---

## Total Implementation Delta

### Files to Create/Modify
1. **NEW**: `src/utils/saliency.py` (~200 lines)
   - `compute_saliency_map()`
   - `compute_smoothed_saliency()`
   - `compute_integrated_gradients()`
   - `overlay_saliency_on_image()`

2. **MODIFY**: `src/tabs/disease_detector.py` (+50 lines)
   - Add visualization method selector
   - Integrate saliency map generation
   - Display saliency results

3. **NEW/MODIFY**: `jupyter_notebooks/09_model_interpretation.ipynb`
   - Side-by-side Grad-CAM vs Saliency comparison
   - Analysis of differences
   - Statistical comparison of attribution methods

4. **MODIFY**: `README.md` (Update from "planned" to "implemented")

### Effort Estimate
| Component | Lines of Code | Time Estimate |
|-----------|---------------|---------------|
| Vanilla Saliency | 50 | 1-2 hours |
| SmoothGrad | 30 | 1 hour |
| Integrated Gradients | 80 | 2-3 hours |
| Visualization Utils | 30 | 0.5 hours |
| Dashboard Integration | 50 | 1 hour |
| Notebook Analysis | 200 | 3-4 hours |
| Testing & Debugging | - | 2 hours |
| **TOTAL** | **~440 lines** | **11-14 hours** |

---

## Comparison: Grad-CAM vs Saliency Maps

### Grad-CAM (Already Implemented ✅)
- **Granularity**: Region-level (coarse, 7x7 feature map upsampled to 224x224)
- **Computation**: Uses last convolutional layer activations
- **Speed**: Fast (single forward + backward pass)
- **Interpretation**: Shows which REGIONS influenced decision
- **Best For**: High-level understanding ("model looked at upper-right lung")

### Saliency Maps (Not Implemented ❌)
- **Granularity**: Pixel-level (fine, 224x224)
- **Computation**: Uses input image gradients
- **Speed**: Fast for vanilla, slower for SmoothGrad (50x forward passes)
- **Interpretation**: Shows which PIXELS are most important
- **Best For**: Fine-grained analysis ("model sensitive to these specific edges")

### When to Use Each
| Scenario | Grad-CAM | Saliency | Both |
|----------|----------|----------|------|
| Quick interpretation | ✅ | | |
| Clinical presentation | ✅ | | |
| Research analysis | | | ✅ |
| Debugging model focus | ✅ | ✅ | ✅ |
| Fine-grained attribution | | ✅ | |
| Publication figures | | | ✅ |

---

## Recommendation

### For Submission (Now)
**Action**: ✅ **Keep as-is**

**Rationale**:
1. Grad-CAM is SUFFICIENT for demonstrating model interpretation (LO4, LO8)
2. Dashboard has working visualization (meets requirements)
3. Saliency would be NICE-TO-HAVE but not required
4. 11-14 hours is significant time investment close to submission

**README Status**: ✅ Currently lists saliency as planned feature, not implemented
- This is HONEST - shows awareness of technique
- Shows future improvement planning (LO10, LO11)
- No false claims

### For Future Enhancement
**Priority**: MEDIUM 📊

**Rationale**:
1. Would complement existing Grad-CAM well
2. Provides finer-grained analysis for research
3. Integrated Gradients more robust than vanilla Grad-CAM
4. Good for publication-quality figures

**Add to Future Improvements**:
> **4. Enhanced Explainability Methods**
>    - Implement pixel-level saliency maps (vanilla, SmoothGrad)
>    - Add Integrated Gradients for faithful attribution
>    - Create side-by-side comparison (Grad-CAM vs Saliency)
>    - Analyze differences: where do methods agree/disagree?

---

## Summary

**Gap**: Saliency maps mentioned in README but NOT implemented

**Reality Check**:
- ✅ Grad-CAM: FULLY IMPLEMENTED (production-ready)
- ❌ Saliency Maps: NOT IMPLEMENTED (listed as planned)
- ✅ README: Honest (doesn't claim saliency is done)

**Delta to Implement**: ~440 lines, 11-14 hours

**Recommendation**: 
- **NOW**: Keep as-is, Grad-CAM sufficient for submission ✅
- **FUTURE**: Add saliency in post-submission improvements 📊

**Learning Objectives**: Already met without saliency
- LO4 (AI integration): Grad-CAM provides visual explanation ✅
- LO8 (Communication): Dashboard visualizations work ✅
- LO11 (Adaptability): Awareness of alternative methods ✅

