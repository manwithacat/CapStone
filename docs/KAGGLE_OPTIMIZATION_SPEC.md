# Kaggle Training Optimization Requirements

**Date:** 2025-10-29
**Status:** Pending Implementation
**Source:** Colleague recommendations for maximizing Kaggle P100/T4 GPU performance

---

## Executive Summary

Optimization strategies to improve Kaggle cloud training performance, reduce training time, and better utilize GPU resources. Estimated potential speedup: **3-5x** with all optimizations applied.

---

## 1. High-Priority Quick Wins (5-10 min implementation)

### 1.1 Mixed Precision Training (AMP)
**Benefit:** 1.5-2x faster training, reduced memory usage
**Effort:** 1 line of code
**Compatibility:** Works on P100 (no Tensor Cores but still beneficial)

**Implementation:**
```python
# Add before model compilation
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Loss must use float32 for numerical stability
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1, dtype='float32')
```

**Notes:**
- Monitor training stability
- Loss computation stays in float32
- Activations/weights use float16

---

### 1.2 Enable Class Weights
**Benefit:** Better handling of class imbalance, faster convergence
**Effort:** 1 parameter change
**Status:** We already compute class weights but don't use them!

**Implementation:**
```python
# In model.fit()
model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights_dict,  # ← ADD THIS
    ...
)
```

**Notes:**
- `class_weights_dict` already loaded from `preprocessing_config.json`
- No Finding: 53.84%, Infiltration: 17.65%, etc.

---

### 1.3 Increase Batch Size
**Benefit:** Better GPU utilization, faster throughput
**Effort:** Change 1 config value
**Current:** `batch_size=32`
**Recommended:** `batch_size=64` or `128`

**Implementation:**
```python
CONFIG = {
    'batch_size': 64,  # Test 64 first, then try 128
    ...
}
```

**Testing Required:**
- Monitor GPU memory usage (P100 has 16GB)
- If OOM → reduce to 64 or use gradient accumulation

---

### 1.4 Label Smoothing
**Benefit:** Stabilizes half-precision training, reduces overfitting
**Effort:** 1 parameter change
**Recommended:** 0.05-0.1

**Implementation:**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    ...
)
```

**Notes:**
- Prevents model from being overconfident (predicting exact 0/1)
- Particularly useful for multi-label classification

---

## 2. Medium-Priority Optimizations (1-2 hours implementation)

### 2.1 Replace ImageDataGenerator with tf.data Pipeline
**Benefit:** 2-3x faster data loading, better CPU utilization
**Effort:** Moderate refactor
**Current:** `ImageDataGenerator` (deprecated, slow)
**Recommended:** `tf.data.Dataset` API

**Implementation:**
```python
def create_dataset(df, augment=False):
    def load_and_preprocess(path, labels):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float16) / 255.0

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            # Add more augmentations

        return img, labels

    dataset = tf.data.Dataset.from_tensor_slices((
        df['full_path'].values,
        df[disease_classes].values
    ))

    dataset = dataset.map(load_and_preprocess,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()  # Cache decoded images in memory
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(CONFIG['batch_size'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

**Benefits:**
- `cache()`: Decode images once, reuse across epochs
- `prefetch()`: Load next batch while GPU trains current
- `num_parallel_calls=AUTOTUNE`: Multi-threaded loading
- Better integration with mixed precision

---

### 2.2 Better Learning Rate Schedule
**Benefit:** Faster convergence, better final performance
**Effort:** Replace ReduceLROnPlateau callback
**Current:** ReduceLROnPlateau (reactive, slow)
**Recommended:** OneCycle or Cosine Decay with Warmup

**Implementation (OneCycle):**
```python
from tensorflow.keras.optimizers.schedules import CosineDecay

steps_per_epoch = len(train_df) // CONFIG['batch_size']
total_steps = steps_per_epoch * CONFIG['epochs']

lr_schedule = CosineDecay(
    initial_learning_rate=CONFIG['learning_rate'],
    decay_steps=total_steps,
    alpha=0.0  # Final LR = 0
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

**Benefits:**
- Proactive schedule (vs reactive ReduceLR)
- Warmup prevents early divergence
- Cosine decay prevents overfitting at end

---

### 2.3 Exponential Moving Average (EMA) of Weights
**Benefit:** Better generalization, free quality lift
**Effort:** Add custom callback
**Recommended:** EMA decay = 0.999

**Implementation:**
```python
class EMACallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None

    def on_train_begin(self, logs=None):
        self.ema_weights = [tf.Variable(w) for w in self.model.get_weights()]

    def on_train_batch_end(self, batch, logs=None):
        for ema_w, w in zip(self.ema_weights, self.model.get_weights()):
            ema_w.assign(self.decay * ema_w + (1 - self.decay) * w)

    def on_epoch_end(self, epoch, logs=None):
        # Swap to EMA weights for validation
        original = self.model.get_weights()
        self.model.set_weights([w.numpy() for w in self.ema_weights])
        # Validation happens here
        self.model.set_weights(original)
```

---

## 3. High-Impact Infrastructure Change (2-4 hours)

### 3.1 Pre-resize Image Dataset
**Benefit:** 10-16x less I/O, massive data loading speedup
**Effort:** One-time preprocessing + dataset upload
**Current:** Load 1024x1024 → resize to 224x224 every epoch
**Recommended:** Pre-resize once, upload 224x224 dataset

**Size Comparison:**
- Original: ~47 GB (1024x1024 PNG)
- Resized: ~2-3 GB (224x224 PNG or JPG q=95)
- **16x smaller files = 16x faster I/O**

**Implementation Steps:**

1. **Local Preprocessing Script** (`scripts/create_resized_dataset.py`):
```python
from PIL import Image
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path('data/raw/')
OUTPUT_DIR = Path('data/resized_224/')
TARGET_SIZE = (224, 224)

for img_path in tqdm(list(INPUT_DIR.rglob('*.png'))):
    # Maintain directory structure
    rel_path = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resize and save
    img = Image.open(img_path).convert('RGB')
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img.save(out_path, 'PNG', optimize=True)
```

2. **Package for Kaggle:**
```bash
# Create dataset metadata
cat > kaggle_dataset/dataset-metadata.json << EOF
{
  "title": "NIH Chest X-Ray 224px (Preprocessed)",
  "subtitle": "Pre-resized to 224x224 for faster training",
  "id": "yourusername/nih-chest-xray-224px",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Copy resized images
cp -r data/resized_224/* kaggle_dataset/

# Upload
kaggle datasets create -p kaggle_dataset
```

3. **Update Notebook:**
```python
# Change DATA_DIR
DATA_DIR = Path('/kaggle/input/nih-chest-xray-224px')

# Remove target_size from generators (images already 224x224)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='full_path',
    y_col=disease_classes,
    # target_size REMOVED - images already correct size
    batch_size=CONFIG['batch_size'],
    class_mode='raw',
)
```

**Benefits:**
- No resize overhead during training
- 16x less disk I/O
- Can use higher quality resizing (LANCZOS) offline
- Enables caching full dataset in memory

---

## 4. Kaggle Resource Management

### 4.1 GPU Quota Management
**Limits:** ~30 GPU-hours/week, 9-hour max session
**Strategy:**
- Run full training (50 epochs) = ~1-2 hours with optimizations
- Reserve quota for experimentation
- Monitor via Kaggle settings

### 4.2 Session Best Practices
- Always enable GPU in kernel settings
- P100 > T4 > K80 (request P100 when available)
- Save checkpoints every N epochs (not just best)
- Use `/kaggle/working/` for outputs (writeable)

---

## 5. For Notebook 07 - Transfer Learning

### 5.1 Efficient Architectures for P100
**Current Plan:** ResNet50, DenseNet121
**Better Options for P100:**
- EfficientNetV2-S (depthwise convs, fast on P100)
- ConvNeXt-Tiny (modern ConvNet, efficient)
- RegNetY-* (optimized for older GPUs)

### 5.2 Progressive Unfreezing Strategy
```python
# Phase 1: Freeze backbone, train head (3-5 epochs)
base_model.trainable = False
model.compile(...)
model.fit(..., epochs=5)

# Phase 2: Unfreeze top layers (5-10 epochs, lower LR)
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5))  # 10x lower LR
model.fit(..., epochs=10)
```

---

## 6. Implementation Priority

### Phase 1: Immediate (Current Notebook 06b)
- [ ] Mixed precision training
- [ ] Class weights
- [ ] Batch size 64
- [ ] Label smoothing 0.1

**Estimated Time:** 5 minutes
**Estimated Speedup:** 1.5-2x

### Phase 2: Next Iteration (Notebook 06c or 06b v2)
- [ ] tf.data pipeline
- [ ] Cosine decay LR schedule
- [ ] EMA weights
- [ ] Increase epochs to 20-30 (with better convergence)

**Estimated Time:** 1-2 hours
**Estimated Speedup:** 2-3x (cumulative with Phase 1)

### Phase 3: Infrastructure (Optional, High Value)
- [ ] Pre-resize images locally (224x224)
- [ ] Upload resized dataset to Kaggle
- [ ] Update paths in notebook
- [ ] Remove target_size from generators

**Estimated Time:** 2-4 hours (one-time)
**Estimated Speedup:** 3-5x (cumulative)

---

## 7. Success Metrics

**Before Optimizations (Current):**
- Sample (5K images): ~3-5 min/epoch
- Full (78K images): ~30-45 min/epoch
- 50 epochs: ~25-40 hours

**After Phase 1+2 Optimizations:**
- Sample: ~1-2 min/epoch
- Full: ~10-15 min/epoch
- 50 epochs: ~8-12 hours

**After Phase 3 (Pre-resized Images):**
- Sample: ~30-60 sec/epoch
- Full: ~3-5 min/epoch
- 50 epochs: ~2.5-4 hours

---

## 8. Testing Plan

1. **Baseline Run (Current State):**
   - 1 epoch, full dataset
   - Record: time/epoch, GPU utilization, memory usage

2. **Phase 1 Test:**
   - Apply quick wins
   - 1 epoch, full dataset
   - Compare metrics

3. **Phase 2 Test:**
   - Add tf.data + LR schedule
   - 5 epochs, full dataset
   - Verify convergence quality

4. **Phase 3 Test:**
   - Pre-resized dataset
   - Full 50 epoch training
   - Final model evaluation

---

## 9. Risks and Mitigation

### Risk: Mixed Precision Instability
**Mitigation:** Keep loss in float32, monitor for NaN gradients

### Risk: OOM with Larger Batches
**Mitigation:** Start with 64, gradient accumulation if needed

### Risk: Pre-resized Dataset Quality
**Mitigation:** Use high-quality LANCZOS resize, compare sample images

### Risk: Time Investment vs Benefit
**Mitigation:** Implement incrementally, measure each phase

---

## 10. References

- TensorFlow Mixed Precision: https://www.tensorflow.org/guide/mixed_precision
- tf.data Performance Guide: https://www.tensorflow.org/guide/data_performance
- Kaggle GPU Quotas: https://www.kaggle.com/docs/notebooks#technical-specifications
- OneCycle Learning: Smith, "A disciplined approach to neural network hyper-parameters" (2018)

---

## Appendix A: Current Configuration

```python
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'channels': 3,
    'batch_size': 32,      # ← Increase to 64
    'epochs': 50,
    'learning_rate': 0.001,
    'filters': [32, 64, 128, 256],
    'dense_units': 512,
    'dropout_rate': 0.5,
    'l2_reg': 0.0001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'num_classes': 14,
    'use_sample': False,
    'sample_size': 5000,
    'random_state': 42
}
```

---

## Appendix B: Quick Implementation Checklist

**5-Minute Quick Wins:**
```python
# 1. Mixed precision (before model creation)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 2. Larger batch size
CONFIG['batch_size'] = 64

# 3. Label smoothing + class weights (in compile)
model.compile(
    optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.1,
        dtype='float32'
    ),
    ...
)

# 4. Class weights (in fit)
model.fit(
    train_generator,
    class_weight=class_weights_dict,
    ...
)
```

---

**END OF SPECIFICATION**
