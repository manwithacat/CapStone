# CNN Training Optimization Guide - Fix 37.8hr → 10hr

## Current Performance Problem

**Target**: 50 epochs in <12 hours
**Current**: 37.8 hours for 50 epochs (3× too slow!)
**Throughput**: 28.9 images/second (should be 200-400 on P100)

## Root Causes & Solutions

### 1. Batch Size Too Small ⚠️ CRITICAL

**Current**: `batch_size=32`
**Problem**: Only using ~10% of P100's 16GB memory
**Solution**: Increase to 128 or 256

```python
CONFIG = {
    'batch_size': 128,  # Was 32 - increase 4x
    # ... rest stays same
}
```

**Impact**: 3-4× speedup (reduces to ~10 hours)

### 2. Replace ImageDataGenerator with tf.data ⚠️ CRITICAL

**Current**: `ImageDataGenerator` (slow, CPU-bound)
**Problem**: No prefetching, no GPU parallelization
**Solution**: Use `tf.data.Dataset` with AUTOTUNE

```python
import tensorflow as tf

def load_and_preprocess_image(image_path, label):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    # Augmentation (runs on GPU!)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

# Create dataset
train_paths = train_df['full_path'].values
train_labels = train_df[disease_classes].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(128)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
```

**Impact**: 2-3× speedup

### 3. Enable Mixed Precision Training

**Current**: FP32 (full precision)
**Problem**: Not using P100 Tensor Cores
**Solution**: Enable FP16 mixed precision

```python
from tensorflow.keras import mixed_precision

# Enable at start of notebook
mixed_precision.set_global_policy('mixed_float16')

# Model output layer needs float32
model = models.Sequential([
    # ... conv layers ...
    layers.Dense(num_classes, activation='sigmoid', dtype='float32')  # Note dtype
])
```

**Impact**: 1.5-2× speedup

### 4. Reduce Unnecessary Augmentation

**Current**: Rotation, shifts, zoom all enabled
**Problem**: Complex augmentation slows training
**Solution**: Use only horizontal flip for medical images

```python
# Medical X-rays should NOT be rotated/zoomed heavily
# Only flip horizontally (chest is symmetric)
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label
```

**Impact**: 20-30% speedup

### 5. Simplify Model Architecture (Optional)

**Current**: 4 blocks with 2 Conv layers each = 8 conv layers
**Problem**: Deep custom CNN may not be necessary
**Solution**: Try 3 blocks or transfer learning

```python
# Simpler architecture
filters = [64, 128, 256]  # Was [32, 64, 128, 256]
# Remove one conv block
```

**Impact**: 15-25% speedup

## Optimized Configuration

```python
# CNN Configuration - OPTIMIZED
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'channels': 3,
    'batch_size': 128,  # ← Increased from 32
    'epochs': 50,
    'learning_rate': 0.001,
    'filters': [64, 128, 256],  # ← Simplified from [32, 64, 128, 256]
    'dense_units': 512,
    'dropout_rate': 0.5,
    'l2_reg': 0.0001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'num_classes': 14,
}
```

## Expected Performance Gains

| Optimization | Speedup | Cumulative Time |
|--------------|---------|-----------------|
| **Baseline** | 1.0× | 37.8 hours |
| Batch size 32→128 | 3.5× | **10.8 hours** ✓ |
| + tf.data pipeline | 2.2× | **4.9 hours** ✓✓ |
| + Mixed precision | 1.7× | **2.9 hours** ✓✓✓ |
| + Simplified augmentation | 1.2× | **2.4 hours** ✓✓✓✓ |

**Conservative estimate**: 10-12 hours for 50 epochs (just within Kaggle limit)
**With all optimizations**: 3-5 hours for 50 epochs

## Implementation Priority

### Phase 1: Quick Win (10 min to implement)
1. Change `batch_size: 32 → 128` in CONFIG
2. Re-upload to Kaggle and run

**Expected**: 50 epochs in ~10-11 hours

### Phase 2: Major Optimization (30 min to implement)
1. Replace ImageDataGenerator with tf.data
2. Enable mixed precision
3. Simplify augmentation to horizontal flip only

**Expected**: 50 epochs in ~4-5 hours

### Phase 3: Advanced (if still needed)
1. Reduce model complexity
2. Use gradient accumulation for even larger effective batch size
3. Profile with TensorBoard to find remaining bottlenecks

## Example: Optimized Training Cell

```python
# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Create tf.data pipeline
def make_dataset(df, disease_classes, batch_size, shuffle=True, augment=True):
    paths = df['full_path'].values
    labels = df[disease_classes].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0

        if augment:
            image = tf.image.random_flip_left_right(image)

        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create datasets
train_dataset = make_dataset(train_df, disease_classes, batch_size=128, shuffle=True, augment=True)
val_dataset = make_dataset(val_df, disease_classes, batch_size=128, shuffle=False, augment=False)

# Train with optimized pipeline
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callback_list,
    class_weight=class_weights_dict,
    verbose=1
)
```

## Monitoring Performance

After implementing optimizations, check:

```python
# At start of training
print(f"Steps per epoch: {len(train_dataset)}")
print(f"Expected time: ~{len(train_dataset) * 0.5 / 60:.1f} min/epoch")
# Should see: ~5-8 min/epoch instead of 45 min/epoch
```

## Bottom Line

**Just changing batch_size from 32 → 128 will get you to ~10-11 hours for 50 epochs** (within Kaggle limit).

The full optimization suite should achieve 3-5 hours for 50 epochs, giving you plenty of buffer.
