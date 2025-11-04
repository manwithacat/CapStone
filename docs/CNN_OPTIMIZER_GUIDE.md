# CNN Optimizer Configuration Guide

**Notebook 06 Training Optimizations**

## Overview

This guide explains the advanced training techniques implemented in notebook 06 for improved CNN performance and stability.

## Implemented Optimizations

### ✅ 1. Label Smoothing (0.1)
**Status**: Already implemented

**What it does:**
- Softens hard targets (0, 1) → (0.1, 0.9)
- Prevents model overconfidence
- Improves generalization

**Why it helps:**
- Medical imaging has label noise (NLP-extracted labels)
- Reduces overfitting on imperfect labels
- Standard practice for ImageNet models

---

### ✅ 2. AdamW Optimizer
**Status**: ✅ **NEW - IMPLEMENTED**

**What it does:**
- Decouples weight decay from gradient updates
- More effective regularization than L2 penalty
- Better convergence properties

**Configuration:**
```python
CONFIG = {
    'optimizer': 'adamw',  # 'adam' or 'adamw'
    'weight_decay': 0.01,  # AdamW weight decay (typical range: 0.001-0.1)
}
```

**Why it helps:**
- Adam's weight decay interacts poorly with adaptive learning rates
- AdamW fixes this by applying weight decay independently
- Proven effective on vision tasks (ResNet, EfficientNet use it)
- Often 1-2% AUC improvement over standard Adam

**When to use:**
- ✅ **Always recommended** for CNN training
- Especially beneficial for:
  - Long training runs (50+ epochs)
  - Models with many parameters
  - Medical imaging (benefits from better regularization)

---

### ✅ 3. Cosine Decay Learning Rate
**Status**: ✅ **NEW - IMPLEMENTED**

**What it does:**
- Gradually reduces learning rate following cosine curve
- Smooth decay from initial LR to 0 over training
- Replaces ReduceLROnPlateau (predetermined schedule)

**Configuration:**
```python
CONFIG = {
    'use_cosine_decay': True,  # Enable cosine annealing
    'learning_rate': 0.001,    # Initial LR
    'epochs': 50,              # Decay over full training
}
```

**Learning rate trajectory:**
```
Epoch 0:  LR = 0.001000
Epoch 10: LR = 0.000905
Epoch 20: LR = 0.000655
Epoch 30: LR = 0.000345
Epoch 40: LR = 0.000095
Epoch 50: LR = 0.000000
```

**Why it helps:**
- Smoother convergence than step decay (ReduceLROnPlateau)
- Allows model to explore solution space early (high LR)
- Fine-tunes parameters late in training (low LR)
- No manual tuning of LR reduction triggers needed
- Standard practice for state-of-the-art models

**When to use:**
- ✅ **Recommended for production training**
- Best for:
  - Fixed epoch count (50 epochs)
  - Long training runs
  - Known good hyperparameters
- NOT recommended for:
  - Exploratory training (unknown epoch count)
  - Very short runs (<10 epochs)

**Comparison with ReduceLROnPlateau:**

| Feature | CosineDecay | ReduceLROnPlateau |
|---------|-------------|-------------------|
| **Schedule** | Predetermined | Reactive to plateaus |
| **Smoothness** | Very smooth | Sudden drops |
| **Tuning** | No tuning needed | Requires patience tuning |
| **Predictability** | Fully predictable | Depends on training dynamics |
| **Best for** | Production training | Exploratory training |

---

### ⚠️ 4. Gradient Clipping
**Status**: ⚠️ **OPTIONAL - MONITOR FIRST**

**What it does:**
- Clips gradient norms to prevent exploding gradients
- Stabilizes training when gradients spike

**Configuration:**
```python
CONFIG = {
    'gradient_clip_norm': None,  # Set to 1.0 to enable
}
```

**Why it helps:**
- Prevents gradient explosions (NaN losses)
- Stabilizes training with:
  - High learning rates
  - Class imbalance (large weight updates)
  - Deep networks
  - Mixed precision training

**When to use:**
- ⚠️ **Only if training is unstable**
- Signs you need it:
  - Loss suddenly spikes or becomes NaN
  - Training diverges after early epochs
  - Large oscillations in validation metrics
- Our setup (batch_size=128, class_weights, FP16) has moderate risk
- **Recommendation**: Start without it, add if problems arise

**If enabling:**
```python
'gradient_clip_norm': 1.0,  # Clip gradients to norm 1.0
```

---

## Configuration Examples

### 1. Production Training (Recommended)
```python
CONFIG = {
    # Core settings
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,

    # Optimizer
    'optimizer': 'adamw',        # ✅ Better than Adam
    'weight_decay': 0.01,        # AdamW weight decay

    # Learning rate schedule
    'use_cosine_decay': True,    # ✅ Smooth LR decay

    # Stability
    'gradient_clip_norm': None,  # Start without, add if needed

    # Other
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,     # Ignored when use_cosine_decay=True
}
```

**Best for:**
- Final 50-epoch training run
- Full dataset (78K images)
- Production model
- Known to work well with medical imaging

---

### 2. Exploratory Training (Fast Iteration)
```python
CONFIG = {
    # Core settings
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,

    # Optimizer
    'optimizer': 'adam',          # Faster to experiment with
    'weight_decay': 0.01,         # Ignored for Adam

    # Learning rate schedule
    'use_cosine_decay': False,    # Use ReduceLROnPlateau instead

    # Stability
    'gradient_clip_norm': None,

    # Other
    'early_stopping_patience': 5,  # Exit earlier
    'reduce_lr_patience': 3,       # More aggressive LR reduction
}
```

**Best for:**
- Testing architecture changes
- Hyperparameter search
- Quick experiments on samples
- Unknown optimal epoch count

---

### 3. Unstable Training (Emergency Config)
```python
CONFIG = {
    # Core settings
    'batch_size': 64,            # Reduce if still unstable
    'epochs': 50,
    'learning_rate': 0.0005,     # Lower LR for stability

    # Optimizer
    'optimizer': 'adamw',
    'weight_decay': 0.01,

    # Learning rate schedule
    'use_cosine_decay': True,

    # Stability
    'gradient_clip_norm': 1.0,   # ✅ Enable clipping

    # Other
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
}
```

**Use if:**
- Loss becomes NaN
- Training diverges
- Extreme gradient spikes
- Class weights causing instability

---

## Expected Improvements

### Performance Gains (Estimated)

| Optimization | AUC Improvement | Training Time Impact |
|--------------|-----------------|----------------------|
| **AdamW** | +1-2% | No change |
| **Cosine Decay** | +0.5-1.5% | No change |
| **Label Smoothing** | +0.5-1% | No change (already enabled) |
| **Gradient Clipping** | Stability only | Minimal (~1-2% slower) |
| **Combined** | **+2-4% AUC** | No significant change |

### Convergence Behavior

**Before (Adam + ReduceLROnPlateau):**
- Training loss: Smooth with occasional plateaus
- Validation AUC: Plateaus trigger LR drops
- Final performance: Good but may plateau early

**After (AdamW + CosineDecay):**
- Training loss: Very smooth, gradual convergence
- Validation AUC: Steady improvement throughout training
- Final performance: Better generalization, higher final AUC

---

## Monitoring Training

### Key Metrics to Watch

1. **Learning Rate**
   - Check MLflow or training logs
   - Should smoothly decay with cosine schedule

2. **Gradient Norms** (if issues arise)
   ```python
   # Add to callbacks if needed
   callbacks.TensorBoard(histogram_freq=1)
   ```

3. **Validation AUC**
   - Should improve steadily
   - No sudden drops (indicates stability)

4. **Training Loss**
   - Should decrease smoothly
   - No spikes or NaN values

### Warning Signs

⚠️ **Enable gradient clipping if you see:**
- Loss suddenly jumps to NaN
- Validation metrics oscillate wildly
- Training loss spikes then recovers
- Gradient norms > 10 (check with TensorBoard)

---

## MLflow Tracking

All optimization settings are automatically logged:

```python
mlflow.log_params({
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'use_cosine_decay': True,
    'gradient_clip_norm': None,
    'label_smoothing': 0.1,
})
```

Compare runs:
```bash
mlflow ui  # Open http://localhost:5000
```

---

## References

1. **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
2. **Cosine Annealing**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017
3. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
4. **Gradient Clipping**: Pascanu et al., "On the difficulty of training RNNs", ICML 2013

---

## Quick Start

**For 50-epoch production training:**

1. Use the recommended config (AdamW + CosineDecay)
2. Start training
3. Monitor for 2-3 epochs
4. If stable → continue
5. If unstable → add `gradient_clip_norm=1.0`

**Success criteria:**
- ✅ No NaN losses
- ✅ Validation AUC improves steadily
- ✅ Training completes without divergence
- ✅ Final AUC > baseline + 2-4%
