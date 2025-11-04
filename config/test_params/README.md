# Test Parameters Directory

This directory contains JSON parameter files for testing notebooks with different configurations.

## Purpose

Test parameter files allow you to run notebooks with specific configurations without modifying the notebook itself. This is useful for:

- **Pre-flight testing**: Quick validation before pushing to cloud GPUs
- **Kaggle configuration**: Parameters specific to Kaggle environment
- **Local testing**: Different configs for local development
- **Warm starts**: Resume training from checkpoints

## File Naming Convention

```
test_params_<purpose>_<platform>.json
```

**Examples:**
- `test_params_v10_kaggle.json` - Version 10 Kaggle configuration
- `test_params_local_preflight.json` - Local pre-flight test
- `test_params_v9_preflight.json` - Version 9 quick test

## Structure

All parameter files should follow this structure:

```json
{
  "MODELS_TO_TRAIN": ["resnet50", "densenet121", "efficientnetb3"],
  "RUN_NAME": "descriptive_run_name",
  "USE_MLFLOW": true,
  "RETRAIN_MODELS": false,
  "CONFIG": {
    "img_height": 224,
    "img_width": 224,
    "channels": 3,
    "batch_size": 32,
    "epochs_stage1": 10,
    "learning_rate_stage1": 0.001,
    "epochs_stage2": 10,
    "learning_rate_stage2": 0.0001,
    "unfreeze_layers": 20,
    "dense_units": 512,
    "dropout_rate": 0.5,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "num_classes": 14,
    "use_sample": false,
    "sample_size": 1000,
    "random_state": 42
  }
}
```

## Usage

### With Papermill

```bash
papermill jupyter_notebooks/07_transfer_learning.ipynb \
    output.ipynb \
    -f config/test_params/test_params_v10_kaggle.json
```

### Pre-flight Testing

```bash
./scripts/test_notebook_07_preflight.sh
```

The preflight script uses local preflight parameters for quick validation.

### Common Configurations

**Quick Test (Preflight):**
- Models: `["densenet121"]` (single model)
- Epochs: 1-2 per stage
- Sample: Small subset
- MLflow: Disabled

**Full Kaggle Training:**
- Models: All three (ResNet50, DenseNet121, EfficientNetB3)
- Epochs: 10+ per stage
- Sample: Full dataset
- MLflow: Enabled

**Warm Start:**
- `RETRAIN_MODELS`: false (load existing models)
- Continue training from checkpoint

## Git Handling

⚠️ **Important**: Test parameter files are git-ignored by default (see `.gitignore`).

This is intentional because:
- Parameters change frequently during experimentation
- Different developers may need different configs
- Avoid cluttering git history

To track specific parameter files (like production configs), add them explicitly:

```bash
git add -f config/test_params/test_params_production.json
```

## Related Documentation

- [NBPUSH_CLI.md](../../docs/NBPUSH_CLI.md) - Push notebooks with specific configs
- [KAGGLE_GUIDE.md](../../docs/KAGGLE_GUIDE.md) - Kaggle training workflow
- [Notebook 07](../../jupyter_notebooks/07_transfer_learning.ipynb) - Transfer learning notebook
