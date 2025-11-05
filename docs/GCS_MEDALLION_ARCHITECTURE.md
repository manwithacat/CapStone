# GCS Medallion Architecture - NIH Chest X-Ray Project

## Overview

The `nih-xrays` bucket follows a **Medallion-style, multi-zone data layout** for ML data management.

## Structure

```
gs://nih-xrays/
├── 00_raw/           # Immutable source data
├── 10_bronze/        # Lightly cleaned data
├── 20_silver/        # Preprocessed data
├── 30_gold/          # Feature-ready data
├── 40_models/        # Model registry
├── 50_artifacts/     # Evaluation outputs
├── 60_logs/          # Training logs
├── 70_cfg/           # Environment configs
├── 80_docs/          # Documentation
└── 90_tmp/           # Scratch (auto-cleaned after 7 days)
```

## Layers Explained

### 00_raw/ - Raw Data (Immutable)

**Purpose**: Original, unmodified source data
**Never modify or delete**: This is your source of truth

```
00_raw/
└── nih-cxr/
    ├── images/
    │   ├── images_001/
    │   │   └── images/
    │   │       ├── 00000001_000.png
    │   │       └── ... (112,120 images total)
    │   ├── images_002/
    │   └── ... (images_012/)
    └── metadata/
        ├── Data_Entry_2017.csv
        ├── BBox_List_2017.csv
        └── expert_labels/ (if applicable)
```

**Size**: ~47 GB
**Update frequency**: Never (immutable)
**Access pattern**: Read-only

### 10_bronze/ - Light Cleaning

**Purpose**: Lightly processed, validated data with minimal transformations
**Examples**: Train/val/test splits, validated CSVs

```
10_bronze/
└── nih-cxr/
    └── manifests/
        ├── train_split.csv       # 78,831 samples
        ├── val_split.csv         # 16,383 samples
        ├── test_split.csv        # 16,890 samples
        └── preprocessing_config.json
```

**Size**: ~6.6 MB
**Update frequency**: When splits change
**Access pattern**: Read frequently (every training run)

**Key transformations from raw**:
- Patient-level stratified splits (no data leakage)
- Binary disease labels (0/1)
- Removed full_path column (built at runtime)

### 20_silver/ - Preprocessed Data (Future)

**Purpose**: Resized, normalized, augmented data ready for training
**Not yet implemented** - streaming from raw is fast enough

```
20_silver/
└── nih-cxr/
    ├── images-224/          # Resized to 224x224
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── images-512/          # For higher-res models
    └── tfrecords/           # Sharded for distributed training
        ├── train-0000.tfrecord
        └── ...
```

**When to create**:
- If preprocessing becomes bottleneck
- For distributed training (sharded TFRecords)
- For models requiring specific sizes

### 30_gold/ - Feature-Ready (Future)

**Purpose**: Extracted features, embeddings, predictions
**Not yet implemented**

```
30_gold/
└── nih-cxr/
    ├── features/
    │   ├── resnet50-embeddings/
    │   └── densenet-embeddings/
    └── predictions/
        └── ensemble-predictions.csv
```

**When to create**:
- For feature-based models (using pre-extracted embeddings)
- For ensemble models
- For transfer learning experiments

### 40_models/ - Model Registry

**Purpose**: Trained model checkpoints, configs, model cards

```
40_models/
└── nih-cxr/
    ├── resnet50-transfer/
    │   ├── runs/
    │   │   ├── 2025-11-04_2130/
    │   │   │   ├── best_model.keras
    │   │   │   ├── config.json
    │   │   │   ├── training_history.csv
    │   │   │   └── model_card.md
    │   │   └── 2025-11-05_1045/
    │   └── production/
    │       └── v1.0.0/
    ├── densenet121-transfer/
    └── efficientnetb3-transfer/
```

**Access pattern**: Write during training, read for inference
**Versioning**: Timestamped runs + semantic versioned production

### 50_artifacts/ - Evaluation Outputs

**Purpose**: Metrics, plots, confusion matrices, reports

```
50_artifacts/
└── nih-cxr/
    ├── metrics/
    │   ├── 2025-11-04_resnet50_metrics.json
    │   └── model_comparison.csv
    ├── plots/
    │   ├── roc_curves.png
    │   ├── confusion_matrix.png
    │   └── training_history.png
    └── reports/
        ├── experiment_001.md
        └── final_evaluation.pdf
```

**Download to local MLflow**: Use these for tracking and comparison

### 60_logs/ - Training Logs

**Purpose**: TensorBoard logs, MLflow artifacts

```
60_logs/
└── nih-cxr/
    ├── tensorboard/
    │   ├── run_20251104_2130/
    │   └── run_20251105_1045/
    └── mlflow/
        ├── experiments/
        └── runs/
```

**Sync to local**: Download after training for analysis

### 70_cfg/ - Environment Configs

**Purpose**: Reproducibility - exact package versions used

```
70_cfg/
├── requirements.txt          # Python packages
├── environment.yml           # Conda env
├── Dockerfile               # Container image
└── runs/
    └── 2025-11-04/
        ├── requirements.txt  # Snapshot for this run
        └── conda_list.txt
```

**Best practice**: Snapshot configs with each training run

### 80_docs/ - Documentation

**Purpose**: Datasheets, licenses, ethics documentation

```
80_docs/
├── README.md
├── DATASHEET.md              # Data documentation
├── MODEL_CARD.md             # Model documentation
├── ETHICS.md                 # Ethical considerations
├── CITATIONS.txt             # Papers to cite
└── licenses/
    └── NIH_LICENSE.txt
```

### 90_tmp/ - Scratch Space

**Purpose**: Temporary files, experiments, debugging
**Auto-cleaned**: Lifecycle rule deletes files >7 days old

```
90_tmp/
├── debug_samples/
├── test_uploads/
└── scratch/
```

**Warning**: Don't rely on this for persistent data!

## Workflow

### Training Workflow

```
1. Load splits:        10_bronze/nih-cxr/manifests/
2. Stream images:      00_raw/nih-cxr/images/
3. Save model:         40_models/nih-cxr/{model}/runs/{timestamp}/
4. Save metrics:       50_artifacts/nih-cxr/metrics/
5. Save logs:          60_logs/nih-cxr/tensorboard/
6. Save config:        70_cfg/runs/{timestamp}/
```

### Download for MLflow

```bash
# After training in Colab
gsutil -m rsync -r gs://nih-xrays/50_artifacts/ ./outputs/artifacts/
gsutil -m rsync -r gs://nih-xrays/40_models/ ./models/

# Import to MLflow
python scripts/gcs_to_mlflow.py
```

## Cost Optimization

**Storage costs** (~$0.020/GB/month):
- 00_raw: 47 GB = $0.94/month (immutable)
- 10_bronze: 0.01 GB = negligible
- 40_models: ~10 GB = $0.20/month (grows over time)
- 50_artifacts: ~1 GB = $0.02/month

**Total: ~$1.20/month initially, grows to ~$2-3/month**

**Lifecycle rules**:
- 90_tmp/: Auto-delete after 7 days
- Old model runs: Consider archiving to Coldline after 90 days

## Access Patterns

| Layer | Read Frequency | Write Frequency | Size Growth |
|-------|----------------|-----------------|-------------|
| 00_raw | Every run | Never | Static |
| 10_bronze | Every run | Rarely | Minimal |
| 20_silver | Future | Future | TBD |
| 30_gold | Future | Future | TBD |
| 40_models | As needed | Per run | Linear |
| 50_artifacts | Post-run | Per run | Linear |
| 60_logs | Debug | Per run | Linear |
| 70_cfg | Reference | Per run | Minimal |
| 80_docs | Reference | Rarely | Minimal |
| 90_tmp | Debug | Frequently | Controlled |

## Commands

### List structure
```bash
gsutil ls gs://nih-xrays/
```

### Check sizes
```bash
gsutil du -sh gs://nih-xrays/*
```

### Download artifacts
```bash
gsutil -m rsync -r gs://nih-xrays/50_artifacts/ ./local_artifacts/
```

### Upload model
```bash
gsutil cp model.keras gs://nih-xrays/40_models/resnet50/runs/$(date +%Y-%m-%d_%H%M)/
```

## Benefits

✅ **Clear data lineage**: Raw → Bronze → Silver → Gold
✅ **Immutability**: Raw data never changes
✅ **Reproducibility**: Configs snapshots with each run
✅ **Cost control**: Lifecycle rules for temp data
✅ **Collaboration**: Clear zones for different uses
✅ **Scalability**: Add new layers as needed

## References

- [Medallion Architecture (Databricks)](https://www.databricks.com/glossary/medallion-architecture)
- [Data Mesh Principles](https://www.datamesh-architecture.com/)
- [ML Ops Best Practices](https://ml-ops.org/)
