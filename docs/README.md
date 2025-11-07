# Documentation Index

**Project**: NIH Chest X-Ray Disease Detection
**Last Updated**: November 4, 2025

## Table of Contents

- [Quick Start](#quick-start)
- [Platform Guides](#platform-guides)
- [Workflow & Tools](#workflow--tools)
- [MLflow & Experiment Tracking](#mlflow--experiment-tracking)
- [Notebook Development](#notebook-development)
- [Optimization & Training](#optimization--training)
- [Deployment](#deployment)
- [Domain Knowledge](#domain-knowledge)
- [Project Planning](#project-planning)

---

## Quick Start

Start here for essential workflows:

| Document | Purpose | Audience |
|----------|---------|----------|
| [PLATFORM_ORGANIZATION.md](PLATFORM_ORGANIZATION.md) | **Directory structure & platform separation** | All developers |
| [NBPUSH_CLI.md](NBPUSH_CLI.md) | **CLI tool for pushing notebooks to cloud GPUs** | ML developers |
| [MLFLOW_QUICKSTART.md](MLFLOW_QUICKSTART.md) | **Quick start for experiment tracking** | ML developers |

---

## Platform Guides

### Cloud GPU Training

| Document | Description |
|----------|-------------|
| [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) | Complete Kaggle workflow (push/monitor/download) |
| [KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md) | Alternative Kaggle workflow guide |
| [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md) | Kaggle-specific training setup |
| [KAGGLE_07_TRAINING.md](KAGGLE_07_TRAINING.md) | Notebook 07 transfer learning on Kaggle |
| [HEADLESS_KAGGLE_TRAINING.md](HEADLESS_KAGGLE_TRAINING.md) | Automated Kaggle training without UI |
| [COLAB_GUIDE.md](COLAB_GUIDE.md) | Complete Colab workflow |
| [COLAB_PRO_SETUP.md](COLAB_PRO_SETUP.md) | Colab Pro configuration guide |
| [../colab/COLAB_NO_DRIVE_SETUP.md](../colab/COLAB_NO_DRIVE_SETUP.md) | Colab without Google Drive setup |
| [../colab/COLAB_PYDRIVE2_SETUP.md](../colab/COLAB_PYDRIVE2_SETUP.md) | PyDrive2 integration for Colab |
| [../colab/COLAB_REFERENCE.md](../colab/COLAB_REFERENCE.md) | Colab API reference |
| [../colab/COLAB_SETUP_GUIDE.md](../colab/COLAB_SETUP_GUIDE.md) | Comprehensive Colab setup |
| [../colab/GCS_SETUP.md](../colab/GCS_SETUP.md) | Google Cloud Storage setup |
| [../colab/GCS_SETUP_GUIDE.md](../colab/GCS_SETUP_GUIDE.md) | Detailed GCS configuration |
| [../colab/OAUTH_SETUP.md](../colab/OAUTH_SETUP.md) | OAuth credentials for Colab |
| [../colab/SETUP_INSTRUCTIONS.md](../colab/SETUP_INSTRUCTIONS.md) | General Colab setup instructions |
| [../colab/MLFLOW_ENRICHMENT_SUMMARY.md](../colab/MLFLOW_ENRICHMENT_SUMMARY.md) | MLflow integration with Colab results |

### Kaggle-Specific Documentation

| Document | Description |
|----------|-------------|
| [KAGGLE_NOTEBOOK_CHANGES.md](KAGGLE_NOTEBOOK_CHANGES.md) | Modifications for Kaggle compatibility |
| [KAGGLE_OPTIMIZATION_SPEC.md](KAGGLE_OPTIMIZATION_SPEC.md) | Requirements for Kaggle optimization |
| [KAGGLE_PROGRESSIVE_TRAINING.md](KAGGLE_PROGRESSIVE_TRAINING.md) | Progressive training strategy |

---

## Workflow & Tools

### Development Tools

| Document | Description |
|----------|-------------|
| [NBPUSH_CLI.md](NBPUSH_CLI.md) | **CLI for notebook cloud push** (interactive UI, activity logging) |
| [JUPYTEXT_WORKFLOW.md](JUPYTEXT_WORKFLOW.md) | Notebook sync between .ipynb and .py |
| [PYLANCE_SETUP.md](PYLANCE_SETUP.md) | VS Code Python language server setup |

### Data & Files

| Document | Description |
|----------|-------------|
| [LARGE_FILES_STRATEGY.md](LARGE_FILES_STRATEGY.md) | Handling large models and datasets |
| [image_pipeline_requirements.md](image_pipeline_requirements.md) | Image preprocessing pipeline spec |

---

## MLflow & Experiment Tracking

### Core MLflow Documentation

| Document | Description |
|----------|-------------|
| [MLFLOW_QUICKSTART.md](MLFLOW_QUICKSTART.md) | **Quick start guide** - Start here |
| [MLFLOW_IMPLEMENTATION_STRATEGY.md](MLFLOW_IMPLEMENTATION_STRATEGY.md) | Overall MLflow architecture |
| [MLFLOW_PORT_NOTES.md](MLFLOW_PORT_NOTES.md) | Port configuration notes |
| [MLFLOW_SQLITE_MIGRATION.md](MLFLOW_SQLITE_MIGRATION.md) | Migrating to SQLite backend |
| [MLFLOW_VERSIONING_GUIDE.md](MLFLOW_VERSIONING_GUIDE.md) | Model versioning strategy |

### Kaggle MLflow Integration

| Document | Description |
|----------|-------------|
| [MLFLOW_KAGGLE_INTEGRATION.md](MLFLOW_KAGGLE_INTEGRATION.md) | Integrating MLflow with Kaggle |
| [KAGGLE_MLFLOW_IMPORT.md](KAGGLE_MLFLOW_IMPORT.md) | Importing Kaggle runs to local MLflow |
| [CSV_TO_MLFLOW.md](CSV_TO_MLFLOW.md) | Importing CSV training histories |

---

## Notebook Development

### Notebook Documentation

| Document | Description |
|----------|-------------|
| [NOTEBOOK_07B_PYTORCH_SUMMARY.md](NOTEBOOK_07B_PYTORCH_SUMMARY.md) | PyTorch transfer learning notebook summary |

---

## Optimization & Training

### Optimization Guides

| Document | Description |
|----------|-------------|
| [CNN_OPTIMIZER_GUIDE.md](CNN_OPTIMIZER_GUIDE.md) | CNN optimization strategies |
| [tensorflow_optimisations_spec.md](tensorflow_optimisations_spec.md) | TensorFlow-specific optimizations |
| [PHASE1_OPTIMIZATIONS_APPLIED.md](PHASE1_OPTIMIZATIONS_APPLIED.md) | Phase 1 optimization summary |

---

## Deployment

| Document | Description |
|----------|-------------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Streamlit dashboard deployment guide |
| [STREAMLIT_INTEGRATION_PLAN.md](STREAMLIT_INTEGRATION_PLAN.md) | DenseNet121 model integration plan |
| [streamlit_histogram_binning_spec.md](streamlit_histogram_binning_spec.md) | Histogram binning specification |

---

## Domain Knowledge

### Medical/Radiology

| Document | Description |
|----------|-------------|
| [radiology_for_dummies.md](radiology_for_dummies.md) | Medical imaging basics for developers |

---

## Project Planning

### Assessment & Setup

| Document | Description |
|----------|-------------|
| [Assessment_Handbook.md](Assessment_Handbook.md) | Code Institute assessment criteria |
| [LEARNING_OBJECTIVES_VERIFICATION.md](LEARNING_OBJECTIVES_VERIFICATION.md) | **Complete LO1-LO11 verification** ✅ |
| [AI_USAGE_STATEMENT.md](AI_USAGE_STATEMENT.md) | **AI usage, challenges, benefits, and ethical considerations** (for submission form) |
| [SUBMISSION_PREPARATION_SUMMARY.md](SUBMISSION_PREPARATION_SUMMARY.md) | Final submission checklist and summary |
| [PROJECT_SETUP_COMPLETE.md](PROJECT_SETUP_COMPLETE.md) | Initial setup completion notes |
| [COMMIT_SUMMARY.md](COMMIT_SUMMARY.md) | Git commit history summary |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Future development plans |
| [FUTURE_SALIENCY_MAPS_GAP_ANALYSIS.md](FUTURE_SALIENCY_MAPS_GAP_ANALYSIS.md) | Implementation plan for pixel-level saliency maps (future enhancement) |

---

## Document Categories

### By Use Case

**Starting a new training run:**
1. [PLATFORM_ORGANIZATION.md](PLATFORM_ORGANIZATION.md) - Understand directory structure
2. [NBPUSH_CLI.md](NBPUSH_CLI.md) - Push notebook to cloud
3. [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) or [COLAB_GUIDE.md](COLAB_GUIDE.md) - Monitor training
4. [MLFLOW_QUICKSTART.md](MLFLOW_QUICKSTART.md) - Track experiments

**Debugging Kaggle issues:**
1. [KAGGLE_NOTEBOOK_CHANGES.md](KAGGLE_NOTEBOOK_CHANGES.md) - Check compatibility
2. [HEADLESS_KAGGLE_TRAINING.md](HEADLESS_KAGGLE_TRAINING.md) - Automated debugging
3. [KAGGLE_MLFLOW_IMPORT.md](KAGGLE_MLFLOW_IMPORT.md) - Import results

**Optimizing models:**
1. [CNN_OPTIMIZER_GUIDE.md](CNN_OPTIMIZER_GUIDE.md) - CNN strategies
2. [tensorflow_optimisations_spec.md](tensorflow_optimisations_spec.md) - TensorFlow tips
3. [KAGGLE_PROGRESSIVE_TRAINING.md](KAGGLE_PROGRESSIVE_TRAINING.md) - Training strategy

---

## File Conventions

- **UPPERCASE_GUIDE.md**: Platform or tool guides (step-by-step workflows)
- **lowercase_spec.md**: Technical specifications and requirements
- **UPPERCASE_SUMMARY.md**: Summary documents and retrospectives
- **lowercase_for_dummies.md**: Educational/beginner content

---

## Contributing

When creating new documentation:

1. Add entry to appropriate section in this index
2. Use descriptive filename matching conventions above
3. Include title, purpose, and target audience in document header
4. Cross-reference related documents

---

## Related Documentation

- Main project README: [../README.md](../README.md)
- Project strategy: [../.claude/CLAUDE.md](../.claude/CLAUDE.md)
- Notebook structure: [../jupyter_notebooks/README.md](../jupyter_notebooks/README.md)

---

**Total Documents**: 48 (docs/ + colab/)
**Last Scan**: November 7, 2025
**Status**: ✅ Ready for submission

---

## Future Enhancements

Documentation for planned improvements:

| Document | Description |
|----------|-------------|
| [FUTURE_SALIENCY_MAPS_GAP_ANALYSIS.md](FUTURE_SALIENCY_MAPS_GAP_ANALYSIS.md) | Pixel-level saliency maps implementation plan (~440 LOC, 11-14 hours) |
| [NEXT_STEPS.md](NEXT_STEPS.md) | General future development roadmap |
