#!/usr/bin/env python3
"""
Enrich MLflow runs with comprehensive metadata.

Adds:
- Dataset information (NIH X-Ray dataset details)
- Model architecture details (params, layers, size)
- Training environment (TensorFlow version, hardware)
- Registered models in MLflow Model Registry
- Additional tags and metadata
"""

import os
import json
import mlflow
import mlflow.tensorflow
from pathlib import Path
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
COLAB_RESULTS_DIR = PROJECT_ROOT / "colab" / "results"

# MLflow settings
EXPERIMENT_NAME = "NIH-XRay-Transfer-Learning"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"

# Dataset metadata
DATASET_INFO = {
    "name": "NIH Chest X-Ray Dataset",
    "version": "2017",
    "source": "NIH Clinical Center",
    "total_images": 112120,
    "total_patients": 30805,
    "num_classes": 14,
    "classes": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ],
    "image_size": "1024x1024 (original)",
    "train_samples": 78566,
    "val_samples": 17063,
    "test_samples": 16491,
    "split_strategy": "patient-level",
    "train_patients": 21563,
    "val_patients": 4677,
    "test_patients": 4565,
    "data_format": "grayscale PNG",
    "label_type": "multi-label",
    "label_source": "NLP extraction from radiology reports",
}

def get_model_info(model_path: Path):
    """Extract detailed model information."""
    try:
        model = keras.models.load_model(model_path)

        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        # Get model size
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Layer info
        num_layers = len(model.layers)

        # Input/output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape

        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "non_trainable_params": int(non_trainable_params),
            "num_layers": num_layers,
            "model_size_mb": round(model_size_mb, 2),
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
        }
    except Exception as e:
        print(f"    ⚠️  Could not extract model info: {e}")
        return {}

def enrich_run(run_id: str, model_name: str, config: dict, model_path: Path):
    """Enrich a single MLflow run with comprehensive metadata."""

    print(f"\n🔧 Enriching {model_name} (run_id: {run_id[:8]}...)")

    with mlflow.start_run(run_id=run_id):
        # 1. Dataset information
        print("   📊 Adding dataset metadata...")
        mlflow.set_tag("mlflow.dataset.name", DATASET_INFO["name"])
        mlflow.set_tag("mlflow.dataset.version", DATASET_INFO["version"])
        mlflow.log_param("dataset_total_images", DATASET_INFO["total_images"])
        mlflow.log_param("dataset_total_patients", DATASET_INFO["total_patients"])
        mlflow.log_param("dataset_num_classes", DATASET_INFO["num_classes"])
        mlflow.log_param("dataset_train_samples", DATASET_INFO["train_samples"])
        mlflow.log_param("dataset_val_samples", DATASET_INFO["val_samples"])
        mlflow.log_param("dataset_test_samples", DATASET_INFO["test_samples"])
        mlflow.log_param("dataset_split_strategy", DATASET_INFO["split_strategy"])
        mlflow.log_param("label_type", DATASET_INFO["label_type"])

        # Log class names
        for i, class_name in enumerate(DATASET_INFO["classes"]):
            mlflow.set_tag(f"class_{i}", class_name)

        # 2. Model architecture details
        print("   🏗️  Extracting model architecture...")
        if model_path and model_path.exists():
            model_info = get_model_info(model_path)
            for key, value in model_info.items():
                if isinstance(value, (int, float)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.set_tag(key, value)

            print(f"      ✓ Total params: {model_info.get('total_params', 'N/A'):,}")
            print(f"      ✓ Model size: {model_info.get('model_size_mb', 'N/A')} MB")

        # 3. Training environment
        print("   🖥️  Adding environment info...")
        mlflow.set_tag("framework", "TensorFlow/Keras")
        mlflow.set_tag("framework_version", tf.__version__)
        mlflow.set_tag("python_version", f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}")
        mlflow.set_tag("training_hardware", "Google Colab A100 GPU")
        mlflow.set_tag("gpu_memory", "80GB")
        mlflow.set_tag("training_location", "Google Cloud Platform")

        # 4. Training strategy details
        print("   📚 Adding training strategy...")
        mlflow.set_tag("training_strategy", "two-stage-transfer-learning")
        mlflow.set_tag("stage1_strategy", "feature_extraction")
        mlflow.set_tag("stage2_strategy", "fine_tuning")
        mlflow.log_param("pretrained_weights", "ImageNet")
        mlflow.log_param("input_preprocessing", "ImageNet normalization")

        # 5. Additional metadata from config
        if config:
            # Calculate total epochs
            total_epochs = config.get("epochs_stage1", 0) + config.get("epochs_stage2", 0)
            mlflow.log_param("total_epochs", total_epochs)

            # Add data augmentation info
            mlflow.set_tag("data_augmentation", "horizontal_flip,rotation,zoom,shift")

            # Training config summary
            mlflow.set_tag("optimizer_stage1", "Adam")
            mlflow.set_tag("optimizer_stage2", "Adam")
            mlflow.set_tag("loss_function", "binary_crossentropy")
            mlflow.set_tag("metrics", "accuracy,AUC")

        # 6. Project metadata
        print("   📝 Adding project metadata...")
        mlflow.set_tag("project_name", "NIH-Chest-XRay-Disease-Detection")
        mlflow.set_tag("task_type", "multi-label-classification")
        mlflow.set_tag("domain", "medical-imaging")
        mlflow.set_tag("use_case", "chest-xray-disease-detection")
        mlflow.set_tag("model_purpose", "research")

        # 7. Data provenance
        mlflow.set_tag("data_source", "NIH Clinical Center")
        mlflow.set_tag("data_collection_period", "1992-2015")
        mlflow.set_tag("data_publication_year", "2017")
        mlflow.set_tag("data_citation", "Wang et al., IEEE CVPR 2017")

        # 8. Model versioning info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("enrichment_timestamp", timestamp)
        mlflow.set_tag("mlflow_version", mlflow.__version__)

        print("   ✅ Enrichment complete")

def register_models():
    """Register models in MLflow Model Registry."""
    print("\n📦 Registering models in Model Registry...")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    for _, run in runs.iterrows():
        model_arch = run.get('tags.model_architecture', '')
        run_id = run['run_id']

        if not model_arch:
            continue

        # Register model with version
        model_name = f"NIH-XRay-{model_arch.upper()}"

        try:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            # Add model version tags
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                model_name,
                mv.version,
                "stage",
                "staging"
            )
            client.set_model_version_tag(
                model_name,
                mv.version,
                "platform",
                "colab"
            )

            print(f"   ✅ Registered {model_name} version {mv.version}")

        except Exception as e:
            print(f"   ⚠️  Could not register {model_name}: {e}")

def main():
    """Main enrichment function."""
    print("🚀 MLflow Run Enrichment")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get all runs
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    print(f"\n✓ Found {len(runs)} runs to enrich")

    # Enrich each run
    for _, run in runs.iterrows():
        run_id = run['run_id']
        model_arch = run.get('tags.model_architecture', 'unknown')

        # Find model file
        model_files = list(COLAB_RESULTS_DIR.glob(f"models/{model_arch}-transfer/**/*.keras"))
        model_path = model_files[0] if model_files else None

        # Load config
        config = {}
        config_files = list(COLAB_RESULTS_DIR.glob(f"models/{model_arch}-transfer/**/config.json"))
        if config_files:
            with open(config_files[0], 'r') as f:
                config = json.load(f)

        # Enrich run
        try:
            enrich_run(run_id, model_arch, config, model_path)
        except Exception as e:
            print(f"   ❌ Failed to enrich {model_arch}: {e}")

    # Register models
    register_models()

    print("\n" + "=" * 60)
    print("✅ Enrichment Complete!")
    print("=" * 60)
    print(f"\n📊 View enriched runs:")
    print(f"   MLflow UI: http://localhost:5001")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print(f"\n💾 Data stored in:")
    print(f"   {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()
