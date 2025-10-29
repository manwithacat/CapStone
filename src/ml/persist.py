"""
Model persistence utilities.

Implements best practices for saving/loading:
- Preprocessor saved with joblib
- XGBoost models saved as JSON (portable, platform-independent)
- Manifest with metadata and version tracking
"""
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def save_pipeline(
    pipeline: Pipeline,
    output_dir: Path,
    disease_classes: list[str],
    config: dict,
    metrics: dict[str, Any] = None
) -> None:
    """
    Save a trained pipeline to disk using best practices.

    Structure:
        output_dir/
        ├── preprocessor.joblib (FeatureUnion + StandardScaler)
        ├── class_0.json (XGBoost trees for disease 0)
        ├── class_1.json
        ├── ...
        ├── class_N.json
        └── manifest.json (metadata, params, versions)

    Args:
        pipeline: Trained Pipeline with 'prep' and 'clf' steps
        output_dir: Directory to save artifacts
        disease_classes: List of disease class names
        config: Configuration dictionary used for training
        metrics: Optional dictionary of evaluation metrics

    Raises:
        ValueError: If pipeline structure is invalid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate pipeline structure
    if 'prep' not in pipeline.named_steps or 'clf' not in pipeline.named_steps:
        raise ValueError("Pipeline must have 'prep' and 'clf' steps")

    preprocessor = pipeline.named_steps['prep']
    classifier = pipeline.named_steps['clf']

    if not isinstance(classifier, MultiOutputClassifier):
        raise ValueError("Classifier must be MultiOutputClassifier")

    # Save preprocessor with joblib
    preprocessor_path = output_dir / 'preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_path)

    # Save each XGBoost estimator as JSON
    for i, estimator in enumerate(classifier.estimators_):
        if not isinstance(estimator, XGBClassifier):
            raise ValueError(f"Estimator {i} is not XGBClassifier")

        json_path = output_dir / f'class_{i}.json'
        estimator.save_model(str(json_path))

    # Create manifest
    manifest = {
        'model_type': 'MultiOutputClassifier',
        'base_estimator': 'XGBClassifier',
        'n_outputs': len(classifier.estimators_),
        'disease_classes': disease_classes,

        # XGBoost parameters (from first estimator, all are identical)
        'xgb_params': classifier.estimators_[0].get_params(deep=False),

        # Classes per output
        'classes_per_output': [
            np.asarray(classes).tolist()
            for classes in classifier.classes_
        ],

        # Library versions
        'versions': {
            'python': f"{np.__version__}",  # Using numpy as proxy
            'scikit_learn': sklearn.__version__,
            'xgboost': xgb.__version__,
            'numpy': np.__version__,
        },

        # Configuration snapshot
        'config': config,

        # Optional metrics
        'metrics': metrics or {},
    }

    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Pipeline saved to {output_dir}")
    print(f"  - preprocessor.joblib")
    print(f"  - class_{{0-{len(classifier.estimators_)-1}}}.json")
    print(f"  - manifest.json")


def load_pipeline(model_dir: Path) -> Pipeline:
    """
    Load a saved pipeline from disk.

    Args:
        model_dir: Directory containing saved pipeline artifacts

    Returns:
        Reconstructed Pipeline with preprocessor and classifier

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If manifest is invalid
    """
    model_dir = Path(model_dir)

    # Load manifest
    manifest_path = model_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Validate manifest
    if manifest['model_type'] != 'MultiOutputClassifier':
        raise ValueError(f"Unsupported model type: {manifest['model_type']}")

    if manifest['base_estimator'] != 'XGBClassifier':
        raise ValueError(f"Unsupported base estimator: {manifest['base_estimator']}")

    # Load preprocessor
    preprocessor_path = model_dir / 'preprocessor.joblib'
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    preprocessor = joblib.load(preprocessor_path)

    # Load XGBoost estimators
    n_outputs = manifest['n_outputs']
    xgb_params = manifest['xgb_params']
    classes_per_output = manifest['classes_per_output']

    estimators = []
    for i in range(n_outputs):
        json_path = model_dir / f'class_{i}.json'
        if not json_path.exists():
            raise FileNotFoundError(f"Booster not found: {json_path}")

        # Create XGBClassifier with same params
        estimator = XGBClassifier(**xgb_params)

        # Load trained model
        estimator.load_model(str(json_path))

        # Set classes (required for sklearn compatibility)
        estimator.classes_ = np.array(classes_per_output[i])

        estimators.append(estimator)

    # Reconstruct MultiOutputClassifier
    multi_output = MultiOutputClassifier(
        estimator=XGBClassifier(**xgb_params),
        n_jobs=1
    )
    multi_output.estimators_ = estimators
    multi_output.classes_ = [np.array(c) for c in classes_per_output]
    multi_output.n_outputs_ = n_outputs

    # Reconstruct Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', multi_output)
    ])

    print(f"✓ Pipeline loaded from {model_dir}")
    print(f"  - {n_outputs} disease classifiers")
    print(f"  - XGBoost version: {manifest['versions']['xgboost']}")

    return pipeline


def get_manifest(model_dir: Path) -> dict:
    """
    Load and return the manifest for a saved model.

    Args:
        model_dir: Directory containing saved pipeline

    Returns:
        Manifest dictionary
    """
    manifest_path = Path(model_dir) / 'manifest.json'

    with open(manifest_path, 'r') as f:
        return json.load(f)
