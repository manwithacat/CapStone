"""
Pipeline factory for image-based XGBoost classification.
"""
from typing import Optional

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import GLCMTexture, HOGFeatures, StatisticalPixels


def build_pipeline(config: dict) -> Pipeline:
    """
    Build a scikit-learn Pipeline for image → features → XGBoost.

    Pipeline structure:
        1. FeatureUnion: Parallel feature extraction (HOG + GLCM + Stats)
        2. StandardScaler: Feature standardization
        3. MultiOutputClassifier(XGBoost): Multi-label classification

    Args:
        config: Configuration dictionary with keys:
            - image: {img_size: tuple}
            - hog: {orientations, pixels_per_cell, cells_per_block}
            - glcm: {levels, distances, angles, properties}
            - stats: {canny_threshold1, canny_threshold2}
            - xgb: {n_estimators, max_depth, learning_rate, etc.}
            - runtime: {n_jobs, random_state}

    Returns:
        Fitted Pipeline ready for fit/predict on image paths
    """
    # Extract configs
    img_config = config.get('image', {})
    hog_config = config.get('hog', {})
    glcm_config = config.get('glcm', {})
    stats_config = config.get('stats', {})
    xgb_config = config.get('xgb', {})
    runtime_config = config.get('runtime', {})

    # Shared image size
    img_size = tuple(img_config.get('img_size', [128, 128]))

    # Feature extractors
    hog_extractor = HOGFeatures(
        img_size=img_size,
        orientations=hog_config.get('orientations', 9),
        pixels_per_cell=tuple(hog_config.get('pixels_per_cell', [8, 8])),
        cells_per_block=tuple(hog_config.get('cells_per_block', [2, 2])),
        block_norm=hog_config.get('block_norm', 'L2-Hys')
    )

    glcm_extractor = GLCMTexture(
        img_size=img_size,
        levels=glcm_config.get('levels', 8),
        distances=tuple(glcm_config.get('distances', [1])),
        angles=tuple(glcm_config.get('angles', [0, 0.785398, 1.570796, 2.356194])),  # π/4, π/2, 3π/4
        properties=tuple(glcm_config.get('properties', ['contrast', 'homogeneity', 'energy', 'correlation']))
    )

    stats_extractor = StatisticalPixels(
        img_size=img_size,
        canny_threshold1=stats_config.get('canny_threshold1', 50),
        canny_threshold2=stats_config.get('canny_threshold2', 150)
    )

    # Parallel feature extraction
    n_jobs = runtime_config.get('n_jobs', -1)
    feature_union = FeatureUnion(
        transformer_list=[
            ('hog', hog_extractor),
            ('glcm', glcm_extractor),
            ('stats', stats_extractor)
        ],
        n_jobs=n_jobs
    )

    # Preprocessor: features + scaling
    preprocessor = Pipeline([
        ('features', feature_union),
        ('scaler', StandardScaler())
    ])

    # XGBoost multi-output classifier
    xgb_estimator = XGBClassifier(
        n_estimators=xgb_config.get('n_estimators', 100),
        max_depth=xgb_config.get('max_depth', 6),
        learning_rate=xgb_config.get('learning_rate', 0.1),
        random_state=runtime_config.get('random_state', 42),
        n_jobs=runtime_config.get('n_jobs', -1),
        eval_metric=xgb_config.get('eval_metric', 'logloss'),
        tree_method=xgb_config.get('tree_method', 'hist'),
        verbosity=xgb_config.get('verbosity', 0)
    )

    multi_output = MultiOutputClassifier(xgb_estimator, n_jobs=1)  # XGB handles parallelism internally

    # Full pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', multi_output)
    ])

    return pipeline
