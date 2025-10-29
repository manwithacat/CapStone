Refactor Spec: Reproducible Image-Features → XGBoost Pipeline (v1)

1) Objectives
	•	Replace ad-hoc notebook functions with a scikit-learn Pipeline that:
	•	Accepts image paths as X.
	•	Extracts HOG + GLCM + statistical features in a composable, tunable way.
	•	Scales features and trains multi-output XGBoost classifiers.
	•	Enable reproducible training/evaluation (CV, seeds, configs).
	•	Standardise persistence:
	•	Preprocessing saved with joblib.
	•	XGBoost trees saved with save_model() (JSON/UBJ), not pickle.
	•	Provide programmatic + CLI entry points for train/eval/infer.
	•	Prepare for future hyperparameter search and experiment tracking.

2) Scope
	•	Code extraction into a small Python package/module.
	•	Feature transformers as sklearn components.
	•	End-to-end Pipeline assembly.
	•	Model saving/loading utilities.
	•	Basic CV/training loop + metrics.
	•	Minimal CLI wrappers.
	•	Tests for transformers, pipeline fit/predict, and save/load round-trip.

3) Deliverables
	•	/ml/ (or /src/) package containing:
	•	features/:
	•	hog.py – HOGFeatures(BaseEstimator, TransformerMixin)
	•	glcm.py – GLCMTexture(BaseEstimator, TransformerMixin)
	•	stats.py – StatisticalPixels(BaseEstimator, TransformerMixin)
	•	io.py – shared _load_gray(path, img_size)
	•	pipeline.py – constructs Pipeline(prep -> MultiOutput(XGBClassifier))
	•	persist.py – save_preprocessor, save_boosters, load_pipeline
	•	train.py – training loop, CV, metrics, manifests
	•	infer.py – batch inference from image paths
	•	config.py – load/validate YAML (or .toml) config into dict
	•	cli.py (Typer/argparse) with commands: train, evaluate, predict, export.
	•	configs/experiment.yaml – parameters (paths, HOG/GLCM/XGB, CV).
	•	requirements.txt (or pyproject.toml) with pinned versions.
	•	tests/ with pytest:
	•	unit tests for each transformer,
	•	pipeline fit/predict on a tiny fixture set,
	•	save→load→predict equality tests.
	•	README.md with usage.

4) Data Flow (high level)

image_paths (list[str|Path]) ──> [HOG | GLCM | Stats] ── FeatureUnion ──> StandardScaler ──> MultiOutput(XGBClassifier)
                                       │                                   │
                                       └──────── tunable via config ───────┘


5) Components & Interfaces

5.1 Transformers

All transform classes must:
	•	Inherit BaseEstimator, TransformerMixin.
	•	Accept constructor kwargs for all tunables (to support GridSearchCV).
	•	Expect X as a 1D list/array of image paths; return np.ndarray [n_samples, n_features].
	•	Raise a clear ValueError if any image fails to load (do not silently drop rows).

Common params
	•	img_size: tuple[int,int] = (128,128) across transformers.
	•	HOG: orientations, pixels_per_cell, cells_per_block.
	•	GLCM: levels, distances, angles (default distances (1,), angles [0, π/4, π/2, 3π/4]), internal down-quantization.
	•	Stats: canny1, canny2.

5.2 Preprocessor
	•	FeatureUnion([("hog", HOGFeatures(...)), ("glcm", GLCMTexture(...)), ("stats", StatisticalPixels(...))], n_jobs=CONFIG.n_jobs)
	•	StandardScaler(with_mean=True, with_std=True)
	•	Optional joblib.Memory cache hook on the preprocessor pipeline for repeated transforms.

5.3 Estimator
	•	MultiOutputClassifier(XGBClassifier(...))
	•	Required params: n_estimators, max_depth, learning_rate, random_state, n_jobs, eval_metric.
	•	All seeds/random_state pulled from config for determinism.

5.4 Pipeline
	•	Pipeline([("prep", preprocessor), ("clf", MultiOutputClassifier(...))])
	•	Expose factory build_pipeline(cfg: dict) -> Pipeline.

6) Persistence (Best Practice)

Do not pickle the full pipeline.
	•	Preprocessor: joblib.dump(model.named_steps["prep"], preprocessor_path)
	•	Boosters: iterate model.named_steps["clf"].estimators_, call est.save_model(outdir/f"class_{i}.json")
	•	Manifest (manifest.json):
	•	n_outputs, classes_ (per target), xgb_params (shallow), library_versions (xgboost, scikit-learn, numpy, opencv-python, scikit-image)
	•	feature_params snapshot (HOG/GLCM/Stats)
	•	img_size, scaler_config
	•	optional: training metadata (dataset hash, date, metrics, early stopping)

Loading
	•	prep = joblib.load(preprocessor_path)
	•	Recreate XGBClassifier(**xgb_params) per output; clf.load_model(json_path)
	•	Rehydrate MultiOutputClassifier (set estimators_, classes_, n_outputs_)
	•	Return a full Pipeline([("prep", prep), ("clf", mo)])

7) Configuration
	•	Single YAML (or TOML) file with:
	•	Paths: train/val/test lists or glob roots.
	•	Image: img_size.
	•	HOG params, GLCM params, Stats params.
	•	XGB params.
	•	CV params: cv_folds, scoring, n_jobs.
	•	Runtime: seeds, n_jobs, cache dir.
	•	Output dirs and run name/version.

Validation via a lightweight schema (pydantic optional).

8) Training & Evaluation
	•	Minimal routine:
	•	Load datasets: X_train_paths, y_train, etc.
	•	Build pipeline from config.
	•	fit on train.
	•	Evaluate on val/test:
	•	For multi-label: per-label ROC-AUC, precision/recall, macro/micro averages.
	•	For multi-class per output: accuracy, macro-F1, ROC-AUC(ovr) if applicable.
	•	Persist preprocessor + boosters + manifest.
	•	Emit a metrics.json and a short REPORT.md.
	•	Optional CV:
	•	StratifiedKFold per output if feasible, otherwise standard KFold with caution.
	•	Report mean±std of chosen metrics.

9) Hyperparameter Tuning (Phase 2 ready)
	•	Make all HOG/GLCM/Stats/XGB params grid-/random-searchable.
	•	Provide example grid in config (commented).
	•	Ensure transformers are deterministic across runs given random_state.

10) Reproducibility
	•	Pin library versions.
	•	Capture random_state everywhere (XGB + any randomness in future transforms).
	•	Save manifest.json + requirements.txt snapshot (or pip freeze) alongside artefacts.
	•	Optionally write a small DATASET.md with counts and class distribution.

11) CLI
	•	python -m project.cli train --config configs/experiment.yaml
	•	python -m project.cli evaluate --model-dir artifacts/baseline_xgb_img_v1 --data val.csv
	•	python -m project.cli predict --model-dir ... --inputs paths.txt --out preds.csv
	•	python -m project.cli export --model-dir ... (prints manifest, params)

(Use Typer or argparse. All commands non-interactive, exit codes on failure.)

12) Logging
	•	Use logging (INFO for progress; DEBUG for shapes and timing).
	•	Log feature matrix shapes, per-stage timings, and cache hits if joblib.Memory enabled.

13) Tests (pytest)
	•	Transformers: given 3–5 tiny test images, assert transform shape and finiteness.
	•	Pipeline: fit + predict round-trip on toy set.
	•	Persistence: save, load, and assert identical predictions on a fixed batch.
	•	CLI: smoke tests with temporary dirs and fixtures.

14) Performance Notes
	•	FeatureUnion(n_jobs=…) for parallel extractor branches.
	•	IO can bottleneck; consider:
	•	optional image read caching (LRU in process),
	•	or on-disk joblib.Memory for repeated transforms in CV.
	•	Keep inputs as paths to avoid ballooning memory.

15) Security/Robustness
	•	Validate paths exist; fail fast on missing/corrupt images.
	•	Disallow silent NaN/Inf in features; raise with actionable message.
	•	No untrusted pickle loads; only joblib for our own artefacts + XGB JSON.

16) Acceptance Criteria
	•	train produces:
	•	preprocessor.joblib
	•	class_*.json boosters
	•	manifest.json with classes and params
	•	metrics.json and REPORT.md
	•	predict on the loaded model equals predictions from in-memory model within a tolerance of zero (deterministic).
	•	Tests pass locally (pytest -q), CI green.
	•	README shows a runnable example end-to-end.

17) Migration Plan
	1.	Lift notebook functions into features/ transformers with identical logic.
	2.	Implement pipeline.build_pipeline(cfg).
	3.	Implement persist.save_* and persist.load_pipeline.
	4.	Replace notebook training calls with cli train.
	5.	Add metrics reporting and manifests.
	6.	Wire minimal tests; pin versions; document.

⸻

Notes for the developer
	•	Prefer XGBoost model.json/model.ubj over pickling the sklearn wrapper.
	•	Keep classes_ per output in the manifest to avoid label drift.
	•	Expose img_size in a single source of truth (config); pass to all transformers.
	•	If early stopping is added later, persist eval history in the manifest.