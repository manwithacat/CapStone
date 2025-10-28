# Simple project workflow — run `make help` to see targets.

PY ?= python

.PHONY: help install lint format typecheck pre-commit etl features pipeline app clean

help:
	@echo "Usage:"
	@echo "  make install    - install dev tools and project deps"
	@echo "  make lint       - run Ruff lint only"
	@echo "  make format     - run Black format only"
	@echo "  make typecheck  - run Pyright type checker (investigate Pylance errors)"
	@echo "  make pre-commit - run lint and format (recommended before commit)"
	@echo "  make etl        - run ETL pipeline (raw → cleaned.parquet)"
	@echo "  make features   - run feature engineering (cleaned → features.parquet)"
	@echo "  make pipeline   - run full pipeline (ETL + features)"
	@echo "  make app        - run the Streamlit dashboard locally"
	@echo "  make clean      - remove caches and temp files"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check .

format:
	black .

typecheck:
	@echo "Running Pyright type checker..."
	@echo "NOTE: This is the same engine that powers VS Code Pylance"
	@echo ""
	pyright src/ || true
	@echo ""
	@echo "💡 TIP: For full Pylance integration in VS Code:"
	@echo "   1. Reload VS Code window (Cmd+Shift+P → 'Reload Window')"
	@echo "   2. Check .vscode/settings.json has src/ in extraPaths"
	@echo "   3. Install package in editable mode: pip install -e ."

pre-commit: lint format
	@echo "✅ Pre-commit checks complete"

etl:
	$(PY) src/etl.py

features:
	$(PY) src/features.py

pipeline: etl features
	@echo "✅ Full data pipeline complete"

app:
	streamlit run app.py

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name ".ruff_cache" -type d -prune -exec rm -rf {} +
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
