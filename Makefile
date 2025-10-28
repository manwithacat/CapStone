# Simple project workflow — run `make help` to see targets.

PY ?= python

.PHONY: help install lint format typecheck check-pylance test test-fast test-all test-notebooks pre-commit etl features pipeline app clean

help:
	@echo "Usage:"
	@echo "  make install       - install dev tools and project deps"
	@echo "  make lint          - run Ruff lint only"
	@echo "  make format        - run Black format only"
	@echo "  make typecheck     - run Pyright type checker (investigate Pylance errors)"
	@echo "  make check-pylance  - diagnose Pylance/import configuration issues"
	@echo "  make test          - run notebook tests (excluding slow)"
	@echo "  make test-fast     - run only fast notebooks"
	@echo "  make test-all      - run ALL notebooks (including slow/data downloads)"
	@echo "  make test-notebooks - prepare notebooks for testing"
	@echo "  make pre-commit    - run lint and format (recommended before commit)"
	@echo "  make etl           - run ETL pipeline (raw → cleaned.parquet)"
	@echo "  make features      - run feature engineering (cleaned → features.parquet)"
	@echo "  make pipeline      - run full pipeline (ETL + features)"
	@echo "  make app           - run the Streamlit dashboard locally"
	@echo "  make clean         - remove caches and temp files"

install:
	@echo "Installing project dependencies..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo ""
	@echo "Installing project package in editable mode..."
	$(PY) -m pip install -e .
	@echo ""
	@echo "Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		cp .env.example .env && \
		echo "✓ Created .env file from .env.example"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Reload VS Code window (Cmd+Shift+P → 'Reload Window')"
	@echo "  2. Run: make check-pylance (to verify setup)"
	@echo "  3. Start working: jupyter notebook or make app"

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

check-pylance:
	@echo "Running Pylance diagnostic script..."
	@echo ""
	$(PY) scripts/check_pylance.py

test-notebooks:
	@echo "Preparing notebooks for testing..."
	@echo ""
	$(PY) scripts/prepare_notebooks_for_testing.py
	@echo ""
	@echo "✓ Notebooks prepared"

test:
	@echo "Running notebook tests (excluding slow)..."
	@echo ""
	pytest -m "not slow and not data_download" --nbmake-timeout=300
	@echo ""
	@echo "✅ Notebook tests passed"

test-fast:
	@echo "Running fast notebook tests only..."
	@echo ""
	pytest jupyter_notebooks/02_exploratory_data_analysis.ipynb jupyter_notebooks/04_hypothesis_testing.ipynb
	@echo ""
	@echo "✅ Fast tests passed"

test-all:
	@echo "Running ALL notebook tests (including slow)..."
	@echo ""
	@echo "⚠️  This may take a long time and download large datasets!"
	@echo ""
	pytest
	@echo ""
	@echo "✅ All notebook tests passed"

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
