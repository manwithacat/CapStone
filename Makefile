# NIH Chest X-Ray Disease Detection - Project Makefile
# Run `make help` to see available targets

PY ?= python

.PHONY: help install lint format typecheck test test-fast test-notebooks pre-commit app clean

help:
	@echo "Usage:"
	@echo "  make install       - install dev dependencies and project package"
	@echo "  make lint          - run Ruff linter"
	@echo "  make format        - run Black formatter"
	@echo "  make typecheck     - run Pyright type checker"
	@echo "  make test          - run notebook tests (excluding slow)"
	@echo "  make test-fast     - run only fast notebook tests"
	@echo "  make test-notebooks - prepare notebooks for testing"
	@echo "  make pre-commit    - run lint and format checks"
	@echo "  make app           - run the Streamlit dashboard locally"
	@echo "  make clean         - remove caches and temp files"

install:
	@echo "Installing development dependencies..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
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
	@echo "  2. Start working: jupyter notebook or make app"
	@echo ""
	@echo "💡 For production deployment, use: pip install -r requirements.txt"

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

pre-commit: lint format
	@echo "✅ Pre-commit checks complete"

app:
	streamlit run app.py

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name ".ruff_cache" -type d -prune -exec rm -rf {} +
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
	@echo "✅ Cleaned up cache files"
