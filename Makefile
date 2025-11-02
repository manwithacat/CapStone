# NIH Chest X-Ray Disease Detection - Project Makefile
# Run `make help` to see available targets

PY ?= python

.PHONY: help install lint format typecheck test test-fast test-notebooks pre-commit app clean \
        mlflow-start mlflow-stop mlflow-ui mlflow-test mlflow-compare mlflow-clean

help:
	@echo "Usage:"
	@echo ""
	@echo "Development:"
	@echo "  make install       - install dev dependencies and project package"
	@echo "  make lint          - run Ruff linter"
	@echo "  make format        - run Black formatter"
	@echo "  make typecheck     - run Pyright type checker"
	@echo "  make test          - run notebook tests (excluding slow)"
	@echo "  make test-fast     - run only fast notebook tests"
	@echo "  make test-notebooks - prepare notebooks for testing"
	@echo "  make pre-commit    - run lint and format checks"
	@echo "  make clean         - remove caches and temp files"
	@echo ""
	@echo "Application:"
	@echo "  make app           - run the Streamlit dashboard locally"
	@echo ""
	@echo "MLflow Experiment Tracking:"
	@echo "  make mlflow-start  - start MLflow UI server (http://localhost:5001)"
	@echo "  make mlflow-stop   - stop MLflow UI server"
	@echo "  make mlflow-ui     - open MLflow UI in browser"
	@echo "  make mlflow-test   - run MLflow tracking test"
	@echo "  make mlflow-compare - compare experiments (top 5 runs)"
	@echo "  make mlflow-clean  - clean MLflow experiment data (WARNING: deletes all!)"

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

# ============================================================================
# MLflow Experiment Tracking Commands
# ============================================================================

mlflow-start:
	@echo "🚀 Starting MLflow UI server..."
	@echo ""
	@if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "⚠️  MLflow is already running on port 5001"; \
		echo "   Access at: http://localhost:5001"; \
	else \
		./scripts/mlflow_start.sh & \
		sleep 3; \
		if curl -s http://localhost:5001 >/dev/null 2>&1; then \
			echo "✅ MLflow UI started successfully!"; \
			echo "   Access at: http://localhost:5001"; \
		else \
			echo "❌ Failed to start MLflow server"; \
			echo "   Try manually: ./scripts/mlflow_start.sh"; \
		fi; \
	fi

mlflow-stop:
	@echo "🛑 Stopping MLflow server..."
	@./scripts/mlflow_stop.sh

mlflow-ui:
	@echo "📊 Opening MLflow UI in browser..."
	@if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then \
		open http://localhost:5001 || xdg-open http://localhost:5001 || echo "Open manually: http://localhost:5001"; \
	else \
		echo "⚠️  MLflow server is not running"; \
		echo "   Start it with: make mlflow-start"; \
	fi

mlflow-test:
	@echo "🧪 Testing MLflow tracking..."
	@echo ""
	@$(PY) test_mlflow_simple.py
	@echo ""
	@echo "✨ View results: make mlflow-ui"

mlflow-compare:
	@echo "📊 Comparing experiments..."
	@echo ""
	@if [ -d mlruns ]; then \
		$(PY) scripts/mlflow_compare.py --experiment cnn-custom --top 5 2>/dev/null || \
		echo "No experiments found yet. Track some experiments first!"; \
	else \
		echo "No MLflow data found. Run some experiments first."; \
	fi

mlflow-clean:
	@echo "⚠️  WARNING: This will delete ALL MLflow experiment data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirmation
	@rm -rf mlruns/ mlflow/mlflow.db mlflow/.mlflow_server.pid
	@echo "✅ MLflow data cleaned"
	@echo "   Run 'make mlflow-test' to create new test data"
