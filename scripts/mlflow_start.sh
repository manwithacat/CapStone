#!/bin/bash
# Start MLflow tracking server
#
# Usage:
#   ./scripts/mlflow_start.sh [port]
#
# Default port: 5001 (5000 conflicts with macOS AirPlay Receiver)
# Access UI at: http://localhost:5001

set -e

# Detect project root (script is in PROJECT_ROOT/scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PORT=${1:-5001}
BACKEND_STORE="sqlite:///$PROJECT_ROOT/mlflow.db"
ARTIFACT_ROOT="$PROJECT_ROOT/mlruns"

echo "🚀 Starting MLflow Tracking Server (SQLite Backend)"
echo "======================================================"
echo ""
echo "Configuration:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Port: $PORT"
echo "  Backend Store: $BACKEND_STORE"
echo "  Artifact Root: $ARTIFACT_ROOT"
echo ""

# Create directories if they don't exist
mkdir -p "$PROJECT_ROOT/mlflow/artifacts"
mkdir -p "$PROJECT_ROOT/mlruns"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $PORT is already in use"
    echo ""
    read -p "Kill existing process and restart? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing process on port $PORT..."
        lsof -ti:$PORT | xargs kill -9
        sleep 1
    else
        echo "Aborted. Use a different port: ./scripts/mlflow_start.sh <port>"
        exit 1
    fi
fi

echo "Starting server..."
echo ""

# Start MLflow server
mlflow server \
    --backend-store-uri "$BACKEND_STORE" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port $PORT \
    &

# Get PID
MLFLOW_PID=$!

# Wait a moment for server to start
sleep 2

# Check if server started successfully
if ps -p $MLFLOW_PID > /dev/null; then
    echo "✅ MLflow server started successfully!"
    echo ""
    echo "📊 Access the UI at:"
    echo "   http://localhost:$PORT"
    echo ""
    echo "🛑 To stop the server:"
    echo "   kill $MLFLOW_PID"
    echo "   OR press Ctrl+C"
    echo ""
    echo "Server is running in the background (PID: $MLFLOW_PID)"
    echo ""

    # Save PID to file
    echo $MLFLOW_PID > "$PROJECT_ROOT/mlflow/.mlflow_server.pid"
    echo "PID saved to: $PROJECT_ROOT/mlflow/.mlflow_server.pid"

    # Wait for user interrupt
    wait $MLFLOW_PID
else
    echo "❌ Failed to start MLflow server"
    exit 1
fi
