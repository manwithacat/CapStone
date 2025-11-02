#!/bin/bash
# Stop MLflow tracking server

set -e

PID_FILE="mlflow/.mlflow_server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")

    if ps -p $PID > /dev/null 2>&1; then
        echo "🛑 Stopping MLflow server (PID: $PID)..."
        kill $PID
        rm "$PID_FILE"
        echo "✅ MLflow server stopped"
    else
        echo "⚠️  No process found with PID: $PID"
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. Checking for MLflow processes..."

    # Find MLflow processes
    PIDS=$(pgrep -f "mlflow server" || true)

    if [ -n "$PIDS" ]; then
        echo "Found MLflow processes: $PIDS"
        read -p "Kill these processes? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$PIDS" | xargs kill
            echo "✅ MLflow processes stopped"
        fi
    else
        echo "No MLflow server processes found"
    fi
fi
