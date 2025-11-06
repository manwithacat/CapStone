#!/bin/bash
# Import MLflow runs from Kaggle output downloads
#
# Usage:
#   ./scripts/import_kaggle_mlflow.sh <path_to_kaggle_output.zip>
#   OR
#   ./scripts/import_kaggle_mlflow.sh <path_to_extracted_mlruns_dir>
#
# Example:
#   ./scripts/import_kaggle_mlflow.sh ~/Downloads/kaggle_output.zip
#   ./scripts/import_kaggle_mlflow.sh ~/Downloads/kaggle_output/mlruns

set -e

if [ $# -eq 0 ]; then
    echo "❌ Error: No input file or directory specified"
    echo ""
    echo "Usage:"
    echo "  ./scripts/import_kaggle_mlflow.sh <kaggle_output.zip>"
    echo "  ./scripts/import_kaggle_mlflow.sh <path/to/mlruns/>"
    echo ""
    exit 1
fi

INPUT="$1"

echo "🔍 Importing Kaggle MLflow runs..."
echo "=================================="
echo ""

# Check if input is a zip file
if [[ "$INPUT" == *.zip ]]; then
    echo "📦 Detected zip file: $INPUT"

    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    echo "📂 Extracting to temporary directory: $TEMP_DIR"

    # Extract zip
    unzip -q "$INPUT" -d "$TEMP_DIR"

    # Find mlruns directory
    MLRUNS_SOURCE=$(find "$TEMP_DIR" -type d -name "mlruns" | head -n 1)

    if [ -z "$MLRUNS_SOURCE" ]; then
        echo "❌ Error: No mlruns directory found in $INPUT"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "✓ Found mlruns directory: $MLRUNS_SOURCE"

elif [ -d "$INPUT" ]; then
    echo "📂 Detected directory: $INPUT"

    # Check if this IS the mlruns directory
    if [[ "$(basename "$INPUT")" == "mlruns" ]]; then
        MLRUNS_SOURCE="$INPUT"
    else
        # Look for mlruns inside the provided directory
        MLRUNS_SOURCE=$(find "$INPUT" -type d -name "mlruns" | head -n 1)

        if [ -z "$MLRUNS_SOURCE" ]; then
            echo "❌ Error: No mlruns directory found in $INPUT"
            exit 1
        fi
    fi

    echo "✓ Using mlruns directory: $MLRUNS_SOURCE"

else
    echo "❌ Error: $INPUT is not a valid file or directory"
    exit 1
fi

# Define destination
MLRUNS_DEST="./mlruns"

echo ""
echo "📋 Import Summary:"
echo "  Source: $MLRUNS_SOURCE"
echo "  Destination: $MLRUNS_DEST"
echo ""

# Check if destination exists
if [ ! -d "$MLRUNS_DEST" ]; then
    echo "⚠️  Local mlruns directory does not exist"
    read -p "Create $MLRUNS_DEST and copy Kaggle runs? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$MLRUNS_DEST"
        echo "✓ Created $MLRUNS_DEST"
    else
        echo "Aborted"
        [ -n "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
        exit 1
    fi
fi

echo "🔄 Merging Kaggle runs into local MLflow tracking..."
echo ""

# Copy experiment directories (avoiding conflicts)
for exp_dir in "$MLRUNS_SOURCE"/*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")

        # Skip .trash directory
        if [ "$exp_name" == ".trash" ]; then
            continue
        fi

        if [ -d "$MLRUNS_DEST/$exp_name" ]; then
            echo "  📁 $exp_name: Merging with existing experiment..."

            # Copy run directories
            for run_dir in "$exp_dir"/*; do
                if [ -d "$run_dir" ]; then
                    run_id=$(basename "$run_dir")

                    if [ -d "$MLRUNS_DEST/$exp_name/$run_id" ]; then
                        echo "     ⚠️  Run $run_id already exists, skipping"
                    else
                        cp -r "$run_dir" "$MLRUNS_DEST/$exp_name/"
                        echo "     ✓ Imported run $run_id"
                    fi
                fi
            done

            # Update meta.yaml if needed
            if [ -f "$exp_dir/meta.yaml" ]; then
                cp -f "$exp_dir/meta.yaml" "$MLRUNS_DEST/$exp_name/"
            fi
        else
            echo "  📁 $exp_name: Creating new experiment..."
            cp -r "$exp_dir" "$MLRUNS_DEST/"
            echo "     ✓ Imported all runs"
        fi
    fi
done

echo ""
echo "✅ Import complete!"
echo ""

# Clean up temp directory if we created one
if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "🧹 Cleaned up temporary files"
    echo ""
fi

# Show imported runs
echo "📊 Imported Runs Summary:"
echo "========================="
echo ""

python3 << 'PYTHON_EOF'
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all experiments
experiments = client.search_experiments()

# Count runs with platform=kaggle tag
kaggle_runs = []
for exp in experiments:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.platform = 'kaggle'"
    )
    kaggle_runs.extend(runs)

if kaggle_runs:
    print(f"Found {len(kaggle_runs)} Kaggle run(s):\n")
    for run in kaggle_runs:
        exp = client.get_experiment(run.info.experiment_id)
        tags = run.data.tags
        print(f"  🚀 {run.info.run_name or run.info.run_id[:8]}")
        print(f"     Experiment: {exp.name}")
        print(f"     Status: {run.info.status}")
        print(f"     Platform: {tags.get('platform', 'N/A')}")
        print(f"     GPU: {tags.get('gpu', 'N/A')}")
        print(f"     Source: {tags.get('mlflow.source.name', 'N/A')}")
        print()
else:
    print("No Kaggle runs found (runs should be tagged with 'platform=kaggle')")
    print()
PYTHON_EOF

echo ""
echo "💡 View runs in MLflow UI:"
echo "   make mlflow-ui"
echo "   OR"
echo "   ./scripts/mlflow_start.sh"
echo ""
echo "   Then open: http://localhost:5001"
echo "   Filter by tag: platform = 'kaggle'"
echo ""
