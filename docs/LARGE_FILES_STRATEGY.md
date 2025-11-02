# Large Files Strategy - NIH Chest X-Ray Project

**Problem**: GitHub rejects files >100MB and warns about files >50MB. Trained ML models (308MB+) cannot be pushed to GitHub.

**Solution**: Exclude large files from git, use external storage, and document retrieval methods.

---

## Table of Contents

1. [Current Situation](#current-situation)
2. [Resolution](#resolution)
3. [Prevention Strategy](#prevention-strategy)
4. [Model Distribution Options](#model-distribution-options)
5. [Best Practices](#best-practices)
6. [Recovery Procedures](#recovery-procedures)

---

## Current Situation

### Large Files in This Project

| File Type | Size | Location | Status |
|-----------|------|----------|--------|
| Trained CNN models | 308MB+ | `models/saved_models/*.keras` | **Excluded** |
| Trained CNN models | 308MB+ | `models/saved_models/*.h5` | **Excluded** |
| Kaggle downloads | 308MB+ | `outputs/kaggle_cloud_training/` | **Excluded** |
| Raw X-ray images | ~47GB | `data/raw/images_*/` | **Excluded** |
| Expert labels | ~500MB | `data/raw/expert_labels/` | **Excluded** |

### GitHub Limits

- **Hard limit**: 100MB per file (push will fail)
- **Warning**: 50MB per file (warning during push)
- **Recommended**: Keep repos under 1GB total

### What Happened

On commit `aa25ac9`, two large model files were accidentally added:
- `models/saved_models/cnn_custom_best.h5` (308MB)
- `models/saved_models/cnn_custom_best.keras` (308MB)

**Impact**: Git push to GitHub failed with "file too large" error.

**Resolution**:
1. ✅ Reset commit (`git reset --soft HEAD~1`)
2. ✅ Unstaged large files
3. ✅ Updated `.gitignore` to exclude model files
4. ✅ Will commit without large files

---

## Resolution

### Immediate Fix (Applied)

```bash
# 1. Undo problematic commit (keep changes)
git reset --soft HEAD~1

# 2. Unstage large files
git reset HEAD models/saved_models/cnn_custom_best.h5
git reset HEAD models/saved_models/cnn_custom_best.keras

# 3. Update .gitignore (already done)
# 4. Commit without large files
git add <other files>
git commit -m "feat: add CNN optimization infrastructure (excluding large models)"

# 5. Push successfully
git push origin main
```

### Updated .gitignore

Added comprehensive model file exclusions:

```gitignore
# Machine Learning Models & Artifacts (LARGE FILES)
models/saved_models/*.keras
models/saved_models/*.h5
models/saved_models/*.hdf5
models/saved_models/*.pb
models/saved_models/*.pt
models/saved_models/*.pth
models/saved_models/*.ckpt
models/saved_models/*.weights

# Kaggle downloaded outputs
outputs/kaggle_cloud_training/

# Allow small metadata
!models/saved_models/*.json
!models/saved_models/*.yaml
!models/saved_models/README.md
```

---

## Prevention Strategy

### Git Hooks (Recommended)

Create `.git/hooks/pre-commit` to prevent large files:

```bash
#!/bin/bash
# Pre-commit hook to prevent large files from being committed

MAX_SIZE=50000000  # 50MB in bytes

# Check staged files
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt $MAX_SIZE ]; then
            echo "❌ Error: File $file is too large ($(($size/1024/1024))MB > 50MB)"
            echo "   Large files should not be committed to git."
            echo "   Add to .gitignore or use Git LFS."
            exit 1
        fi
    fi
done
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Automated Checks

Use `git-sizer` to analyze repository size:

```bash
# Install
brew install git-sizer  # macOS
# OR
wget https://github.com/github/git-sizer/releases/download/v1.5.0/git-sizer-1.5.0-linux-amd64.zip

# Run analysis
git-sizer --verbose
```

---

## Model Distribution Options

### Option 1: Kaggle Datasets (RECOMMENDED ✓)

**Pros**:
- Free, unlimited storage for ML datasets
- Already using Kaggle for training
- Easy download via `kaggle datasets download`
- Versioning support

**Cons**:
- Requires Kaggle account
- Public or private (private has visibility limits)

**Implementation**:

```bash
# Create dataset from trained model
kaggle datasets init -p models/saved_models
# Edit dataset-metadata.json
kaggle datasets create -p models/saved_models

# Download when needed
kaggle datasets download <username>/cnn-chest-xray-models -p models/saved_models/
```

**Status**: ✅ Already implemented for training data splits

### Option 2: GitHub Releases (RECOMMENDED ✓)

**Pros**:
- Tied to specific code versions
- 2GB per file limit (plenty for 308MB models)
- Free for public repos
- Easy download via GitHub UI or API

**Cons**:
- Manual upload process
- Not in git history

**Implementation**:

```bash
# Create release with model
gh release create v1.0.0 \
  models/saved_models/cnn_optimized_best.keras \
  --title "CNN Model v1.0 - 50 epochs" \
  --notes "Trained on P100 GPU, Val AUC: 0.792"

# Download when needed
gh release download v1.0.0 \
  --pattern "*.keras" \
  --dir models/saved_models/
```

### Option 3: Git LFS (NOT RECOMMENDED)

**Pros**:
- Models tracked in git history
- Transparent to users (looks like normal git)

**Cons**:
- GitHub LFS limits: 1GB storage, 1GB bandwidth/month (free tier)
- Costs $5/month for 50GB (Pro)
- Complex to set up
- Can hit bandwidth limits quickly

**When to use**: Only if you need version-controlled models AND have budget for LFS.

### Option 4: External Storage (OPTIONAL)

**Options**:
- Google Drive
- Dropbox
- AWS S3
- Hugging Face Hub (ML-specific)

**When to use**: Team sharing, large models >2GB, need public access

---

## Best Practices

### 1. Always Check Before Committing

```bash
# Before committing, check file sizes
git diff --cached --stat | awk '{print $1, $NF}'

# Find large files in staging
git diff --cached --name-only | xargs -I {} du -h {}
```

### 2. Model Versioning

Create a `models/saved_models/README.md`:

```markdown
# Trained Models

Models are NOT stored in git. Download from:

## CNN Custom Model (06c)
- **Version**: v1.0 (50 epochs, optimized)
- **Size**: 308MB
- **Val AUC**: 0.792
- **Download**:
  ```bash
  kaggle kernels output <username>/cnn-optimized-training -p .
  mv cnn_optimized_best.keras models/saved_models/
  ```
  OR
  ```bash
  gh release download v1.0.0 --pattern "*.keras"
  ```

## Baseline Models
- **XGBoost**: `baseline_xgboost_pipeline_v1/` (small, included in git)
- **Random Forest**: Stored on Kaggle Datasets
```

### 3. Documentation

Always document in README where to get large files:

```markdown
## Setup Instructions

1. Clone repository
2. Download trained models (choose one):
   ```bash
   # Option A: From Kaggle
   kaggle kernels output <username>/cnn-optimized-training

   # Option B: From GitHub Releases
   gh release download v1.0.0
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run dashboard: `streamlit run app.py`
```

### 4. Automated Download Scripts

Create `scripts/download_models.sh`:

```bash
#!/bin/bash
# Download trained models from Kaggle

echo "📥 Downloading trained models..."

# Set Kaggle config
export KAGGLE_CONFIG_DIR=".kaggle"

# Download from latest kernel run
kaggle kernels output <username>/cnn-optimized-training \
  --path outputs/kaggle_results

# Copy to models directory
cp outputs/kaggle_results/cnn_optimized_best.keras \
   models/saved_models/

echo "✅ Models downloaded successfully!"
```

---

## Recovery Procedures

### If Large Files Were Already Pushed

**Option 1: BFG Repo-Cleaner (Easiest)**

```bash
# Download BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Remove large files
java -jar bfg-1.14.0.jar --delete-files "*.keras" .
java -jar bfg-1.14.0.jar --delete-files "*.h5" .

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (DESTRUCTIVE!)
git push --force origin main
```

**Option 2: git filter-repo (More Control)**

```bash
# Install
pip install git-filter-repo

# Remove specific files
git filter-repo --path models/saved_models/cnn_custom_best.keras --invert-paths
git filter-repo --path models/saved_models/cnn_custom_best.h5 --invert-paths

# Force push
git push --force origin main
```

**⚠️ WARNING**: Both options rewrite git history. Coordinate with team first!

### If Push Fails with "File Too Large"

```bash
# 1. Check what's too large
git ls-files | xargs ls -lh | sort -k 5 -h | tail -10

# 2. Find in git history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {if ($3 > 50000000) print $3/1024/1024 " MB", $4}'

# 3. Remove from last commit
git reset --soft HEAD~1
git reset HEAD <large-file>
git commit -m "..."

# 4. Update .gitignore
echo "<large-file-pattern>" >> .gitignore
git add .gitignore
git commit --amend --no-edit
```

---

## Summary

**Current Status**: ✅ Fixed
- Large model files excluded from git
- Updated `.gitignore` prevents future issues
- Documentation added for model distribution

**Recommended Workflow**:
1. Train models on Kaggle (GPU access)
2. Download via `kaggle kernels output`
3. Use models locally (excluded from git)
4. Share via GitHub Releases or Kaggle Datasets
5. Document download instructions in README

**Never Commit**:
- ❌ Trained model files (*.keras, *.h5, *.pt)
- ❌ Raw image datasets (GB-sized)
- ❌ Temporary training outputs
- ❌ Kaggle API credentials

**Always Commit**:
- ✅ Model metadata (JSON, YAML configs)
- ✅ Training scripts and notebooks
- ✅ Small CSV files (<50MB)
- ✅ Documentation and READMEs
- ✅ Code and tests

---

**Last Updated**: November 1, 2025
**Status**: Active Prevention Strategy
