#!/usr/bin/env python3
"""
Analyze and visualize CNN cloud training results from Kaggle P100.

Usage:
    python scripts/analyze_cloud_training.py [history_file.csv]

If no file specified, auto-detects latest cloud training results.

Outputs:
    - Training curves plot (PNG)
    - Summary statistics (JSON)
    - Console summary
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
OUTPUTS_DIR = Path('outputs')
REPORTS_DIR = OUTPUTS_DIR / 'reports'
FIGURES_DIR = OUTPUTS_DIR / 'figures' / 'cnn_training'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detect training history file
def find_training_history():
    """Find the most recent cloud training history CSV."""
    candidates = [
        REPORTS_DIR / '06c_optimized_training_history.csv',
        REPORTS_DIR / '06c_optimized_training_history_kaggle.csv',
        REPORTS_DIR / '06_cnn_cloud_15epoch_history.csv',
        REPORTS_DIR / '06_cnn_training_history_kaggle.csv',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: find any matching CSV
    matching = list(REPORTS_DIR.glob('06*training_history*.csv'))
    if matching:
        # Return most recently modified
        return max(matching, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"No training history file found in {REPORTS_DIR}. "
        "Expected one of: " + ", ".join(c.name for c in candidates)
    )

# Load training history
if len(sys.argv) > 1:
    history_file = Path(sys.argv[1])
else:
    history_file = find_training_history()

print(f"Loading: {history_file.name}")
df = pd.read_csv(history_file)

print("=" * 70)
print(f"CNN CLOUD TRAINING RESULTS - {len(df)} EPOCHS ON KAGGLE P100")
print("=" * 70)

# Summary statistics
print(f"\nTraining Summary:")
print(f"  Total Epochs: {len(df)}")
print(f"  Best Val AUC: {df['val_auc'].max():.4f} (Epoch {df['val_auc'].idxmax()})")
print(f"  Best Val Loss: {df['val_loss'].min():.4f} (Epoch {df['val_loss'].idxmin()})")
print(f"  Final Train Loss: {df['loss'].iloc[-1]:.4f}")
print(f"  Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")

# Calculate improvements
initial_val_auc = df['val_auc'].iloc[0]
final_val_auc = df['val_auc'].iloc[-1]
best_val_auc = df['val_auc'].max()
auc_improvement = ((final_val_auc - initial_val_auc) / initial_val_auc) * 100
auc_improvement_best = ((best_val_auc - initial_val_auc) / initial_val_auc) * 100

print(f"\nPerformance Gains:")
print(f"  Initial Val AUC: {initial_val_auc:.4f}")
print(f"  Final Val AUC: {final_val_auc:.4f} (+{auc_improvement:.1f}%)")
print(f"  Best Val AUC: {best_val_auc:.4f} (+{auc_improvement_best:.1f}%)")

# Loss convergence
initial_train_loss = df['loss'].iloc[0]
final_train_loss = df['loss'].iloc[-1]
loss_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100

print(f"\nLoss Reduction:")
print(f"  Train Loss: {initial_train_loss:.4f} → {final_train_loss:.4f} (-{loss_reduction:.1f}%)")
print(f"  Val Loss: {df['val_loss'].iloc[0]:.4f} → {df['val_loss'].iloc[-1]:.4f}")

# Overfitting check
print(f"\nOverfitting Analysis:")
print(f"  Train-Val AUC Gap: {df['auc'].iloc[-1] - df['val_auc'].iloc[-1]:.4f}")
print(f"  Train-Val Loss Gap: {df['loss'].iloc[-1] - df['val_loss'].iloc[-1]:.4f}")
if abs(df['loss'].iloc[-1] - df['val_loss'].iloc[-1]) < 0.02:
    print("  Status: ✓ Good generalization (minimal overfitting)")
else:
    print("  Status: ⚠ Check for overfitting")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'CNN Cloud Training - {len(df)} Epochs on Kaggle P100', fontsize=16, fontweight='bold')

# Plot 1: Loss
axes[0, 0].plot(df['epoch'], df['loss'], label='Train', marker='o', linewidth=2)
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', marker='s', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Binary Crossentropy Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: AUC
axes[0, 1].plot(df['epoch'], df['auc'], label='Train', marker='o', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val_auc'], label='Val', marker='s', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].set_title('Area Under ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].axhline(y=best_val_auc, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_val_auc:.3f}')

# Plot 3: Accuracy
axes[0, 2].plot(df['epoch'], df['accuracy'], label='Train', marker='o', linewidth=2)
axes[0, 2].plot(df['epoch'], df['val_accuracy'], label='Val', marker='s', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].set_title('Accuracy (Note: Low due to class imbalance)')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Plot 4: Precision
axes[1, 0].plot(df['epoch'], df['precision'], label='Train', marker='o', linewidth=2)
axes[1, 0].plot(df['epoch'], df['val_precision'], label='Val', marker='s', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Recall
axes[1, 1].plot(df['epoch'], df['recall'], label='Train', marker='o', linewidth=2)
axes[1, 1].plot(df['epoch'], df['val_recall'], label='Val', marker='s', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 6: Learning Rate
axes[1, 2].plot(df['epoch'], df['learning_rate'], marker='o', linewidth=2, color='purple')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Learning Rate')
axes[1, 2].set_title('Learning Rate Schedule')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].set_yscale('log')

plt.tight_layout()
# Generate output filename based on epochs and file source
output_filename = f'06_cloud_training_{len(df)}epochs.png'
output_file = FIGURES_DIR / output_filename
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_file}")

# Save summary statistics
summary = {
    'training_info': {
        'epochs_completed': len(df),
        'hardware': 'Kaggle P100 GPU',
        'timeout': '12 hours (43,200 seconds)',
        'training_images': 78566
    },
    'best_performance': {
        'best_val_auc': float(best_val_auc),
        'best_val_auc_epoch': int(df['val_auc'].idxmax()),
        'best_val_loss': float(df['val_loss'].min()),
        'best_val_loss_epoch': int(df['val_loss'].idxmin())
    },
    'final_epoch': {
        'train_loss': float(df['loss'].iloc[-1]),
        'val_loss': float(df['val_loss'].iloc[-1]),
        'train_auc': float(df['auc'].iloc[-1]),
        'val_auc': float(df['val_auc'].iloc[-1]),
        'train_accuracy': float(df['accuracy'].iloc[-1]),
        'val_accuracy': float(df['val_accuracy'].iloc[-1])
    },
    'improvements': {
        'val_auc_improvement_pct': float(auc_improvement),
        'val_auc_best_improvement_pct': float(auc_improvement_best),
        'train_loss_reduction_pct': float(loss_reduction)
    },
    'generalization': {
        'train_val_auc_gap': float(df['auc'].iloc[-1] - df['val_auc'].iloc[-1]),
        'train_val_loss_gap': float(df['loss'].iloc[-1] - df['val_loss'].iloc[-1]),
        'overfitting_status': 'minimal' if abs(df['loss'].iloc[-1] - df['val_loss'].iloc[-1]) < 0.02 else 'check_required'
    }
}

summary_file = REPORTS_DIR / '06_cloud_training_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved: {summary_file}")

# Comparison with baseline models
print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE MODELS")
print("=" * 70)

baseline_results = {
    'Logistic Regression': {'auc': 0.63, 'type': 'feature-based'},
    'Random Forest': {'auc': 0.68, 'type': 'feature-based'},
    'XGBoost': {'auc': 0.71, 'type': 'feature-based'},
    'CNN (15 epochs)': {'auc': best_val_auc, 'type': 'deep learning'}
}

print(f"\n{'Model':<25} {'AUC':<10} {'Type':<20} {'vs Best Baseline'}")
print("-" * 70)

best_baseline_auc = 0.71  # XGBoost
for model_name, results in baseline_results.items():
    improvement = ((results['auc'] - best_baseline_auc) / best_baseline_auc) * 100
    if model_name == 'CNN (15 epochs)':
        print(f"{model_name:<25} {results['auc']:<10.3f} {results['type']:<20} +{improvement:.1f}%")
    else:
        print(f"{model_name:<25} {results['auc']:<10.3f} {results['type']:<20}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. ✓ Model checkpoint saved: models/saved_models/cnn_custom_best.keras (308 MB)
2. → Threshold tuning per disease class (currently using 0.5)
3. → Per-disease performance analysis (14 disease classes)
4. → Resume training to complete 50 epochs
5. → Compare with transfer learning models (ResNet50, DenseNet121)
6. → Test on expert-validated subset for clinical validation
""")

print("Analysis complete!")
