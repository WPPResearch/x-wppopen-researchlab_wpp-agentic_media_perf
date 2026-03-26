"""
Test the checkpoint/experiment tracking system.
"""

from src.utils.experiment_tracker import ExperimentTracker, create_experiment_summary
import numpy as np

print("Testing Experiment Tracker...")
print("=" * 70)

# Test the experiment tracker
with ExperimentTracker("test_experiment") as tracker:
    # Save parameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model": "Ridge",
        "embedding_model": "all-mpnet-base-v2",
    }
    tracker.save_params(params)

    # Simulate training
    tracker.log("Starting training...")
    tracker.log("Epoch 1/10 - Loss: 0.5")
    tracker.log("Epoch 2/10 - Loss: 0.3")
    tracker.log("Epoch 3/10 - Loss: 0.2")

    # Save training history
    history = {
        "train_loss": [0.5, 0.3, 0.2],
        "val_loss": [0.6, 0.4, 0.25],
    }
    tracker.save_training_history(history)

    # Save metrics
    metrics = {
        "MAE": 1.23,
        "RMSE": 1.56,
        "R2": 0.42,
    }
    tracker.save_metrics(metrics, stage="test")

    # Save sample predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    tracker.save_predictions(y_true, y_pred, stage="test")

    # Save summary
    summary = create_experiment_summary("test_experiment", params, metrics, "Ridge")
    tracker.save_summary(summary)

    tracker.log("Done!")

print("\n" + "=" * 70)
print(f"✅ Experiment saved to: {tracker.experiment_dir}")
print("=" * 70)

# Show what was saved
print("\nSaved files:")
import os

for root, dirs, files in os.walk(tracker.experiment_dir):
    level = root.replace(str(tracker.experiment_dir), "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

print("\n" + "=" * 70)
print("You can now run your training scripts and they will save checkpoints!")
print("Example: python src/training/train_likes_regression.py")
print("=" * 70)
