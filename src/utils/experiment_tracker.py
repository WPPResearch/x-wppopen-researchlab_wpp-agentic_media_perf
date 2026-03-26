"""
Experiment tracking utility for saving models, parameters, and logs.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ExperimentTracker:
    """Track experiments with timestamps, parameters, and results."""

    def __init__(self, experiment_name: str, base_dir: str = "checkpoints"):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment (e.g., "ridge_regression", "transformer")
            base_dir: Base directory for checkpoints
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)

        # Create experiment folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize log file
        self.log_file = self.logs_dir / "training.log"
        self.metrics_file = self.logs_dir / "metrics.json"

        # Store parameters
        self.params = {}
        self.metrics = {}

        print(f"Experiment directory: {self.experiment_dir}")

    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

    def save_params(self, params: Dict[str, Any]):
        """Save experiment parameters."""
        self.params = params
        params_file = self.experiment_dir / "params.json"

        # Convert non-serializable objects to strings
        serializable_params = {}
        for key, value in params.items():
            try:
                json.dumps(value)
                serializable_params[key] = value
            except (TypeError, ValueError):
                serializable_params[key] = str(value)

        with open(params_file, "w") as f:
            json.dump(serializable_params, f, indent=2)

        self.log(f"Parameters saved to {params_file}")

    def save_model(
        self, model, model_name: str = "best_model", is_pytorch: bool = False
    ):
        """
        Save model to disk.

        Args:
            model: Model object (sklearn, pytorch, etc.)
            model_name: Name for the saved model
            is_pytorch: If True, use torch.save, else use pickle
        """
        if is_pytorch:
            import torch

            model_path = self.models_dir / f"{model_name}.pt"
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        else:
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        self.log(f"Model saved to {model_path}")
        return model_path

    def save_metrics(self, metrics: Dict[str, float], stage: str = "test"):
        """
        Save metrics for a stage (train/val/test).

        Args:
            metrics: Dictionary of metric names and values
            stage: Stage name (train/val/test)
        """
        if stage not in self.metrics:
            self.metrics[stage] = {}

        self.metrics[stage].update(metrics)

        # Save to JSON
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Log metrics
        self.log(f"{stage.upper()} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")

    def save_predictions(self, y_true, y_pred, stage: str = "test"):
        """Save predictions to CSV."""
        import pandas as pd

        predictions_file = self.logs_dir / f"{stage}_predictions.csv"
        df = pd.DataFrame(
            {
                "y_true": y_true.flatten() if hasattr(y_true, "flatten") else y_true,
                "y_pred": y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred,
            }
        )
        df.to_csv(predictions_file, index=False)
        self.log(f"Predictions saved to {predictions_file}")

    def save_training_history(self, history: Dict[str, list]):
        """
        Save training history (e.g., loss curves).

        Args:
            history: Dictionary with lists of values (e.g., {'train_loss': [...], 'val_loss': [...]})
        """
        history_file = self.logs_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        self.log(f"Training history saved to {history_file}")

    def save_summary(self, summary_text: str):
        """Save experiment summary."""
        summary_file = self.experiment_dir / "SUMMARY.txt"
        with open(summary_file, "w") as f:
            f.write(summary_text)
        self.log(f"Summary saved to {summary_file}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.log(f"ERROR: {exc_type.__name__}: {exc_val}")
        else:
            self.log("Experiment completed successfully")


def create_experiment_summary(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    best_model_name: str = None,
) -> str:
    """
    Create a formatted summary of the experiment.

    Args:
        experiment_name: Name of experiment
        params: Parameters used
        metrics: Final metrics
        best_model_name: Name of best model

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 70)
    summary.append(f"EXPERIMENT SUMMARY: {experiment_name}")
    summary.append("=" * 70)
    summary.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary.append("\n\nPARAMETERS:")
    summary.append("-" * 70)
    for key, value in params.items():
        summary.append(f"  {key}: {value}")

    summary.append("\n\nRESULTS:")
    summary.append("-" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            summary.append(f"  {key}: {value:.4f}")
        else:
            summary.append(f"  {key}: {value}")

    if best_model_name:
        summary.append(f"\n\nBest Model: {best_model_name}")

    summary.append("\n" + "=" * 70)

    return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    with ExperimentTracker("test_experiment") as tracker:
        # Save parameters
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model": "Ridge",
        }
        tracker.save_params(params)

        # Simulate training
        tracker.log("Starting training...")
        tracker.log("Epoch 1/10 - Loss: 0.5")
        tracker.log("Epoch 2/10 - Loss: 0.3")

        # Save metrics
        metrics = {
            "MAE": 1.23,
            "RMSE": 1.56,
            "R2": 0.42,
        }
        tracker.save_metrics(metrics, stage="test")

        # Save summary
        summary = create_experiment_summary("test_experiment", params, metrics, "Ridge")
        tracker.save_summary(summary)

        tracker.log("Done!")

    print(f"\nExperiment saved to: {tracker.experiment_dir}")
