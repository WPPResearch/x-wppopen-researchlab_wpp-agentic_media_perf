# Experiment Tracking & Checkpoint System

All training scripts now automatically save experiment results with timestamps, parameters, models, and logs.

## 📁 Directory Structure

When you run a training script, it creates a checkpoint folder:

```
checkpoints/
└── {experiment_name}_{timestamp}/
    ├── SUMMARY.txt                 # Human-readable summary
    ├── params.json                 # All hyperparameters
    ├── models/
    │   └── best_model.pkl/.pt      # Best trained model
    └── logs/
        ├── training.log            # Full training log
        ├── metrics.json            # All metrics (train/val/test)
        ├── training_history.json   # Loss curves per epoch
        └── test_predictions.csv    # Actual vs predicted values
```

## 🚀 Usage

All training scripts automatically use the checkpoint system:

### 1. **Text Embedding Regression**
```bash
python src/training/train_likes_regression.py --size 10k --embedding all-mpnet-base-v2
```
**Saves to**: `checkpoints/text_embedding_regression_{timestamp}/`

### 2. **Feature Engineered Regression**
```bash
python src/training/train_likes_with_features.py --size 10k
```
**Saves to**: `checkpoints/feature_engineered_regression_{timestamp}/`

### 3. **Transformer Fine-tuning**
```bash
python src/training/train_transformer_finetune.py --size 10k --model distilbert-base-uncased --epochs 5
```
**Saves to**: `checkpoints/transformer_finetuning_{timestamp}/`

### 4. **Neural Network Regression**
```bash
python src/training/train_neural_regression.py --size 10k
```
**Note**: This script compares four architectures (SimpleNN, DeepNN, WideNN, ResNet) and prints results to stdout. It does not use the ExperimentTracker, so results are not automatically saved to `checkpoints/`.

## 📊 What Gets Saved

### 1. **Parameters** (`params.json`)
All hyperparameters and configuration:
```json
{
  "n_samples": 10000,
  "embedding_model": "all-mpnet-base-v2",
  "test_size": 0.2,
  "random_state": 42,
  "target_transform": "log(likes + 1)"
}
```

### 2. **Metrics** (`logs/metrics.json`)
Performance metrics for each model:
```json
{
  "test": {
    "Ridge (alpha=0.1)_MAE": 1.19,
    "Ridge (alpha=0.1)_RMSE": 1.51,
    "Ridge (alpha=0.1)_R2": 0.1691,
    "XGBoost_MAE": 1.22,
    "XGBoost_RMSE": 1.55,
    "XGBoost_R2": 0.1287
  }
}
```

### 3. **Training History** (`logs/training_history.json`)
For neural networks/transformers:
```json
{
  "train_loss": [2.5, 2.1, 1.8, 1.6],
  "val_loss": [2.6, 2.3, 2.0, 1.9],
  "val_r2": [0.10, 0.15, 0.18, 0.20]
}
```

### 4. **Training Log** (`logs/training.log`)
Timestamped console output:
```
[2026-02-27 16:41:27] Loaded 10003 samples.
[2026-02-27 16:41:27] Training set size: 8002
[2026-02-27 16:41:27] Test set size: 2001
[2026-02-27 16:41:30] --- Training Ridge (alpha=0.1) ---
[2026-02-27 16:41:32] MAE: 1.19
[2026-02-27 16:41:32] RMSE: 1.51
[2026-02-27 16:41:32] R2 Score: 0.1691
```

### 5. **Best Model** (`models/best_*.pkl` or `.pt`)
- **Sklearn models**: Saved as `.pkl` (pickle)
- **PyTorch models**: Saved as `.pt` (state_dict)

Load them later:
```python
# Sklearn
import pickle
with open('checkpoints/.../models/best_Ridge.pkl', 'rb') as f:
    model = pickle.load(f)

# PyTorch
import torch
model.load_state_dict(torch.load('checkpoints/.../models/best_model.pt'))
```

### 6. **Predictions** (`logs/test_predictions.csv`)
Actual vs predicted for error analysis:
```csv
y_true,y_pred
6.82,7.15
5.15,6.49
3.30,5.96
```

### 7. **Summary** (`SUMMARY.txt`)
Human-readable overview:
```
======================================================================
EXPERIMENT SUMMARY: Feature Engineered Regression
======================================================================

Timestamp: 2026-02-27 16:41:27

PARAMETERS:
----------------------------------------------------------------------
  n_samples: 10003
  embedding_model: all-mpnet-base-v2
  features_used: text_embeddings + post_metadata + influencer_data + temporal

RESULTS:
----------------------------------------------------------------------
  Best_Model: XGBoost
  Best_R2: 0.4567
  XGBoost_MAE: 1.05
  XGBoost_RMSE: 1.32
  XGBoost_R2: 0.4567

Best Model: XGBoost
======================================================================
```

## 🔍 Finding Your Experiments

List all experiments:
```bash
ls -ltr checkpoints/
```

View latest experiment summary:
```bash
cat checkpoints/*/SUMMARY.txt | tail -20
```

Compare experiments:
```bash
# Find all R2 scores
grep "Best_R2" checkpoints/*/logs/metrics.json
```

## 🎯 Example Workflow

```bash
# Run experiment 1: Text embeddings only
python src/training/train_likes_regression.py --size 10k --embedding all-mpnet-base-v2

# Check results
cat checkpoints/text_embedding_regression_*/SUMMARY.txt

# Run experiment 2: With feature engineering
python src/training/train_likes_with_features.py --size 10k

# Compare R2 scores
grep "R2" checkpoints/*/SUMMARY.txt

# Run experiment 3: Transformer fine-tuning
python src/training/train_transformer_finetune.py --size 10k --epochs 10

# View training curves
cat checkpoints/transformer_finetuning_*/logs/training_history.json
```

## 📈 Analyzing Results

### Compare Multiple Experiments
```python
import pandas as pd
import json
import glob

# Load all experiment metrics
results = []
for metrics_file in glob.glob("checkpoints/*/logs/metrics.json"):
    with open(metrics_file) as f:
        metrics = json.load(f)
    experiment_name = metrics_file.split('/')[1]
    results.append({
        'experiment': experiment_name,
        **metrics.get('test', {})
    })

df = pd.DataFrame(results)
print(df.sort_values('Best_R2', ascending=False))
```

### Plot Training Curves
```python
import matplotlib.pyplot as plt

with open('checkpoints/transformer_finetuning_*/logs/training_history.json') as f:
    history = json.load(f)

plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curves.png')
```

## 🛠️ Advanced: Custom Tracking

You can also use the tracker directly in your own scripts:

```python
from src.utils.experiment_tracker import ExperimentTracker

with ExperimentTracker("my_experiment") as tracker:
    # Save params
    tracker.save_params({'lr': 0.001, 'epochs': 10})

    # Log messages
    tracker.log("Starting training...")

    # Save metrics
    tracker.save_metrics({'accuracy': 0.95}, stage='test')

    # Save model
    tracker.save_model(model, model_name='my_model')

    # Save predictions
    tracker.save_predictions(y_true, y_pred)
```

## 📝 Notes

- **Timestamps**: All experiments use format `{name}_{YYYYMMDD_HHMMSS}`
- **No duplicates**: Each run creates a new folder (never overwrites)
- **Automatic cleanup**: Failed experiments still save logs for debugging
- **Git**: Add `checkpoints/` to `.gitignore` (large files)

## 🎉 Benefits

✅ **Reproducibility**: All parameters saved automatically
✅ **Comparison**: Easy to compare multiple experiments
✅ **Debugging**: Full logs for troubleshooting
✅ **Model persistence**: Best models saved for deployment
✅ **Analysis**: CSV predictions for error analysis
✅ **History**: Never lose experiment results

---

Happy experimenting! 🚀
