"""
Inference script for a fine-tuned transformer engagement prediction model.
Evaluates the saved checkpoint on the validation and test splits.
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.utils.metrics import log_sample_predictions
from src.training.train_transformer_finetune import TextDataset, TransformerRegressor, evaluate


def print_metrics(split_name, preds, labels):
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    print(f"\n{'=' * 50}")
    print(f"{split_name} Results (log-space)")
    print(f"{'=' * 50}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Convert back to original like counts
    likes_true = np.exp(labels) - 1
    likes_pred = np.exp(preds) - 1
    likes_mae = mean_absolute_error(likes_true, likes_pred)
    likes_rmse = np.sqrt(mean_squared_error(likes_true, likes_pred))
    print(f"\n{split_name} Results (original like counts)")
    print(f"MAE:  {likes_mae:.1f}")
    print(f"RMSE: {likes_rmse:.1f}")

    log_sample_predictions(labels, preds, n=10, log_fn=print)

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def main():
    from src.utils.cli import create_data_argparser, load_data_from_args

    parser = create_data_argparser(
        "Evaluate a fine-tuned transformer checkpoint on val/test splits."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/transformer_finetuning_20260314_125202",
        help="Path to experiment directory (containing params.json and models/best_model.pt) "
        "or directly to a .pt file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Transformer model name (e.g. distilbert-base-uncased). "
        "Loaded from params.json if not provided.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max token sequence length. Loaded from params.json if not provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Resolve model .pt file and params
    if checkpoint_path.suffix == ".pt":
        model_pt = checkpoint_path
        params_file = checkpoint_path.parent.parent / "params.json"
    else:
        model_pt = checkpoint_path / "models" / "best_model.pt"
        params_file = checkpoint_path / "params.json"

    if not model_pt.exists():
        raise FileNotFoundError(f"Model file not found: {model_pt}")

    # Load params to recover model_name and max_length used during training
    saved_params = {}
    if params_file.exists():
        with open(params_file) as f:
            saved_params = json.load(f)
        print(f"Loaded params from {params_file}")

    model_name = args.model or saved_params.get("model_name", "distilbert-base-uncased")
    max_length = args.max_length or saved_params.get("max_length", 256)

    print(f"Model:      {model_name}")
    print(f"Max length: {max_length}")
    print(f"Checkpoint: {model_pt}")

    df = load_data_from_args(args)

    # Replicate the exact same splits as training
    y = np.log(df["y"] + 1.0)
    texts = df["text"].tolist()

    _, X_temp, _, y_temp = train_test_split(
        texts, y.values, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Val: {len(X_val)}, Test: {len(X_test)}")

    # Tokenizer and datasets
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TransformerRegressor(model_name=model_name)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model = model.to(device)

    # Evaluate
    print("\nEvaluating on validation set...")
    val_preds, val_labels = evaluate(model, val_loader, device)
    print_metrics("Validation", val_preds, val_labels)

    print("\nEvaluating on test set...")
    test_preds, test_labels = evaluate(model, test_loader, device)
    print_metrics("Test", test_preds, test_labels)


if __name__ == "__main__":
    main()
