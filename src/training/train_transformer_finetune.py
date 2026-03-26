"""
Fine-tune a transformer model (BERT/RoBERTa) for engagement prediction.
This trains the entire model end-to-end instead of using frozen embeddings.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm

from src.utils.experiment_tracker import ExperimentTracker, create_experiment_summary
from src.utils.metrics import log_sample_predictions, build_sample_errors

EARLY_STOPPING_PATIENCE = 5


class TextDataset(Dataset):
    """Dataset for text regression."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
        }


class TransformerRegressor(nn.Module):
    """Transformer-based regression model."""

    def __init__(self, model_name="bert-base-uncased", dropout=0.3):
        super(TransformerRegressor, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Predict
        return self.regressor(cls_output).squeeze(-1)


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, criterion=None):
    """Evaluate model. Returns (avg_loss, preds, labels) when criterion is provided,
    or (preds, labels) when criterion is None (inference-only mode)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)

    if criterion is not None:
        avg_loss = total_loss / len(dataloader)
        return avg_loss, preds, labels_arr

    return preds, labels_arr


def train_transformer(
    df,
    model_name="distilbert-base-uncased",
    epochs=5,
    batch_size=16,
    lr=2e-5,
    max_length=256,
    ):
    """Fine-tune a transformer model for engagement prediction."""
    with ExperimentTracker("transformer_finetuning") as tracker:
        tracker.log(f"\n{'=' * 70}")
        tracker.log(f"TRANSFORMER FINE-TUNING: {model_name}")
        tracker.log(f"{'=' * 70}")

        y = np.log(df["y"].values + 1.0)
        texts = np.array(df["text"].tolist())
        post_ids = np.array(df["post_id"].tolist()) if "post_id" in df.columns else None

        tracker.log(f"\nDataset: {len(texts)} samples")

        params = {
            "n_samples": len(texts),
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "target_transform": "log(likes + 1)",
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "warmup_steps": "10% of total",
        }
        tracker.save_params(params)

        if post_ids is not None:
            (
                X_train, X_temp,
                y_train, y_temp,
                ids_train, ids_temp,
                texts_train, texts_temp,
            ) = train_test_split(
                texts, y, post_ids, texts, test_size=0.2, random_state=42
            )
            (
                X_val, X_test,
                y_val, y_test,
                ids_val, ids_test,
                texts_val, texts_test,
            ) = train_test_split(
                X_temp, y_temp, ids_temp, texts_temp, test_size=0.5, random_state=42
            )
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                texts, y, test_size=0.2, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            ids_train = ids_val = ids_test = None
            texts_train = X_train
            texts_test = X_test

        tracker.log(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        tracker.log(f"\nLoading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
        test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tracker.log(f"\nDevice: {device}")

        model = TransformerRegressor(model_name=model_name)
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        tracker.log(f"Total parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = EARLY_STOPPING_PATIENCE
        patience_counter = 0
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_r2": [],
        }

        for epoch in range(epochs):
            tracker.log(f"\n{'=' * 70}")
            tracker.log(f"Epoch {epoch + 1}/{epochs}")
            tracker.log(f"{'=' * 70}")

            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, criterion
            )
            val_loss, val_preds, val_labels = evaluate(
                model, val_loader, device, criterion
            )

            val_mae = mean_absolute_error(val_labels, val_preds)
            val_r2 = r2_score(val_labels, val_preds)

            training_history["train_loss"].append(float(train_loss))
            training_history["val_loss"].append(float(val_loss))
            training_history["val_mae"].append(float(val_mae))
            training_history["val_r2"].append(float(val_r2))

            tracker.log(f"\nTrain Loss: {train_loss:.4f}")
            tracker.log(
                f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                tracker.save_model(model, model_name="best_model", is_pytorch=True)
                tracker.log("✓ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tracker.log(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        tracker.save_training_history(training_history)

        tracker.log(f"\n{'=' * 70}")
        tracker.log("FINAL EVALUATION ON TEST SET")
        tracker.log(f"{'=' * 70}")

        best_model_path = tracker.models_dir / "best_model.pt"
        model.load_state_dict(torch.load(best_model_path))

        test_loss, test_preds, test_labels = evaluate(
            model, test_loader, device, criterion
        )

        test_mae = mean_absolute_error(test_labels, test_preds)
        test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
        test_r2 = r2_score(test_labels, test_preds)

        test_metrics = {
            "test_loss": float(test_loss),
            "test_MAE": float(test_mae),
            "test_RMSE": float(test_rmse),
            "test_R2": float(test_r2),
        }
        tracker.save_metrics(test_metrics, stage="test")
        tracker.save_predictions(test_labels, test_preds, stage="test")

        tracker.log("\nTest Results:")
        tracker.log(f"MAE:  {test_mae:.4f}")
        tracker.log(f"RMSE: {test_rmse:.4f}")
        tracker.log(f"R² Score: {test_r2:.4f}")

        log_sample_predictions(test_labels, test_preds, n=10, log_fn=tracker.log)

        tracker.log(f"\n{'=' * 70}")
        tracker.log("COMPUTING PER-SAMPLE ERRORS (TRAIN + TEST)")
        tracker.log(f"{'=' * 70}")

        train_sample_errors = {}
        test_sample_errors = {}

        if ids_train is not None and ids_test is not None:
            train_eval_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
            train_eval_loader = DataLoader(
                train_eval_dataset, batch_size=batch_size * 2, shuffle=False
            )
            _, train_preds, train_labels = evaluate(
                model, train_eval_loader, device, criterion
            )

            train_sample_errors = build_sample_errors(ids_train, train_labels, train_preds, texts_train)
            test_sample_errors = build_sample_errors(ids_test, test_labels, test_preds, texts_test)

            tracker.log(f"Train sample errors: {len(train_sample_errors)}")
            tracker.log(f"Test sample errors: {len(test_sample_errors)}")

        metrics = {
            "r2": float(test_r2),
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "train_sample_errors": train_sample_errors,
            "test_sample_errors": test_sample_errors,
        }

        summary = create_experiment_summary(
            "Transformer Fine-tuning",
            params,
            test_metrics,
            model_name,
        )
        tracker.save_summary(summary)

        return model, metrics

if __name__ == "__main__":
    from src.utils.cli import create_data_argparser, load_data_from_args

    parser = create_data_argparser("Fine-tune transformer for engagement prediction.")
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="Transformer model (distilbert-base-uncased, bert-base-uncased, roberta-base)",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max-length", type=int, default=256, help="Max sequence length"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    df = load_data_from_args(args)

    train_transformer(
        df,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
