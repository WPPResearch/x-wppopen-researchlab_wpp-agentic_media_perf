import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset.dataloader import load_datav2, convert_to_classification_labels
from src.utils.embeddings import generate_embeddings
from src.utils.safe_path import safe_path_join
from src.config import DATASET_SUBSETS

from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class InstagramPostClassifier(pl.LightningModule):
    def __init__(
        self, input_dim, hidden_dim=128, num_classes=3, lr=1e-3, loss_type="ce"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        if loss_type == "focal":
            self.loss_fn = FocalLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main(args):
    DATA_DIR = args.data_dir

    print(f"Loading data from {DATA_DIR}...")

    data_dir_resolved = str(safe_path_join(
        os.path.dirname(DATA_DIR), os.path.basename(DATA_DIR)
    ))
    df = load_datav2(
        data_dir_resolved, args.text_output_dir, classification_task=args.classification_task
    )

    if df is None or df.empty:
        print("No data loaded. Please check your data directory.")
        return

    # Convert continuous 'y' values to classification labels (3 classes by default)
    if args.classification_task == "engagement_label":
        print("Converting engagement scores to classification labels...")
        df = convert_to_classification_labels(df)

    # Generate embeddings features using Sentence Transformers
    print("Generating sentence embeddings...")
    X = generate_embeddings(df["text"].tolist())
    y = df["y"].values

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(
        f"CLFtask {args.classification_task} Number of classes: {len(df['y'].unique())}"
    )
    # Convert numpy arrays PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Setup PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # Initialize PyTorch Lightning Model
    input_dim = X_train.shape[1]
    model = InstagramPostClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(df["y"].unique()),
        lr=args.lr,
        loss_type=args.loss_type,
    )

    # Setup Early Stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    )

    # Start Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
        logger=True,
        log_every_n_steps=5,
        accelerator="auto",
        devices="auto",
    )

    # Train
    print()
    print("--- Starting Training ---")
    trainer.fit(model, train_loader, val_loader)

    # Predict and evaluate on the test split
    print()
    print("--- Evaluating Model ---")
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print()
    print("Classification Report:")
    report = classification_report(y_test, preds, zero_division=0)
    print(classification_report(y_test, preds, zero_division=0))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    return {"accuracy": acc, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PyTorch Lightning Neural Network for Instagram Post Classification."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SUBSETS.keys()),
        default="top20",
        help="Dataset to use: " + ", ".join(DATASET_SUBSETS.keys()),
    )
    parser.add_argument(
        "--text_output_dir",
        type=str,
        default=None,
        help="Path to text outputs directory (default: <dataset>/text_output_gemini_2.5_flash/responses)",
    )

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument(
        "--loss_type",
        choices=["ce", "focal"],
        default="focal",
        help="Loss function to use (ce or focal)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        choices=[
            "google/embeddinggemma-300m",
            "sentence-transformers/all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
        ],
        help="Sentence embedding model to use",
    )
    parser.add_argument(
        "--classification_task",
        type=str,
        default="engagement_label",
        help="Classification task to perform (e.g., engagement, sentiment)",
        choices=["sponsor_label", "engagement_label"],
    )
    args = parser.parse_args()
    BASE_DIR = Path(os.path.abspath(__file__)).parent.parent.parent
    args.data_dir = str(BASE_DIR / "data" / DATASET_SUBSETS[args.dataset])
    main(args)
