import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.experiment_tracker import ExperimentTracker, create_experiment_summary
from src.utils.metrics import log_sample_predictions


class SimpleNN(nn.Module):
    """Simple 2-layer neural network."""

    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze()


class DeepNN(nn.Module):
    """Deep 4-layer neural network."""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super(DeepNN, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class WideNN(nn.Module):
    """Wide but shallow network."""

    def __init__(self, input_dim, hidden_dim=1024):
        super(WideNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze()


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet(nn.Module):
    """Residual neural network."""

    def __init__(self, input_dim, hidden_dim=512, num_blocks=3):
        super(ResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x).squeeze()


def train_model(
    model, train_loader, val_loader, epochs=50, lr=0.001, device="cpu", patience=10
):
    """Train a neural network model with early stopping."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def train_and_evaluate_neural_nets(df, embedding_model="all-mpnet-base-v2"):
    """Train and compare different neural network architectures."""
    if df.empty:
        print("No data loaded.")
        return

    with ExperimentTracker("neural_regression") as tracker:
        tracker.log(f"\n{'=' * 70}")
        tracker.log("NEURAL NETWORK REGRESSION - TEXT EMBEDDINGS ONLY")
        tracker.log(f"{'=' * 70}")

        tracker.log(f"\nLoaded {len(df)} samples.")
        y = np.log(df["y"] + 1.0)

        tracker.log("\nTarget statistics:")
        tracker.log(str(df["y"].describe()))
        tracker.log("\nLog-transformed target:")
        tracker.log(str(y.describe()))

        # Save global params
        params = {
            "n_samples": len(df),
            "embedding_model": embedding_model,
            "target_transform": "log(likes + 1)",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "patience": 15,
            "optimizer": "Adam",
            "weight_decay": 1e-5,
        }
        tracker.save_params(params)

        # Generate embeddings
        tracker.log(f"\nGenerating text embeddings with {embedding_model}...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model)
        X = model.encode(df["text"].tolist(), show_progress_bar=True)

        tracker.log(f"Embedding dimension: {X.shape[1]}")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        tracker.log(f"\nTraining set: {len(X_train)}")
        tracker.log(f"Validation set: {len(X_val)}")
        tracker.log(f"Test set: {len(X_test)}")

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train.values)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.values)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tracker.log(f"\nUsing device: {device}")

        # Define architectures to test
        input_dim = X.shape[1]
        architectures = {
            "Simple NN (256)": SimpleNN(input_dim, hidden_dim=256),
            "Deep NN (512-256-128)": DeepNN(input_dim, hidden_dims=[512, 256, 128]),
            "Wide NN (1024)": WideNN(input_dim, hidden_dim=1024),
            "ResNet (512, 3 blocks)": ResNet(input_dim, hidden_dim=512, num_blocks=3),
        }

        results = {}
        all_training_histories = {}

        for name, model_arch in architectures.items():
            tracker.log(f"\n{'=' * 70}")
            tracker.log(f"Training: {name}")
            tracker.log(f"{'=' * 70}")

            # Count parameters
            num_params = sum(p.numel() for p in model_arch.parameters())
            tracker.log(f"Total parameters: {num_params:,}")

            # Train
            trained_model, train_losses, val_losses = train_model(
                model_arch,
                train_loader,
                val_loader,
                epochs=100,
                lr=0.001,
                device=device,
                patience=15,
            )

            # Sanitize name for use as file/key identifier
            arch_slug = (
                name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("-", "_")
            )

            # Save training history for this architecture
            all_training_histories[arch_slug] = {
                "train_loss": train_losses,
                "val_loss": val_losses,
            }

            # Save model
            tracker.save_model(trained_model, model_name=arch_slug, is_pytorch=True)

            # Evaluate on test set
            y_pred, metrics = evaluate_model(
                trained_model, X_test, y_test, device=device
            )

            tracker.log("\n--- Test Set Results ---")
            tracker.log(f"MAE:  {metrics['MAE']:.4f}")
            tracker.log(f"RMSE: {metrics['RMSE']:.4f}")
            tracker.log(f"R² Score: {metrics['R2']:.4f}")

            # Save metrics and predictions
            tracker.save_metrics(
                {"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]},
                stage=arch_slug,
            )
            tracker.save_predictions(
                y_test.values if hasattr(y_test, "values") else y_test,
                y_pred,
                stage=arch_slug,
            )

            # Show sample predictions
            log_sample_predictions(y_test, y_pred, n=10, log_fn=tracker.log)

            results[name] = metrics

        # Save all training histories
        tracker.save_training_history(all_training_histories)

        # Summary
        tracker.log(f"\n{'=' * 70}")
        tracker.log("SUMMARY - All Models")
        tracker.log(f"{'=' * 70}")
        tracker.log(f"{'Model':<30} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
        tracker.log(f"{'-' * 70}")

        for name, metrics in sorted(
            results.items(), key=lambda x: x[1]["R2"], reverse=True
        ):
            tracker.log(
                f"{name:<30} {metrics['R2']:>10.4f} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f}"
            )

        best_model_name, best_metrics = max(results.items(), key=lambda x: x[1]["R2"])
        tracker.log(
            f"\nBest Model: {best_model_name} with R² = {best_metrics['R2']:.4f}"
        )

        # Save overall summary
        summary = create_experiment_summary(
            "Neural Network Regression",
            params,
            {
                "best_model": best_model_name,
                "best_R2": best_metrics["R2"],
                "best_MAE": best_metrics["MAE"],
                "best_RMSE": best_metrics["RMSE"],
            },
            best_model_name,
        )
        tracker.save_summary(summary)


if __name__ == "__main__":
    from src.utils.cli import create_data_argparser, load_data_from_args

    parser = create_data_argparser("Train neural networks for engagement prediction.")
    args = parser.parse_args()

    df = load_data_from_args(args)
    train_and_evaluate_neural_nets(df, embedding_model=args.embedding)
