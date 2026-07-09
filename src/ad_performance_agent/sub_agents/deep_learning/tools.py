"""Tools for the deep learning agent to train classification models."""

import logging
import os
import sys
from argparse import Namespace
from typing import Dict, Any

# Add project root to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)

logger = logging.getLogger(__name__)

from src.training.train_transformer_finetune import train_transformer
from src.utils.metrics import log_sample_predictions
from src.utils.safe_path import safe_path_join

from src.training.train_classification_nn import main as train_main
from src.utils.embeddings import generate_embeddings
from src.utils.metrics import build_sample_errors
from src.training.model_configs import DEFAULT_XGBOOST_PARAMS, DEFAULT_LIGHTGBM_PARAMS
from src.dataset.dataloader import load_datav3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np


def train_classification_model(
    data_dir: str = "data",
    size: str = "10k",
    text_output_dir: str = "text_outputs",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    hidden_dim: int = 128,
    loss_type: str = "focal",
    classification_task: str = "sponsor_label",
) -> Dict[str, Any]:
    """
    Train a neural network classification model for Instagram post performance prediction.

    Args:
        size: Dataset size to use ("1k" or "10k"). Default: "10k"
        epochs: Number of training epochs. Default: 30
        batch_size: Batch size for training. Default: 32
        lr: Learning rate. Default: 1e-4
        hidden_dim: Hidden layer dimension. Default: 128
        loss_type: Loss function type ("ce" for cross-entropy or "focal"). Default: "focal"
        classification_task: Type of classification task ("performance" or "sponsorship"). Default: "performance"

    Returns:
        Dictionary containing:
            - accuracy: Test accuracy score
            - report: Classification report string

    Example:
        >>> results = train_classification_model(size="1k", epochs=20, lr=1e-3)
        >>> print(f"Accuracy: {results['accuracy']:.4f}")
    """
    # Validate inputs
    if size not in ["1k", "10k"]:
        raise ValueError(f"size must be '1k' or '10k', got '{size}'")

    if loss_type not in ["ce", "focal"]:
        raise ValueError(f"loss_type must be 'ce' or 'focal', got '{loss_type}'")

    # Create arguments namespace
    args = Namespace(
        data_dir=data_dir,
        size=size,
        epochs=epochs,
        text_output_dir=text_output_dir,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        loss_type=loss_type,
        classification_task=classification_task,
    )

    # Call the training function
    logger.info("Starting training with parameters:")
    logger.info("  Dataset size: %s  Epochs: %d  Batch size: %d  LR: %s  Hidden dim: %d  Loss: %s",
        size, epochs, batch_size, lr, hidden_dim, loss_type)

    results = train_main(args)

    logger.info("Training completed! Final Accuracy: %s", results)

    return results


def train_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a classification model using a configuration dictionary.

    Args:
        config: Dictionary with training configuration. Supported keys:
            - size: Dataset size ("1k" or "10k")
            - epochs: Number of epochs
            - batch_size: Batch size
            - lr: Learning rate
            - hidden_dim: Hidden dimension
            - loss_type: Loss type ("ce" or "focal")

    Returns:
        Dictionary containing accuracy and classification report

    Example:
        >>> config = {
        ...     "size": "1k",
        ...     "epochs": 20,
        ...     "batch_size": 64,
        ...     "lr": 1e-3
        ... }
        >>> results = train_with_config(config)
    """
    # Extract parameters with defaults
    size = config.get("size", "10k")
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 32)
    lr = config.get("lr", 1e-4)
    hidden_dim = config.get("hidden_dim", 128)
    loss_type = config.get("loss_type", "focal")

    return train_classification_model(
        size=size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        loss_type=loss_type,
    )


def _train_tree_regression(
    data_dir: str,
    dataset_size: str,
    text_output_dir: str,
    embedding_model: str,
    influencer_dir: str,
    model_type: str,
) -> Dict[str, Any]:
    """Shared implementation for tree-based regression (XGBoost / LightGBM).

    Args:
        data_dir: Path to the dataset directory.
        dataset_size: Dataset size identifier.
        text_output_dir: Path to text outputs directory.
        embedding_model: Sentence-transformers model name.
        influencer_dir: Path to influencer profiles directory.
        model_type: Either "xgboost" or "lightgbm".

    Returns:
        Dictionary with r2, mae, rmse, best_model_name, model, sample errors.
    """

    if model_type not in ("xgboost", "lightgbm"):
        raise ValueError(f"Unknown model_type: {model_type}. Use 'xgboost' or 'lightgbm'.")

    model_name = "XGBoost" if model_type == "xgboost" else "LightGBM"

    logger.info("=" * 60)
    logger.info("Starting %s Regression Training", model_name)
    logger.info("  Dataset: %s  Text dir: %s  Embedding: %s",
        dataset_size, text_output_dir, embedding_model)

    json_dir = str(safe_path_join(data_dir, "json_subset"))

    logger.info("Loading data from: %s  Text dir: %s  JSON dir: %s",
        data_dir, text_output_dir, json_dir)

    load_kwargs = dict(
        text_dir=text_output_dir,
        json_dir=json_dir,
        classification_task="perf",
        include_user_info=True,
    )
    if influencer_dir is not None:
        load_kwargs["influencer_dir"] = influencer_dir

    df = load_datav3(data_dir, **load_kwargs)

    if df.empty:
        raise ValueError("No data loaded. Check paths.")

    logger.info("Loaded %d samples for regression.", len(df))

    y = np.log(df["y"].values + 1.0)
    texts = np.array(df["text"].tolist())
    post_ids = np.array(df["post_id"].tolist()) if "post_id" in df.columns else None

    logger.info("Target stats (likes):\n%s", df['y'].describe())

    # Generate embeddings BEFORE importing XGBoost/LightGBM so that torch's
    # libomp initialises its thread pool first (defense-in-depth for the
    # dual-libomp conflict on macOS).
    X = generate_embeddings(df["text"].tolist(), model_name=embedding_model)

    if post_ids is not None:
        (
            X_train, X_test,
            y_train, y_test,
            ids_train, ids_test,
            texts_train, texts_test,
        ) = train_test_split(
            X, y, post_ids, texts, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ids_train = ids_test = None
        texts_train = texts_test = None

    logger.info("Training set: %d  Test set: %d", len(X_train), len(X_test))

    # Lazy-import tree regressor AFTER embeddings to avoid loading a second
    # libomp before torch has claimed the OMP runtime.
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        regressor = XGBRegressor(**DEFAULT_XGBOOST_PARAMS)
    else:
        from lightgbm import LGBMRegressor
        lgbm_params = {**DEFAULT_LIGHTGBM_PARAMS, "reg_alpha": 0.1, "reg_lambda": 0.1}
        regressor = LGBMRegressor(**lgbm_params)

    logger.info("Training %s Regressor...", model_name)
    pipeline = Pipeline([("regressor", regressor)])
    try:
        pipeline.fit(X_train, y_train)
    except Exception:
        logger.exception("Failed during %s fit", model_name)
        raise
    logger.info("%s training completed.", model_name)

    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    logger.info("%s Results: MAE=%.4f  RMSE=%.4f  R2=%.4f", model_name, mae, rmse, r2)
    log_sample_predictions(y_test, y_pred_test, n=10, log_fn=logger.info)

    train_sample_errors = {}
    test_sample_errors = {}

    if ids_train is not None and ids_test is not None:
        train_sample_errors = build_sample_errors(ids_train, y_train, y_pred_train, texts_train)
        test_sample_errors = build_sample_errors(ids_test, y_test, y_pred_test, texts_test)

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "best_model_name": model_name,
        "model": pipeline,
        "train_sample_errors": train_sample_errors,
        "test_sample_errors": test_sample_errors,
    }


def train_regression_model(
    data_dir: str,
    dataset_size: str = "10k",
    text_output_dir: str = "text_outputs",
    embedding_model: str = "all-mpnet-base-v2",
    influencer_dir: str = None,
) -> Dict[str, Any]:
    """Train an XGBoost regression model for Instagram post likes prediction."""
    return _train_tree_regression(
        data_dir, dataset_size, text_output_dir, embedding_model, influencer_dir,
        model_type="xgboost",
    )


def train_lightgbm_model(
    data_dir: str,
    dataset_size: str = "10k",
    text_output_dir: str = "text_outputs",
    embedding_model: str = "all-mpnet-base-v2",
    influencer_dir: str = None,
) -> Dict[str, Any]:
    """Train a LightGBM regression model for Instagram post likes prediction."""
    return _train_tree_regression(
        data_dir, dataset_size, text_output_dir, embedding_model, influencer_dir,
        model_type="lightgbm",
    )


def train_transformer_model(
    data_dir: str,
    dataset_size: str = "10k",
    text_output_dir: str = "text_outputs",
    transformer_model_name: str = "distilbert-base-uncased",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 2e-5,
    influencer_dir: str = None,
    device: str = None,
) -> Dict[str, Any]:
    """Train a transformer model for Instagram post likes prediction.

    Args:
        data_dir: Path to the dataset directory.
        dataset_size: Dataset size identifier. Default: "10k"
        text_output_dir: Path to the directory containing text outputs.
        transformer_model_name: HuggingFace model ID. Default: "distilbert-base-uncased"
        epochs: Number of training epochs. Default: 25
        batch_size: Batch size for training. Default: 16
        lr: Learning rate. Default: 2e-5
        influencer_dir: Path to users_influencers_SPOD directory. Default: None (uses config)
        device: Device to train on ("cuda", "mps", "cpu"). Default: None (auto-detect).

    Returns:
        Dictionary containing r2, mae, rmse, train_sample_errors, test_sample_errors.
    """

    logger.info("=" * 60)
    logger.info("Starting Transformer Fine-tuning")
    logger.info("  Model: %s  Dataset: %s  Epochs: %d  Batch size: %d  LR: %s",
        transformer_model_name, dataset_size, epochs, batch_size, lr)

    json_dir = str(safe_path_join(data_dir, "json_subset"))

    load_kwargs = dict(
        text_dir=text_output_dir,
        json_dir=json_dir,
        classification_task="perf",
        include_user_info=True,
    )
    if influencer_dir is not None:
        load_kwargs["influencer_dir"] = influencer_dir

    df = load_datav3(data_dir, **load_kwargs)

    if df.empty:
        raise ValueError("No data loaded. Check paths.")

    logger.info("Loaded %d samples for transformer training.", len(df))

    _, metrics = train_transformer(
        df,
        model_name=transformer_model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    return metrics
