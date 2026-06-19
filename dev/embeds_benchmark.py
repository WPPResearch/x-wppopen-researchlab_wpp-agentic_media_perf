"""Benchmark different embedding models on the best iteration text data."""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from src.dataset.dataloader import load_datav3
from src.utils.embeddings import generate_embeddings
from src.utils.safe_path import safe_path_join
from src.training.model_configs import DEFAULT_XGBOOST_PARAMS

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = [
    # Small / Fast
    "all-MiniLM-L6-v2",                    # 384 dims
    "all-MiniLM-L12-v2",                   # 384 dims
    # Medium
    "all-mpnet-base-v2",                    # 768 dims
    # Large
    "all-roberta-large-v1",                 # 1024 dims
    "sentence-transformers/gtr-t5-large",   # 768 dims
    "thenlper/gte-large",                   # 1024 dims
    "thenlper/gte-base",                    # 768 dims
]


def train_xgboost_with_embeddings(
    data_dir: str,
    text_output_dir: str,
    embedding_model: str,
    ) -> Dict[str, Any]:
    """Train XGBoost with a specific embedding model and return metrics."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    json_dir = str(safe_path_join(data_dir, "json_subset"))

    df = load_datav3(
        data_dir,
        text_dir=text_output_dir,
        json_dir=json_dir,
        classification_task="perf",
        include_user_info=True,
    )

    if df.empty:
        raise ValueError("No data loaded. Check paths.")

    y = np.log(df["y"].values + 1.0)
    texts = df["text"].tolist()
    post_ids = (
        np.array(df["post_id"].tolist()) if "post_id" in df.columns else None
    )

    start_time = time.time()
    X = generate_embeddings(texts, model_name=embedding_model, cleanup_gpu=True)
    embed_time = time.time() - start_time

    if post_ids is not None:
        (
            X_train, X_test,
            y_train, y_test,
            ids_train, ids_test,
        ) = train_test_split(
            X, y, post_ids, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    start_time = time.time()
    model = XGBRegressor(**DEFAULT_XGBOOST_PARAMS)

    pipeline = Pipeline([("regressor", model)])
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred_test = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    return {
        "embedding_model": embedding_model,
        "embedding_dim": X.shape[1],
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "embed_time_s": round(embed_time, 2),
        "train_time_s": round(train_time, 2),
        "n_samples": len(df),
    }


def run_embedding_benchmark(
    data_dir: str,
    text_output_dir: str,
    embedding_models: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    ) -> pd.DataFrame:
    """Benchmark multiple embedding models with XGBoost on the same text data."""
    if embedding_models is None:
        embedding_models = list(EMBEDDING_MODELS)

    results = []

    logger.info("=" * 70)
    logger.info("Embedding Model Benchmark")
    logger.info("  Data dir:        %s", data_dir)
    logger.info("  Text output dir: %s", text_output_dir)
    logger.info("  Models to test:  %d", len(embedding_models))
    logger.info("  GPU available:   %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  GPU device:      %s", torch.cuda.get_device_name(0))

    for i, model_name in enumerate(embedding_models, 1):
        logger.info("[%d/%d] Testing: %s", i, len(embedding_models), model_name)

        try:
            metrics = train_xgboost_with_embeddings(
                data_dir=data_dir,
                text_output_dir=text_output_dir,
                embedding_model=model_name,
            )
            results.append(metrics)

            logger.info("  R2: %.4f  MAE: %.4f  RMSE: %.4f  Dim: %s  Embed: %ss  Train: %ss",
                metrics['r2'], metrics['mae'], metrics['rmse'],
                metrics['embedding_dim'], metrics['embed_time_s'], metrics['train_time_s'])

        except (ImportError, ValueError) as e:
            logger.error("Model %s failed with known error: %s", model_name, e)
            results.append({
                "embedding_model": model_name,
                "embedding_dim": None,
                "r2": None,
                "mae": None,
                "rmse": None,
                "embed_time_s": None,
                "train_time_s": None,
                "n_samples": None,
                "error": str(e),
            })
        except Exception as e:
            logger.exception("Model %s failed with unexpected error", model_name)
            results.append({
                "embedding_model": model_name,
                "embedding_dim": None,
                "r2": None,
                "mae": None,
                "rmse": None,
                "embed_time_s": None,
                "train_time_s": None,
                "n_samples": None,
                "error": str(e),
            })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("r2", ascending=False, na_position="last")
    df_results = df_results.reset_index(drop=True)

    display_cols = [
        "embedding_model", "embedding_dim", "r2", "mae", "rmse",
        "embed_time_s", "train_time_s",
    ]
    if "error" in df_results.columns and df_results["error"].notna().any():
        display_cols.append("error")

    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS (sorted by R2)")
    logger.info("\n%s", df_results[display_cols].to_string(index=False))

    best = df_results.iloc[0]
    if best["r2"] is not None:
        logger.info("Best model: %s  R2: %.4f  MAE: %.4f  RMSE: %.4f",
            best['embedding_model'], best['r2'], best['mae'], best['rmse'])

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"embedding_benchmark_{timestamp}.csv"

    df_results.to_csv(output_path, index=False)
    logger.info("Results saved to: %s", output_path)

    json_path = output_path.replace(".csv", ".json")
    df_results.to_json(json_path, orient="records", indent=2)
    logger.info("Results saved to: %s", json_path)

    return df_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models with XGBoost"
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--text-output-dir",
        default="",
        help="Path to the best iteration text outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific embedding models to test (space-separated).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for results",
    )
    args = parser.parse_args()

    run_embedding_benchmark(
        data_dir=args.data_dir,
        text_output_dir=args.text_output_dir,
        embedding_models=args.models,
        output_path=args.output,
    )
    