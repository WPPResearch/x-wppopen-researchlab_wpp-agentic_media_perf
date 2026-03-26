"""Shared metrics utilities for training and evaluation."""

import logging
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def log_sample_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 10,
    log_fn: Callable[[str], None] = print,
) -> None:
    """Log sample predictions converting from log-space to original like counts.

    Args:
        y_true: Array of true values (log-transformed likes).
        y_pred: Array of predicted values (log-transformed likes).
        n: Number of samples to display.
        log_fn: Logging function (e.g. print, tracker.log, logger.info).
    """
    log_fn("Sample predictions (first %d):" % n)
    for i in range(min(n, len(y_true))):
        actual_val = y_true[i] if not hasattr(y_true, "iloc") else y_true.iloc[i]
        actual_likes = int(np.exp(actual_val) - 1)
        predicted_likes = int(np.exp(y_pred[i]) - 1)
        error_pct = 100 * abs(actual_likes - predicted_likes) / max(1, actual_likes)
        log_fn(
            f"Actual: {actual_likes:7d} | Predicted: {predicted_likes:7d} | Error: {error_pct:5.1f}%"
        )


def build_sample_errors(
    post_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Build per-sample absolute error dictionary.

    Args:
        post_ids: Array of post identifiers.
        y_true: Array of true values.
        y_pred: Array of predicted values.
        texts: Optional array of text descriptions.

    Returns:
        Dictionary mapping post_id -> {"error": float, "text": str}.
    """
    errors = np.abs(y_true - y_pred)
    sample_errors = {}

    for i, (pid, err) in enumerate(zip(post_ids, errors)):
        entry = {"error": float(err)}
        if texts is not None:
            entry["text"] = str(texts[i])
        sample_errors[str(pid)] = entry

    return sample_errors
