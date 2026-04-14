"""Unified sentence embedding generation."""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"


def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress_bar: bool = True,
    trust_remote_code: bool = True,
    cleanup_gpu: bool = False,
) -> np.ndarray:
    """Generate sentence embeddings using SentenceTransformer.

    Args:
        texts: List of text strings to encode.
        model_name: SentenceTransformer model identifier.
            Common options:
            - "all-mpnet-base-v2" (768 dims, balanced quality)
            - "all-MiniLM-L6-v2" (384 dims, fastest)
            - "all-roberta-large-v1" (1024 dims, best quality)
            - "google/embeddinggemma-300m" (Google embedding model)
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to display encoding progress.
        trust_remote_code: Whether to trust remote code for model loading.
        cleanup_gpu: Whether to delete the model and clear GPU cache after encoding.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass

    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code, device=device)
    dim = model.get_embedding_dimension()
    logger.info("Encoding %d texts with %s (dim=%d)", len(texts), model_name, dim)

    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
    )

    if cleanup_gpu:
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return embeddings