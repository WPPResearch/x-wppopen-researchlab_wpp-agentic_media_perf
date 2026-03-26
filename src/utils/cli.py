"""Shared CLI argument parsing and data path resolution for training scripts."""

import argparse
import os
from pathlib import Path

from src.config import DATASET_SUBSETS
from src.dataset.dataloader import load_dataset


def create_data_argparser(description: str = "Training script") -> argparse.ArgumentParser:
    """Create an ArgumentParser with common dataset arguments.

    Args:
        description: Script description for --help output.

    Returns:
        ArgumentParser with --dataset, --text_output_dir, --include-user, --embedding args.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SUBSETS.keys()),
        default="top100",
        help="Dataset to use: " + ", ".join(DATASET_SUBSETS.keys()),
    )
    parser.add_argument(
        "--text_output_dir",
        type=str,
        default="checkpoints/run_20260314_124407/it_3/text_output",
        help="Path to text outputs directory (relative to dataset dir)",
    )
    parser.add_argument(
        "--include-user",
        action="store_true",
        default=True,
        help="Include user info in text",
    )
    parser.add_argument(
        "--embedding",
        default="all-mpnet-base-v2",
        help="Sentence transformer model (default: all-mpnet-base-v2)",
    )
    return parser


def resolve_data_paths(args: argparse.Namespace):
    """Resolve DATA_DIR, TEXT_DIR, JSON_DIR from parsed arguments.

    Args:
        args: Parsed arguments containing dataset and text_output_dir.

    Returns:
        Tuple of (data_dir, text_dir, json_dir) as strings.
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
    data_dir = str(base_dir / "data" / DATASET_SUBSETS[args.dataset])
    text_dir = str(args.text_output_dir)
    json_dir = os.path.join(data_dir, "json_subset")
    return data_dir, text_dir, json_dir


def load_data_from_args(args: argparse.Namespace):
    """Load dataset DataFrame based on parsed CLI arguments.

    Args:
        args: Parsed arguments with dataset, text_output_dir, include_user fields.

    Returns:
        pandas DataFrame with text and target columns.
    """
    data_dir, text_dir, json_dir = resolve_data_paths(args)
    include_user = getattr(args, "include_user", True)

    print(f"Loading data from {data_dir}...")
    return load_dataset(
        data_dir,
        text_dir=text_dir,
        json_dir=json_dir,
        classification_task="perf",
        include_user_info=include_user,
        include_temporal=include_user,
        include_influencer_details=include_user,
    )
