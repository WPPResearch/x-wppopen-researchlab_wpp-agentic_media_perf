import glob
import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.dataset.loaders import parse_annotation_line, parse_influencer_profile
from src.utils.temporal_features import timestamp_to_text
from src.utils.safe_path import safe_path_join
from src.config import INFLUENCER_DATASET_PATH as _DEFAULT_INFLUENCER_PATH

logger = logging.getLogger(__name__)

MAX_NUMBER_OF_LIKES = 1.0


def load_data(text_dir: str, json_dir: str) -> pd.DataFrame:
    """Load text and JSON data, returning a DataFrame with text and likes."""
    data = []
    text_files = glob.glob(os.path.join(text_dir, "*.txt"))

    logger.info("Found %d text files.", len(text_files))

    for text_file in text_files:
        try:
            file_id = os.path.splitext(os.path.basename(text_file))[0]
            json_file = os.path.join(json_dir, f"{file_id}.json")

            if not os.path.exists(json_file):
                continue

            with open(text_file, "r", encoding="utf-8") as f:
                text_content = f.read().strip()

            with open(json_file, "r", encoding="utf-8") as f:
                json_content = json.load(f)

            if "edge_media_preview_like" in json_content:
                likes = (
                    int(json_content["edge_media_preview_like"]["count"])
                    / MAX_NUMBER_OF_LIKES
                )

            else:
                continue

            data.append(
                {
                    #'id': file_id,
                    "text": text_content,
                    "y": likes,
                }
            )

        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning("Error processing %s: %s", text_file, e)
            continue

    return pd.DataFrame(data)


def load_dataset(
    data_dir: str,
    text_dir: str = None,
    json_dir: str = None,
    classification_task: str = "sponsor_label",
    include_user_info: bool = True,
    include_temporal: bool = True,
    include_influencer_details: bool = True,
    influencer_dir: str = _DEFAULT_INFLUENCER_PATH,
) -> pd.DataFrame:
    """
    Unified data loader for annotation-based datasets.

    Args:
        data_dir: Base directory containing dataset.
        text_dir: Directory with text descriptions (optional, defaults to standard path).
        json_dir: Directory with JSON files (optional, defaults to json_subset).
        classification_task: Either 'sponsor_label' or 'perf' for performance prediction.
        include_user_info: If True, prepends "This post is from {influencer_name}." to text.
        include_temporal: If True, appends temporal context from post timestamp.
        include_influencer_details: If True, appends influencer bio and category.
            If False, appends only influencer text_desc.
        influencer_dir: Path to the users_influencers_SPOD directory.
    """
    if text_dir is None:
        text_dir = os.path.join(data_dir, "text_output_gemini_2.5_flash", "responses")
    if json_dir is None:
        json_dir = os.path.join(data_dir, "json_subset")

    logger.info("Number of txt files: %d", len(glob.glob(os.path.join(text_dir, "*.txt"))))
    logger.info("Number of json files: %d", len(glob.glob(os.path.join(json_dir, "*.json"))))

    data = []
    annotation_file = safe_path_join(data_dir, "subset_annotations.txt")
    if not annotation_file.exists():
        logger.error("Annotation file not found at %s", annotation_file)
        return pd.DataFrame()

    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    logger.info("%d lines found in annotation file.", len(lines))

    for line in lines:
        post_info = parse_annotation_line(line)
        try:
            json_file = str(safe_path_join(json_dir, post_info["json_filename"]))
            text_file = str(safe_path_join(
                text_dir, post_info["json_filename"].replace(".json", ".txt")
            ))
        except ValueError:
            logger.warning("Suspicious filename skipped: %s", post_info.get("json_filename"))
            continue
        sponsor_label = post_info.get("sponsor_label", 0)
        if not os.path.exists(json_file) or not os.path.exists(text_file):
            logger.warning("Missing files for post: %s", post_info.get("post_id", "unknown"))
            continue

        influencer_name = post_info["influencer_name"]
        try:
            influencer_file = safe_path_join(influencer_dir, influencer_name)
        except ValueError:
            logger.warning("Suspicious influencer name skipped: %s", influencer_name)
            continue

        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        # Load JSON to get timestamp and other metadata
        with open(json_file, "r", encoding="utf-8") as f:
            json_content = json.load(f)

        # Add influencer name at the beginning of text
        if include_user_info:
            text_content = (
                f"This post is from {post_info['influencer_name']}. {text_content}"
            )

        # Add temporal context from timestamp
        if include_temporal:
            timestamp = json_content.get("taken_at_timestamp", None)
            if timestamp:
                temporal_text = timestamp_to_text(timestamp)
                text_content += f"\n {temporal_text}"

        influencer_content = parse_influencer_profile(influencer_file)
        if include_influencer_details:
            text_content += f" Influencer profile: {influencer_content['text_desc']} bio {influencer_content['bio']} Category {influencer_content['category']}."
        else:
            text_content += f" Influencer profile: {influencer_content['text_desc']}."

        if "edge_media_preview_like" in json_content:
            likes = (
                int(json_content["edge_media_preview_like"]["count"])
                / MAX_NUMBER_OF_LIKES
            )
            popularity_score = likes
        else:
            logger.info("No likes detected for post: %s", post_info["post_id"])
            continue

        y = (
            sponsor_label
            if classification_task == "sponsor_label"
            else popularity_score
        )
        data.append(
            {
                "post_id": post_info["post_id"],
                "text": text_content,
                "y": y,
                "influencer_name": post_info["influencer_name"],
            }
        )

    return pd.DataFrame(data)


def load_datav2(
    data_dir: str,
    text_dir: str = None,
    json_dir: str = None,
    classification_task: str = "sponsor_label",
    influencer_dir: str = _DEFAULT_INFLUENCER_PATH,
) -> pd.DataFrame:
    """Load data without user info prefix, temporal text, or influencer bio/category details."""
    return load_dataset(
        data_dir,
        text_dir=text_dir,
        json_dir=json_dir,
        classification_task=classification_task,
        include_user_info=False,
        include_temporal=False,
        include_influencer_details=False,
        influencer_dir=influencer_dir,
    )


def load_datav3(
    data_dir: str,
    text_dir: str = None,
    json_dir: str = None,
    classification_task: str = "sponsor_label",
    include_user_info: bool = True,
    influencer_dir: str = _DEFAULT_INFLUENCER_PATH,
) -> pd.DataFrame:
    """Load data with user info, temporal text, and full influencer profile details."""
    return load_dataset(
        data_dir,
        text_dir=text_dir,
        json_dir=json_dir,
        classification_task=classification_task,
        include_user_info=include_user_info,
        include_temporal=True,
        include_influencer_details=True,
        influencer_dir=influencer_dir,
    )


def convert_to_classification_labels(df):
    """
    Converts continuous engagement rate (y) into 3 classes:
    0: Low Performance (Bottom 15%)
    1: Average (Middle 70%)
    2: High Performance (Top 15%)
    """
    if df.empty:
        return df

    # Calculate percentiles
    low_threshold = df["y"].quantile(0.15)
    high_threshold = df["y"].quantile(0.85)

    logger.info("--- Classification Thresholds ---")
    logger.info("Low Performance (< 15%%): y <= %.4f", low_threshold)
    logger.info("High Performance (> 85%%): y >= %.4f", high_threshold)
    logger.info("Average: %.4f < y < %.4f", low_threshold, high_threshold)

    def categorize(val):
        if val <= low_threshold:
            return 0  # Low
        elif val >= high_threshold:
            return 2  # High
        else:
            return 1  # Average

    df["y_original"] = df["y"]
    df["y"] = df["y"].apply(categorize)

    logger.info("Class distribution:\n%s", df["y"].value_counts().sort_index())

    return df
