import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.dataset.loaders import parse_influencer_profile
from src.utils.temporal_features import extract_temporal_features
from src.utils.experiment_tracker import ExperimentTracker, create_experiment_summary
from src.utils.metrics import log_sample_predictions
from src.training.model_configs import DEFAULT_XGBOOST_PARAMS, DEFAULT_LIGHTGBM_PARAMS
from src.config import DATASET_SUBSETS, INFLUENCER_DATASET_PATH as _DEFAULT_INFLUENCER_PATH

logger = logging.getLogger(__name__)


def extract_post_features(json_file: str, text_content: str) -> dict:
    """Extract numerical features from JSON and text content."""
    features = {}

    # Load JSON to get metadata
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Comment count
        features["comment_count"] = data.get("edge_media_to_comment", {}).get(
            "count", 0
        )

        # Is video
        features["is_video"] = 1 if data.get("is_video", False) else 0

        # Image dimensions / aspect ratio
        dims = data.get("dimensions", {})
        height = dims.get("height", 1)
        width = dims.get("width", 1)
        features["aspect_ratio"] = width / height if height > 0 else 1.0
        features["image_area"] = width * height

        # Extract caption text
        caption = ""
        edges = data.get("edge_media_to_caption", {}).get("edges", [])
        if edges and len(edges) > 0:
            caption = edges[0].get("node", {}).get("text", "")

    except (json.JSONDecodeError, KeyError, IOError) as e:
        logger.warning("Error reading JSON %s: %s", json_file, e)
        # Set defaults
        features["comment_count"] = 0
        features["is_video"] = 0
        features["aspect_ratio"] = 1.0
        features["image_area"] = 1000000
        caption = ""

    # Caption features
    features["caption_length"] = len(caption)
    features["hashtag_count"] = caption.count("#")
    features["mention_count"] = caption.count("@")

    # Count emojis (rough approximation)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "]+",
        flags=re.UNICODE,
    )
    features["emoji_count"] = len(emoji_pattern.findall(caption))

    # Text features
    features["text_length"] = len(text_content)

    # Temporal features from timestamp
    try:
        timestamp = data.get("taken_at_timestamp", None)
        if timestamp:
            temporal_features = extract_temporal_features(timestamp)
            # Add selected temporal features (not all, to avoid too many features)
            features["hour_of_day"] = temporal_features["hour_of_day"]
            features["day_of_week"] = temporal_features["day_of_week"]
            features["is_weekend"] = temporal_features["is_weekend"]
            features["is_peak_hours"] = temporal_features["is_peak_hours"]
            features["month"] = temporal_features["month"]
            features["is_summer"] = temporal_features["is_summer"]
            # Cyclical encodings (better for ML models)
            features["hour_sin"] = temporal_features["hour_sin"]
            features["hour_cos"] = temporal_features["hour_cos"]
            features["day_sin"] = temporal_features["day_sin"]
            features["day_cos"] = temporal_features["day_cos"]
        else:
            # Default values if no timestamp
            features["hour_of_day"] = 12
            features["day_of_week"] = 2
            features["is_weekend"] = 0
            features["is_peak_hours"] = 0
            features["month"] = 6
            features["is_summer"] = 0
            features["hour_sin"] = 0
            features["hour_cos"] = 1
            features["day_sin"] = 0
            features["day_cos"] = 1
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Error extracting temporal features: %s", e)
        # Use defaults
        features["hour_of_day"] = 12
        features["day_of_week"] = 2
        features["is_weekend"] = 0
        features["is_peak_hours"] = 0
        features["month"] = 6
        features["is_summer"] = 0
        features["hour_sin"] = 0
        features["hour_cos"] = 1
        features["day_sin"] = 0
        features["day_cos"] = 1

    return features


def load_data_with_features(data_dir: str, influencer_dir: str = _DEFAULT_INFLUENCER_PATH):
    """Load data with both text and numerical features."""
    text_dir = os.path.join(data_dir, "text_output_gemini_2.5_flash", "responses")
    json_dir = os.path.join(data_dir, "json_subset")
    annotation_file = Path(os.path.join(data_dir, "subset_annotations.txt"))

    if not annotation_file.exists():
        logger.error("Annotation file not found at %s", annotation_file)
        return None

    from src.dataset.loaders import parse_annotation_line

    data_records = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info("Processing %d posts...", len(lines))

    for line in lines:
        post_info = parse_annotation_line(line)
        json_file = os.path.join(json_dir, post_info["json_filename"])
        text_file = os.path.join(
            text_dir, post_info["json_filename"].replace(".json", ".txt")
        )

        if not os.path.exists(json_file) or not os.path.exists(text_file):
            continue

        # Read text
        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        # Add influencer name
        text_content = (
            f"This post is from {post_info['influencer_name']}. {text_content}"
        )

        # Get influencer profile
        influencer_name = post_info["influencer_name"]
        influencer_file = (Path(influencer_dir) / influencer_name).resolve()
        if not influencer_file.is_relative_to(Path(influencer_dir).resolve()):
            logger.warning("Suspicious influencer name skipped: %s", influencer_name)
            continue
        influencer_data = parse_influencer_profile(influencer_file)

        if influencer_data is None:
            continue

        # Add influencer profile to text
        text_content += f" Influencer profile: {influencer_data['text_desc']} bio {influencer_data['bio']} Category {influencer_data['category']}."

        # Get likes
        with open(json_file, "r", encoding="utf-8") as f:
            json_content = json.load(f)

        if "edge_media_preview_like" not in json_content:
            continue

        likes = int(json_content["edge_media_preview_like"]["count"])

        # Extract post features
        post_features = extract_post_features(json_file, text_content)

        # Extract influencer features
        influencer_features = {
            "followers": influencer_data["followers"],
            "followees": influencer_data["followees"],
            "posts_count": influencer_data["posts"],
            "follower_followee_ratio": (
                influencer_data["followers"] / max(1, influencer_data["followees"])
            ),
        }

        # Combine all
        record = {
            "text": text_content,
            "likes": likes,
            "influencer_name": post_info["influencer_name"],
            **post_features,
            **influencer_features,
        }

        data_records.append(record)

    df = pd.DataFrame(data_records)
    logger.info("Loaded %d samples with features.", len(df))
    return df


def train_with_features(df: pd.DataFrame, embedding_model: str = "all-mpnet-base-v2") -> None:
    """Train models using text embeddings + numerical features."""
    if df.empty:
        logger.warning("No data loaded.")
        return

    # Create experiment tracker
    with ExperimentTracker("feature_engineered_regression") as tracker:
        tracker.log("\n=== Dataset Statistics ===")
        tracker.log(f"Likes distribution:\n{df['likes'].describe()}")
        tracker.log(
            f"\nLog target distribution:\n{np.log(df['likes'] + 1.0).describe()}"
        )

        # Prepare target (log-transformed)
        y = np.log(df["likes"] + 1.0)

        # Save parameters
        params = {
            "n_samples": len(df),
            "embedding_model": embedding_model,
            "test_size": 0.2,
            "random_state": 42,
            "target_transform": "log(likes + 1)",
            "features_used": "text_embeddings + post_metadata + influencer_data + temporal",
        }
        tracker.save_params(params)

        # Generate text embeddings
        tracker.log("\n=== Generating Embeddings ===")
        model = SentenceTransformer(embedding_model)
        text_embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

        # Get numerical features
        feature_cols = [
            # Post metadata
            "comment_count",
            "is_video",
            "aspect_ratio",
            "caption_length",
            "hashtag_count",
            "mention_count",
            "emoji_count",
            "text_length",
            # Influencer features
            "followers",
            "followees",
            "posts_count",
            "follower_followee_ratio",
            # Temporal features
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_peak_hours",
            "month",
            "is_summer",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]

        numerical_features = df[feature_cols].values

        # Standardize numerical features
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)

        # Combine embeddings + numerical features
        X = np.hstack([text_embeddings, numerical_features_scaled])

        tracker.log("\nFeature dimensions:")
        tracker.log(f"  - Text embeddings: {text_embeddings.shape[1]}")
        tracker.log(f"  - Numerical features: {numerical_features.shape[1]}")
        tracker.log(f"  - Total features: {X.shape[1]}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        tracker.log(f"\nTraining set size: {len(X_train)}")
        tracker.log(f"Test set size: {len(X_test)}")

        # Define models
        models = {
            "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        }

        # Add XGBoost/LightGBM if available
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(**DEFAULT_XGBOOST_PARAMS)
            logger.info("XGBoost available - added to models")
        except ImportError:
            logger.info("XGBoost not installed")

        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(**DEFAULT_LIGHTGBM_PARAMS)
            logger.info("LightGBM available - added to models")
        except ImportError:
            logger.info("LightGBM not installed")

        # Train and evaluate
        best_model_name = ""
        best_r2 = -float("inf")
        best_model = None
        best_predictions = None
        all_model_metrics = {}

        for name, model in models.items():
            tracker.log(f"\n{'=' * 60}")
            tracker.log(f"Training {name}")
            tracker.log(f"{'=' * 60}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store metrics
            model_metrics = {
                f"{name}_MAE": mae,
                f"{name}_RMSE": rmse,
                f"{name}_R2": r2,
            }
            all_model_metrics.update(model_metrics)

            tracker.log(f"MAE: {mae:.4f}")
            tracker.log(f"RMSE: {rmse:.4f}")
            tracker.log(f"R² Score: {r2:.4f}")

            # Show sample predictions
            log_sample_predictions(y_test, y_pred, n=10, log_fn=tracker.log)

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model
                best_predictions = y_pred

        # Save all metrics
        tracker.save_metrics(all_model_metrics, stage="test")

        # Save best model
        if best_model is not None:
            tracker.save_model(
                best_model, model_name=f"best_{best_model_name.replace(' ', '_')}"
            )
            tracker.save_predictions(y_test.values, best_predictions, stage="test")

        # Create and save summary
        summary = create_experiment_summary(
            "Feature Engineered Regression",
            params,
            {"Best_Model": best_model_name, "Best_R2": best_r2, **all_model_metrics},
            best_model_name,
        )
        tracker.save_summary(summary)

        tracker.log(f"\n{'=' * 60}")
        tracker.log(f"BEST MODEL: {best_model_name} with R² = {best_r2:.4f}")
        tracker.log(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train engagement prediction with feature engineering."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SUBSETS.keys()),
        default="10k",
        help="Dataset to use: " + ", ".join(DATASET_SUBSETS.keys()),
    )
    parser.add_argument(
        "--embedding",
        default="all-mpnet-base-v2",
        help="Sentence transformer model",
    )
    parser.add_argument(
        "--influencer-profiles-dir",
        type=str,
        default=None,
        help="Path to users_influencers_SPOD directory. Defaults to INFLUENCER_DATASET_PATH env var.",
    )
    args = parser.parse_args()

    # Define paths
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
    DATA_DIR = str(BASE_DIR / "data" / DATASET_SUBSETS[args.dataset])
    influencer_dir = args.influencer_profiles_dir or _DEFAULT_INFLUENCER_PATH

    logger.info("Loading data from %s...", DATA_DIR)
    df = load_data_with_features(DATA_DIR, influencer_dir=influencer_dir)

    if df is not None and not df.empty:
        print("\nDataset preview:")
        print(df.head())
        print("\nFeature statistics:")
        print(df.describe())
        train_with_features(df, embedding_model=args.embedding)
