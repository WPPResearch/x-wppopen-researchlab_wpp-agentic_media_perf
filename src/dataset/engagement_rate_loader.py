"""
Load data with engagement rate instead of absolute likes.
Engagement rate = likes / followers (more normalized metric)
"""

import logging
import pandas as pd

from src.dataset.loaders import parse_influencer_profile
from pathlib import Path
from src.dataset.dataloader import load_dataset
from src.config import INFLUENCER_DATASET_PATH as _DEFAULT_INFLUENCER_PATH

logger = logging.getLogger(__name__)


def load_engagement_rate_data(
    data_dir: str,
    text_dir: str = None,
    json_dir: str = None,
    include_user_info: bool = True,
    min_followers: int = 100,
    influencer_dir: str = _DEFAULT_INFLUENCER_PATH,
) -> pd.DataFrame:
    """
    Load data with engagement rate as target (likes / followers).

    Uses the unified load_dataset loader, then replaces the target
    with engagement rate = 100 * likes / followers.

    Args:
        data_dir: Base directory
        text_dir: Text files directory
        json_dir: JSON files directory
        include_user_info: Add username to text
        min_followers: Minimum follower count to include (avoid division by small numbers)
        influencer_dir: Path to influencer profiles directory

    Returns:
        DataFrame with 'text', 'y' (engagement_rate), 'likes', 'followers', 'influencer_name'
    """

    # Load base data with likes as target
    df = load_dataset(
        data_dir,
        text_dir=text_dir,
        json_dir=json_dir,
        classification_task="perf",
        include_user_info=include_user_info,
        include_temporal=False,
        include_influencer_details=True,
        influencer_dir=influencer_dir,
    )

    if df.empty:
        return df

    # Add follower counts from influencer profiles
    followers_list = []
    for _, row in df.iterrows():
        influencer_file = (Path(influencer_dir) / row["influencer_name"]).resolve()
        profile = parse_influencer_profile(influencer_file)
        followers_list.append(profile["followers"] if profile else 0)

    df["followers"] = followers_list
    df["likes"] = df["y"].astype(int)

    # Filter out low-follower accounts
    skipped = (df["followers"] < min_followers).sum()
    df = df[df["followers"] >= min_followers].copy()

    # Calculate engagement rate
    df["y"] = 100.0 * df["likes"] / df["followers"]

    logger.info(
        "Loaded %d posts (skipped %d with <%d followers)",
        len(df), skipped, min_followers,
    )

    if not df.empty:
        logger.info("Engagement rate statistics:\n%s", df["y"].describe())
        logger.info("Top engagement rates:\n%s",
            df.nlargest(5, "y")[["influencer_name", "likes", "followers", "y"]])

    return df


if __name__ == "__main__":
    data_dir = "data/influencer_dataset_top20_min100posts_nonsponsored"
    text_dir = "checkpoints/run_20260314_124407/it_3/text_output"
    logging.basicConfig(level=logging.INFO)
    df = load_engagement_rate_data(data_dir, text_dir=text_dir, min_followers=100)
    print(df.head())
