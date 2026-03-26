"""Automatic dataset preparation: split creation and GCS image upload."""

import logging
import os
import re
import shutil
import sys
from pathlib import Path

from google.cloud import storage

from src.config import (
    DATASET_SPLIT_PARAMS,
    GCS_BUCKET_NAME,
    GCS_BUCKET_PREFIX,
    INFLUENCER_DATASET_PATH,
    PROJECT_ROOT,
    RAW_DATASET_PATH,
)

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_URL = (
    "https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset"
)

# Expected files/dirs inside the raw dataset download
RAW_ANNOTATION_FILE = "post_info.txt"
RAW_IMAGES_DIR = "img_resized"
RAW_PROFILES_DIR = "users_influencers_SPOD"

_GCS_BUCKET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,61}[a-z0-9]$")


def _validate_raw_dataset(raw_path: Path) -> bool:
    """Check that the raw dataset directory has the expected structure."""
    required = [
        raw_path / RAW_ANNOTATION_FILE,
        raw_path / RAW_IMAGES_DIR,
        raw_path / RAW_PROFILES_DIR,
    ]
    return all(p.exists() for p in required)


def _validate_path_within_project(path: Path, label: str) -> Path:
    """Resolve a path and verify it's within the project root."""
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(
            f"{label} must be under project root ({project_root}), got: {resolved}"
        )
    return resolved


def _ensure_influencer_profiles(
    raw_path: Path,
    influencer_profiles_path: str = INFLUENCER_DATASET_PATH,
) -> None:
    """Copy users_influencers_SPOD from raw dataset if not already present."""
    dest = _validate_path_within_project(
        Path(influencer_profiles_path), "influencer_profiles_path"
    )
    if dest.exists() and any(dest.iterdir()):
        logger.info("Influencer profiles already present at %s", dest)
        return

    source = raw_path / RAW_PROFILES_DIR
    logger.info("Copying influencer profiles from %s to %s ...", source, dest)
    shutil.copytree(str(source), str(dest), dirs_exist_ok=True)
    logger.info("Influencer profiles copied.")


def ensure_subset_exists(
    subset_dir: Path,
    dataset_size: str,
    raw_dataset_path: str = RAW_DATASET_PATH,
    influencer_profiles_path: str = INFLUENCER_DATASET_PATH,
) -> None:
    """Ensure the dataset split exists, creating it from raw data if needed.

    Args:
        subset_dir: Target directory for the split (e.g. data/influencer_dataset_top20_...)
        dataset_size: Split key (e.g. "top20", "top100")
        raw_dataset_path: Path to the raw downloaded dataset
        influencer_profiles_path: Path to the users_influencers_SPOD directory

    Raises:
        SystemExit: If raw dataset is not found (prints download instructions)
        ValueError: If dataset_size has no split parameters defined
    """
    # Check if split already exists
    annotations = subset_dir / "subset_annotations.txt"
    json_dir = subset_dir / "json_subset"
    img_dir = subset_dir / "img_subset"

    if annotations.exists() and json_dir.exists() and img_dir.exists():
        logger.info("Dataset split '%s' found at %s", dataset_size, subset_dir)
        return

    # Need to create the split from raw data
    logger.info("Dataset split '%s' not found at %s. Will create from raw data.", dataset_size, subset_dir)

    if dataset_size not in DATASET_SPLIT_PARAMS:
        raise ValueError(
            f"No split parameters defined for '{dataset_size}'. "
            f"Available splits: {list(DATASET_SPLIT_PARAMS.keys())}. "
            f"For '1k' or '10k' splits, create them manually using src/utils/subset_creation.py"
        )

    raw_path = Path(raw_dataset_path)
    if not _validate_raw_dataset(raw_path):
        logger.error("Raw dataset not found or incomplete at: %s", raw_path)
        print("\n" + "=" * 70)
        print("DATASET NOT FOUND")
        print("=" * 70)
        print(f"\nThe dataset split '{dataset_size}' does not exist and the raw dataset")
        print(f"was not found at: {raw_path}")
        print(f"\nTo set up the dataset:")
        print(f"  1. Download the Instagram Influencer dataset from:")
        print(f"     {DATASET_DOWNLOAD_URL}")
        print(f"  2. Extract it to: {raw_path}")
        print(f"  3. Verify it contains:")
        print(f"     - {RAW_ANNOTATION_FILE}")
        print(f"     - {RAW_IMAGES_DIR}/")
        print(f"     - {RAW_PROFILES_DIR}/")
        print(f"  4. Re-run this command.")
        print(f"\n  Alternatively, set RAW_DATASET_PATH in .env to point to")
        print(f"  your download location.")
        print("=" * 70 + "\n")
        sys.exit(1)

    # Ensure influencer profiles are available for dataloaders
    _ensure_influencer_profiles(raw_path, influencer_profiles_path)

    # Create the split
    split_params = DATASET_SPLIT_PARAMS[dataset_size]
    _create_split(
        raw_path=raw_path,
        target_dir=subset_dir,
        num_influencers=split_params["num_influencers"],
        min_posts=split_params["min_posts"],
    )


def _create_split(
    raw_path: Path,
    target_dir: Path,
    num_influencers: int,
    min_posts: int,
) -> None:
    """Create a dataset split from the raw dataset."""
    from src.utils.subset_creation import create_ranked_subset_by_followers

    logger.info(
        "Creating dataset split: top %d influencers with >= %d posts",
        num_influencers, min_posts,
    )

    os.makedirs(target_dir, exist_ok=True)

    result = create_ranked_subset_by_followers(
        source_img_dir=str(raw_path / RAW_IMAGES_DIR),
        annotation_file=str(raw_path / RAW_ANNOTATION_FILE),
        influencer_profiles_dir=str(raw_path / RAW_PROFILES_DIR),
        target_dir=str(target_dir),
        min_posts_per_influencer=min_posts,
        num_influencers=num_influencers,
        sponsor_filter=0,  # non-sponsored only
        dry_run=False,
        copy_json=True,
    )

    if result is None:
        raise RuntimeError(f"Failed to create dataset split at {target_dir}")

    logger.info(
        "Dataset split created: %d influencers, %d posts",
        result["num_influencers"], result["num_posts"],
    )


def _validate_bucket_name(bucket_name: str) -> None:
    """Validate GCS bucket name format."""
    if not bucket_name:
        raise ValueError(
            "GCS_BUCKET_NAME is not set. Add it to your .env file.\n"
            "Create a bucket with: gsutil mb -l us-central1 gs://your-bucket-name\n"
            "Then set GCS_BUCKET_NAME=your-bucket-name in .env"
        )
    if not _GCS_BUCKET_NAME_RE.match(bucket_name):
        raise ValueError(
            f"Invalid GCS bucket name: '{bucket_name}'. "
            "Bucket names must be 3-63 characters, lowercase letters, numbers, hyphens, underscores, and dots."
        )


def ensure_images_on_gcs(
    subset_dir: Path,
    bucket_name: str = GCS_BUCKET_NAME,
    bucket_prefix: str = GCS_BUCKET_PREFIX,
) -> None:
    """Upload subset images to GCS if not already present.

    Args:
        subset_dir: Local path to the subset directory
        bucket_name: GCS bucket name
        bucket_prefix: Prefix path within the bucket

    Raises:
        ValueError: If GCS_BUCKET_NAME is not configured or invalid
    """
    _validate_bucket_name(bucket_name)

    subset_name = subset_dir.name
    gcs_img_prefix = f"{bucket_prefix}/{subset_name}/img_subset"
    local_img_dir = subset_dir / "img_subset"

    if not local_img_dir.exists():
        raise FileNotFoundError(f"Local image directory not found: {local_img_dir}")

    # Check if images already exist on GCS by listing a few blobs
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_img_prefix, max_results=5))

        if len(blobs) > 0:
            logger.info(
                "Images already present on GCS at gs://%s/%s (%d+ files found)",
                bucket_name, gcs_img_prefix, len(blobs),
            )
            return
    except Exception as e:
        logger.warning("Could not check GCS bucket: %s. Will attempt upload.", e)

    # Upload images
    logger.info(
        "Uploading images from %s to gs://%s/%s ...",
        local_img_dir, bucket_name, gcs_img_prefix,
    )

    from src.utils.gcp_utils import upload_folder_to_gcs

    uploaded = upload_folder_to_gcs(
        bucket_name=bucket_name,
        local_folder_path=str(local_img_dir),
        gcs_destination_folder=gcs_img_prefix,
    )

    if not uploaded:
        raise RuntimeError(
            f"Failed to upload images to gs://{bucket_name}/{gcs_img_prefix}. "
            "Check your GCS credentials and bucket permissions."
        )

    logger.info("Uploaded %d images to GCS.", len(uploaded))
