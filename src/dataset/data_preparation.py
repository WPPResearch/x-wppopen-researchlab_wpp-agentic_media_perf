"""Automatic dataset preparation: split creation and GCS image upload."""

import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from google.cloud import storage
from PIL import Image
from tqdm import tqdm

from src.config import (
    DATASET_SPLIT_PARAMS,
    GCS_BUCKET_NAME,
    GCS_BUCKET_PREFIX,
    INFLUENCER_DATASET_PATH,
    PROJECT_ROOT,
    RAW_DATASET_PATH,
)
from src.utils.safe_path import safe_path_join

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_URL = (
    "https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset"
)

# Expected files/dirs inside the raw dataset download
RAW_ANNOTATION_FILE = "post_info.txt"
RAW_IMAGES_DIR = "img_resized"
RAW_PROFILES_DIR = "users_influencers_SPOD"
RAW_JSON_DIR = "json"

# Zip file patterns in the raw download
_IMAGE_ZIP_PATTERN = re.compile(r"^img_fi\d+\.zip$")
_JSON_ZIP = "json_files.zip"
_PROFILES_INFLUENCERS_ZIP = "profiles_influencers.zip"
_PROFILES_BRANDS_ZIP = "profiles_brands.zip"

_IMAGE_RESIZE_TARGET = (256, 256)

_GCS_BUCKET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,61}[a-z0-9]$")


def _extract_zip_flat(zip_path: Path, dest_dir: Path, *, extensions: tuple[str, ...] | None = None) -> int:
    """Extract files from a zip into dest_dir, flattening any subdirectories.

    Args:
        zip_path: Path to the zip file.
        dest_dir: Directory to extract into.
        extensions: If provided, only extract files with these extensions (e.g. (".jpg", ".png")).

    Returns:
        Number of files extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            filename = Path(member.filename).name
            if not filename:
                continue
            if extensions and not filename.lower().endswith(extensions):
                continue
            try:
                target = safe_path_join(str(dest_dir), filename)
            except ValueError:
                logger.warning("Suspicious zip member name skipped: %s", member.filename)
                continue
            if not target.exists():
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    return count


def _resize_images(img_dir: Path, target_size: tuple[int, int] = _IMAGE_RESIZE_TARGET) -> int:
    """Resize all images in a directory to target_size.

    Returns:
        Number of images resized.
    """
    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not image_files:
        return 0

    resized = 0
    for img_path in tqdm(image_files, desc="Resizing images"):
        try:
            with Image.open(img_path) as img:
                if img.size == target_size:
                    continue
                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(img_path)
                resized += 1
        except Exception as e:
            logger.warning("Failed to resize %s: %s", img_path.name, e)
    return resized


def _extract_raw_zips(raw_path: Path) -> bool:
    """Extract zip files in the raw dataset directory if not already extracted.

    Handles:
        - img_fi01.zip ... img_fi16.zip -> img_resized/ (flattened + resized to 256x256)
        - json_files.zip -> json/
        - profiles_influencers.zip -> users_influencers_SPOD/
        - profiles_brands.zip -> users_brands_SPOD/

    Returns:
        True if extraction was performed or data already exists; False if no zips found.
    """
    # Detect available zips
    image_zips = sorted(
        p for p in raw_path.iterdir()
        if p.is_file() and _IMAGE_ZIP_PATTERN.match(p.name)
    )
    json_zip = raw_path / _JSON_ZIP
    profiles_inf_zip = raw_path / _PROFILES_INFLUENCERS_ZIP
    profiles_brands_zip = raw_path / _PROFILES_BRANDS_ZIP

    has_any_zip = bool(image_zips) or json_zip.exists() or profiles_inf_zip.exists()
    if not has_any_zip:
        return False

    logger.info("Found zip files in %s — checking if extraction is needed...", raw_path)

    # --- Image zips -> img_resized/ ---
    img_dir = raw_path / RAW_IMAGES_DIR
    if image_zips and not (img_dir.exists() and any(img_dir.iterdir())):
        logger.info("Extracting %d image zip(s) into %s ...", len(image_zips), img_dir)
        total = 0
        for zp in tqdm(image_zips, desc="Extracting image zips"):
            total += _extract_zip_flat(zp, img_dir, extensions=(".jpg", ".jpeg", ".png"))
        logger.info("Extracted %d images.", total)

        logger.info("Resizing images to %s ...", _IMAGE_RESIZE_TARGET)
        resized = _resize_images(img_dir, _IMAGE_RESIZE_TARGET)
        logger.info("Resized %d images.", resized)
    elif img_dir.exists():
        logger.info("Image directory already exists at %s, skipping extraction.", img_dir)

    # --- json_files.zip -> json/ ---
    json_dir = raw_path / RAW_JSON_DIR
    if json_zip.exists() and not (json_dir.exists() and any(json_dir.iterdir())):
        logger.info("Extracting %s ...", json_zip.name)
        count = _extract_zip_flat(json_zip, json_dir, extensions=(".json",))
        logger.info("Extracted %d JSON files.", count)
    elif json_dir.exists():
        logger.info("JSON directory already exists at %s, skipping extraction.", json_dir)

    # --- profiles_influencers.zip -> users_influencers_SPOD/ ---
    profiles_inf_dir = raw_path / RAW_PROFILES_DIR
    if profiles_inf_zip.exists() and not (profiles_inf_dir.exists() and any(profiles_inf_dir.iterdir())):
        logger.info("Extracting %s ...", profiles_inf_zip.name)
        count = _extract_zip_flat(profiles_inf_zip, profiles_inf_dir)
        logger.info("Extracted %d influencer profiles.", count)
    elif profiles_inf_dir.exists():
        logger.info("Influencer profiles already exist at %s, skipping extraction.", profiles_inf_dir)

    # --- profiles_brands.zip -> users_brands_SPOD/ ---
    brands_dir = raw_path / RAW_BRANDS_DIR
    if profiles_brands_zip.exists() and not (brands_dir.exists() and any(brands_dir.iterdir())):
        logger.info("Extracting %s ...", profiles_brands_zip.name)
        count = _extract_zip_flat(profiles_brands_zip, brands_dir)
        logger.info("Extracted %d brand profiles.", count)
    elif brands_dir.exists():
        logger.info("Brand profiles already exist at %s, skipping extraction.", brands_dir)

    return True


RAW_BRANDS_DIR = "users_brands_SPOD"


def _validate_raw_dataset(raw_path: Path) -> bool:
    """Check that the raw dataset directory has the expected structure."""
    required = [
        raw_path / RAW_ANNOTATION_FILE,
        raw_path / RAW_IMAGES_DIR,
        raw_path / RAW_JSON_DIR,
        raw_path / RAW_PROFILES_DIR,
        raw_path / RAW_BRANDS_DIR,
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
    raw_path = Path(raw_dataset_path)

    # Auto-extract zips if present but not yet extracted
    if raw_path.exists() and not _validate_raw_dataset(raw_path):
        _extract_raw_zips(raw_path)

    # Ensure influencer profiles are copied to data/ for dataloaders
    if raw_path.exists() and (raw_path / RAW_PROFILES_DIR).exists():
        _ensure_influencer_profiles(raw_path, influencer_profiles_path)

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
        print(f"  2. Place the downloaded files (zip files + post_info.txt) in: {raw_path}")
        print(f"     The zip files (img_fi*.zip, json_files.zip, profiles_*.zip)")
        print(f"     will be automatically extracted on the next run.")
        print(f"  3. Re-run this command.")
        print(f"\n  Alternatively, set RAW_DATASET_PATH in .env to point to")
        print(f"  your download location.")
        print("=" * 70 + "\n")
        sys.exit(1)

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
