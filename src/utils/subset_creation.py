import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import ast

from src.dataset.loaders import parse_influencer_profile
from src.utils.safe_path import safe_path_join

logger = logging.getLogger(__name__)


def create_dataset_subset(
    source_img_dir,
    annotation_file,
    target_dir,
    subset_size=10000,
    dry_run=False,
    copy_json=True,
):
    """
    Creates a subset of an image dataset based on a text annotation file.

    Args:
        source_img_dir (str): Path to 'img_resized' folder.
        annotation_file (str): Path to the large .txt file (e.g., post_info.txt).
        target_dir (str): Where to save the sampled images and new txt file.
        subset_size (int): Number of entries to sample.
        dry_run (bool): If True, checks for file existence without copying or writing.
        copy_json (bool): If True, also copies the .json files listed in column 4.
    """
    if dry_run:
        print("--- DRY RUN MODE ACTIVE: No files will be copied or written ---")

    # 1. Setup paths
    subset_img_dir = os.path.join(target_dir, "img_subset")
    subset_json_dir = os.path.join(target_dir, "json_subset")
    subset_annotation_path = str(safe_path_join(target_dir, "subset_annotations.txt"))

    if not dry_run:
        os.makedirs(subset_img_dir, exist_ok=True)
        if copy_json:
            os.makedirs(subset_json_dir, exist_ok=True)

    print(f"Reading annotation file: {annotation_file}...")

    # 2. Read all lines
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {annotation_file} not found.")
        return

    total_entries = len(all_lines)
    print(f"Found {total_entries} total annotations.")

    # 3. Randomly sample lines
    if total_entries < subset_size:
        print(f"Warning: Dataset only has {total_entries} entries. Processing all.")
        sampled_lines = all_lines
    else:
        sampled_lines = random.sample(all_lines, subset_size)

    print(f"Processing {len(sampled_lines)} entries...")

    # 4. Process sampled lines
    successful_posts = 0
    missing_images = 0
    missing_jsons = 0

    out_f = None
    if not dry_run:
        out_f = open(subset_annotation_path, "w", encoding="utf-8")

    try:
        for line in tqdm(sampled_lines):
            line = line.strip()
            if not line:
                continue

            # --- PARSING LOGIC ---
            image_filenames = []
            json_filename = None
            parsed = False

            # Try parsing the whole line as a Python list first (Snippet format)
            # Only if it looks like a list to avoid syntax errors on plain text
            if line.startswith("["):
                try:
                    row_data = ast.literal_eval(line)
                    if isinstance(row_data, list) and len(row_data) >= 5:
                        json_filename = row_data[3]
                        img_col = row_data[4]
                        # If image column is a string representation of a list, parse it
                        if isinstance(img_col, str) and img_col.startswith("["):
                            image_filenames = ast.literal_eval(img_col)
                        elif isinstance(img_col, list):
                            image_filenames = img_col
                        else:
                            image_filenames = [img_col]
                        parsed = True
                except (ValueError, SyntaxError):
                    pass

            if not parsed:
                # Fallback to space splitting (Screenshot format)
                parts = line.split()
                if len(parts) >= 5:
                    json_filename = parts[3]
                    # Reconstruct image list string if it was split by spaces
                    list_start = -1
                    for i, p in enumerate(parts):
                        if p.startswith("["):
                            list_start = i
                            break

                    if list_start != -1:
                        list_str = " ".join(parts[list_start:])
                        try:
                            image_filenames = ast.literal_eval(list_str)
                        except (ValueError, SyntaxError):
                            pass

            if not image_filenames:
                continue

            # --- FILE VALIDATION & COPYING ---
            all_files_ok = True

            # Check Images
            valid_images = []
            for img_name in image_filenames:
                try:
                    src = str(safe_path_join(source_img_dir, img_name))
                    dst = str(safe_path_join(subset_img_dir, img_name))
                except ValueError:
                    logger.warning("Suspicious image name skipped: %s", img_name)
                    all_files_ok = False
                    missing_images += 1
                    continue
                if os.path.exists(src):
                    valid_images.append((src, dst))
                else:
                    all_files_ok = False
                    missing_images += 1

            # Check JSON (if column 4 is valid)
            json_task = None
            if copy_json and json_filename and json_filename.endswith(".json"):
                try:
                    json_candidates = [
                        str(safe_path_join(
                            os.path.dirname(source_img_dir), os.path.join("json", json_filename)
                        )),
                        str(safe_path_join(
                            os.path.dirname(source_img_dir), os.path.join("json_data", json_filename)
                        )),
                        str(safe_path_join(source_img_dir, json_filename)),
                    ]
                except ValueError:
                    logger.warning("Suspicious JSON filename skipped: %s", json_filename)
                    missing_jsons += 1
                    json_candidates = []

                found_json = False
                for json_src in json_candidates:
                    if os.path.exists(json_src):
                        json_task = (
                            json_src,
                            str(safe_path_join(subset_json_dir, json_filename)),
                        )
                        found_json = True
                        break

                if not found_json and json_candidates:
                    missing_jsons += 1

            # Perform Copy
            if all_files_ok:
                if not dry_run:
                    for s, d in valid_images:
                        shutil.copy2(s, d)
                    if json_task:
                        shutil.copy2(json_task[0], json_task[1])
                    out_f.write(line + "\n")
                successful_posts += 1

    finally:
        if out_f:
            out_f.close()

    print("\n" + "=" * 30)
    print("SUBSET CREATION SUMMARY")
    print("=" * 30)
    if dry_run:
        print(f"Dry run finished. Found {successful_posts} valid complete posts.")
    else:
        print(f"Successfully created subset with {successful_posts} posts.")
        print(f"Annotations: {subset_annotation_path}")
        print(f"Images:      {subset_img_dir}")
        if copy_json:
            print(f"JSONs:       {subset_json_dir}")

    print(f"Total missing images encountered: {missing_images}")
    print(f"Total missing JSONs encountered:  {missing_jsons}")
    print("=" * 30)


def create_ranked_subset_by_followers(
    source_img_dir,
    annotation_file,
    influencer_profiles_dir,
    target_dir,
    min_posts_per_influencer=5,
    num_influencers=100,
    sponsor_filter=None,
    dry_run=False,
    copy_json=True,
):
    """
    Creates a dataset subset by selecting top influencers ranked by follower count,
    where each influencer has at least K posts.

    Args:
        source_img_dir (str): Path to 'img_resized' folder.
        annotation_file (str): Path to the annotation file (e.g., post_info.txt).
        influencer_profiles_dir (str): Path to directory with influencer profile files.
        target_dir (str): Where to save the subset.
        min_posts_per_influencer (int): Minimum number of posts required per influencer (K).
        num_influencers (int): Number of influencers to include in subset (M).
        sponsor_filter (int or None): Filter by sponsorship label.
            - None: include all posts (default)
            - 0: only non-sponsored posts
            - 1: only sponsored posts
        dry_run (bool): If True, checks without copying files.
        copy_json (bool): If True, also copies the .json files.

    Returns:
        dict: Summary statistics about the created subset.
    """
    if dry_run:
        print("--- DRY RUN MODE ACTIVE: No files will be copied or written ---")

    print("\n" + "=" * 60)
    print("RANKED SUBSET CREATION BY FOLLOWERS")
    print("=" * 60)
    print(f"Min posts per influencer (K): {min_posts_per_influencer}")
    print(f"Target number of influencers (M): {num_influencers}")
    if sponsor_filter is not None:
        sponsor_label = "sponsored" if sponsor_filter == 1 else "non-sponsored"
        print(f"Sponsor filter: {sponsor_label} posts only")
    else:
        print("Sponsor filter: all posts")
    print("=" * 60 + "\n")

    # Setup paths
    subset_img_dir = os.path.join(target_dir, "img_subset")
    subset_json_dir = os.path.join(target_dir, "json_subset")
    subset_annotation_path = str(safe_path_join(target_dir, "subset_annotations.txt"))

    if not dry_run:
        os.makedirs(subset_img_dir, exist_ok=True)
        if copy_json:
            os.makedirs(subset_json_dir, exist_ok=True)

    # Step 1: Read annotation file and group by influencer
    print(f"Reading annotation file: {annotation_file}...")
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {annotation_file} not found.")
        return None

    print(f"Found {len(all_lines)} total annotations.")

    # Group posts by influencer (with sponsor filtering)
    print("Grouping posts by influencer...")
    influencer_posts = defaultdict(list)
    filtered_out = 0

    for line in tqdm(all_lines, desc="Parsing annotations"):
        line = line.strip()
        if not line:
            continue

        # Parse line (tab-separated format)
        parts = line.split("\t")
        if len(parts) < 4:
            continue

        influencer_name = parts[1]

        # Apply sponsor filter if specified
        if sponsor_filter is not None:
            try:
                sponsor_label = int(parts[2])
                assert sponsor_label in (0, 1)
                if sponsor_label != sponsor_filter:
                    filtered_out += 1
                    continue
            except (ValueError, IndexError):
                continue

        influencer_posts[influencer_name].append(line)

    if sponsor_filter is not None:
        print(f"Filtered out {filtered_out} posts based on sponsor filter.")

    print(f"Found {len(influencer_posts)} unique influencers.")

    # Step 2: Filter influencers with at least K posts
    print(f"\nFiltering influencers with at least {min_posts_per_influencer} posts...")
    qualified_influencers = {
        name: posts
        for name, posts in influencer_posts.items()
        if len(posts) >= min_posts_per_influencer
    }

    print(
        f"Found {len(qualified_influencers)} influencers with >= {min_posts_per_influencer} posts."
    )

    if len(qualified_influencers) == 0:
        print("Error: No influencers meet the minimum post requirement.")
        return None

    # Step 3: Load follower counts for qualified influencers
    print("\nLoading follower counts...")
    influencer_data = []

    for name in tqdm(qualified_influencers.keys(), desc="Loading profiles"):
        try:
            profile_path = safe_path_join(influencer_profiles_dir, name)
        except ValueError:
            logger.warning("Suspicious influencer name skipped: %s", name)
            continue
        profile = parse_influencer_profile(profile_path)

        if profile is not None:
            influencer_data.append(
                {
                    "name": name,
                    "followers": profile["followers"],
                    "category": profile["category"],
                    "num_posts": len(qualified_influencers[name]),
                    "posts": qualified_influencers[name],
                }
            )

    print(f"Successfully loaded {len(influencer_data)} influencer profiles.")

    if len(influencer_data) == 0:
        print("Error: Could not load any influencer profiles.")
        return None

    # Step 4: Rank by followers and select top M
    print("\nRanking influencers by follower count...")
    influencer_data.sort(key=lambda x: x["followers"], reverse=True)

    selected_count = min(num_influencers, len(influencer_data))
    selected_influencers = influencer_data[:selected_count]

    print(f"\nSelected top {selected_count} influencers:")
    print(
        f"  Follower range: {selected_influencers[-1]['followers']:,} - {selected_influencers[0]['followers']:,}"
    )
    print(f"  Total posts: {sum(inf['num_posts'] for inf in selected_influencers):,}")

    # Step 5: Export posts from selected influencers
    print("\nExporting posts from selected influencers...")

    successful_posts = 0
    missing_images = 0
    missing_jsons = 0

    out_f = None
    if not dry_run:
        out_f = open(subset_annotation_path, "w", encoding="utf-8")

    try:
        for influencer in tqdm(selected_influencers, desc="Processing influencers"):
            for line in influencer["posts"]:
                # Parse the line
                parts = line.split("\t")
                if len(parts) < 5:
                    continue

                json_filename = parts[3]

                # Parse image filenames
                try:
                    image_filenames = ast.literal_eval(parts[4])
                    if not isinstance(image_filenames, list):
                        image_filenames = [image_filenames]
                except (ValueError, SyntaxError):
                    continue

                # Validate and copy files
                all_files_ok = True
                valid_images = []

                # Check images
                for img_name in image_filenames:
                    try:
                        src = str(safe_path_join(source_img_dir, img_name))
                        dst = str(safe_path_join(subset_img_dir, img_name))
                    except ValueError:
                        logger.warning("Suspicious image name skipped: %s", img_name)
                        all_files_ok = False
                        missing_images += 1
                        continue
                    if os.path.exists(src):
                        valid_images.append((src, dst))
                    else:
                        all_files_ok = False
                        missing_images += 1

                # Check JSON
                json_task = None
                if copy_json and json_filename and json_filename.endswith(".json"):
                    try:
                        json_candidates = [
                            str(safe_path_join(
                                os.path.dirname(source_img_dir), os.path.join("json", json_filename)
                            )),
                            str(safe_path_join(
                                os.path.dirname(source_img_dir), os.path.join("json_data", json_filename)
                            )),
                            str(safe_path_join(source_img_dir, json_filename)),
                        ]
                    except ValueError:
                        logger.warning("Suspicious JSON filename skipped: %s", json_filename)
                        missing_jsons += 1
                        json_candidates = []

                    found_json = False
                    for json_src in json_candidates:
                        if os.path.exists(json_src):
                            json_task = (
                                json_src,
                                str(safe_path_join(subset_json_dir, json_filename)),
                            )
                            found_json = True
                            break

                    if not found_json and json_candidates:
                        missing_jsons += 1

                # Copy files
                if all_files_ok:
                    if not dry_run:
                        for s, d in valid_images:
                            shutil.copy2(s, d)
                        if json_task:
                            shutil.copy2(json_task[0], json_task[1])
                        out_f.write(line + "\n")
                    successful_posts += 1

    finally:
        if out_f:
            out_f.close()

    # Print summary
    print("\n" + "=" * 60)
    print("RANKED SUBSET CREATION SUMMARY")
    print("=" * 60)
    print("Selection criteria:")
    print(f"  Min posts per influencer: {min_posts_per_influencer}")
    print(f"  Target influencers: {num_influencers}")
    if sponsor_filter is not None:
        sponsor_label = "sponsored" if sponsor_filter == 1 else "non-sponsored"
        print(f"  Sponsor filter: {sponsor_label} posts only")
    else:
        print("  Sponsor filter: all posts")
    print("\nResults:")
    print(f"  Influencers selected: {selected_count}")
    print(f"  Total posts exported: {successful_posts}")
    print(
        f"  Follower range: {selected_influencers[-1]['followers']:,} - {selected_influencers[0]['followers']:,}"
    )
    print(f"  Avg posts per influencer: {successful_posts / selected_count:.1f}")

    if not dry_run:
        print("\nOutput files:")
        print(f"  Annotations: {subset_annotation_path}")
        print(f"  Images: {subset_img_dir}")
        if copy_json:
            print(f"  JSONs: {subset_json_dir}")

    print("\nMissing files:")
    print(f"  Images: {missing_images}")
    print(f"  JSONs: {missing_jsons}")
    print("=" * 60)

    return {
        "num_influencers": selected_count,
        "num_posts": successful_posts,
        "min_followers": selected_influencers[-1]["followers"],
        "max_followers": selected_influencers[0]["followers"],
        "missing_images": missing_images,
        "missing_jsons": missing_jsons,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create dataset subsets from the raw Instagram Influencer dataset."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/Influencer brand dataset",
        help="Path to the raw dataset directory (containing img_resized/, post_info.txt, users_influencers_SPOD/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory for the subset (default: data/)",
    )
    parser.add_argument(
        "--num-influencers", "-M",
        type=int, default=100,
        help="Number of top influencers to select (default: 100)",
    )
    parser.add_argument(
        "--min-posts", "-K",
        type=int, default=100,
        help="Minimum posts per influencer (default: 100)",
    )
    parser.add_argument(
        "--sponsor-filter",
        type=int, default=0, choices=[0, 1],
        help="Filter: 0=non-sponsored only, 1=sponsored only (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check file existence without copying",
    )
    args = parser.parse_args()

    source_images = str(safe_path_join(args.raw_dir, "img_resized"))
    annotation_file = str(safe_path_join(args.raw_dir, "post_info.txt"))
    influencer_profiles = str(safe_path_join(args.raw_dir, "users_influencers_SPOD"))

    sponsor_suffix = ""
    if args.sponsor_filter == 0:
        sponsor_suffix = "_nonsponsored"
    elif args.sponsor_filter == 1:
        sponsor_suffix = "_sponsored"

    target_folder = os.path.join(
        args.output_dir,
        f"influencer_dataset_top{args.num_influencers}_min{args.min_posts}posts{sponsor_suffix}",
    )
    os.makedirs(target_folder, exist_ok=True)

    create_ranked_subset_by_followers(
        source_img_dir=source_images,
        annotation_file=annotation_file,
        influencer_profiles_dir=influencer_profiles,
        target_dir=target_folder,
        min_posts_per_influencer=args.min_posts,
        num_influencers=args.num_influencers,
        sponsor_filter=args.sponsor_filter,
        dry_run=args.dry_run,
        copy_json=True,
    )
