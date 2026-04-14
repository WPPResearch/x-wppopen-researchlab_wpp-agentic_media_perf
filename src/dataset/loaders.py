import ast

import json
from pathlib import Path
from typing import List, Union, Dict, Optional, Any
from pydantic import ValidationError

from src.dataset.datamodel import InstagramPost, Influencer


class InMemMediaLoader:
    def __init__(self, source: Union[str, Path]):
        self.source = Path(source)
        self.records: List[InstagramPost] = []
        self.failed_files: List[Dict] = []

        # Trigger the full load immediately
        self._load_all()

    def _load_all(self):
        """Internal method to crawl files and populate the records list."""
        files = [self.source] if self.source.is_file() else self.source.glob("*.json")

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Standardize into a list if it's just one object
                items = data if isinstance(data, list) else [data]

                for item in items:
                    try:
                        self.records.append(InstagramPost(**item))
                    except ValidationError as e:
                        print(file_path.name, e.json())
                        self.failed_files.append(
                            {"file": file_path.name, "error": e.json()}
                        )
            except (json.JSONDecodeError, IOError) as e:
                self.failed_files.append({"file": file_path.name, "error": str(e)})

    # --- In-Memory Utilities ---

    def sort_by_date(self, reverse: bool = True):
        """Sorts all loaded records by their timestamp."""
        self.records.sort(key=lambda x: x.taken_at_timestamp, reverse=reverse)
        return self

    def filter_by_min_likes(self, count: int) -> List[InstagramPost]:
        """Returns a subset of records with likes >= count."""
        return [r for r in self.records if r.edge_media_preview_like.count >= count]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def parse_annotation_line(line):
    """
    Parses a line from the annotation file to extract image filenames and JSON filename.
    Handles both Python list string format and space-separated format.
    """
    line = line.strip()
    if not line:
        return None, None

    image_filenames = []
    json_filename = None
    # 1. Try parsing as Python literal (list)
    if line.startswith("["):
        try:
            row_data = ast.literal_eval(line)
            if isinstance(row_data, list) and len(row_data) >= 5:
                post_id = row_data[0]
                influencer_name = row_data[1]
                sponsor_label = int(row_data[2])
                json_filename = row_data[3]
                img_col = row_data[4]
                if isinstance(img_col, str) and img_col.startswith("["):
                    image_filenames = ast.literal_eval(img_col)
                elif isinstance(img_col, list):
                    image_filenames = img_col
                else:
                    image_filenames = [img_col]
                return image_filenames, json_filename
        except (ValueError, SyntaxError):
            pass

    # 2. Fallback to space splitting
    parts = line.split()
    if len(parts) >= 5:
        post_id = parts[0]
        influencer_name = parts[1]
        sponsor_label = int(parts[2])
        json_filename = parts[3]
        # Reconstruct list string if split by spaces
        list_start = -1
        for i, p in enumerate(parts):
            if p.startswith("["):
                list_start = i
                break
        if list_start != -1:
            try:
                list_str = " ".join(parts[list_start:])
                image_filenames = ast.literal_eval(list_str)
            except (ValueError, SyntaxError):
                pass
    output = {
        "post_id": post_id,
        "influencer_name": influencer_name,
        "sponsor_label": sponsor_label,
        "json_filename": json_filename,
        "image_filenames": image_filenames,
    }
    return output


def parse_influencer_profile(file_path: Path) -> Optional[Influencer | Dict | Any]:
    """
    Parses a tab-separated influencer profile file.
    Expected format:
    [Name] [Followers] [Followees] [Posts] [URL] [T/F] [Category] [Bio] [E-mail] [Phone] [Profile_pic]
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split("\t")

    # We expect at least 11 fields based on the description
    if len(parts) < 11:
        # Fallback: sometimes editors convert tabs to spaces, or multiple spaces.
        # But strict requirement says "separated by Tab".
        # We will try to be slightly lenient if tabs fail but length is correct-ish?
        # No, strict tab splitting is safer for described format.
        return None

    try:
        # Extract fields by index
        name = parts[0].strip()
        followers_str = parts[1].strip()
        followees_str = parts[2].strip()
        posts_str = parts[3].strip()
        url = parts[4].strip()
        # tf = parts[5] # Ignored
        category = parts[6].strip()
        bio = parts[7].strip()
        email = parts[8].strip()
        phone = parts[9].strip()
        profile_pic = parts[10].strip()
        text_desc = f"{name} is an influencer in the {category} category with {followers_str} followers and {posts_str} posts. Bio: {bio}"

        # Helper to safely parse ints (remove commas if any)
        def parse_int(s):
            s = s.replace(",", "").replace(
                ".", ""
            )  # Assuming 1.000 or 1,000 for thousands? Or 1.5k?
            # Start simple: digits only
            # If "1.2m", this will fail. Let's assume raw numbers for now.
            if not s:
                return 1
            try:
                # specific handling for "k", "m" suffixes could be added if needed
                # For now assume standard integer string
                return int(float(s))
            except ValueError:
                return 1

        return dict(
            Influencer(
                name=name,
                followers=parse_int(followers_str),
                followees=parse_int(followees_str),
                posts=parse_int(posts_str),
                url=url if url else None,
                category=category if category else None,
                bio=bio if bio else None,
                email=email if email else None,
                phone=phone if phone else None,
                profile_pic_url=profile_pic if profile_pic else None,
                text_desc=text_desc if text_desc else None,
            )
        )
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    # 1. Initialize and load everything
    loader = InMemMediaLoader(
        "data/influencer_dataset_top20_min100posts_nonsponsored/json_subset"
    )

    print(f"Successfully loaded {len(loader)} records.")
    if loader.failed_files:
        print(f"Warning: {len(loader.failed_files)} items failed validation.")

    # 2. Access data like a list
    first_post = loader[0]
    print(f"First ID in memory: {first_post.id}")

    # 3. Use the built-in utilities
    recent_posts = loader.sort_by_date().records
    popular_posts = loader.filter_by_min_likes(1000)

    for post in popular_posts:
        print(
            f"Popular Post: {post.shortcode} - Likes: {post.edge_media_preview_like.count}"
        )
