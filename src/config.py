import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("REGION")
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").upper() == "TRUE"
API_KEY = os.getenv("GOOGLE_API_KEY", "")
AGENT_LLM_MODEL: str = os.getenv("AGENT_LLM_MODEL", "gemini-3-pro-preview")
CAPTION_LLM_MODEL: str = os.getenv("CAPTION_LLM_MODEL", "gemini-2.5-flash")
JOB_WAIT_TIME: int = 60

# GCS configuration
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
GCS_BUCKET_PREFIX: str = os.getenv("GCS_BUCKET_PREFIX", "agentic-media-perf")

# Dataset paths
RAW_DATASET_PATH: str = os.getenv(
    "RAW_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "raw"),
)
INFLUENCER_DATASET_PATH: str = os.getenv(
    "INFLUENCER_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "users_influencers_SPOD"),
)

DATASET_SUBSETS: dict[str, str] = {
    "1k": "influencer_dataset_subset_1k",
    "10k": "influencer_dataset_subset_10k",
    "top100": "influencer_dataset_top100_min100posts_nonsponsored",
    "top20": "influencer_dataset_top20_min100posts_nonsponsored",
}

DATASET_SPLIT_PARAMS: dict[str, dict] = {
    "top20": {"num_influencers": 20, "min_posts": 100},
    "top100": {"num_influencers": 100, "min_posts": 100},
}
