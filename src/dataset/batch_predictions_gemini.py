import argparse
import json
import os
import time
from pathlib import Path

import gcsfs
from dotenv import load_dotenv
from google.genai.types import CreateBatchJobConfig
from tqdm import tqdm

from src.config import GCS_BUCKET_NAME, GCS_BUCKET_PREFIX, JOB_WAIT_TIME, PROJECT_ID
from src.dataset.loaders import parse_annotation_line
from src.dataset.text_conversion import save_predictions_to_txt
from src.llm.prompts import select_prompt
from src.llm.utils import initialize_gemini_client
from src.utils.gcp_utils import download_file_from_gcs, upload_file_to_gcs
from src.utils.safe_path import safe_path_join
from src.utils.logger import logger

load_dotenv()
client = initialize_gemini_client()


def generate_batched_jsonl(
    subset_dir: str | Path,
    output_dir: str | Path,
    prompt_template: str,
    temperature: float = 0.8,
    bucket_name: str = "",
    gcs_base_path: str = None,
):
    if not bucket_name:
        bucket_name = GCS_BUCKET_NAME
    """Generates a JSONL file with batched entries for Gemini processing.
    Each entry contains the image paths, JSON context, and the prompt.
    Args:
        subset_dir (str): Path to the subset directory (containing img_subset, json_subset, subset_annotations.txt).
        output_dir (str): Path to save the output JSONL file and config.
        prompt_template (str): The prompt instructions for Gemini.
        temperature (float): Model temperature.
        bucket_name (str): GCS bucket name.
        gcs_base_path (str): Base path in GCS for images. If None, derives from subset_dir name.
    """
    subset_dir = Path(subset_dir)
    output_dir = Path(output_dir)
    json_dir = safe_path_join(str(subset_dir), "json_subset")
    annotation_file = safe_path_join(str(subset_dir), "subset_annotations.txt")

    # Derive GCS base path if not provided
    if gcs_base_path is None:
        subset_name = subset_dir.name
        gcs_base_path = f"{GCS_BUCKET_PREFIX}/{subset_name}/img_subset"
    if not annotation_file.exists():
        logger.error(f"Annotation file not found at {annotation_file}")
        return
    jsonl_data = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    logger.info(f"Found {len(lines)} entries in annotation file.")
    for line in tqdm(lines, desc="Preparing JSONL"):
        post_info = parse_annotation_line(line)
        images = post_info.get("image_filenames", [])
        json_file = post_info.get("json_filename", None)

        caption = " "
        if json_file:
            try:
                json_path = safe_path_join(str(json_dir), json_file)
            except ValueError:
                logger.warning("Suspicious JSON filename skipped: %s", json_file)
                continue
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        caption = data.get("edge_media_to_caption", {}).get("edges", {})
                        if caption and isinstance(caption, list) and len(caption) > 0:
                            caption = caption[0].get("node", {}).get("text", "")
                        caption = f"\n\n[Post Caption]\n{caption}"
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error("Error parsing JSON file %s: %s", json_path, e)
        images_dict = [{"text": prompt_template + caption}]
        for img_name in images:
            try:
                safe_path_join(gcs_base_path, img_name)
            except ValueError:
                logger.warning("Suspicious image name skipped in GCS URI: %s", img_name)
                continue
            cloud_uri = f"gs://{bucket_name}/{gcs_base_path}/{img_name}"
            images_dict.append(
                {"file_data": {"file_uri": cloud_uri, "mime_type": "image/jpeg"}}
            )
        jsonl_entry = {
            "id": int(json_file.replace(".json", "")),
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": images_dict,
                    }
                ],
                "generationConfig": {"temperature": temperature},
            },
        }
        jsonl_data.append(jsonl_entry)
    output_jsonl_path = safe_path_join(str(output_dir), "gemini_input.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for entry in jsonl_data:
            json.dump(entry, out_f)
            out_f.write("\n")
    gcs_url = upload_file_to_gcs(
        bucket_name=bucket_name,
        local_file_path=str(output_jsonl_path),
        gcs_destination_blob_name=f"{GCS_BUCKET_PREFIX}/batch_jobs/gemini_input.jsonl",
    )
    return gcs_url


def inline_batch_job(
    inline_data,
    model_name: str = "models/gemini-2.5-flash",
    BUCKET_URI: str = "",
):
    if not BUCKET_URI:
        BUCKET_URI = f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_PREFIX}/batch_jobs/"
    inline_batch_job = client.batches.create(
        model=model_name,
        src=inline_data,
        config=CreateBatchJobConfig(dest=BUCKET_URI),
    )

    logger.info(f"Created batch job: {inline_batch_job.name}")
    job_name = inline_batch_job.name
    logger.info(f"Polling status for job: {job_name}")

    while True:
        batch_job_inline = client.batches.get(name=job_name)
        if batch_job_inline.state.name in (
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        ):
            break
        logger.info(
            f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds..."
        )

        time.sleep(JOB_WAIT_TIME)
    if batch_job_inline.state.name == "JOB_STATE_SUCCEEDED":
        logger.info("Batch job succeeded.")
        fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
        file_paths = fs.glob(
            f"{BUCKET_URI.replace('gs://', '')}**/**/predictions.jsonl"
        )
        return file_paths[0]
    else:
        logger.error("Batch job failed with state: %s", batch_job_inline.state.name)
        return None


def arguments():
    argparser = argparse.ArgumentParser(
        description="Batch Gemini Job for Influencer Dataset"
    )
    argparser.add_argument(
        "--subset_dir",
        type=str,
        default="",
        help="Path to the subset directory",
    )
    argparser.add_argument(
        "--temperature", type=float, default=0.8, help="Model temperature"
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (e.g., gemini-2.5-flash)",
    )
    argparser.add_argument(
        "--bucket_uri",
        type=str,
        default="",
        help="GCS bucket URI for batch job output (default: from GCS_BUCKET_NAME env var)",
    )
    argparser.add_argument(
        "--bucket_name",
        type=str,
        default="",
        help="GCS bucket name (default: from GCS_BUCKET_NAME env var)",
    )
    argparser.add_argument(
        "--gcs_base_path",
        type=str,
        default=None,
        help="Base path in GCS for images (auto-derived from subset_dir if not specified)",
    )
    argparser.add_argument(
        "--prompt_type",
        type=str,
        default="social_media",
        help="Type of prompt to use (e.g., social_media)",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    args = arguments()

    # 1. Source Subset Directory
    SUBSET_DIR = args.subset_dir

    # 2. Output Directory
    OUTPUT_DIR = os.path.join(SUBSET_DIR, "batched_jsonl_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Prompt
    prompt = select_prompt(args.prompt_type)

    # 4. Model Settings
    TEMPERATURE = args.temperature
    MODEL_NAME = args.model_name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    BUCKET_URI = args.bucket_uri + f"{timestamp}/"
    jsonl_data = generate_batched_jsonl(
        subset_dir=SUBSET_DIR,
        output_dir=OUTPUT_DIR,
        prompt_template=prompt,
        temperature=TEMPERATURE,
        bucket_name=args.bucket_name,
        gcs_base_path=args.gcs_base_path,
    )

    output_file = inline_batch_job(
        jsonl_data,
        model_name=MODEL_NAME,
        BUCKET_URI=BUCKET_URI,
    )

    bucket_name = output_file.split("gs://")[-1].split("/")[0]

    file_path = output_file.split("gs://")[-1].rsplit("/")[1:]
    file_path = "/".join(file_path)

    pred_path = safe_path_join(OUTPUT_DIR, "pred.jsonl")
    download_file_from_gcs(
        bucket_name=bucket_name,
        gcs_blob_name=file_path,
        local_destination_path=str(pred_path),
    )
    print("Converting batch output to text files...")
    text_output_path = safe_path_join(OUTPUT_DIR, os.path.join("text_output", timestamp))
    save_predictions_to_txt(
        jsonl_path=str(pred_path),
        output_dir=str(text_output_path),
    )
