import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

from src.dataset.loaders import parse_annotation_line
from src.llm.utils import initialize_gemini_client
from src.llm.multimodal_to_text import llm_post_to_text

load_dotenv()


def convert_subset_to_text(
    subset_dir,
    output_dir,
    prompt_template,
    temperature=0.0,
    model_name="gemini-2.5-flash",
    limit: int = 100,
):
    """
    Converts a subset of images and JSONs to text descriptions using Gemini.

    Args:
        subset_dir (str): Path to the subset directory (containing img_subset, json_subset, subset_annotations.txt).
        output_dir (str): Path to save the output text files and config.
        prompt_template (str): The prompt instructions for Gemini.
        temperature (float): Model temperature.
        model_name (str): Gemini model name.
    """
    subset_dir = Path(subset_dir)
    output_dir = Path(output_dir)
    img_dir = subset_dir / "img_subset"
    json_dir = subset_dir / "json_subset"
    annotation_file = subset_dir / "subset_annotations.txt"

    if not annotation_file.exists():
        print(f"Error: Annotation file not found at {annotation_file}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir

    try:
        client = initialize_gemini_client()
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return

    # Read annotations
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit:
        lines = lines[:limit]
    print(f"Found {len(lines)} entries in annotation file.")
    print(f"Processing images using model: {model_name}...")

    success_count = 0
    fail_count = 0

    for line in tqdm(lines, desc="Converting"):
        post_info = parse_annotation_line(line)
        images = post_info.get("image_filenames", [])
        json_file = post_info.get("json_filename", None)
        if not images:
            continue

        # Load JSON context
        json_content = ""
        if json_file:
            json_path = json_dir / json_file
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        # Load and dump to ensure valid JSON string, or just read text?
                        # Reading text is safer to preserve exact content, but formatting helps.
                        data = json.load(jf)
                        json_content = json.dumps(data, indent=2)
                except Exception:
                    print("Error parsing JSON file:", json_path)
                    # Ignore JSON errors, proceed with image only

        # Process each image in the entry
        image_prompts = []
        for img_name in images:
            img_path = img_dir / img_name
            if not img_path.exists():
                continue
            try:
                # Open image
                image = Image.open(img_path)
                image_prompts.append(image)
            except Exception as e:
                print(f"Error opening image {img_name}: {e}")
                fail_count += 1
                continue
            # Check if output already exists (optional: skip if exists? No, overwrite for now)
        out_path = responses_dir / f"{Path(json_file).stem}.txt"

        try:
            # Open image

            # Construct prompt with context
            full_prompt = prompt_template
            if json_content:
                full_prompt += f"\n\n[Context JSON]\n{json_content}"

            # Call Gemini
            response_text = llm_post_to_text(
                client,
                image_prompts,
                full_prompt,
                model_name=model_name,
                temperature=temperature,
            )

            if response_text:
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(response_text)
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"Error processing images for {json_file}: {e}")

    print("\n" + "=" * 30)
    print("CONVERSION SUMMARY")
    print("=" * 30)
    print(f"Processed successfully: {success_count}")
    print(f"Failed/Skipped:       {fail_count}")
    print(f"Results saved in:     {responses_dir}")
    print("=" * 30)


def save_predictions_to_txt(jsonl_path, output_dir):
    """
    Reads responses from a JSONL file and saves each one as a .txt file named {id}.txt.

    Args:
        jsonl_path (str): Path to the predictions.jsonl file.
        output_dir (str): Path to the directory where .txt files will be saved.
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        print(f"Error: JSONL file not found at {jsonl_path}")
        return

    success_count = 0
    fail_count = 0

    print(f"Reading predictions from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Saving TXT files"):
            try:
                data = json.loads(line)
                post_id = data.get("id")

                # Extract response text
                # Path: .response.candidates[0].content.parts[0].text
                response = data.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    fail_count += 1
                    continue

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    fail_count += 1
                    continue

                text = parts[0].get("text", "")

                if post_id and text:
                    out_path = output_dir / f"{post_id}.txt"
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        out_f.write(text)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                fail_count += 1

    print("\nFinished saving text files.")
    print(f"Successfully saved: {success_count}")
    print(f"Failed/Skipped:    {fail_count}")
    print(f"Files saved in:    {output_dir}")


if __name__ == "__main__":
    from src.llm.prompts import PROMPT_SOCIAL_MEDIA

    SUBSET_DIR = "data/influencer_dataset_top20_min100posts_nonsponsored"
    OUTPUT_DIR = os.path.join(SUBSET_DIR, "text_output_v3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    convert_subset_to_text(
        subset_dir=SUBSET_DIR,
        output_dir=OUTPUT_DIR,
        prompt_template=PROMPT_SOCIAL_MEDIA,
        temperature=0.8,
        model_name="gemini-2.5-flash",
        limit=10000,
    )
