import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.dataset.text_conversion import save_predictions_to_txt

if __name__ == "__main__":
    BASE_DIR = Path(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    jsonl_path = BASE_DIR / "data/influencer_dataset_subset_10k/predictions.jsonl"
    output_dir = (
        BASE_DIR
        / "data/influencer_dataset_subset_10k/text_output_gemini_2.5_flash/responses"
    )
    print(jsonl_path, output_dir)

    save_predictions_to_txt(str(jsonl_path), str(output_dir))
