import faulthandler
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

faulthandler.enable()

from src.ad_performance_agent.sub_agents.deep_learning.tools import (
    train_lightgbm_model,
    train_regression_model,
    train_transformer_model,
)

import pandas as pd
import vertexai
from dotenv import load_dotenv
from google import genai
from google.genai import types
from vertexai import types as vtx_types
from src.config import (
    AGENT_LLM_MODEL, CAPTION_LLM_MODEL, DATASET_SUBSETS,
    GCS_BUCKET_NAME, GCS_BUCKET_PREFIX, INFLUENCER_DATASET_PATH,
    PROJECT_ID, RAW_DATASET_PATH, REGION,
)
from src.dataset.batch_predictions_gemini import (
    generate_batched_jsonl,
    inline_batch_job,
)
from src.dataset.data_preparation import ensure_images_on_gcs, ensure_subset_exists
from src.dataset.text_conversion import convert_subset_to_text, save_predictions_to_txt
from src.llm.prompts import PROMPT_SIMPLE
from src.utils.gcp_utils import download_file_from_gcs

load_dotenv()

logger = logging.getLogger(__name__)

FEW_SHOT_EXAMPLE_COUNT = 20


def resolve_subset_dir(base_dir: Path, dataset_size: str) -> Path:
    if dataset_size not in DATASET_SUBSETS:
        raise ValueError(
            f"Unknown dataset_size '{dataset_size}'. "
            f"Choose from: {list(DATASET_SUBSETS.keys())}"
        )
    return base_dir / "data" / DATASET_SUBSETS[dataset_size]


class AdPerformanceAgent:
    def __init__(
        self,
        dataset_size="10k",
        iterations=3,
        model_type="xgboost",
        transformer_model_name="distilbert-base-uncased",
        epochs=25,
        batch_size=16,
        lr=2e-5,
        optimize_method="few_shot",
        initial_prompt=None,
        execution_mode="batch",
        influencer_profiles_dir=None,
    ):
        """Initialize the Ad Performance Agent for agentic prompt optimization.

        Args:
            dataset_size (str): Dataset subset to use. Options: "top20", "top100", "1k", "10k".
                Default: "10k"
            iterations (int): Number of optimization iterations to run. Default: 3
            model_type (str): Downstream regression model type. Options: "xgboost", "lightgbm", "transformer".
                Default: "xgboost"
            transformer_model_name (str): HuggingFace transformer model ID for transformer-based training.
                Default: "distilbert-base-uncased"
            epochs (int): Number of training epochs for transformer or neural models. Default: 25
            batch_size (int): Batch size for training. Default: 16
            lr (float): Learning rate for transformer fine-tuning. Default: 2e-5
            optimize_method (str): Prompt optimization strategy. Options: "llm" (Gemini-based),
                "few_shot" (Vertex AI Few-Shot Optimizer). Default: "few_shot"
            initial_prompt (str, optional): Custom initial prompt for predictions. If None, uses PROMPT_SIMPLE
                from src.llm.prompts. Default: None
            execution_mode (str): Prediction execution method. Options:
                - "batch": Uses Google Batch API (efficient for large datasets, requires GCS)
                - "single": Direct Gemini API calls per item (slower, no GCS needed)
                Default: "batch"

        Raises:
            ValueError: If model_type, optimize_method, or execution_mode is not a valid choice.

        Notes:
            - On initialization, automatically:
              1. Creates dataset split if it doesn't exist (from data/raw/)
              2. For batch mode: uploads images to GCS if missing
            - Checkpoint directory created at: checkpoints/run_{timestamp}/
        """
        if model_type not in ("xgboost", "lightgbm", "transformer"):
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose 'xgboost', 'lightgbm', or 'transformer'."
            )
        if optimize_method not in ("llm", "few_shot"):
            raise ValueError(
                f"Unknown optimize_method '{optimize_method}'. Choose 'llm' or 'few_shot'."
            )
        if execution_mode not in ("batch", "single"):
            raise ValueError(
                f"Unknown execution_mode '{execution_mode}'. Choose 'batch' or 'single'."
            )
        self.dataset_size = dataset_size
        self.iterations = iterations
        self.model_type = model_type
        self.optimize_method = optimize_method
        self.initial_prompt = initial_prompt
        self.execution_mode = execution_mode
        self.influencer_profiles_dir = influencer_profiles_dir or INFLUENCER_DATASET_PATH
        self.transformer_model_name = transformer_model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.client = genai.Client()
        self.vertex_client = vertexai.Client(project=PROJECT_ID, location=REGION)
        self.base_dir = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.subset_dir = resolve_subset_dir(self.base_dir, self.dataset_size)

        # Ensure dataset split exists (create from raw if needed)
        ensure_subset_exists(
            self.subset_dir, self.dataset_size, RAW_DATASET_PATH,
            influencer_profiles_path=self.influencer_profiles_dir,
        )
        # Ensure images are uploaded to GCS (for batch predictions)
        ensure_images_on_gcs(self.subset_dir, GCS_BUCKET_NAME, GCS_BUCKET_PREFIX)

        self.output_dir = self.subset_dir / "batched_jsonl_output"
        self.text_output_dir = None
        self.batch_output_file = Path("gemini_batch_output.jsonl")
        self.history = ""
        self.model_name = CAPTION_LLM_MODEL
        self.agent_model_name = AGENT_LLM_MODEL
        os.makedirs(self.output_dir, exist_ok=True)

        # Create checkpoint directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.base_dir / f"checkpoints/run_{timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_prompt_checkpoint(self, iteration, prompt, metrics=None, timestamp=None):
        """Saves the prompt and metadata to a checkpoint file.

        Args:
            iteration (int): Current iteration number
            prompt (str): The prompt text to save
            metrics (dict): Optional metrics from training
            timestamp (str): Optional timestamp for the checkpoint
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create checkpoint data (exclude non-serializable objects like trained models)
        # Convert NaN/Inf floats to None so JSON remains valid
        def _safe_value(v):
            try:
                if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                    return None
            except (TypeError, ValueError):
                pass
            return v

        serializable_metrics = (
            {
                k: _safe_value(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            if metrics
            else None
        )
        checkpoint_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "prompt": prompt,
            "metrics": serializable_metrics,
        }

        # Save as JSON
        checkpoint_file = self.checkpoint_dir / f"iteration_{iteration:03d}.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        # Also save prompt as plain text for easy viewing
        prompt_file = self.checkpoint_dir / f"iteration_{iteration:03d}_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        logger.info("Saved checkpoint to %s", checkpoint_file)

    def run_training(self):
        """Runs training and returns metrics dict with r2, mae, rmse, and train/test sample_errors."""
        if self.model_type == "transformer":
            return train_transformer_model(
                data_dir=str(self.subset_dir),
                dataset_size=self.dataset_size,
                text_output_dir=str(self.text_output_dir),
                transformer_model_name=self.transformer_model_name,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                influencer_dir=self.influencer_profiles_dir,
            )

        elif self.model_type == "lightgbm":
            return train_lightgbm_model(
                data_dir=str(self.subset_dir),
                dataset_size=self.dataset_size,
                text_output_dir=str(self.text_output_dir),
                embedding_model="all-mpnet-base-v2",
                influencer_dir=self.influencer_profiles_dir,
            )

        else:
            return train_regression_model(
                data_dir=str(self.subset_dir),
                dataset_size=self.dataset_size,
                text_output_dir=str(self.text_output_dir),
                embedding_model="all-mpnet-base-v2",
                influencer_dir=self.influencer_profiles_dir,
            )
        
    def _optimize_prompt_llm(self, current_prompt: str, metrics: Dict[str, Any]) -> str:
        """Uses Gemini to propose a new prompt based on the regression metrics."""
        downstream_model = (
            f"fine-tuned {self.transformer_model_name} transformer"
            if self.model_type == "transformer"
            else "XGBoost regression"
        )
        system_instruction = (
            "You are an expert prompt engineer. Your task is to optimize a prompt used for a multimodal model "
            f"that describes Instagram posts. The output of this model is used to train a downstream {downstream_model} model "
            "to predict post popularity (number of likes). You will be given the current prompt and its resulting downstream R2 score (coefficient of determination). "
            "A higher R2 score (closer to 1.0) means better predictions. "
            "Propose a new prompt that you believe will extract better features and improve the downstream R2 score. "
            "Ensure the prompt still requests the model to output a description of the image, caption, hashtags, and any structural information "
            "useful for predicting success. Only output the raw text of the new prompt, with no markdown formatting or explanations."
        )

        r2_score = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 0.0)
        rmse = metrics.get("rmse", 0.0)

        user_message = (
            f"History of prompts and metrics:\n{self.history}\n\n"
            f"Current Prompt:\n{current_prompt}\n\n"
            f"Current Metrics:\n"
            f"  R2 Score: {r2_score:.4f}\n"
            f"  MAE: {mae:.4f}\n"
            f"  RMSE: {rmse:.4f}\n\n"
            f"Provide the next improved prompt:"
        )

        response = self.client.models.generate_content(
            model=AGENT_LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
        )

        new_prompt = response.text.strip()
        if new_prompt.startswith("```"):
            new_prompt = new_prompt.split("\n", 1)[-1]
        if new_prompt.endswith("```"):
            new_prompt = new_prompt.rsplit("\n", 1)[0]

        return new_prompt.strip()

    def _optimize_prompt_few_shot(self, current_prompt: str, metrics: Dict[str, Any]) -> str:
        """Uses the Vertex AI Few-Shot Prompt Optimizer with training sample errors.

        Ranks responses by per-sample prediction error from the downstream model.
        Low-error = good (description captured useful features).
        High-error = bad (description missed important features).
        Selects 20 best + 20 worst from training set only.
        """
        train_sample_errors = metrics.get("train_sample_errors", {})

        if not train_sample_errors:
            raise RuntimeError("No train sample errors available for optimization.")

        # Build scored list directly from metrics
        scored = [
            (v["error"], v["text"])
            for v in train_sample_errors.values()
        ]

        # Sort by error: low first (good), high last (bad)
        scored.sort(key=lambda x: x[0])

        n_good = min(FEW_SHOT_EXAMPLE_COUNT, len(scored) // 2)
        n_bad = min(FEW_SHOT_EXAMPLE_COUNT, len(scored) // 2)
        good = scored[:n_good]
        bad = scored[-n_bad:]

        logger.info(
            "Selected %d good (error %.4f-%.4f) and %d bad (error %.4f-%.4f) examples.",
            n_good, good[0][0], good[-1][0], n_bad, bad[0][0], bad[-1][0],
        )

        rows = []
        for error, text in good:
            rows.append({
                "prompt": current_prompt,
                "model_response": text,
                "rubrics": str(["Acceptable prediction error", "Absolute prediction error value"]),
                "rubrics_evaluations": str([True, error]),
            })
        for error, text in bad:
            rows.append({
                "prompt": current_prompt,
                "model_response": text,
                "rubrics": str(["Acceptable prediction error", "Absolute prediction error value"]),
                "rubrics_evaluations": str([False, error]),
            })

        logger.info("Built %d examples for optimizer.", len(rows))
        df = pd.DataFrame(rows)

        config = vtx_types.OptimizeConfig(
            optimization_target=vtx_types.OptimizeTarget.OPTIMIZATION_TARGET_FEW_SHOT_RUBRICS,
            examples_dataframe=df,
        )

        logger.info("Calling Vertex AI Few-Shot Prompt Optimizer...")
        response = self.vertex_client.prompts.optimize(
            prompt=current_prompt,
            config=config,
        )

        new_prompt = response.parsed_response.suggested_prompt
        logger.info("Optimizer returned new prompt (%d chars).", len(new_prompt))
        return new_prompt
    
    def optimize_prompt(self, current_prompt: str, metrics: Dict[str, Any]) -> str:
        """Dispatches to the selected optimization method."""
        if self.optimize_method == "few_shot":
            return self._optimize_prompt_few_shot(current_prompt, metrics)
        else:
            return self._optimize_prompt_llm(current_prompt, metrics)

    def _run_batch_predictions(self, prompt, timestamp, iteration):
        """Runs predictions via Google Batch API for efficient bulk processing.

        Uses the Gemini Batch API to process all items in parallel:
        1. Generates JSONL batch job file with prompt and images
        2. Images loaded from GCS (uploaded on agent initialization)
        3. Submits batch job to Google Batch API
        4. Downloads results and converts to text files

        Args:
            prompt (str): The prompt template for predictions
            timestamp (str): Timestamp for organizing output files
            iteration (int): Current iteration number (1-indexed)

        Raises:
            Exception: If batch job fails to produce output

        Output:
            Text descriptions saved to subset_dir/text_output/{timestamp}/it_{iteration}/responses/
        """
        logger.info("Generating batched JSONL...")
        jsonl_data = generate_batched_jsonl(
            subset_dir=self.subset_dir,
            output_dir=self.output_dir,
            prompt_template=prompt,
            temperature=0.8,
            bucket_name=GCS_BUCKET_NAME,
        )

        logger.info("Running batch job...")
        bucket_uri = (
            f"gs://{GCS_BUCKET_NAME}/{GCS_BUCKET_PREFIX}/batch_jobs/"
            f"{timestamp}/"
        )
        output_file = inline_batch_job(
            inline_data=jsonl_data,
            model_name=self.model_name,
            BUCKET_URI=bucket_uri,
        )
        if not output_file:
            raise Exception("Batch job failed to produce an output file.")
        bucket_name = output_file.split("gs://")[-1].split("/")[0]

        file_path = output_file.split("gs://")[-1].rsplit("/")[1:]
        file_path = "/".join(file_path)

        download_file_from_gcs(
            bucket_name=bucket_name,
            gcs_blob_name=file_path,
            local_destination_path=str(self.output_dir / "pred.jsonl"),
        )

        logger.info("Converting batch output to text files...")
        text_out = self.checkpoint_dir / f"it_{iteration}" / "text_output"
        save_predictions_to_txt(
            jsonl_path=str(self.output_dir / "pred.jsonl"),
            output_dir=str(text_out),
        )

        self.text_output_dir = text_out

    def _run_single_predictions(self, prompt, timestamp, iteration):
        """Runs predictions one item at a time via direct Gemini API calls.

        Processes each item sequentially using convert_subset_to_text():
        1. Loads images from subset_dir/img_subset/
        2. Loads JSON metadata from subset_dir/json_subset/
        3. Calls Gemini API per item with prompt
        4. Saves text responses to output directory

        This mode is slower than batch but avoids GCS dependency and batch API costs.
        Useful for testing, small datasets, or environments without GCS access.

        Args:
            prompt (str): The prompt template for predictions
            timestamp (str): Timestamp for organizing output files
            iteration (int): Current iteration number (1-indexed)

        Output:
            Text descriptions saved to subset_dir/text_output/{timestamp}/it_{iteration}/responses/
        """
        text_out = self.checkpoint_dir / f"it_{iteration}" / "text_output"
        logger.info("Running single-item predictions to %s ...", text_out)
        convert_subset_to_text(
            subset_dir=self.subset_dir,
            output_dir=str(text_out),
            prompt_template=prompt,
            temperature=0.8,
            model_name=self.model_name,
            limit=999999,
        )
        self.text_output_dir = text_out 

    def run_loop(self):
        """Run the agentic optimization loop.

        Executes iterative prompt optimization cycles:
        1. Generate predictions (batch or single mode based on execution_mode)
        2. Train downstream model on prediction text
        3. Evaluate metrics (R2, MAE, RMSE)
        4. Optimize prompt based on per-sample prediction errors
        5. Save checkpoints (metrics, prompts, JSON)

        The loop uses initial_prompt if provided, otherwise defaults to PROMPT_SIMPLE.
        For each iteration, the optimized prompt from the previous iteration is used.

        Output artifacts saved to checkpoints/run_{timestamp}/:
        - iteration_{N}.json: Prompt text and metrics
        - iteration_{N}_prompt.txt: Plain text prompt for manual review
        - run_summary.txt: Full run summary with all iteration metrics

        Raises:
            Exception: If batch job fails to produce output (batch mode only)
        """
        os.makedirs(self.output_dir, exist_ok=True)
        current_prompt = self.initial_prompt if self.initial_prompt is not None else PROMPT_SIMPLE

        for i in range(self.iterations):
            logger.info("=" * 40)
            logger.info("Iteration %d/%d", i + 1, self.iterations)
            logger.info("Using Prompt:\n%s...", current_prompt[:200])

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # 1. Run predictions (batch or single-item)
            if self.execution_mode == "single":
                self._run_single_predictions(current_prompt, timestamp, i + 1)
            else:
                self._run_batch_predictions(current_prompt, timestamp, i + 1)

            # 2. Train downstream model and get metrics
            metrics = self.run_training()

            # 5. Save checkpoint with prompt and metrics
            self.save_prompt_checkpoint(
                iteration=i + 1,
                prompt=current_prompt,
                metrics=metrics,
                timestamp=timestamp,
            )

            # 6. Record History
            r2 = metrics.get("r2", 0.0)
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            self.history += (
                f"Prompt: {current_prompt}\n"
                f"R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n\n"
            )

            # 7. Optimize Prompt for next iteration
            if i < self.iterations - 1:
                logger.info("Optimizing prompt for next iteration...")
                current_prompt = self.optimize_prompt(current_prompt, metrics)

        # Save final summary
        summary_file = self.checkpoint_dir / "run_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Ad Performance Agent Run Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Dataset Size: {self.dataset_size}\n")
            f.write(f"Iterations: {self.iterations}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(self.history)

        logger.info("=" * 50)
        logger.info("Run completed! Checkpoints saved to: %s", self.checkpoint_dir)
        logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ad Performance Agent")
    parser.add_argument("--dataset-size", default="top20")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--model-type",
        choices=["xgboost", "transformer", "lightgbm"],
        default="xgboost",
        help="Downstream model to train: xgboost or transformer",
    )
    parser.add_argument("--transformer-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optimize-method",
        choices=["llm", "few_shot"],
        default="llm",
        help="Prompt optimization method: 'llm' (Gemini-based) or 'few_shot' (Vertex AI Few-Shot Optimizer)",
    )
    parser.add_argument("--initial-prompt",
        type=str,
        default=None,
        help="Custom initial prompt text. If not provided, uses PROMPT_SIMPLE.",
    )
    parser.add_argument("--execution-mode",
        choices=["batch", "single"],
        default="single",
        help="Prediction mode: 'batch' (Google Batch API) or 'single' (one item at a time via direct API calls)",
    )
    parser.add_argument("--influencer-profiles-dir",
        type=str,
        default=None,
        help="Path to users_influencers_SPOD directory. If not provided, uses INFLUENCER_DATASET_PATH env var or data/users_influencers_SPOD.",
    )
    args = parser.parse_args()

    agent = AdPerformanceAgent(
        dataset_size=args.dataset_size,
        iterations=args.iterations,
        model_type=args.model_type,
        transformer_model_name=args.transformer_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimize_method=args.optimize_method,
        initial_prompt=args.initial_prompt,
        execution_mode=args.execution_mode,
        influencer_profiles_dir=args.influencer_profiles_dir,
    )
    agent.run_loop()
