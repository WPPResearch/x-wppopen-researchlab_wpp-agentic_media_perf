# Agentic Media Performance

An agentic framework for analyzing Instagram influencer post performance using Gemini AI and deep learning.

## Table of Contents
- [Quick Start](#quick-start)
- [Setup](#setup)
  - [Install UV](#install-uv)
  - [Create Virtual Environment](#create-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [1. Agentic Optimization Loop (Main Entry Point)](#1-agentic-optimization-loop-main-entry-point)
  - [2. Dataset Subset Creation](#2-dataset-subset-creation)
  - [3. Batch Predictions with Gemini](#3-batch-predictions-with-gemini)
  - [4. Training Models](#4-training-models)
  - [5. Transformer Inference](#5-transformer-inference)
- [Project Structure](#project-structure)
- [Shared Utilities](#shared-utilities)

---

## Quick Start

Complete workflow from clone to running the agent:

```bash
# 1. Clone and install
git clone <repo-url>
cd agentic-media-perf
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Google Cloud project ID and GCS bucket name

# 3. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 4. Download the Instagram Influencer dataset
# From: https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
# Place all downloaded files (zips + post_info.txt) in data/raw/
# Zips are auto-extracted and images resized to 256x256 on first run

# 5. Run the agent (main entry point)
# (auto-creates dataset split + uploads images to GCS on first run)
uv run python -m src.ad_performance_agent.agent \
    --dataset-size top20 \
    --model-type xgboost \
    --iterations 20
```

---

## Setup

### Install UV

UV is a fast Python package installer and resolver. Install it using one of these methods:

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

**Verify installation:**
```bash
uv --version
```

### Create Virtual Environment

Create a virtual environment with UV:

```bash
# Navigate to project directory
cd /path/to/agentic-media-perf

# Create virtual environment with Python 3.12+
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Install Dependencies

Install project dependencies using UV:

```bash
# Install main dependencies
uv pip install -e .

# Install development dependencies (for notebooks)
uv pip install -e ".[dev]"

# Or install all dependencies at once
uv sync
```

**Note:** UV is much faster than pip and automatically resolves dependencies.

### Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

```bash
# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
REGION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true

# GCS Bucket (for batch predictions -- you must create this)
# gsutil mb -l us-central1 gs://your-bucket-name
GCS_BUCKET_NAME=your-bucket-name
```

Optional variables (with defaults):

```bash
# Alternative: use Gemini API key instead of Vertex AI (simpler, no GCP project needed)
# GOOGLE_API_KEY=your-gemini-api-key

# GCS prefix for uploaded images/data
# GCS_BUCKET_PREFIX=agentic-media-perf

# LLM models
# AGENT_LLM_MODEL=gemini-3-pro-preview
# CAPTION_LLM_MODEL=gemini-2.5-flash

# Dataset paths
# RAW_DATASET_PATH=data/raw
# INFLUENCER_DATASET_PATH=data/users_influencers_SPOD

# HuggingFace token (needed for some embedding models)
# HF_TOKEN=your-huggingface-token
```

### Google Cloud Setup

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Create a GCS bucket for batch predictions
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

### Download the Dataset

This project uses the **Instagram Influencer Dataset** by Kim et al. You must download it from:
https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset

Specifically, you need **II. Influencer and Brand Dataset: Sponsorship Detection**, which requires requesting access from the dataset authors. Follow the instructions on the page to request access.

Once you have the dataset, place all the downloaded files into `data/raw/`:

```
data/raw/
├── post_info.txt              # Annotation file (download as-is)
├── img_fi01.zip               # Image zip files (up to 16)
├── img_fi02.zip
├── ...
├── img_fi16.zip
├── json_files.zip             # Post JSON metadata
├── profiles_influencers.zip   # Influencer profile files
└── profiles_brands.zip        # Brand profile files
```

**Automatic extraction:** On first run, the agent will automatically extract the zip files and resize all images to 256x256. After extraction, the directory will contain:

```
data/raw/
├── post_info.txt
├── img_resized/               # Extracted + resized post images (256x256)
├── json/                      # Extracted JSON metadata
├── users_influencers_SPOD/    # Extracted influencer profiles
├── users_brands_SPOD/         # Extracted brand profiles
├── img_fi01.zip               # Original zips (can be deleted after extraction)
├── ...
└── profiles_brands.zip
```

Extraction is idempotent — if the target directories already exist and are non-empty, they are skipped.

The `users_influencers_SPOD/` directory contains influencer profile metadata needed for training. If you place it somewhere other than the default (`data/users_influencers_SPOD`), either:
- Set `INFLUENCER_DATASET_PATH` in your `.env` file, or
- Pass `--influencer-profiles-dir /path/to/users_influencers_SPOD` when running the agent

> **Tip:** You can place the raw dataset elsewhere and set `RAW_DATASET_PATH` in `.env`.

---

## Usage

### 1. Agentic Optimization Loop (Main Entry Point)

The agent is the main entry point for this project. It iteratively optimizes prompts based on downstream model performance, and automatically handles dataset creation and GCS uploads on first run.

```bash
uv run python -m src.ad_performance_agent.agent \
    --dataset-size top20 \
    --model-type xgboost \
    --iterations 20 \
    --optimize-method llm
```

**All options:**
```bash
uv run python -m src.ad_performance_agent.agent \
    --dataset-size top20 \         # top20, top100, 1k, 10k
    --model-type xgboost \         # xgboost, lightgbm, transformer
    --iterations 20 \              # number of optimization iterations
    --optimize-method llm \        # llm or few_shot
    --transformer-model distilbert-base-uncased \
    --epochs 25 \
    --batch-size 16 \
    --lr 2e-5 \
    --initial-prompt "Your custom prompt here" \  # optional
    --execution-mode batch \       # batch or single
    --influencer-profiles-dir /path/to/users_influencers_SPOD  # optional
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-size` | `top20` | Dataset subset: `top20`, `top100` |
| `--model-type` | `xgboost` | Downstream model: `xgboost`, `lightgbm`, or `transformer` |
| `--iterations` | `20` | Number of prompt optimization iterations |
| `--optimize-method` | `llm` | Optimization approach: `llm` (Gemini-based) or `few_shot` (Vertex AI Few-Shot Optimizer) |
| `--transformer-model` | `distilbert-base-uncased` | HuggingFace transformer model ID (if `--model-type transformer`) |
| `--epochs` | `25` | Training epochs for transformer/neural models |
| `--batch-size` | `16` | Batch size for training |
| `--lr` | `2e-5` | Learning rate |
| `--initial-prompt` | `None` | Custom initial prompt text (uses `PROMPT_SIMPLE` if not provided) |
| `--execution-mode` | `batch` | Prediction execution mode (see below) |
| `--influencer-profiles-dir` | `None` | Path to `users_influencers_SPOD/` directory (defaults to `INFLUENCER_DATASET_PATH` env var or `data/users_influencers_SPOD`) |

**Execution Modes:**

`--execution-mode` controls how predictions are generated:

- **`batch`** (default): Uses Google Batch API for bulk predictions
  - More efficient for large datasets
  - Requires GCS bucket for images
  - Uploads images to GCS on startup if missing
  - Recommended for production runs

- **`single`**: Direct Gemini API calls, one item at a time
  - Slower but no GCS dependency
  - Loads local images from `subset_dir/img_subset/`
  - Loads JSON metadata from `subset_dir/json_subset/`
  - Uses `convert_subset_to_text()` function
  - Useful for testing or small datasets

**What happens on startup:**
1. Checks if the dataset split exists (e.g. `data/influencer_dataset_top20_min100posts_nonsponsored/`)
2. If missing, creates it automatically from the raw dataset in `data/raw/`
3. If using `--execution-mode batch`:
   - Checks if images are uploaded to your GCS bucket
   - If missing, uploads them automatically

**What each iteration does:**
1. Generates predictions based on `--execution-mode`:
   - **batch**: Google Batch API for bulk predictions (multimodal: images + prompt)
   - **single**: Direct Gemini API calls, one item at a time
2. Converts predictions to text descriptions
3. Trains the downstream model (XGBoost / LightGBM / Transformer)
4. Evaluates model performance (R2, MAE, RMSE)
5. Optimizes the prompt using per-sample error feedback
6. Repeats with the improved prompt

**Prompt Templates:**

The agent uses prompt templates from `src/llm/prompts.py`:
- `PROMPT_SIMPLE` (default) -- concise analysis prompt
- `PROMPT_SOCIAL_MEDIA` -- detailed social media analysis
- `PROMPT_V0` -- original baseline prompt
- `PROMPT_SPONSORSHIP` -- sponsorship detection focused

Use `--initial-prompt` to pass custom prompt text, or modify `prompts.py` to add new templates.

**Output:**
- Checkpoints saved to `checkpoints/run_{timestamp}/`
- Each iteration saves:
  - `iteration_{NNN}.json`: Prompt and metrics (zero-padded)
  - `iteration_{NNN}_prompt.txt`: Plain text prompt
  - `it_{N}/text_output/`: Generated text descriptions for this iteration
- `run_summary.txt`: Full run summary

---

### 2. Dataset Subset Creation

> **Note:** Dataset splits are created **automatically** when you run the agent. You only need this section for manual/advanced usage.

You can create splits manually:

```bash
# Top 20 influencers with 100+ non-sponsored posts each
uv run python src/utils/subset_creation.py \
    --raw-dir data/raw \
    --output-dir data \
    --num-influencers 20 \
    --min-posts 100 \
    --sponsor-filter 0

# Top 100 influencers
uv run python src/utils/subset_creation.py \
    --raw-dir data/raw \
    --output-dir data \
    --num-influencers 100 \
    --min-posts 100 \
    --sponsor-filter 0

# Dry run (check without copying files)
uv run python src/utils/subset_creation.py \
    --raw-dir data/raw \
    --num-influencers 20 \
    --min-posts 100 \
    --dry-run
```

**Output:** Creates a subset directory at `data/influencer_dataset_top{M}_min{K}posts_nonsponsored/`

---

### 3. Batch Predictions with Gemini

> **Note:** The agent handles predictions automatically. Use this section only if you want to run batch predictions independently.

Generate predictions for Instagram posts using Gemini AI in batch mode.

**Basic usage:**
```bash
uv run python -m src.dataset.batch_predictions_gemini \
    --subset_dir data/influencer_dataset_subset_10k \
    --bucket_name your-gcs-bucket \
    --bucket_uri gs://your-bucket/batch-output/ \
    --prompt_type social_media
```

**Parameters:**
- `--subset_dir`: Path to dataset subset directory
- `--temperature`: Model temperature (0.0-1.0, default: 0.8)
- `--model_name`: Gemini model to use (default: gemini-2.5-flash)
- `--bucket_uri`: GCS URI for batch job output
- `--bucket_name`: GCS bucket name for images/data
- `--gcs_base_path`: Base path in GCS for images (auto-derived if not specified)
- `--prompt_type`: Prompt template type (e.g., social_media, sponsorship)

**Output:** Creates JSONL batch job files and submits to Gemini API

---

### 4. Training Models

> **Note:** The agent trains models automatically as part of its optimization loop. Use these scripts only if you want to train models independently or experiment with different configurations.

All training scripts share common CLI arguments via `src/utils/cli.py`:

| Common Argument | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `top100` | Dataset subset: `top20`, `top100`, `1k`, `10k` |
| `--text_output_dir` | checkpoint path | Path to text outputs directory |
| `--include-user` | `True` | Prepend `"This post is from {username}."` to each text |
| `--embedding` | `all-mpnet-base-v2` | SentenceTransformer model name |

---

#### Script 1: Classification Neural Network (`train_classification_nn.py`)

Trains a PyTorch Lightning MLP classifier for sponsor detection or engagement-tier prediction using frozen sentence embeddings as input features.

```bash
uv run python -m src.training.train_classification_nn \
    --dataset top20 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --loss_type focal \
    --classification_task engagement_label
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | `30` | Max training epochs (early stopping at patience=5) |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--hidden_dim` | `128` | Hidden layer dimension |
| `--loss_type` | `focal` | `ce` (cross-entropy) or `focal` (focal loss) |
| `--embedding_model` | `all-mpnet-base-v2` | Embedding model (also supports `google/embeddinggemma-300m`, `sentence-transformers/all-MiniLM-L12-v2`) |
| `--classification_task` | `engagement_label` | `sponsor_label` (2-class) or `engagement_label` (3-class) |

**Architecture:** 2-layer MLP with dropout (input -> hidden -> hidden/2 -> num_classes). Uses PyTorch Lightning with early stopping on `val_loss`.

---

#### Script 2: Text Embedding Regression (`train_likes_regression.py`)

Trains scikit-learn regression models to predict likes using sentence embeddings as the sole input features. Compares Ridge regression against gradient boosting methods.

```bash
uv run python -m src.training.train_likes_regression \
    --dataset top20 \
    --embedding all-mpnet-base-v2
```

**Models trained:** Ridge (alpha=0.1), Ridge (alpha=1.0), XGBoost, LightGBM.

**Target:** `log(likes + 1)` -- log-transformed to reduce skew.

**Output:** Saved to `checkpoints/text_embedding_regression_{timestamp}/`

---

#### Script 3: Neural Network Regression (`train_neural_regression.py`)

Compares four PyTorch neural network architectures for likes regression, all trained on frozen sentence embeddings.

```bash
uv run python -m src.training.train_neural_regression \
    --dataset top20 \
    --embedding all-mpnet-base-v2
```

**Architectures compared:**

| Architecture | Description |
|---|---|
| `Simple NN (256)` | 2-layer MLP, 256 hidden units |
| `Deep NN (512-256-128)` | 4-layer MLP with BatchNorm |
| `Wide NN (1024)` | 2-layer wide MLP, 1024 hidden units |
| `ResNet (512, 3 blocks)` | Residual network with skip connections |

**Training details:** Adam optimizer with weight decay (`1e-5`), `ReduceLROnPlateau` scheduler, early stopping (patience=15), up to 100 epochs. Data split: 70% train / 15% val / 15% test.

---

#### Script 4: Transformer Fine-tuning (`train_transformer_finetune.py`)

Fine-tunes a pretrained transformer (DistilBERT, BERT, RoBERTa) end-to-end for engagement prediction. Unlike the embedding-based scripts, the full model is trained rather than using frozen features.

```bash
uv run python -m src.training.train_transformer_finetune \
    --dataset top20 \
    --model distilbert-base-uncased \
    --epochs 25 \
    --batch-size 16 \
    --lr 2e-5 \
    --max-length 256
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `distilbert-base-uncased` | HuggingFace model ID |
| `--epochs` | `25` | Max training epochs (early stopping at patience=5) |
| `--batch-size` | `16` | Batch size |
| `--lr` | `2e-5` | Learning rate |
| `--max-length` | `256` | Max token sequence length |

**Supported models:** `distilbert-base-uncased` (fast), `bert-base-uncased` (balanced), `roberta-base` (best quality).

**Training details:** AdamW optimizer (weight decay=0.01), linear warmup scheduler (10% warmup), gradient clipping (max_norm=1.0). Best model checkpoint saved automatically.

**Output:** Saved to `checkpoints/transformer_finetuning_{timestamp}/`

---

#### Script 5: Feature-Engineered Regression (`train_likes_with_features.py`)

Trains regression models on a rich feature set that combines text embeddings with structured post metadata, influencer profile data, and temporal features. Typically achieves the highest R2 scores.

```bash
uv run python -m src.training.train_likes_with_features \
    --dataset top20 \
    --embedding all-mpnet-base-v2
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--influencer-profiles-dir` | from env | Path to `users_influencers_SPOD/` directory |

**Feature groups used:**

| Group | Features |
|---|---|
| Text embeddings | SentenceTransformer output (768 dims default) |
| Post metadata | `comment_count`, `is_video`, `aspect_ratio`, `caption_length`, `hashtag_count`, `mention_count`, `emoji_count` |
| Influencer data | `followers`, `followees`, `posts_count`, `follower_followee_ratio` |
| Temporal | `hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hours`, `month`, `is_summer`, cyclical encodings |

**Models trained:** Ridge (alpha=1.0), XGBoost, LightGBM. Numerical features are standardized with `StandardScaler` before concatenation with embeddings.

---

#### Choosing a Training Script

| Use Case | Recommended Script |
|---|---|
| Sponsor detection | `train_classification_nn.py` (`--classification_task sponsor_label`) |
| Engagement tier classification | `train_classification_nn.py` (`--classification_task engagement_label`) |
| Likes prediction (text only, fast baseline) | `train_likes_regression.py` |
| Likes prediction (neural architectures) | `train_neural_regression.py` |
| Likes prediction (end-to-end, GPU recommended) | `train_transformer_finetune.py` |
| Likes prediction (highest accuracy) | `train_likes_with_features.py` |

---

### 5. Transformer Inference

Evaluate a trained transformer checkpoint on validation and test splits:

```bash
uv run python -m src.inference.infer_transformer \
    --checkpoint checkpoints/transformer_finetuning_YYYYMMDD_HHMMSS \
    --dataset top20
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | `checkpoints/transformer_finetuning` | Path to experiment directory or `.pt` file |
| `--model` | from `params.json` | Transformer model name (auto-loaded from checkpoint) |
| `--max-length` | from `params.json` | Max token sequence length (auto-loaded) |
| `--batch-size` | `32` | Inference batch size |

The script replicates the exact train/val/test split from training and reports metrics on both validation and test sets.

---

## Project Structure

```
agentic-media-perf/
├── src/
│   ├── ad_performance_agent/           # Main agentic loop
│   │   ├── agent.py                   # Agent orchestration (main entry point)
│   │   └── sub_agents/
│   │       └── deep_learning/
│   │           ├── agent.py           # Deep learning sub-agent
│   │           ├── tools.py           # Programmatic training API
│   │           └── prompt.py          # Agent instructions
│   ├── dataset/                        # Data handling
│   │   ├── batch_predictions_gemini.py # Batch Gemini API calls
│   │   ├── data_preparation.py        # Auto dataset split & GCS upload
│   │   ├── dataloader.py             # Unified data loading (load_dataset + wrappers)
│   │   ├── datamodel.py              # Pydantic data models
│   │   ├── engagement_rate_loader.py  # Engagement rate target (likes/followers)
│   │   ├── loaders.py                # Low-level parsers (annotations, influencer profiles)
│   │   └── text_conversion.py        # Prediction format conversion
│   ├── training/                       # Model training scripts
│   │   ├── train_classification_nn.py  # MLP classifier (PyTorch Lightning)
│   │   ├── train_likes_regression.py   # Text-embedding regression (sklearn)
│   │   ├── train_neural_regression.py  # Neural net regression (PyTorch)
│   │   ├── train_transformer_finetune.py # Transformer fine-tuning
│   │   ├── train_likes_with_features.py  # Feature-engineered regression
│   │   ├── model_configs.py           # Shared XGBoost/LightGBM hyperparameters
│   │   └── visualize_data.py          # Dataset visualizations
│   ├── inference/                      # Model inference
│   │   └── infer_transformer.py       # Transformer checkpoint evaluation
│   ├── utils/                          # Shared utilities
│   │   ├── cli.py                     # Shared CLI argument parsing & data paths
│   │   ├── embeddings.py             # Unified sentence embedding generation
│   │   ├── metrics.py                # Shared metrics (sample predictions, errors)
│   │   ├── experiment_tracker.py     # Checkpoint & metrics tracking
│   │   ├── temporal_features.py      # Timestamp feature extraction & text
│   │   ├── subset_creation.py        # Dataset subset tools
│   │   ├── gcp_utils.py              # GCP/GCS helpers
│   │   └── logger.py                 # Logging utilities
│   ├── llm/                            # LLM utilities
│   │   ├── prompts.py                 # Prompt templates
│   │   ├── multimodal_to_text.py      # Multimodal response handling
│   │   └── utils.py                   # LLM helpers
│   └── config.py                       # Global configuration
├── dev/                                # Development scripts & notebooks
│   ├── embeds_benchmark.py            # Embedding model benchmark
│   ├── test_loader_check.py           # Data loader check script
├── data/
│   └── raw/                            # Raw dataset (download here)
├── checkpoints/                        # Training experiment outputs
├── .env.example                        # Environment variable template
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # UV lock file
└── README.md                           # This file
```

---

## Shared Utilities

The codebase uses shared utility modules to avoid duplication across training scripts.

### Unified Data Loader (`src/dataset/dataloader.py`)

`load_dataset()` is the canonical data loading function. It reads annotation files, text descriptions, JSON metadata, and influencer profiles, returning a DataFrame with `post_id`, `text`, `y`, and `influencer_name` columns.

Configuration flags:
- `include_user_info` -- prepend influencer name to text
- `include_temporal` -- append posting time context
- `include_influencer_details` -- append influencer bio and category

`load_datav2` and `load_datav3` are thin wrappers with preset flags for backward compatibility.

### Embeddings (`src/utils/embeddings.py`)

`generate_embeddings(texts, model_name, batch_size, cleanup_gpu)` -- unified embedding generation using SentenceTransformer. Supports all sentence-transformers models.

### Metrics (`src/utils/metrics.py`)

- `log_sample_predictions(y_true, y_pred, n, log_fn)` -- log sample predictions in human-readable format
- `build_sample_errors(post_ids, y_true, y_pred, texts)` -- build per-sample error dictionaries

### CLI (`src/utils/cli.py`)

- `create_data_argparser(description)` -- create ArgumentParser with common dataset args
- `resolve_data_paths(args)` -- resolve data/text/json directories from parsed args
- `load_data_from_args(args)` -- load dataset DataFrame from CLI arguments

### Model Configs (`src/training/model_configs.py`)

`DEFAULT_XGBOOST_PARAMS` and `DEFAULT_LIGHTGBM_PARAMS` -- shared hyperparameter dictionaries used by all training scripts and the agent's training tools.

---

## Development

**Run notebooks:**
```bash
jupyter notebook dev/notebooks/dataset_visualization.ipynb
```

**Run embedding benchmark:**
```bash
uv run python dev/embeds_benchmark.py --data-dir data/influencer_dataset_top20_min100posts_nonsponsored
```

**View dataset statistics:**
The visualization notebook includes:
- Likes distribution analysis
- Follower distribution analysis
- Posts per influencer analysis
- Engagement rate calculations
- Correlation analysis

---

## Troubleshooting

**UV not found:**
```bash
# Add UV to PATH or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Google Cloud authentication:**
```bash
gcloud auth application-default login
```

**Missing HuggingFace token:**
1. Create account at https://huggingface.co
2. Generate token at https://huggingface.co/settings/tokens
3. Add to `.env` file: `HF_TOKEN=your_token`

---

## License

Apache 2.0 - See LICENSE file for details.
