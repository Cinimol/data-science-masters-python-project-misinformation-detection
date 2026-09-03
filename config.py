"""
Central configuration for the multimodal misinformation detection artefact.

All paths are resolved relative to the repository root so that the pipeline
runs unchanged on macOS, Linux and Windows.

Author: (MSc Data Science project, UWE Bristol, UFCF9Y-60-M)
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (RAW, INTERIM, PROCESSED, IMAGES, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
# Fakeddit (Nakamura, Levy and Wang, 2020). We use a 100k public mirror of the
# multimodal split that retains the full Reddit metadata schema.
FAKEDDIT_TSV_URL = (
    "https://huggingface.co/datasets/rtfarchitect/fakeddit_sample/"
    "resolve/main/multimodal_train_100k.tsv"
)
FAKEDDIT_TSV = RAW / "fakeddit_100k.tsv"

# Maximum number of posts sampled per subreddit. Chosen so that the corpus is
# large enough for stable transformer fine-tuning yet small enough to build and
# encode on a laptop CPU in a couple of hours.
MAX_PER_SUBREDDIT = 1600
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Evaluation protocols
# --------------------------------------------------------------------------- #
# Protocol A: stratified random split (the convention in the published
#             Fakeddit literature).
# Protocol B: source-disjoint split -- entire subreddits are held out, so no
#             test-time source appears in training. This removes the
#             subreddit -> label shortcut documented in Chapter 4.
# Protocol C: temporal split -- train on older posts, test on newer ones.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}

SOURCE_DISJOINT = {
    # held-out validation sources (one real, one fake family)
    "val": ["upliftingnews", "misleadingthumbnails"],
    # held-out test sources
    "test": ["nottheonion", "usnews", "usanews", "savedyouaclick",
             "confusing_perspective"],
    # everything else falls into train
}

TEMPORAL_QUANTILES = {"train": 0.70, "val": 0.80}  # by created_utc

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "distilroberta-base"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_TEXT_TOKENS = 64          # Fakeddit titles are short (median 51 characters)
IMAGE_SIZE = 224

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
SEEDS = [13, 42, 7]           # repeats, for confidence intervals
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6
HIDDEN = 256
DROPOUT = 0.3
N_BOOTSTRAP = 2000
ADV_LAMBDA = 0.5        # max weight on the domain-adversarial objective

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Return the best available torch device.

    Apple-silicon Macs expose the Metal Performance Shaders (MPS) backend,
    which gives a large speed-up over CPU for the frozen encoders. We fall
    back to CUDA where present and finally to CPU, so the same code runs
    unchanged everywhere.
    """
    import torch

    if os.environ.get("FORCE_CPU"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


N_DOWNLOAD_WORKERS = int(os.environ.get("N_DOWNLOAD_WORKERS", 32))
