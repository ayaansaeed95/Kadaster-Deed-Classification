"""
shared/config.py — single source of truth for all paths, device, and hyperparameters.
Every notebook imports from here. To change a setting, edit bert_config.json only.
"""

from pathlib import Path
import json
import torch
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
ARTIFACTS    = ROOT / "artifacts"
RESULTS      = ROOT / "results"

TRAIN_PATH   = DATA_DIR / "ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl"
TEST_PATH    = DATA_DIR / "ai-challenge-data_ai_challenge_data_anonymized_1920.jsonl"

SPLITS_DIR   = ARTIFACTS / "splits"
PREDS_DIR    = ARTIFACTS / "predictions"
MODELS_DIR   = ARTIFACTS / "models"

LABEL_SPACE_PATH = SPLITS_DIR / "label_space.json"
TRAIN_IDX_PATH   = SPLITS_DIR / "train_idx.npy"
VAL_IDX_PATH     = SPLITS_DIR / "val_idx.npy"

for _d in [SPLITS_DIR, PREDS_DIR, MODELS_DIR, RESULTS]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device — MPS → CUDA → CPU fallback
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VAL_SIZE     = 0.10

# ---------------------------------------------------------------------------
# BERT hyperparameters
# All values live in bert_config.json — edit that file, not this one.
# ---------------------------------------------------------------------------

@dataclass
class BERTConfig:
    # --- model ---
    model_name   : str

    # --- pooling ---
    pooling      : str   # "cls" | "mean" | "max"

    # --- chunking ---
    content_size : int
    stride       : int
    max_chunks   : int

    # --- training ---
    epochs       : int
    lr           : float
    warmup_ratio : float
    weight_decay : float
    batch_size   : int
    grad_accum   : int

    # --- loss ---
    loss_type      : str   # "bce" | "focal"
    pos_weight_cap : float
    focal_gamma    : float

    # --- threshold optimisation ---
    thresh_min_support : int

    # --- optional overrides ---
    gradient_checkpointing : bool         # True = less memory, ~20-30% slower
    sample_size            : Optional[int]  # None = full dataset; int = quick test run
    freeze_layers : int   = 0

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def is_test_run(self) -> bool:
        return self.sample_size is not None


def load_bert_config(path=None) -> BERTConfig:
    """
    Load BERTConfig from a JSON file.
    Raises FileNotFoundError if the file doesn't exist — all values are required.
    The nested loss field {"type": ..., "gamma": ...} is flattened automatically.
    """
    if path is None:
        path = ROOT / "sweeps" / "robbert.json"

    if not Path(path).exists():
        raise FileNotFoundError(
            f"No config file found at {path}. "
            "Create bert_config.json before running."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    loss = raw.pop("loss", {})
    raw["loss_type"]   = loss["type"]
    raw["focal_gamma"] = loss["gamma"]

    cfg = BERTConfig(**raw)
    print(f"BERTConfig loaded from {path}")
    return cfg


BERT_CFG = load_bert_config()

# ---------------------------------------------------------------------------
# Regex / classical
# ---------------------------------------------------------------------------
REGEX_PATTERNS_DIR     = ROOT / "models" / "regex" / "patterns"
REGEX_HIGH_PREC_TARGET = 0.95

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
POLICY_PATH  = ROOT / "models" / "orchestration" / "policy.yaml"


