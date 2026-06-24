"""
shared/io.py — prediction contract read/write.

The contract: one parquet file per method, long format, one row per (deed, code).

Schema
------
akteId    : str    — the deed id
code      : int    — a rechtsfeitcode
score     : float  — confidence for that code (calibrated / normalised, 0–1)
predicted : bool   — did this method assign it (score ≥ its threshold)?
method    : str    — "bert" | "regex" | "tfidf" | "orchestration" | "llm"

Why long format (all codes per deed, not just predicted ones)?
The orchestration layer needs the full score distribution to compute the 1st-vs-2nd margin.
A row with predicted=False and score=0.03 tells the orchestration layer "this method is
confident it's NOT this code." A row with predicted=False and score=0.48
tells the orchestration layer "this method was almost certain — worth escalating."

Usage
-----
    from shared.io import write_predictions, read_predictions

    # BERT / RegEx — after inference:
    write_predictions(akteIds, scores_matrix, predicted_matrix, classes, method="bert")

    # Orchestration — to consume:
    df = read_predictions("bert")
    # df has columns: akteId, code, score, predicted, method
"""

import numpy as np
import pandas as pd
from pathlib import Path

from shared.config import PREDS_DIR


def write_predictions(
    akteIds:   list,
    scores:    np.ndarray,
    predicted: np.ndarray,
    classes:   list,
    method:    str,
) -> Path:
    """
    Write the long-format prediction parquet for one method.

    Parameters
    ----------
    akteIds   : list of deed id strings, length N
    scores    : (N, num_classes) float array, values in [0, 1]
    predicted : (N, num_classes) bool array
    classes   : ordered label list from get_mlb()
    method    : "bert" | "regex" | "orchestration" | …

    Returns
    -------
    Path to the written parquet file.
    """
    assert scores.shape == predicted.shape, "scores and predicted must have the same shape"
    assert scores.shape == (len(akteIds), len(classes)), \
        f"Expected ({len(akteIds)}, {len(classes)}), got {scores.shape}"
    assert scores.min() >= 0.0 and scores.max() <= 1.0, \
        "Scores must be in [0, 1]. Calibrate / normalise before writing."

    N, C = scores.shape
    rows = {
        "akteId"   : np.repeat(akteIds, C),
        "code"     : np.tile([int(c) if str(c).isdigit() else c for c in classes], N),
        "score"    : scores.flatten().astype(np.float32),
        "predicted": predicted.flatten().astype(bool),
        "method"   : method,
    }
    df = pd.DataFrame(rows)

    out_path = PREDS_DIR / f"{method}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Predictions written → {out_path}  ({len(df):,} rows, {N} deeds, {C} codes)")
    return out_path


def read_predictions(method: str) -> pd.DataFrame:
    """
    Load the long-format prediction parquet for a method.

    Parameters
    ----------
    method : "bert" | "regex" | "orchestration" | …

    Returns
    -------
    DataFrame with columns: akteId, code, score, predicted, method
    """
    path = PREDS_DIR / f"{method}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the {method} notebook/predict.py first."
        )
    df = pd.read_parquet(path)
    print(f"Predictions loaded ← {path}  ({len(df):,} rows)")
    return df


def predictions_to_matrix(df: pd.DataFrame, classes: list) -> tuple:
    """
    Convert a long-format predictions DataFrame back to wide matrices.
    Useful for passing into harness.evaluate().

    Returns
    -------
    akteIds   : list of unique deed ids (sorted)
    scores    : (N, num_classes) float32 array
    predicted : (N, num_classes) bool array
    """
    code_to_idx = {c: i for i, c in enumerate(classes)}
    akteIds = sorted(df["akteId"].unique())
    id_to_idx = {a: i for i, a in enumerate(akteIds)}

    N = len(akteIds)
    C = len(classes)
    scores    = np.zeros((N, C), dtype=np.float32)
    predicted = np.zeros((N, C), dtype=bool)

    for row in df.itertuples(index=False):
        i = id_to_idx.get(row.akteId)
        j = code_to_idx.get(row.code)
        if i is not None and j is not None:
            scores[i, j]    = row.score
            predicted[i, j] = row.predicted

    return akteIds, scores, predicted


def roundtrip_check(classes: list, method: str = "_test") -> bool:
    """
    Smoke-test: write dummy predictions and read them back.
    Call this in 00_splits.ipynb after agreeing the contract.
    """
    N = 5
    dummy_ids   = [f"deed_{i}" for i in range(N)]
    dummy_scores = np.random.rand(N, len(classes)).astype(np.float32)
    dummy_pred   = dummy_scores > 0.5

    path = write_predictions(dummy_ids, dummy_scores, dummy_pred, classes, method=method)
    df   = read_predictions(method)

    assert set(df.columns) == {"akteId", "code", "score", "predicted", "method"}, \
        "Schema mismatch"
    assert len(df) == N * len(classes), f"Expected {N * len(classes)} rows, got {len(df)}"

    # clean up test file
    path.unlink()
    print("✓ roundtrip_check passed")
    return True
