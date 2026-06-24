"""
shared/splits.py — stratified train/val split, frozen once.

Why this file exists
--------------------
If each notebook re-runs the split step, there is a chance (especially
after library updates or random-state bugs) that train and val indices
drift.  Worse: any notebook that accidentally leaks val data into training
will look better than it is, and you'll never know.

The contract: run freeze_split() once in 00_splits.ipynb. Every notebook
then calls load_split() which returns the same indices every time.

Usage
-----
    from shared.splits import freeze_split, load_split

    # ONCE in 00_splits.ipynb:
    freeze_split(train_full_df, y_full)

    # Everywhere else:
    train_idx, val_idx = load_split()
    train_df = train_full_df.iloc[train_idx].reset_index(drop=True)
    val_df   = train_full_df.iloc[val_idx].reset_index(drop=True)
"""

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from shared.config import RANDOM_STATE, VAL_SIZE, TRAIN_IDX_PATH, VAL_IDX_PATH


def freeze_split(train_full_df, y_full: np.ndarray) -> tuple:
    """
    Run a single stratified split and save the indices to disk.

    Parameters
    ----------
    train_full_df : DataFrame  (used only for length)
    y_full        : np.ndarray of shape (N, num_classes), float32

    Returns train_idx, val_idx as numpy arrays.
    Raises FileExistsError if the split files already exist.
    """
    if TRAIN_IDX_PATH.exists() or VAL_IDX_PATH.exists():
        raise FileExistsError(
            f"Split files already exist at {TRAIN_IDX_PATH.parent}. "
            "Delete them explicitly to re-split."
        )

    X_idx = np.arange(len(train_full_df))
    msss  = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(msss.split(X_idx, y_full))

    TRAIN_IDX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_IDX_PATH, train_idx)
    np.save(VAL_IDX_PATH,   val_idx)

    # quick sanity print
    ml_full  = (y_full.sum(axis=1) > 1).mean()
    ml_train = (y_full[train_idx].sum(axis=1) > 1).mean()
    ml_val   = (y_full[val_idx].sum(axis=1) > 1).mean()
    print(f"Split frozen → {TRAIN_IDX_PATH.parent}")
    print(f"  Train : {len(train_idx):,} docs   multi-label: {ml_train:.1%}")
    print(f"  Val   : {len(val_idx):,}  docs   multi-label: {ml_val:.1%}")
    print(f"  Full  : {len(train_full_df):,} docs   multi-label: {ml_full:.1%}")

    return train_idx, val_idx


def load_split() -> tuple:
    """
    Load the frozen train/val indices from disk.

    Returns
    -------
    train_idx, val_idx : np.ndarray
    """
    if not TRAIN_IDX_PATH.exists() or not VAL_IDX_PATH.exists():
        raise FileNotFoundError(
            f"Split files not found at {TRAIN_IDX_PATH.parent}. "
            "Run freeze_split() in 00_splits.ipynb first."
        )
    train_idx = np.load(TRAIN_IDX_PATH)
    val_idx   = np.load(VAL_IDX_PATH)
    print(f"Split loaded: {len(train_idx):,} train / {len(val_idx):,} val")
    return train_idx, val_idx
