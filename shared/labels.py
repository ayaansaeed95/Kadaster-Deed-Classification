"""
shared/labels.py — canonical label space.

Why this file exists
--------------------
Each notebook used to call mlb.fit(df["rechtsfeitcodes"]) independently.
If the data slice seen at fit-time differs even slightly (e.g. a rare code
that only appears in train but not in a particular data slice),
the class ordering diverges and the binary matrices are no longer aligned.
Once that happens, macro-F1 numbers can't be compared across methods.

This module freezes the label space once (from the full train file) and
saves it to artifacts/splits/label_space.json.  Every subsequent notebook
calls get_mlb() which loads that frozen file — guaranteed identical ordering.

Usage
-----
    from shared.labels import build_label_space, get_mlb

    # Run ONCE (in 00_splits.ipynb) to freeze:
    build_label_space(train_full_df)

    # In every other notebook:
    mlb, classes, num_classes = get_mlb()
    y = mlb.transform(df["rechtsfeitcodes"])
"""

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from shared.config import LABEL_SPACE_PATH


def build_label_space(train_full_df) -> list:
    """
    Fit a MultiLabelBinarizer on the full train DataFrame and save the
    ordered class list to LABEL_SPACE_PATH.

    Call this ONCE in 00_splits.ipynb.  Raises if the file already exists
    to prevent accidental overwrites.
    """
    if LABEL_SPACE_PATH.exists():
        raise FileExistsError(
            f"{LABEL_SPACE_PATH} already exists. "
            "Delete it explicitly if you really want to re-fit the label space."
        )

    mlb = MultiLabelBinarizer()
    mlb.fit(train_full_df["rechtsfeitcodes"])
    classes = [int(c) if str(c).isdigit() else str(c) for c in mlb.classes_]

    LABEL_SPACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_SPACE_PATH.open("w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"Label space frozen: {len(classes)} classes → {LABEL_SPACE_PATH}")
    return classes


def get_mlb():
    """
    Load the frozen label space and return a fitted MultiLabelBinarizer,
    the class list, and the number of classes.

    Returns
    -------
    mlb        : fitted MultiLabelBinarizer
    classes    : list of rechtsfeitcodes (same order every time)
    num_classes: int
    """
    if not LABEL_SPACE_PATH.exists():
        raise FileNotFoundError(
            f"{LABEL_SPACE_PATH} not found. "
            "Run build_label_space() in 00_splits.ipynb first."
        )

    with LABEL_SPACE_PATH.open("r", encoding="utf-8") as f:
        classes = json.load(f)

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])   # fit with known classes so ordering is locked

    num_classes = len(classes)
    print(f"Label space loaded: {num_classes} classes")
    return mlb, classes, num_classes
