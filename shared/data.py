"""
shared/data.py — canonical data loader.
Every notebook calls load_jsonl() to read the raw data files.
Returns a DataFrame with exactly three columns: akteId, text, rechtsfeitcodes.
"""

import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a .jsonl file into a DataFrame.

    Returns columns: akteId (str), text (str), rechtsfeitcodes (list[int|str]).
    rechtsfeitcodes is always a list — never a bare scalar or NaN.
    """
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)[["akteId", "text", "rechtsfeitcodes"]]

    # Normalise: bare scalar → list; NaN / None → empty list
    df["rechtsfeitcodes"] = df["rechtsfeitcodes"].apply(
        lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else [])
    )
    return df


def load_train_test(train_path: Path, test_path: Path):
    """Convenience: load both files and print shapes."""
    train_df = load_jsonl(train_path)
    test_df  = load_jsonl(test_path)
    print(f"Train : {len(train_df):,} documents")
    print(f"Test  : {len(test_df):,} documents")
    return train_df, test_df
