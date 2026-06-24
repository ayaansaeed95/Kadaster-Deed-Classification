"""
shared/harness.py — the ONE scorer.

Every method's predictions go through evaluate() here.
Nobody computes metrics inline in a personal notebook.

Why this matters: if one method computes macro-F1 with zero_division=1 and another
uses zero_division=0, the numbers look different even on identical predictions.
One harness, one number, comparable results.

Usage
-----
    from shared.harness import evaluate, per_class_matrix, print_summary

    metrics = evaluate(y_true, y_pred, classes, method="bert")
    # → writes results/bert_metrics.json
    # → returns dict with micro_f1, macro_f1, per_class rows

    per_class_matrix(methods_dict, classes)
    # methods_dict = {"bert": (y_true, y_pred), "regex": (y_true, y_pred)}
    # → writes results/per_class_matrix.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)

from shared.config import RESULTS


def evaluate(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    classes: list,
    method:  str,
    save:    bool = True,
) -> dict:
    """
    Compute micro-F1, macro-F1, and per-class P/R/F1/support.

    Parameters
    ----------
    y_true   : (N, num_classes) binary int array
    y_pred   : (N, num_classes) binary int array
    classes  : ordered list of rechtsfeitcodes (from get_mlb())
    method   : name string written into the output files ("bert", "regex", …)
    save     : if True, write results/{method}_metrics.json

    Returns
    -------
    dict with keys: micro_f1, macro_f1, active_labels, per_class (list of dicts)
    """
    support = y_true.sum(axis=0).astype(int)
    active  = support > 0                        # labels with ≥1 true example

    micro_f1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    macro_f1 = f1_score(
        y_true[:, active], y_pred[:, active], average="macro", zero_division=0
    )

    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(   y_true, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(       y_true, y_pred, average=None, zero_division=0)

    per_class = [
        {
            "code"    : int(c) if str(c).isdigit() else c,
            "precision": float(round(per_class_p[i], 4)),
            "recall"  : float(round(per_class_r[i], 4)),
            "f1"      : float(round(per_class_f[i], 4)),
            "support" : int(support[i]),
            "method"  : method,
        }
        for i, c in enumerate(classes)
    ]

    result = {
        "method"       : method,
        "micro_f1"     : float(round(micro_f1, 4)),
        "macro_f1"     : float(round(macro_f1, 4)),
        "active_labels": int(active.sum()),
        "per_class"    : per_class,
    }

    if save:
        out_path = RESULTS / f"{method}_metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Metrics saved → {out_path}")

    print(f"[{method}]  micro-F1: {micro_f1:.4f}   macro-F1: {macro_f1:.4f}   "
          f"active labels: {active.sum()}")
    return result


def per_class_matrix(methods_dict: dict, classes: list, save: bool = True) -> pd.DataFrame:
    """
    Build the per-class performance matrix across all methods.

    Parameters
    ----------
    methods_dict : {"bert": (y_true, y_pred), "regex": (y_true, y_pred), …}
    classes      : ordered label list from get_mlb()
    save         : if True, write results/per_class_matrix.csv

    Returns
    -------
    Wide DataFrame: one row per code, one column-group per method (P/R/F1/support)
    """
    rows = []
    for method, (y_true, y_pred) in methods_dict.items():
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(   y_true, y_pred, average=None, zero_division=0)
        f = f1_score(       y_true, y_pred, average=None, zero_division=0)
        s = y_true.sum(axis=0).astype(int)
        for i, c in enumerate(classes):
            rows.append({
                "code"     : int(c) if str(c).isdigit() else c,
                "method"   : method,
                "precision": round(float(p[i]), 4),
                "recall"   : round(float(r[i]), 4),
                "f1"       : round(float(f[i]), 4),
                "support"  : int(s[i]),
            })

    long_df = pd.DataFrame(rows)

    # pivot to wide: code × (method_f1, method_precision, …)
    wide_df = long_df.pivot(index="code", columns="method",
                            values=["precision", "recall", "f1", "support"])
    wide_df.columns = [f"{m}_{metric}" for metric, m in wide_df.columns]
    wide_df = wide_df.reset_index().sort_values("code")

    # add a "best_method" column (by F1)
    f1_cols = [c for c in wide_df.columns if c.endswith("_f1")]
    if f1_cols:
        wide_df["best_method"] = wide_df[f1_cols].idxmax(axis=1).str.replace("_f1", "")

    if save:
        out_path = RESULTS / "per_class_matrix.csv"
        wide_df.to_csv(out_path, index=False)
        print(f"Per-class matrix saved → {out_path}")

    return wide_df


def print_summary(metrics_list: list[dict]) -> None:
    """
    Print a comparison table of methods side-by-side.

    Parameters
    ----------
    metrics_list : list of dicts returned by evaluate()
    """
    print(f"\n{'Method':<18} {'micro-F1':>10} {'macro-F1':>10} {'active labels':>14}")
    print("-" * 56)
    for m in metrics_list:
        print(f"{m['method']:<18} {m['micro_f1']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['active_labels']:>14}")
