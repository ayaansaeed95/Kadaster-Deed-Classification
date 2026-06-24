"""
models/regex/pipeline.py — the hybrid regex classifier (L2).

The classifier blends two variants and picks a champion per code:

  V1  full-train patterns over ALL codes. Strong on the high-volume head
      codes (606, 537, ...) that have a clear standardised title phrase.

  V2  tail patterns mined on the "overige" bucket — deeds carrying none of the
      large (>= CUTOFF) codes. V2 only sees the rare codes, so its TF-IDF / chi2
      mining is not drowned out by the head, which lifts the long tail.

  champion gating   for each code, keep whichever variant has the higher
                    out-of-fold (5-fold CV) F1 on the train split.

  post-processing   koop tie-break (651/652/696) is applied inside each variant;
                    the mutex pass (545/572, 564/585, 606/532) runs on the final
                    champion predictions.

Deployability note: routing a deed to V2 depends on whether V1 *predicts* no
large code — never on the true labels (which are unknown at inference). This is
the one change from the exploratory notebook, and it makes the val/test numbers
honest.

Public API
----------
    result = run_hybrid(train_df, val_df, test_df, classes)
    result["val"]["scores01"], result["val"]["pred"]      # (N_val, C)
    result["test"]["scores01"], result["test"]["pred"]    # (N_test, C)
    result["champion"], result["large_codes"], result["tail_codes"], ...

``scores01`` are in [0, 1] on one shared scale across val + test, ready for the
prediction contract (shared/io.write_predictions).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from models.regex.patterns import build_patterns, CUTOFF
from models.regex.scoring import (
    strip_kadaster_prefix, score_corpus, build_y_matrix, tune_thresholds,
    apply_thresholds, cv_predict, apply_koop_tiebreak, apply_mutex, per_code_f1,
    scores_to_unit_interval,
)


def _strip(df):
    """Return a copy with the Kadaster registration prefix stripped from text."""
    out = df.copy()
    out["text"] = out["text"].astype(str).apply(strip_kadaster_prefix)
    return out


def run_hybrid(train_df, val_df, test_df, classes) -> dict:
    """Fit the hybrid on ``train_df`` and predict ``val_df`` + ``test_df``.

    All three frames need ``text`` and ``rechtsfeitcodes`` columns. ``classes``
    is the frozen ordered label list from shared.labels.get_mlb().
    """
    all_codes = [int(c) for c in classes]
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    train_df, val_df, test_df = _strip(train_df), _strip(val_df), _strip(test_df)

    # large vs tail split, from the train split's support
    sup = Counter()
    for cs in train_df["rechtsfeitcodes"]:
        sup.update(int(c) for c in cs)
    large_codes = {c for c in all_codes if sup.get(c, 0) >= CUTOFF}
    tail_codes = sorted(c for c in all_codes if sup.get(c, 0) < CUTOFF)
    code_to_idx_tail = {c: i for i, c in enumerate(tail_codes)}
    large_idx = [code_to_idx[c] for c in large_codes]

    y_tr = build_y_matrix(train_df, code_to_idx)

    # ---- V1: all-code patterns mined on the full train split ----
    patterns_v1, _ = build_patterns(all_codes, train_df)
    ls1 = [set(int(c) for c in cs) for cs in train_df["rechtsfeitcodes"]]
    sc_tr_v1, _, _, prox_v1 = score_corpus(
        train_df["text"].tolist(), patterns_v1, code_to_idx, ls1)
    th_v1 = tune_thresholds(sc_tr_v1, y_tr)

    def v1_score_pred(df):
        s, _, _, _ = score_corpus(df["text"].tolist(), patterns_v1, code_to_idx, None, prox_v1)
        p = apply_thresholds(s, th_v1)
        p = apply_koop_tiebreak(p, df["text"].tolist(), code_to_idx)
        return s, p

    # ---- V2: tail patterns mined on the 'overige' bucket of the train split ----
    ov_mask_tr = train_df["rechtsfeitcodes"].apply(
        lambda cs: not (set(int(c) for c in cs) & large_codes)).to_numpy()
    tr_ov = train_df[ov_mask_tr].reset_index(drop=True)
    patterns_v2, accepted_vl = build_patterns(tail_codes, tr_ov)
    ls2 = [set(int(c) for c in cs) for cs in tr_ov["rechtsfeitcodes"]]
    sc_to_v2, _, _, prox_v2 = score_corpus(
        tr_ov["text"].tolist(), patterns_v2, code_to_idx_tail, ls2)
    y_to = build_y_matrix(tr_ov, code_to_idx_tail)
    th_v2 = tune_thresholds(sc_to_v2, y_to)

    def v2_score_pred(df, v1_pred):
        # Deployable routing: 'overige' = V1 predicts NO large code (not true labels)
        mask = (v1_pred[:, large_idx].sum(axis=1) == 0)
        ov = df[mask].reset_index(drop=True)
        s_o, _, _, _ = score_corpus(
            ov["text"].tolist(), patterns_v2, code_to_idx_tail, None, prox_v2)
        p_o = apply_thresholds(s_o, th_v2)
        p_o = apply_koop_tiebreak(p_o, ov["text"].tolist(), code_to_idx_tail)
        sc = np.zeros((len(df), len(all_codes)), dtype=float)
        pr = np.zeros((len(df), len(all_codes)), dtype=np.int8)
        sct = np.zeros((len(df), len(tail_codes)), dtype=float); sct[mask] = s_o
        prt = np.zeros((len(df), len(tail_codes)), dtype=np.int8); prt[mask] = p_o
        for c, j in code_to_idx_tail.items():
            sc[:, code_to_idx[c]] = sct[:, j]
            pr[:, code_to_idx[c]] = prt[:, j]
        return sc, pr

    # ---- champion per code (out-of-fold CV F1 on the train split) ----
    pred_tr_v1 = apply_koop_tiebreak(
        cv_predict(sc_tr_v1, y_tr), train_df["text"].tolist(), code_to_idx)
    pred_to_v2 = apply_koop_tiebreak(
        cv_predict(sc_to_v2, y_to), tr_ov["text"].tolist(), code_to_idx_tail)
    pred_tr_v2 = np.zeros_like(y_tr)
    prt = np.zeros((len(train_df), len(tail_codes)), dtype=np.int8)
    prt[ov_mask_tr] = pred_to_v2
    for c, j in code_to_idx_tail.items():
        pred_tr_v2[:, code_to_idx[c]] = prt[:, j]
    champ = np.where(per_code_f1(y_tr, pred_tr_v1) >= per_code_f1(y_tr, pred_tr_v2), 1, 2)

    def hybrid_scores(df):
        s1, p1 = v1_score_pred(df)
        s2, p2 = v2_score_pred(df, p1)
        pred = apply_mutex(np.where(champ[None, :] == 1, p1, p2), s1, code_to_idx)
        sc = np.where(champ[None, :] == 1, s1, s2)
        return sc, pred

    sc_val, pred_val = hybrid_scores(val_df)
    sc_test, pred_test = hybrid_scores(test_df)

    # one shared [0, 1] scale across val + test
    norm = max(float(sc_val.max()), float(sc_test.max()), 1.0)
    val01, _ = scores_to_unit_interval(sc_val, norm)
    test01, _ = scores_to_unit_interval(sc_test, norm)

    return {
        "classes": all_codes,
        "code_to_idx": code_to_idx,
        "large_codes": sorted(large_codes),
        "tail_codes": tail_codes,
        "champion": champ,
        "n_champion_v1": int((champ == 1).sum()),
        "n_champion_v2": int((champ == 2).sum()),
        "patterns_v1": patterns_v1,
        "patterns_v2": patterns_v2,
        "accepted_valuelist": accepted_vl,
        "val":  {"akteId": val_df["akteId"].tolist(), "scores01": val01, "pred": pred_val.astype(bool)},
        "test": {"akteId": test_df["akteId"].tolist(), "scores01": test01, "pred": pred_test.astype(bool)},
    }
