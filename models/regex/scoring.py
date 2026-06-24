"""
models/regex/scoring.py — scoring, threshold tuning, and post-processing.

Hybrid regex engine. Given a pattern table and
a corpus, it produces a precision-weighted (deed × code) score matrix, tunes per-code
F1-maximising thresholds, and applies two deterministic post-processors:

  · koop tie-break  — 651/652/696 are near-identical koopovereenkomst variants;
                      keep exactly one based on the cited statute.
  · mutex pairs     — sibling codes that should not co-fire (e.g. 545/572);
                      keep the higher-scoring one.

``scores_to_unit_interval`` squashes the raw unbounded score into [0, 1] for
the shared prediction contract (shared/io.py).
"""

from __future__ import annotations

import re
from collections import defaultdict

import ahocorasick
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    _HAVE_ITERSTRAT = True
except ImportError:  # pragma: no cover
    from sklearn.model_selection import KFold
    _HAVE_ITERSTRAT = False

# 5-fold CV is used to tune thresholds out-of-fold on the train split.
# RANDOM_STATE matches shared.config.RANDOM_STATE (the frozen-split seed); it is
# duplicated here so the GPU-free regex layer imports without pulling in torch.
N_SPLITS = 5
RANDOM_STATE = 42

# Raw fuzzy-gap regex carries this marker (e.g. ".{0,80}").
FUZZY_MARKER = ".{0,"

# Sibling code pairs that should not both fire on one deed (lever L6).
MUTEX_PAIRS = [(545, 572), (564, 585), (606, 532)]

_PREFIX_END_RE = re.compile(r"De\s+bewaarder\.?\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text-zone helpers
# ---------------------------------------------------------------------------
def strip_kadaster_prefix(text: str) -> str:
    """Drop everything up to and including 'De bewaarder.' if present.

    Training deeds carry a Kadaster registration prefix ("Het bijgevoegde stuk
    is ingeschreven ... De bewaarder.") that is NOT part of the original
    notarial text. Stripping it stops TF-IDF / chi2 mining from anchoring on
    leak words like 'Hyp3' or 'register', and makes audit numbers transfer 1-1
    to production deeds that lack the prefix.
    """
    m = _PREFIX_END_RE.search(text[:500])
    return text[m.end():] if m else text


def title_zone(text: str, length: int = 600) -> str:
    """First `length` chars after the Kadaster prefix (idempotent if pre-stripped)."""
    m = _PREFIX_END_RE.search(text[:500])
    start = m.end() if m else 0
    return text[start: start + length]


def _split_fuzzy_parts(source_text: str) -> list[str]:
    parts = re.split(r"\s*(?:…|\.{2,})\s*", source_text)
    return [p.strip().lower() for p in parts if p.strip()]


def _word_bounded(text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not text[start - 1].isalpha()
    right_ok = end == len(text) - 1 or not text[end + 1].isalpha()
    return left_ok and right_ok


# ---------------------------------------------------------------------------
# Matching streams: Aho-Corasick for simple terms, candidate-filtered regex
# for fuzzy multi-part phrases.
# ---------------------------------------------------------------------------
def _ac_match(pat_idxs, text_array, source_texts, M_rows, M_cols):
    if len(pat_idxs) == 0:
        return
    term_to_pidxs: dict[str, list[int]] = defaultdict(list)
    for p in pat_idxs:
        term_to_pidxs[source_texts[p].lower()].append(int(p))
    A = ahocorasick.Automaton()
    for term in term_to_pidxs:
        A.add_word(term, term)
    A.make_automaton()
    for d_idx, txt in enumerate(text_array):
        seen_terms = set()
        for end_idx, term in A.iter(txt):
            start = end_idx - len(term) + 1
            if _word_bounded(txt, start, end_idx):
                seen_terms.add(term)
        for term in seen_terms:
            for p in term_to_pidxs[term]:
                M_rows.append(d_idx)
                M_cols.append(p)


def _fuzzy_match(pat_idxs, text_array, source_texts, regex_strs, M_rows, M_cols):
    if len(pat_idxs) == 0:
        return
    fuzzy_parts_per_pat = [_split_fuzzy_parts(source_texts[i]) for i in pat_idxs]
    all_parts = sorted({pt for parts in fuzzy_parts_per_pat for pt in parts if pt})
    part_to_docs: dict[str, set[int]] = {p: set() for p in all_parts}
    if all_parts:
        A_f = ahocorasick.Automaton()
        for term in all_parts:
            A_f.add_word(term, term)
        A_f.make_automaton()
        for d_idx, txt in enumerate(text_array):
            for end_idx, term in A_f.iter(txt):
                start = end_idx - len(term) + 1
                if _word_bounded(txt, start, end_idx):
                    part_to_docs[term].add(d_idx)
    for fp_i, p_idx in enumerate(pat_idxs):
        parts = fuzzy_parts_per_pat[fp_i]
        if not parts:
            continue
        candidates: set[int] | None = None
        for part in parts:
            docs = part_to_docs.get(part, set())
            candidates = docs if candidates is None else candidates & docs
            if not candidates:
                break
        if not candidates:
            continue
        rgx = re.compile(regex_strs[p_idx].replace("(?is)", "(?s)").replace("(?i)", ""))
        for d_idx in candidates:
            if rgx.search(text_array[d_idx]):
                M_rows.append(int(d_idx))
                M_cols.append(int(p_idx))


# ---------------------------------------------------------------------------
# Corpus scoring
# ---------------------------------------------------------------------------
def score_corpus(texts, patterns_df, code_to_idx, label_sets=None, weights=None):
    """Precision-weighted scoring. Returns (score, pattern_hits, hits_pos, prox).

    Pass ``label_sets`` (training) to *learn* per-pattern precision weights, or
    pass pre-computed ``weights`` (test/inference) to reuse the training
    weights and avoid label leakage. ``patterns_df`` must carry a ``match_zone``
    column ("full" or "title").
    """
    n_docs, n_codes, n_patterns = len(texts), len(code_to_idx), len(patterns_df)
    texts_lower = [t.lower() for t in texts]
    title_zones_lower = [title_zone(t).lower() for t in texts]
    regex_strs = patterns_df["regex"].tolist()
    source_texts = patterns_df["source_text"].tolist()
    pattern_codes = patterns_df["code"].to_numpy()

    is_fuzzy = np.array([FUZZY_MARKER in s for s in regex_strs])
    is_title = np.array([z == "title" for z in patterns_df["match_zone"]])
    simple_full = np.where(~is_fuzzy & ~is_title)[0]
    simple_title = np.where(~is_fuzzy & is_title)[0]
    fuzzy_full = np.where(is_fuzzy & ~is_title)[0]
    fuzzy_title = np.where(is_fuzzy & is_title)[0]

    M_rows: list[int] = []
    M_cols: list[int] = []
    _ac_match(simple_full, texts_lower, source_texts, M_rows, M_cols)
    _ac_match(simple_title, title_zones_lower, source_texts, M_rows, M_cols)
    _fuzzy_match(fuzzy_full, texts_lower, source_texts, regex_strs, M_rows, M_cols)
    _fuzzy_match(fuzzy_title, title_zones_lower, source_texts, regex_strs, M_rows, M_cols)

    M = csr_matrix(
        (np.ones(len(M_rows), dtype=np.int8), (M_rows, M_cols)),
        shape=(n_docs, n_patterns), dtype=np.int8,
    )
    M.sum_duplicates()
    M.data[:] = 1
    pattern_hits = np.asarray(M.sum(axis=0)).ravel().astype(np.int32)
    pattern_hits_pos = np.zeros(n_patterns, dtype=np.int32)
    if weights is not None:
        precision_proxy = np.asarray(weights, dtype=np.float32)
    else:
        if label_sets is None:
            raise ValueError("Need either label_sets (to fit weights) or weights.")
        M_csc = M.tocsc()
        for p in range(n_patterns):
            c = int(pattern_codes[p])
            nz_rows = M_csc.getcol(p).indices
            if len(nz_rows) == 0:
                continue
            pattern_hits_pos[p] = sum(1 for d in nz_rows if c in label_sets[d])
        precision_proxy = (pattern_hits_pos / np.maximum(pattern_hits, 1)).astype(np.float32)

    pattern_code_idx = np.array([code_to_idx[int(c)] for c in pattern_codes])
    W = csr_matrix(
        (precision_proxy.astype(np.float32),
         (np.arange(n_patterns), pattern_code_idx)),
        shape=(n_patterns, n_codes), dtype=np.float32,
    )
    score = (M.astype(np.float32) @ W).toarray()
    return score, pattern_hits, pattern_hits_pos, precision_proxy


# ---------------------------------------------------------------------------
# Label matrix + threshold tuning (5-fold CV)
# ---------------------------------------------------------------------------
def build_y_matrix(df, code_to_idx) -> np.ndarray:
    """Binary (n_deeds × n_codes) label matrix aligned to ``code_to_idx``."""
    y = np.zeros((len(df), len(code_to_idx)), dtype=np.int8)
    for i, codes in enumerate(df["rechtsfeitcodes"]):
        for c in codes:
            j = code_to_idx.get(int(c))
            if j is not None:
                y[i, j] = 1
    return y


def tune_thresholds(score, y_true):
    """Per-class threshold maximising F1 on the given (score, y) pair.

    Codes with no positives / no signal get ``inf`` (never predicted).
    """
    n_codes = score.shape[1]
    thresholds = np.full(n_codes, np.inf, dtype=np.float64)
    for c in range(n_codes):
        col, y_c = score[:, c], y_true[:, c]
        if y_c.sum() == 0 or col.max() == 0:
            continue
        nz = col[col > 0]
        if len(nz) == 0:
            continue
        best_f1, best_t = 0.0, np.inf
        for t in np.unique(nz):
            pred = (col >= t).astype(np.int8)
            tp = int((pred & y_c).sum())
            if tp == 0:
                continue
            fp = int(pred.sum() - tp)
            fn = int(y_c.sum() - tp)
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def apply_thresholds(score, thresholds):
    """Apply per-class thresholds to a score block → int8 predictions."""
    finite = np.isfinite(thresholds)
    if not finite.any():
        return np.zeros_like(score, dtype=np.int8)
    t = thresholds.copy()
    t[~finite] = float(score.max()) + 1.0
    return (score >= t[None, :]).astype(np.int8)


def cv_predict(score, y_true):
    """5-fold CV held-out predictions: tune on 4 folds, predict on the held one."""
    y_pred = np.zeros_like(y_true, dtype=np.int8)
    if _HAVE_ITERSTRAT:
        splitter = MultilabelStratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        folds = splitter.split(score, y_true)
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        folds = splitter.split(score)
    for tune_idx, hold_idx in folds:
        th = tune_thresholds(score[tune_idx], y_true[tune_idx])
        y_pred[hold_idx] = apply_thresholds(score[hold_idx], th)
    return y_pred


def per_code_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-code F1 vector (used for champion selection between two variants)."""
    out = np.zeros(y_true.shape[1])
    for j in range(y_true.shape[1]):
        a, b = y_true[:, j], y_pred[:, j]
        tp = int((a & b).sum())
        fp = int(b.sum() - tp)
        fn = int(a.sum() - tp)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out[j] = 2 * p * r / (p + r) if (p + r) else 0.0
    return out


# ---------------------------------------------------------------------------
# Post-processors
# ---------------------------------------------------------------------------
_RX_OMG = re.compile(
    r"\b(?:art(?:ikel)?\.?\s*)?9\.9\b.{0,30}Omgevingswet", re.IGNORECASE | re.DOTALL)
_RX_WVG = re.compile(
    r"\b(?:art(?:ikel)?\.?\s*)?10\b.{0,20}WVG\b", re.IGNORECASE | re.DOTALL)


def apply_koop_tiebreak(y_pred, texts, code_to_idx):
    """If any of {651, 652, 696} is predicted, keep ONE based on the cited statute.

    Order: 9.9 Omgevingswet → 696 · 10 WVG → 651 · otherwise → 652.
    """
    target_codes = [c for c in (651, 652, 696) if c in code_to_idx]
    if len(target_codes) < 2:
        return y_pred
    idxs = [code_to_idx[c] for c in target_codes]
    y2 = y_pred.copy()
    for d in range(y2.shape[0]):
        on = [i for i in idxs if y2[d, i] == 1]
        if not on:
            continue
        winner = None
        txt = texts[d]
        if _RX_OMG.search(txt) and 696 in code_to_idx:
            winner = code_to_idx[696]
        elif _RX_WVG.search(txt) and 651 in code_to_idx:
            winner = code_to_idx[651]
        elif 652 in code_to_idx:
            winner = code_to_idx[652]
        if winner is not None:
            for i in idxs:
                y2[d, i] = 1 if i == winner else 0
    return y2


def apply_mutex(y_pred, score, code_to_idx, pairs=MUTEX_PAIRS):
    """For each sibling pair, if both fire on a deed keep the higher-scoring one."""
    y2 = y_pred.copy()
    for a, b in pairs:
        if a not in code_to_idx or b not in code_to_idx:
            continue
        ja, jb = code_to_idx[a], code_to_idx[b]
        both = (y2[:, ja] == 1) & (y2[:, jb] == 1)
        if not both.any():
            continue
        a_wins = score[:, ja] >= score[:, jb]
        y2[both & ~a_wins, ja] = 0
        y2[both & a_wins, jb] = 0
    return y2


# ---------------------------------------------------------------------------
# Contract helper: bound the raw score into [0, 1]
# ---------------------------------------------------------------------------
def scores_to_unit_interval(score: np.ndarray, norm: float | None = None):
    """Min-max squash raw precision-weighted scores into [0, 1].

    A single monotonic global scalar (``norm``) preserves ordering within a
    deed (1st-vs-2nd margin) and between deeds. Returns (score01, norm);
    pass the returned ``norm`` back when scoring a second split so both share
    one scale.
    """
    if norm is None:
        norm = float(score.max()) if score.size and score.max() > 0 else 1.0
    score01 = np.clip(score / norm, 0.0, 1.0).astype(np.float32)
    return score01, norm
