"""
models/regex/patterns.py — build the pattern table from five sources.

``build_patterns(codes, mining_df)`` combines, for the requested ``codes``:

  · Source A — Kadaster Excel curated phrases       (pattern_sources.EXCEL_CODES)
  · Source B — Pothoven docx secondary triggers     (pattern_sources.POTHOVEN_CODES)
  · Sources C & D — TF-IDF + chi2 terms mined from ``mining_df``
                    (Yang & Pedersen 1997 feature selection)
  · Source E — hand-curated + mined title patterns and runtime-validated
               valuelist candidates

It returns ``(patterns_df, accepted_valuelist)``. The same builder produces
both variants of the hybrid pipeline:

  · V1 — ``build_patterns(all_codes, train_split)``  (every code, full train)
  · V2 — ``build_patterns(tail_codes, overige_bucket)`` (tail codes only)
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

from models.regex.pattern_sources import (
    DUTCH_STOPWORDS, EXCEL_CODES, POTHOVEN_CODES, NOTEBOOK_MANUAL_PATTERNS,
    MINED_TITLE_PATTERNS, VALUELIST_CANDIDATES,
)
from models.regex.scoring import title_zone

# --------------------------------------------------------------------------
# Config — the "large vs tail" split and the mining hyperparameters.
# --------------------------------------------------------------------------
# A code is "large" if it has >= CUTOFF deeds in the train split; the base
# transformer handles those, so the tail variant (V2) only mines codes below it.
CUTOFF = 500

# Data-derived mining (Sources C & D)
TFIDF_TOP_K = 25
TFIDF_MIN_SAMPLES = 1     # mine even for codes with < 5 deeds
TFIDF_MIN_DF = 2          # overige corpus is small; min_df must scale down
TFIDF_MAX_DF = 0.5

# Runtime validation of valuelist candidates (Source E)
VL_MIN_PRECISION = 0.85
VL_MIN_HITS = 3


# --------------------------------------------------------------------------
# Phrase → regex, pattern-row builder
# --------------------------------------------------------------------------
def to_word_regex(s: str, fuzzy_gap: bool = True) -> str:
    """Turn a piece of source text into a regex string.

    Single keyword → word-bounded, case-insensitive match.
    Multi-part phrase with `...` or `…` → parts in order, up to 80 chars apart.
    """
    parts = re.split(r"\s*(?:…|\.{2,})\s*", s) if fuzzy_gap else [s]
    parts = [re.escape(p.strip()) for p in parts if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return rf"(?i)\b{parts[0]}\b"
    return r"(?is)" + r".{0,80}".join(parts)


def make_pattern_row(code, source, source_text, regex, name, seq, **extra) -> dict:
    """Build one pattern-row dict and bump the (code, source) sequence counter."""
    seq[(code, source)] += 1
    row = {
        "code": code,
        "pattern_id": f"{code}-{source}-{seq[(code, source)]:03d}",
        "source": source,
        "source_text": source_text,
        "regex": regex,
        "name": name,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# Sources C & D — TF-IDF and chi2 mining
# --------------------------------------------------------------------------
def derive_ngrams_per_code(df, codes):
    """Top-K n-grams per code, ranked by mean-TF-IDF difference (with vs without)."""
    texts = df["text"].fillna("").tolist()
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3), max_features=50_000,
        min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, stop_words=DUTCH_STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{3,}\b",
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    label_sets = [set(c) for c in df["rechtsfeitcodes"]]
    out: dict[int, list[tuple[str, float]]] = {}
    for code in codes:
        mask = np.array([code in s for s in label_sets])
        if mask.sum() < TFIDF_MIN_SAMPLES:
            out[code] = []
            continue
        pos_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        neg_mean = np.asarray(X[~mask].mean(axis=0)).ravel()
        diff = pos_mean - neg_mean
        top_idx = np.argsort(-diff)[:TFIDF_TOP_K]
        out[code] = [(vocab[i], float(diff[i])) for i in top_idx if diff[i] > 0]
    return out


def derive_chi2_per_code(df, codes):
    """Top-K candidate terms per code via one-vs-rest chi2 (Yang & Pedersen 1997)."""
    texts = df["text"].fillna("").tolist()
    cv = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3), max_features=50_000,
        min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, stop_words=DUTCH_STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{3,}\b",
        use_idf=False, norm=None,
    )
    X = cv.fit_transform(texts)
    X_bin = (X > 0).astype(np.int8)
    vocab = np.array(cv.get_feature_names_out())
    label_sets = [set(c) for c in df["rechtsfeitcodes"]]
    out: dict[int, list[tuple[str, float]]] = {}
    for code in codes:
        y_c = np.array([code in s for s in label_sets], dtype=bool)
        if int(y_c.sum()) < TFIDF_MIN_SAMPLES:
            out[code] = []
            continue
        scores, _ = chi2(X_bin, y_c.astype(np.int8))
        scores = np.nan_to_num(np.asarray(scores), nan=0.0, posinf=0.0, neginf=0.0)
        pos_hits = np.asarray(X_bin[y_c].sum(axis=0)).ravel()
        scores = np.where(pos_hits > 0, scores, 0.0)
        top_idx = np.argsort(-scores)[:TFIDF_TOP_K]
        out[code] = [(vocab[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    return out


# --------------------------------------------------------------------------
# Source E — runtime validation of valuelist candidates
# --------------------------------------------------------------------------
def _validate_valuelist_candidates(mining_df, code_set):
    """Keep only valuelist candidates whose title-zone precision passes."""
    titles = [title_zone(t).lower() for t in mining_df["text"].fillna("")]
    label_sets = [set(int(c) for c in cs) for cs in mining_df["rechtsfeitcodes"]]
    accepted = []
    for code, phrase, name in VALUELIST_CANDIDATES:
        if code not in code_set:
            continue
        rgx = re.compile(rf"\b{re.escape(phrase.lower())}\b", re.IGNORECASE)
        hits_total = hits_pos = 0
        for t, ls in zip(titles, label_sets):
            if rgx.search(t):
                hits_total += 1
                if code in ls:
                    hits_pos += 1
        if hits_total < VL_MIN_HITS:
            continue
        prec = hits_pos / hits_total
        if prec < VL_MIN_PRECISION:
            continue
        accepted.append({
            "code": code, "phrase": phrase, "name": name,
            "precision": round(prec, 3),
            "hits_pos": hits_pos, "hits_total": hits_total,
        })
    return accepted


# --------------------------------------------------------------------------
# Pattern-table builder — combine all five sources for the requested codes
# --------------------------------------------------------------------------
def build_patterns(codes, mining_df):
    """Combine all five sources into one patterns_df, restricted to ``codes``.

    Parameters
    ----------
    codes      : iterable of rechtsfeitcodes to build patterns for
    mining_df  : DataFrame (text + rechtsfeitcodes) the data-derived sources
                 (TF-IDF, chi2, valuelist validation) are mined from

    Returns (patterns_df, accepted_valuelist).
    """
    code_set = set(codes)
    seq: dict = defaultdict(int)
    rows: list[dict] = []

    # Source A — Excel curated phrases.
    for code, blob in EXCEL_CODES.items():
        if code not in code_set:
            continue
        for phrase in blob.get("declaratief", []):
            rgx = to_word_regex(phrase, fuzzy_gap=True)
            if rgx:
                rows.append(make_pattern_row(code, "excel", phrase, rgx, blob["name"], seq))
        for word in blob.get("synoniemen", []):
            rgx = to_word_regex(word, fuzzy_gap=False)
            if rgx:
                rows.append(make_pattern_row(code, "excel", word, rgx, blob["name"], seq))

    # Source B — Pothoven docx phrases.
    for code, blob in POTHOVEN_CODES.items():
        if code not in code_set:
            continue
        for phrase in blob.get("tekst", []):
            rgx = to_word_regex(phrase, fuzzy_gap=True)
            if rgx:
                rows.append(make_pattern_row(code, "pothoven", phrase, rgx, blob["name"], seq))

    # Sources C & D — TF-IDF and chi2 mined from the mining corpus.
    ngrams = derive_ngrams_per_code(mining_df, codes)
    chi2c = derive_chi2_per_code(mining_df, codes)
    all_terms = sorted({t for v in ngrams.values() for t, _ in v}
                       | {t for v in chi2c.values() for t, _ in v})
    has_hits: dict[str, bool] = {}
    if all_terms:
        cvz = CountVectorizer(
            vocabulary=all_terms, ngram_range=(1, 3), lowercase=True,
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ]{3,}\b",
        )
        Xb = cvz.fit_transform(mining_df["text"].fillna("").tolist())
        Xb.data[:] = 1
        hits = np.asarray(Xb.sum(axis=0)).ravel()
        has_hits = {t: int(hits[i]) > 0 for i, t in enumerate(all_terms)}
    for code in codes:
        chi2_kept: set[str] = set()
        for term, _ in chi2c.get(code, []):
            if has_hits.get(term):
                chi2_kept.add(term)
                rows.append(make_pattern_row(
                    code, "data-derived-chi2", term,
                    rf"(?i)\b{re.escape(term)}\b", None, seq))
        for term, _ in ngrams.get(code, []):
            if term in chi2_kept or not has_hits.get(term):
                continue
            rows.append(make_pattern_row(
                code, "data-derived-tfidf", term,
                rf"(?i)\b{re.escape(term)}\b", None, seq))

    # Source E (notebook) — original manual patterns, code-filtered.
    for entry in NOTEBOOK_MANUAL_PATTERNS:
        if entry["code"] not in code_set:
            continue
        if "regex" in entry:
            rgx, src = entry["regex"], entry.get("source_text", entry["regex"])
        else:
            rgx, src = to_word_regex(entry["phrase"], fuzzy_gap=True), entry["phrase"]
        if rgx:
            rows.append(make_pattern_row(
                entry["code"], "manual-notebook", src, rgx, entry["name"], seq,
                match_zone=entry.get("match_zone", "full")))

    # Source E (mined) — mined title-zone patterns.
    for entry in MINED_TITLE_PATTERNS:
        if entry["code"] not in code_set:
            continue
        rgx = to_word_regex(entry["phrase"], fuzzy_gap=True)
        if rgx:
            rows.append(make_pattern_row(
                entry["code"], "manual-mined", entry["phrase"], rgx,
                entry["name"], seq, match_zone="title"))

    # Source E (valuelist) — candidates validated at runtime against mining_df.
    accepted_vl = _validate_valuelist_candidates(mining_df, code_set)
    seen_phrases = {(p["code"], p["phrase"].lower()) for p in MINED_TITLE_PATTERNS}
    for a in accepted_vl:
        key = (a["code"], a["phrase"].lower())
        if key in seen_phrases:
            continue
        rgx = to_word_regex(a["phrase"], fuzzy_gap=True)
        if rgx:
            rows.append(make_pattern_row(
                a["code"], "manual-valuelist", a["phrase"], rgx, a["name"], seq,
                match_zone="title"))

    patterns_df = pd.DataFrame(rows)
    if len(patterns_df):
        patterns_df["match_zone"] = patterns_df.get("match_zone", "full")
        patterns_df["match_zone"] = patterns_df["match_zone"].fillna("full")
    return patterns_df, accepted_vl
