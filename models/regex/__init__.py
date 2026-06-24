"""
models/regex — L2 regex / classical classifier (hybrid).

The logic lives here so notebooks/regex.ipynb stays a thin runner.

Pipeline (see pipeline.run_hybrid)
----------------------------------
1. patterns.build_patterns(codes, mining_df)
       → one DataFrame combining five sources: Excel phrases, Pothoven docx
         triggers, TF-IDF + chi2 mined terms (Yang & Pedersen 1997), and
         hand-curated / mined / valuelist title patterns.
2. scoring.score_corpus(...)
       → precision-weighted (deed × code) score matrix (Aho-Corasick for simple
         terms, candidate-filtered regex for fuzzy phrases).
3. scoring.tune_thresholds / cv_predict
       → per-code F1-maximising cutoffs, tuned out-of-fold on the train split.
4. pipeline.run_hybrid
       → V1 (all codes, full train) + V2 (tail codes, overige bucket),
         champion-per-code gating, koop tie-break and mutex post-processing,
         and scores squashed into [0, 1] for the prediction contract.
"""

from models.regex.patterns import build_patterns, CUTOFF
from models.regex.scoring import (
    score_corpus,
    build_y_matrix,
    tune_thresholds,
    apply_thresholds,
    cv_predict,
    apply_koop_tiebreak,
    apply_mutex,
    per_code_f1,
    scores_to_unit_interval,
    strip_kadaster_prefix,
    title_zone,
)
from models.regex.pipeline import run_hybrid

__all__ = [
    "build_patterns",
    "CUTOFF",
    "score_corpus",
    "build_y_matrix",
    "tune_thresholds",
    "apply_thresholds",
    "cv_predict",
    "apply_koop_tiebreak",
    "apply_mutex",
    "per_code_f1",
    "scores_to_unit_interval",
    "strip_kadaster_prefix",
    "title_zone",
    "run_hybrid",
]
