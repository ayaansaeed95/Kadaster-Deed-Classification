
## RobBERT — DTAI-KULeuven/robbert-2023-dutch-base

**Config**
| param | value |
|---|---|
| model | DTAI-KULeuven/robbert-2023-dutch-base |
| pooling | mean |
| epochs | 15 |
| lr | 2e-05 |
| weight_decay | 0.01 |
| effective_batch | 64 |
| loss | bce |
| max_chunks | 8 |
| content_size | 510 |
| stride | 460 |

**Training curve**
| Epoch | Train Loss | Val Loss | Val micro-F1 | Val macro-F1 |
|---|---|---|---|---|
| 1 | nan | 0.545 | 0.3083 | 0.1208 |
| 2 | nan | 0.301 | 0.4151 | 0.2448 |
| 3 | nan | 0.226 | 0.5586 | 0.3388 |
| 4 | nan | 0.194 | 0.6169 | 0.3953 |
| 5 | nan | 0.175 | 0.6134 | 0.3899 |
| 6 | nan | 0.169 | 0.6429 | 0.3984 |
| 7 | nan | 0.163 | 0.7005 | 0.4623 |
| 8 | nan | 0.157 | 0.6482 | 0.4323 |
| 9 | nan | 0.153 | 0.7138 | 0.4733 |
| 10 | nan | 0.159 | 0.7386 | 0.5169 |
| 11 | nan | 0.153 | 0.7318 | 0.4948 |
| 12 | nan | 0.159 | 0.7468 | 0.5236 |
| 13 | nan | 0.161 | 0.7620 | 0.5334 |
| 14 | nan | 0.160 | 0.7660 | 0.5379 |
| 15 | nan | 0.160 | 0.7592 | 0.5380 |

**Test results**
| threshold | micro-F1 | macro-F1 |
|---|---|---|
| tuned | 0.8204 | 0.5519 |
| flat 0.5 | 0.7688 | 0.5239 |

**Bottom 10 classes by F1 (support ≥ 1)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    588 | 0      |         1 |
|    604 | 0      |         2 |
|    674 | 0      |         1 |
|    662 | 0      |         1 |
|    619 | 0      |         1 |
|    622 | 0      |         1 |
|    633 | 0      |         6 |
|    611 | 0      |         1 |
|    617 | 0.125  |         1 |
|    601 | 0.1333 |         2 |

**Top 10 classes by F1 (support ≥ 15)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    545 | 0.737  |       229 |
|    671 | 0.7451 |        23 |
|    538 | 0.8356 |        64 |
|    564 | 0.85   |        97 |
|    516 | 0.9278 |        46 |
|    518 | 0.9583 |        25 |
|    606 | 0.977  |       829 |
|    527 | 0.9817 |       107 |
|    585 | 0.9909 |       163 |
|    537 | 0.9912 |       454 |
---

## Calibration — robbert

**Method:** temperature scaling on validation set (pooled across all deed×code pairs).

**Optimal temperature:** T = 2.4000, MACE = 0.2103

Model was overconfident (T > 1) — scores pulled towards 0.5.

See `results/plots/robbert_reliability_before.png` and `_after.png` for the reliability diagrams.

Calibrated scores have been written to `artifacts/predictions/robbert.parquet`.

---
## front512 — DTAI-KULeuven/robbert-2023-dutch-base

**Config:** pooling=mean | loss=bce | max_chunks=1 | epochs=20 | lr=2e-05 | batch=64

**Training time:** 213.1 min

**Training curve**
| Epoch | Train Loss | Val Loss | Val micro-F1 | Val macro-F1 |
|---|---|---|---|---|
| 1 | nan | 0.611 | 0.2272 | 0.0984 |
| 2 | nan | 0.350 | 0.4056 | 0.2288 |
| 3 | nan | 0.247 | 0.4846 | 0.2558 |
| 4 | nan | 0.206 | 0.5605 | 0.3324 |
| 5 | nan | 0.186 | 0.6112 | 0.3620 |
| 6 | nan | 0.180 | 0.6666 | 0.4472 |
| 7 | nan | 0.162 | 0.6785 | 0.4529 |
| 8 | nan | 0.161 | 0.6579 | 0.4280 |
| 9 | nan | 0.201 | 0.7182 | 0.5134 |
| 10 | nan | 0.205 | 0.7205 | 0.5169 |
| 11 | nan | 0.200 | 0.7284 | 0.4967 |
| 12 | nan | 0.222 | 0.7584 | 0.5430 |
| 13 | nan | 0.212 | 0.7414 | 0.5345 |
| 14 | nan | 0.214 | 0.7560 | 0.5387 |
| 15 | nan | 0.237 | 0.7781 | 0.5689 |
| 16 | nan | 0.238 | 0.7834 | 0.5623 |
| 17 | nan | 0.246 | 0.7782 | 0.5696 |
| 18 | nan | 0.263 | 0.8001 | 0.5781 |
| 19 | nan | 0.261 | 0.8066 | 0.5700 |
| 20 | nan | 0.261 | 0.7996 | 0.5707 |

**Test results**
| threshold | micro-F1 | macro-F1 |
|---|---|---|
| tuned | 0.8225 | 0.5680 |
| flat 0.5 | 0.8138 | 0.5676 |

**Bottom 10 classes by F1 (support ≥ 1)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    509 | 0      |         3 |
|    604 | 0      |         2 |
|    588 | 0      |         1 |
|    662 | 0      |         1 |
|    619 | 0      |         1 |
|    622 | 0      |         1 |
|    633 | 0      |         6 |
|    611 | 0      |         1 |
|    674 | 0      |         1 |
|    642 | 0.0455 |        12 |

**Top 10 classes by F1 (support ≥ 15)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    545 | 0.7828 |       229 |
|    516 | 0.8485 |        46 |
|    518 | 0.88   |        25 |
|    564 | 0.9231 |        97 |
|    581 | 0.9254 |        33 |
|    538 | 0.9302 |        64 |
|    527 | 0.963  |       107 |
|    606 | 0.9716 |       829 |
|    585 | 0.9908 |       163 |
|    537 | 0.9923 |       454 |
---

## Calibration — front512

**Method:** temperature scaling on validation set (pooled across all deed×code pairs).

**Optimal temperature:** T = 3.7000, MACE = 0.2204

Model was overconfident (T > 1) — scores pulled towards 0.5.

See `results/plots/front512_reliability_before.png` and `_after.png` for the reliability diagrams.

Calibrated scores have been written to `artifacts/predictions/front512.parquet`.

---
## bertje — GroNLP/bert-base-dutch-cased

**Config:** pooling=mean | loss=bce | max_chunks=3 | epochs=20 | lr=3e-05 | batch=32

**Training time:** 625.1 min

**Training curve**
| Epoch | Train Loss | Val Loss | Val micro-F1 | Val macro-F1 |
|---|---|---|---|---|
| 1 | nan | 0.418 | 0.3434 | 0.1505 |
| 2 | nan | 0.213 | 0.4172 | 0.2577 |
| 3 | nan | 0.156 | 0.6417 | 0.3777 |
| 4 | nan | 0.151 | 0.6722 | 0.4607 |
| 5 | nan | 0.148 | 0.6738 | 0.4382 |
| 6 | nan | 0.171 | 0.7747 | 0.5328 |
| 7 | nan | 0.159 | 0.7779 | 0.5398 |
| 8 | nan | 0.194 | 0.8076 | 0.5690 |
| 9 | nan | 0.203 | 0.8341 | 0.5873 |
| 10 | nan | 0.218 | 0.8209 | 0.6063 |
| 11 | nan | 0.253 | 0.8529 | 0.6239 |
| 12 | nan | 0.258 | 0.8558 | 0.6343 |
| 13 | nan | 0.236 | 0.8533 | 0.6273 |
| 14 | nan | 0.296 | 0.8756 | 0.6375 |
| 15 | nan | 0.290 | 0.8749 | 0.6596 |
| 16 | nan | 0.313 | 0.8856 | 0.6641 |
| 17 | nan | 0.321 | 0.8875 | 0.6389 |
| 18 | nan | 0.328 | 0.8841 | 0.6561 |
| 19 | nan | 0.339 | 0.8877 | 0.6697 |
| 20 | nan | 0.342 | 0.8879 | 0.6617 |

**Test results**
| threshold | micro-F1 | macro-F1 |
|---|---|---|
| tuned | 0.8955 | 0.6030 |
| flat 0.5 | 0.8968 | 0.6061 |

**Bottom 10 classes by F1 (support ≥ 1)**
|   code |   f1 |   support |
|-------:|-----:|----------:|
|    604 |    0 |         2 |
|    588 |    0 |         1 |
|    597 |    0 |         1 |
|    662 |    0 |         1 |
|    619 |    0 |         1 |
|    622 |    0 |         1 |
|    633 |    0 |         6 |
|    611 |    0 |         1 |
|    674 |    0 |         1 |
|    656 |    0 |         2 |

**Top 10 classes by F1 (support ≥ 15)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    543 | 0.8511 |        23 |
|    516 | 0.8687 |        46 |
|    564 | 0.9128 |        97 |
|    538 | 0.918  |        64 |
|    518 | 0.92   |        25 |
|    581 | 0.9275 |        33 |
|    527 | 0.9722 |       107 |
|    606 | 0.9788 |       829 |
|    585 | 0.9848 |       163 |
|    537 | 0.9891 |       454 |
---

## Calibration — bertje

**Method:** temperature scaling on validation set (pooled across all deed×code pairs).

**Optimal temperature:** T = 3.5000, MACE = 0.1621

Model was overconfident (T > 1) — scores pulled towards 0.5.

See `results/plots/bertje_reliability_before.png` and `_after.png` for the reliability diagrams.

Calibrated scores have been written to `artifacts/predictions/bertje.parquet`.

---
## focal — DTAI-KULeuven/robbert-2023-dutch-base

**Config:** pooling=mean | loss=focal | max_chunks=3 | epochs=20 | lr=2e-05 | batch=32

**Training time:** 537.9 min

**Training curve**
| Epoch | Train Loss | Val Loss | Val micro-F1 | Val macro-F1 |
|---|---|---|---|---|
| 1 | nan | 0.011 | 0.7242 | 0.0525 |
| 2 | nan | 0.005 | 0.8176 | 0.1676 |
| 3 | nan | 0.004 | 0.8700 | 0.3340 |
| 4 | nan | 0.003 | 0.8781 | 0.3857 |
| 5 | nan | 0.003 | 0.8809 | 0.4257 |
| 6 | nan | 0.003 | 0.8913 | 0.4878 |
| 7 | nan | 0.003 | 0.8919 | 0.5217 |
| 8 | nan | 0.003 | 0.8929 | 0.5262 |
| 9 | nan | 0.003 | 0.8860 | 0.5218 |
| 10 | nan | 0.003 | 0.8924 | 0.5669 |
| 11 | nan | 0.003 | 0.9018 | 0.5715 |
| 12 | nan | 0.003 | 0.8956 | 0.5713 |
| 13 | nan | 0.003 | 0.8932 | 0.5774 |
| 14 | nan | 0.003 | 0.9055 | 0.5964 |
| 15 | nan | 0.003 | 0.9046 | 0.6078 |
| 16 | nan | 0.003 | 0.9048 | 0.5939 |
| 17 | nan | 0.003 | 0.9021 | 0.6039 |

**Test results**
| threshold | micro-F1 | macro-F1 |
|---|---|---|
| tuned | 0.8862 | 0.5849 |
| flat 0.5 | 0.9117 | 0.5756 |

**Bottom 10 classes by F1 (support ≥ 1)**
|   code |   f1 |   support |
|-------:|-----:|----------:|
|    547 |    0 |         2 |
|    597 |    0 |         1 |
|    604 |    0 |         2 |
|    601 |    0 |         2 |
|    588 |    0 |         1 |
|    617 |    0 |         1 |
|    614 |    0 |         2 |
|    611 |    0 |         1 |
|    619 |    0 |         1 |
|    662 |    0 |         1 |

**Top 10 classes by F1 (support ≥ 15)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    671 | 0.8372 |        23 |
|    516 | 0.8866 |        46 |
|    564 | 0.91   |        97 |
|    518 | 0.9167 |        25 |
|    538 | 0.9206 |        64 |
|    581 | 0.9552 |        33 |
|    527 | 0.9811 |       107 |
|    606 | 0.9813 |       829 |
|    585 | 0.9877 |       163 |
|    537 | 0.9923 |       454 |
---

## Calibration — focal

**Method:** temperature scaling on validation set (pooled across all deed×code pairs).

**Optimal temperature:** T = 0.6000, MACE = 0.0354

Model was underconfident (T < 1) — scores pushed away from 0.5.

See `results/plots/focal_reliability_before.png` and `_after.png` for the reliability diagrams.

Calibrated scores have been written to `artifacts/predictions/focal.parquet`.

---
## EuroBERT /EuroBERT-210m

**Config**
| param | value |
|---|---|
| model | EuroBERT/EuroBERT-210m |
| pooling | mean |
| epochs | 4 |
| lr | 2e-05 |
| weight_decay | 0.01 |
| effective_batch | 128 |
| sample_size | None |
| loss | focal |
| max_chunks | 3 |
| content_size | 8190 |
| stride | 4096 |

**Training curve**
| Epoch | Train Loss | Val Loss | Val micro-F1 | Val macro-F1 |
|---|---|---|---|---|
| 1 | nan | 0.002 | 0.8334 | 0.2788 |
| 2 | nan | 0.002 | 0.8721 | 0.4716 |
| 3 | nan | 0.001 | 0.9048 | 0.5894 |
| 4 | nan | 0.001 | 0.9251 | 0.6287 |

**Test results**
| threshold | micro-F1 | macro-F1 |
|---|---|---|
| tuned | 0.9144 | 0.6205 |
| flat 0.5 | 0.9221 | 0.6129 |

**Bottom 10 classes by F1 (support ≥ 1)**
|   code |   f1 |   support |
|-------:|-----:|----------:|
|    509 |    0 |         3 |
|    547 |    0 |         2 |
|    598 |    0 |         2 |
|    588 |    0 |         1 |
|    604 |    0 |         2 |
|    597 |    0 |         1 |
|    601 |    0 |         2 |
|    611 |    0 |         1 |
|    619 |    0 |         1 |
|    622 |    0 |         1 |

**Top 10 classes by F1 (support ≥ 15)**
|   code |     f1 |   support |
|-------:|-------:|----------:|
|    543 | 0.8846 |        23 |
|    538 | 0.9016 |        64 |
|    516 | 0.9348 |        46 |
|    581 | 0.9538 |        33 |
|    518 | 0.9583 |        25 |
|    527 | 0.968  |       107 |
|    564 | 0.9688 |        97 |
|    606 | 0.9782 |       829 |
|    585 | 0.9785 |       163 |
|    537 | 0.9933 |       454 |
---

## Calibration — eurobert

**Method:** temperature scaling on validation set (pooled across all deed×code pairs).

**Optimal temperature:** T = 0.4000, MACE = 0.0237

Model was underconfident (T < 1) — scores pushed away from 0.5.

See `results/plots/eurobert_reliability_before.png` and `_after.png` for the reliability diagrams.

Calibrated scores have been written to `artifacts/predictions/eurobert.parquet`.

---
## RegEx (hybrid)

The regex layer is a **hybrid**: V1 (all-code patterns, full train) + V2 (tail
patterns mined on the *overige* bucket), with a **champion-per-code** choice by
out-of-fold F1 and koop/mutex post-processing. See `notebooks/regex.ipynb`
and `models/regex/`.

| Metric | Value |
|--------|-------|
| **micro-F1** | **0.842** |
| **macro-F1** | **0.472** |
| Codes present in test (support > 0) | 56 / 86 |
| Codes ever predicted by regex | 44 / 86 |
| Test deeds with ≥1 predicted code | 1,842 / 1,920 (95.9%) |
| — of present codes the regex scores F1 > 0 | 38 |
| — present codes fully missed (F1 = 0) | 18 |

## The one-line story

Regex is **strong on the frequent, well-defined deed types and weak on the rare long tail**. That is exactly why micro-F1 (0.842, weights every deed equally) is far above macro-F1 (0.472, weights every code equally): a handful of high-volume codes are nearly perfect, while many rare codes get no usable pattern and score 0. The hybrid's V2 tail variant lifts several rare codes over a single full-train regex, but the deep tail still needs the learned model — which is exactly what the orchestration layer is for.

## Where regex works well (high support, high F1)

| Code | Rechtsfeit | Support | P | R | F1 |
|------|-----------|--------:|----:|----:|----:|
| 537 | Hypotheek | 454 | 0.99 | 0.99 | **0.99** |
| 527 | Erfpachtcanon (wijziging) | 107 | 0.98 | 0.95 | **0.97** |
| 606 | Overdracht | 829 | 0.97 | 0.95 | **0.96** |
| 516 | Beperkt recht (wijzigen) | 46 | 0.95 | 0.91 | **0.93** |
| 538 | Hypotheek (doorhaling) | 64 | 0.93 | 0.80 | 0.86 |
| 585 | Verklaring van erfrecht | 163 | 0.83 | 0.90 | 0.86 |
| 564 | Verbetering | 97 | 0.82 | 0.76 | 0.79 |

These are deeds with a clear, standardised title phrase — perfect for high-precision rules.

## Where regex struggles (decent support, low F1)

| Code | Rechtsfeit | Support | P | R | F1 |
|------|-----------|--------:|----:|----:|----:|
| 545 | Kwalitatieve verplichting | 229 | 0.59 | 0.71 | 0.64 |
| 532 | Aanvullende akte | 61 | 0.60 | 0.54 | 0.57 |
| 572 | Erfdienstbaarheden | 104 | 0.43 | 0.47 | 0.45 |
| 580 | Verdeling gemeenschap (gezamenlijk) | 33 | 0.78 | 0.55 | 0.64 |
| 581 | Verdeling gemeenschap (huwelijk) | 33 | 0.88 | 0.70 | 0.78 |
| 518 | Beslag (doorhaling) | 25 | 0.83 | 0.60 | 0.70 |

These either lack a unique title phrase (the rule fires on too many deeds → low precision) or hide inside the body text (the rule misses them → low recall).

## Takeaways for the meeting

- **Regex is a fast, transparent, high-precision baseline for the top codes** — for 537, 527, 606, 516 it is essentially solved (F1 ≥ 0.93).
- **42 of 86 codes are never predicted** and 18 more score 0 despite appearing in the test set: the rare tail needs the learned model, not rules.
- **micro vs macro is the key contrast to present:** 0.842 vs 0.472 quantifies the head-vs-tail gap.
- Numbers are on the **held-out 1,920-deed test set**; reproduce by running `notebooks/regex.ipynb` (`SAMPLE_SIZE = None`).
---
