# Kadaster Deed Classification — BSc Thesis

Automated multi-label classification of Dutch notarial deeds (`rechtsfeitcodes`) for the Dutch Land Registry (Kadaster). This project builds a three-layer framework that automates confident predictions, escalates uncertain ones to a human reviewer, and lets Kadaster tune the escalation policy without touching code.

---

## The problem

Kadaster is the Dutch Land Registry. Every notarial deed (mortgage, property transfer, easement, etc.) that is registered must be assigned one or more `rechtsfeitcodes` — legal classification codes that describe what legal act the deed records. This assignment is currently done manually by trained staff.

The goal of this project is to automate that classification. Each deed is a Dutch-language document, typically 500–8,000 tokens long. A single deed can be assigned multiple codes simultaneously (multi-label). There are 86 possible codes in the label space, but the distribution is highly skewed — a handful of codes (mortgage, transfer) account for the majority of deeds, while many codes appear fewer than 10 times in the test set.

**Dataset:**
- Train: 19,744 deeds (JSONL)
- Test: 1,920 deeds (JSONL, held-out)
- Format: one deed per line with fields `akteId`, `text`, `rechtsfeitcodes`
- Labels: list of integer codes, e.g. `[537, 606]`

The data is confidential and not included in this repo. See the Data section below.

---

## Three-layer framework

```
L1 — BERT
     Fine-tuned Dutch transformer. Splits long documents into overlapping
     chunks, encodes each chunk independently, pools chunk embeddings (mean),
     and outputs a per-class probability for all 86 codes. Threshold is
     optimised per class on the validation set. Multiple BERT variants were
     trained (see Results).

L2 — RegEx / Classical
     Interpretable, GPU-free rule engine. Uses Aho-Corasick pattern matching
     weighted by training precision. Two variants (V1: all codes on full train;
     V2: tail codes on the overige bucket) are blended with a champion-per-code
     selection. Post-processors handle near-identical code pairs (koop tie-break)
     and mutually exclusive codes (mutex). High-precision auto-accept layer.

L3 — Orchestration
     For each deed and each code, routes prediction to whichever method (BERT
     or RegEx) achieved higher per-class F1 on the validation set. A
     confidence escalation layer flags deeds where neither method is confident
     (1st-vs-2nd score margin below a tunable threshold) and marks them for
     human review. The full policy is written to policy.yaml — Kadaster can
     adjust thresholds without touching code.
```

---

## Key design decisions

**Frozen split** — the train/val split (90/10) is run once and the indices committed to `artifacts/splits/`. Every method trains and evaluates on identical documents. Re-splitting is blocked by a `FileExistsError` safeguard to prevent accidental invalidation of results.

**Frozen label space** — the ordered list of 86 rechtsfeitcodes is frozen at the same time as the split (`artifacts/splits/label_space.json`). Every method uses the same `MultiLabelBinarizer` with identical class ordering, so all binary prediction matrices are directly comparable.

**One harness** — all methods compute metrics through `shared/harness.py`. micro-F1 and macro-F1 are guaranteed to be computed identically across all methods. micro-F1 weights every deed equally; macro-F1 weights every code equally and is more sensitive to rare-code performance.

**Prediction contract** — each method writes a long-format parquet (`artifacts/predictions/{method}.parquet`) with one row per (deed, code) pair, carrying the full confidence score (not just predicted=True rows). This allows the orchestration layer to compute per-deed score margins and detect disagreements between methods.

**Per-class threshold optimisation** — after training, the classification threshold is swept per class on the validation set (not 0.5 flat) to maximise per-class F1. This consistently improves macro-F1 on rare classes.

**Temperature calibration** — after threshold optimisation, a single temperature scalar is fitted on validation probabilities. Overconfident models (T > 1) have scores pulled towards 0.5; underconfident models (T < 1) have scores pushed away from 0.5. Calibrated scores are written back to the parquet so the orchestration layer can use them for margin computations.

---

## Results

All numbers are on the held-out 1,920-deed test set. Reproduce by running the notebooks in order (see How to run).

### BERT models (tuned thresholds)

| Model | micro-F1 | macro-F1 | Notes |
|---|---|---|---|
| RobBERT (chunk, 3×510) | 0.820 | 0.552 | Dutch RobBERT, mean pooling, BCE loss |
| RobBERT front-512 | 0.822 | 0.568 | First 512 tokens only, no chunking |
| BERTje (chunk, 3×510) | 0.896 | 0.603 | Dutch BERTje, mean pooling, BCE loss |
| RobBERT + focal loss | 0.886 | 0.585 | Focal loss (γ=2) for class imbalance |
| **EuroBERT-210m** | **0.914** | **0.621** | Long-context (8190 tok/chunk), focal loss |

EuroBERT is the best single model. It uses a 210M-parameter multilingual transformer with a context window of 8,192 tokens, allowing entire deeds to be encoded in fewer chunks.

### RegEx / Classical

| Metric | Value |
|---|---|
| micro-F1 | 0.842 |
| macro-F1 | 0.472 |

RegEx excels on frequent, well-defined deed types (e.g. Hypotheek: F1=0.99, Overdracht: F1=0.96) but struggles on the rare long tail. 18 out of 86 codes are present in the test set but score F1=0 with regex — these require the learned model.

### Calibration

All BERT models were overconfident (temperature T > 1) except focal-loss variants which were underconfident (T < 1). Calibrated scores are used downstream in orchestration for reliable margin computation.

| Model | Temperature | Pre-cal MACE | Post-cal MACE |
|---|---|---|---|
| RobBERT | 2.4 | — | 0.210 |
| front512 | 3.7 | — | 0.220 |
| BERTje | 3.5 | — | 0.162 |
| focal | 0.6 | — | 0.035 |
| EuroBERT | 0.4 | — | 0.024 |

Full per-class breakdowns and training curves are in `results/SUMMARY.md` and `results/plots/`.

The fine-tuned EuroBERT model (best result) is publicly available on HuggingFace Hub: **[ay44n04712/kadaster-eurobert](https://huggingface.co/ay44n04712/kadaster-eurobert)**. Use `notebooks/predict.ipynb` to generate predictions from it without retraining.

---

## Repo structure

```
kadaster-deed-classification/
├── shared/                       # shared library — imported by all notebooks
│   ├── config.py                 # all paths, device fallback, BERTConfig dataclass
│   ├── data.py                   # canonical JSONL loader → (akteId, text, codes)
│   ├── labels.py                 # frozen label space loader (get_mlb)
│   ├── splits.py                 # frozen train/val split loader (load_split)
│   ├── harness.py                # the one scorer — evaluate(), per_class_matrix()
│   └── io.py                     # prediction contract read/write (parquet)
├── models/
│   └── regex/                    # L2 regex classifier
│       ├── pattern_sources.py    # all pattern definitions (Excel, mined, curated)
│       ├── patterns.py           # builds the combined pattern table
│       ├── scoring.py            # Aho-Corasick scoring + threshold tuning
│       └── pipeline.py           # V1+V2 hybrid, champion selection, post-processing
├── notebooks/
│   ├── 00_splits.ipynb           # run once — freezes split and label space
│   ├── bert.ipynb                # BERT fine-tuning — runs all configs in sweeps/
│   ├── predict.ipynb             # inference only — loads EuroBERT from HuggingFace Hub
│   ├── calibration.ipynb         # temperature scaling on val predictions
│   ├── regex.ipynb               # RegEx / classical classifier
│   └── orchestration.ipynb       # per-class routing, confidence escalation, policy output
├── sweeps/                       # one JSON config per BERT model variant
│   ├── robbert.json              # RobBERT baseline (also the default config)
│   ├── bertje.json               # BERTje variant
│   ├── focal.json                # RobBERT with focal loss
│   ├── front512.json             # RobBERT front-512 truncation baseline
│   └── eurobert.json             # EuroBERT-210m (long-context)
├── artifacts/
│   └── splits/                   # committed — frozen split indices + label space
│       ├── label_space.json      # ordered list of 86 rechtsfeitcodes
│       ├── train_idx.npy         # indices into the train JSONL for training
│       └── val_idx.npy           # indices into the train JSONL for validation
├── results/                      # committed — metrics, plots, and summary
│   ├── SUMMARY.md                # full training curves and per-class F1 for all runs
│   ├── sweep_results.csv         # side-by-side comparison of all BERT sweeps
│   ├── {model}_metrics.json      # micro/macro F1 per model
│   └── plots/                    # training curves and reliability diagrams
├── data/                         # NOT committed — confidential Kadaster data
├── requirements.txt
├── install.sh                    # Unix/Mac install script
├── install.bat                   # Windows install script
└── .gitignore
```

---

## Data

The data files are confidential and not included in this repo. Contact the Kadaster project supervisor to obtain access. Place the following files in the `data/` folder before running:

```
data/ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl   (train — 19,744 deeds)
data/ai-challenge-data_ai_challenge_data_anonymized_1920.jsonl    (test  —  1,920 deeds)
```

Each line is a JSON object with three fields:
```json
{"akteId": "deed_001", "text": "Dutch deed text...", "rechtsfeitcodes": [537, 606]}
```

---

## Setup

### Requirements

- Python 3.14
- PyTorch (installed separately via install script — platform-specific wheels)
- A HuggingFace account with a read token (free) to download the BERT models

### Unix / Mac

```bash
bash install.sh
```

### Windows

```bat
install.bat
```

### Manual

```bash
# Install PyTorch first (CPU/MPS for Mac, CUDA for Linux/Windows)
pip install torch

# Then install everything else
pip install -r requirements.txt
```

### HuggingFace token

Create a `.env` file in the repo root:
```
HF_TOKEN=hf_your_token_here
```
Get a free read token at huggingface.co → Settings → Access Tokens.

---

## How to run

Run the notebooks in order. Each notebook is self-contained and imports everything from `shared/`.

| Step | Notebook | When to run |
|---|---|---|
| 1 | `notebooks/00_splits.ipynb` | **Once** — freezes the train/val split and label space |
| 2 | `notebooks/bert.ipynb` | Every time — trains all BERT configs in `sweeps/` |
| 2b | `notebooks/predict.ipynb` | **Alternative to step 2** — loads fine-tuned EuroBERT from HuggingFace Hub, skips retraining |
| 3 | `notebooks/calibration.ipynb` | After step 2 — temperature-scales a saved model's predictions |
| 4 | `notebooks/regex.ipynb` | Every time — runs the RegEx classifier |
| 5 | `notebooks/orchestration.ipynb` | After steps 2–4 — per-class routing, escalation, policy output |

### Step 1 — Freeze the split (run once)

`00_splits.ipynb` loads the training JSONL, stratifies a 90/10 train/val split preserving multi-label co-occurrence, and saves the indices and label space to `artifacts/splits/`. It will raise `FileExistsError` if the split already exists — this is intentional. Only delete `artifacts/splits/` and re-run if you explicitly want to re-split (this invalidates all previous results).

### Step 2b — Predict without retraining (alternative to step 2)

`predict.ipynb` downloads the fine-tuned EuroBERT model and its per-class thresholds directly from HuggingFace Hub (`ay44n04712/kadaster-eurobert`) and runs inference on any JSONL input. This skips the multi-hour training step and produces the same `artifacts/predictions/eurobert.parquet` that orchestration expects. Set `INPUT_PATH` at the top of the notebook to point to any JSONL file with the same three-field schema.

### Step 2 — Train BERT models

`bert.ipynb` loops over every JSON config in `sweeps/` and trains a separate model for each. For each config it:
1. Loads the frozen split
2. Pre-tokenizes documents into overlapping chunks
3. Fine-tunes the transformer end-to-end with a classification head
4. Optimises per-class thresholds on the validation set
5. Evaluates on the test set via the shared harness
6. Writes `artifacts/predictions/{model}.parquet` and `artifacts/predictions/{model}_val.parquet`
7. Saves `artifacts/models/{model}/thresholds.json`
8. Appends a section to `results/SUMMARY.md`

Control variables at the top of the notebook:
```python
SWEEP_MODE = True      # True = all sweeps/; False = only sweeps/robbert.json
RUN_ONLY   = None      # e.g. ["eurobert.json"] to run one config; None = all
```

**Training time:** EuroBERT takes ~4–8 hours on a GPU. RobBERT variants take ~2–4 hours each. Set `sample_size` in a sweep config to a small integer (e.g. 500) for a fast test run.

### Step 3 — Calibrate (optional)

`calibration.ipynb` fits a temperature scalar on the validation predictions of a single saved model and writes calibrated scores back to the parquet. Set `RUN_NAME` at the top of the notebook to the model you want to calibrate (e.g. `"eurobert"`).

### Step 4 — RegEx classifier

`regex.ipynb` trains and evaluates the hybrid regex pipeline on the same frozen split. Writes `artifacts/predictions/regex.parquet` and `artifacts/predictions/regex_val.parquet`.

### Step 5 — Orchestration

`orchestration.ipynb` reads the prediction parquets from both BERT and RegEx, builds a per-class routing map (fitted on val), sweeps the confidence escalation threshold, exports disagreement instances to `results/disagreement_instances.xlsx`, and writes `models/orchestration/policy.yaml`. The policy file is the single artefact Kadaster uses to tune the system.

---

## Changing hyperparameters

Each BERT model variant has its own JSON config in `sweeps/`. Edit the relevant file and re-run `bert.ipynb` — no code changes needed.

| Model | Config |
|---|---|
| RobBERT (default) | `sweeps/robbert.json` |
| BERTje | `sweeps/bertje.json` |
| RobBERT + focal loss | `sweeps/focal.json` |
| RobBERT front-512 | `sweeps/front512.json` |
| EuroBERT-210m | `sweeps/eurobert.json` |

Key parameters in each config:

| Parameter | Description |
|---|---|
| `model_name` | HuggingFace model ID |
| `pooling` | How to aggregate chunk embeddings: `"mean"` / `"cls"` / `"max"` |
| `content_size` | Tokens per chunk (excluding CLS/SEP) |
| `stride` | Overlap between consecutive chunks |
| `max_chunks` | Maximum number of chunks per document |
| `epochs` | Training epochs |
| `lr` | Learning rate |
| `batch_size` / `grad_accum` | Effective batch = batch_size × grad_accum |
| `loss.type` | `"bce"` (binary cross-entropy) or `"focal"` (focal loss for class imbalance) |
| `loss.gamma` | Focal loss γ parameter (higher = more focus on hard examples) |
| `gradient_checkpointing` | `true` to reduce memory at the cost of ~20% slower training |
| `freeze_layers` | Number of transformer layers to freeze (EuroBERT only) |
| `sample_size` | `null` = full dataset; integer = fast test run on N documents |

---

## Adapting to new data

To run on a different dataset:

1. Place new JSONL files in `data/` with the same three-field schema (`akteId`, `text`, `rechtsfeitcodes`)
2. Update `TRAIN_PATH` and `TEST_PATH` in `shared/config.py` to point to the new filenames
3. Delete `artifacts/splits/` (to force a new split and label space)
4. Re-run all notebooks from step 1

If the new data uses different field names, update line 27 of `shared/data.py`:
```python
df = pd.DataFrame(records)[["akteId", "text", "rechtsfeitcodes"]]
```
Rename the three columns to match your schema — everything downstream adapts automatically.
