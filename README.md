# RF-Based Wi-Fi Evil Twin Detection

Source-code submission for **Lightweight Random Forest-Based Detection of Wi-Fi Evil Twin Attacks Using Beacon-Frame Fingerprints**.

The submitted detector is RF-only. It parses IEEE 802.11 Beacon frames, groups observations by claimed BSSID, builds causal non-overlapping 20-Beacon windows (W20), transforms the fixed feature schema, and applies the frozen Random Forest at attack threshold `0.30`. Ground-truth labels are never used during prediction.

## Reported scope

| Dataset | Role | Result |
|---|---|---|
| AWID3 Evil Twin | chronological within-capture temporal test | TN 86, FP 0, FN 0, TP 72, F1 1.000 |
| D05-A-R2 | previously inspected cross-dataset evaluation | TN 489, FP 0, FN 72, TP 289, F1 0.889 |

AWID3 is split by capture time: 60% train, 20% validation, and 20% internal test. Boundary-crossing windows are dropped. D05-A-R2 is not used for fitting, preprocessing, or threshold selection, but it was inspected during the wider project and must not be described as an untouched independent test.

## Repository contents

- `src/evil_twin/`: Beacon parsing, W20 feature construction, preprocessing, training, evaluation, and inference.
- `models/clean_rebuild_v1/`: frozen RF artifact and metadata.
- `scripts/reproduce_paper_metrics.py`: recomputes AWID3 and D05-A-R2 results from the included W20 tables.
- `scripts/run_clean_rebuild_v1.py`: clean development/freeze workflow when the original PCAPs are available.
- `scripts/evaluate_knn_lstm.py`: KNN/LSTM baseline experiment.
- `scripts/build_rf_research_package.py`: publication table and figure generator.
- `results/`: paper-facing metrics, figures, manifests, and pre-extracted AWID3/D05-A-R2 feature tables.
- `paper/`: LaTeX source and Figure 1.
- `tests/`: focused parser, window, leakage, and ML regression tests.

Raw PCAPs are intentionally not committed because of size and dataset redistribution/provenance constraints. Pre-extracted W20 tables required to verify the reported metrics are included. Download AWID3 from its official source for full PCAP-to-result reproduction. D05-A-R2 is a controlled local capture and is not represented as a public benchmark.

## Installation

The frozen artifact was produced and verified with Python 3.14.7 and scikit-learn 1.9.1. Using the locked environment is recommended because `joblib` model artifacts are version-sensitive.

```bash
git clone https://github.com/Tomie1111/CybercrimeProj.git
cd CybercrimeProj
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

Do not load an untrusted `joblib` file. The included model is checked by SHA-256 in the reproduction script.

## Verify the frozen model

```bash
evil-twin-rf verify-model
```

Expected SHA-256:

```text
8a1e1717f77ded865336c7d4d9c3db81559bb1300cb979e73c803f02f26ed145
```

## Reproduce the paper metrics

This performs actual inference with the frozen pipeline; the reported predictions are not hard-coded.

```bash
python scripts/reproduce_paper_metrics.py \
  --output outputs/reproduced_metrics.json
```

Expected key results:

- AWID3 test: `TN=86, FP=0, FN=0, TP=72, F1=1.0`
- D05-A-R2: `TN=489, FP=0, FN=72, TP=289, F1≈0.8892`

## Run on a PCAP

The capture must contain monitor-mode IEEE 802.11 frames with radiotap headers.

```bash
evil-twin-rf predict path/to/capture.pcap \
  --model models/clean_rebuild_v1/selected_model.joblib \
  --output outputs/predictions.csv
```

Output is one row per complete W20 and includes `attack_probability` and `prediction` (`1` means Evil Twin suspected). This command is offline PCAP post-processing, not a claim of live streaming deployment.

Feature extraction without prediction:

```bash
evil-twin-rf extract path/to/capture.pcap \
  --size 20 --stride 20 \
  --output outputs/windows.csv
```

## Evaluate a labeled W20 table

```bash
evil-twin-rf evaluate \
  --data results/awid3_evil_twin_test_w20.csv \
  --output outputs/awid3_evaluation.json
```

## Training and split protocol

Model selection uses only the chronological AWID3 train and validation tables:

```bash
evil-twin-rf train \
  --train results/awid3_evil_twin_train_w20.csv \
  --validation results/awid3_evil_twin_validation_w20.csv \
  --models outputs/models \
  --output outputs/model_comparison.csv
```

This creates a new development artifact. It does not overwrite the submitted frozen model unless the user explicitly replaces it. The complete original protocol is recorded in `experiments/clean_rebuild_v1/` and `results/split_manifest.json`.

## Tests

Core tests do not require the raw PCAP datasets:

```bash
pytest tests/test_parsers.py tests/test_windows.py tests/test_integrity.py
```

The optional LSTM baseline test requires PyTorch:

```bash
python -m pip install -e '.[baselines]'
pytest tests/test_ml_sequence.py
```

## Reproducibility notes

- Prediction unit: causal non-overlapping W20 grouped by claimed BSSID.
- Frozen RF: 100 trees, maximum depth 6, minimum leaf size 1.
- Threshold: 0.30.
- Seed: 20261116.
- Forbidden predictors include BSSID, SSID, labels, source file, absolute window bounds, frame membership, membership hash, and label purity.
- AWID3 test is within the same capture/session and is not an independent-device test.
- D05-A-R2 uses operator-confirmed AP roles and excludes unrelated ambient APs from labeled metrics.
- The RF output is statistical evidence, not cryptographic proof of physical transmitter identity.

## Ethical use

Use only on networks and captures that you own or are authorized to inspect. The detector is passive and the submitted RF path does not transmit deauthentication or association frames.
