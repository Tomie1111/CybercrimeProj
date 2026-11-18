#!/usr/bin/env python3
"""Recompute the paper's RF metrics from included pre-extracted W20 tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from evil_twin.ml.evaluate import evaluate_predictions


EXPECTED_MODEL_SHA256 = "8a1e1717f77ded865336c7d4d9c3db81559bb1300cb979e73c803f02f26ed145"
ZYXEL_ATTACK_BSSID = "28:28:5d:a6:1c:28"


def score(artifact: dict, table: pd.DataFrame) -> dict:
    probability = artifact["pipeline"].predict_proba(table.drop(columns=["label"]))[:, 1]
    return evaluate_predictions(table.label.astype(int), probability, artifact["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    model = root / "models/clean_rebuild_v1/selected_model.joblib"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Frozen model hash mismatch: {digest}")
    artifact = joblib.load(model)

    awid3 = pd.read_csv(root / "results/awid3_evil_twin_test_w20.csv", low_memory=False)
    d05 = pd.read_csv(root / "results/d05_ar2_w20_features.csv", low_memory=False)
    d05["label"] = d05.bssid.astype(str).str.lower().eq(ZYXEL_ATTACK_BSSID).astype(int)

    result = {
        "model_sha256": digest,
        "threshold": artifact["threshold"],
        "AWID3_temporal_test": score(artifact, awid3),
        "D05-A-R2_previously_inspected_cross_dataset": score(artifact, d05),
        "provenance_note": "D05-A-R2 labels use operator-confirmed AP roles; it is not an untouched independent test.",
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
