#!/usr/bin/env python3
"""Develop, then post-freeze evaluate, the clean_rebuild_v1 detector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from evil_twin.data.real_capture_scope import DECO_BSSID, ZYXEL_BSSID
from evil_twin.experiments.clean_rebuild import (
    assert_stage_allowed,
    load_protocol,
    verify_freeze_manifest,
)
from evil_twin.features.windows import make_windows
from evil_twin.inference.predict import extract_frames_fast
from evil_twin.ml.evaluate import evaluate_predictions
from evil_twin.ml.train import train_select


ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "../Dataset").resolve()
PROTOCOL_PATH = ROOT / "experiments/clean_rebuild_v1/protocol.yaml"
MODEL_DIR = ROOT / "models/clean_rebuild_v1"
DEV_OUT = ROOT / "results/clean_rebuild_v1/development"
D05_OUT = ROOT / "results/clean_rebuild_v1/d05_previously_inspected_test"
SEED = 20261116


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_parts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    start = float(frame.window_start.min())
    end = float(frame.window_end.max())
    train_end = start + 0.60 * (end - start)
    validation_end = start + 0.80 * (end - start)
    return {
        "train": frame[frame.window_end.le(train_end)].copy(),
        "validation": frame[
            frame.window_start.gt(train_end) & frame.window_end.le(validation_end)
        ].copy(),
        "test": frame[frame.window_start.gt(validation_end)].copy(),
    }


def target_windows(filename: str, allowed_bssids: set[str]) -> pd.DataFrame:
    path = DATA / filename
    frames = [
        row for row in extract_frames_fast(path)
        if str(row.get("bssid", "")).lower() in allowed_bssids
    ]
    return pd.DataFrame(make_windows(frames, 20, 20, "profile"))


def benign_positive_rate(probabilities: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(probabilities) >= threshold)) if len(probabilities) else float("nan")


def develop() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    lock_path = ROOT / protocol["freeze_policy"]["lock_path"]
    if lock_path.exists():
        raise RuntimeError("clean_rebuild_v1 is already frozen; create a new version to develop again")
    for dataset, stage in (
        ("AWID3_EVIL_TWIN", "training"),
        ("AWID3_EVIL_TWIN", "hyperparameter_selection"),
        ("D01", "identity_enrollment"),
        ("D02", "threshold_selection"),
        ("D03", "development_validation"),
    ):
        assert_stage_allowed(protocol, dataset, stage)

    DEV_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    awid3 = pd.read_csv(ROOT / "results/awid3_evil_twin_profile_w20_labeled.csv")
    parts = temporal_parts(awid3)

    # D01 is extracted and recorded solely as an identity/reference source.
    d01_tables = []
    for filename in protocol["dataset_roles"]["D01"]["files"]:
        current = target_windows(filename, {DECO_BSSID})
        current.insert(0, "capture", filename)
        d01_tables.append(current)
    d01 = pd.concat(d01_tables, ignore_index=True) if d01_tables else pd.DataFrame()
    d01.to_csv(DEV_OUT / "d01_reference_features.csv", index=False)

    d02 = target_windows(protocol["dataset_roles"]["D02"]["files"][0], {DECO_BSSID})
    d03 = target_windows(protocol["dataset_roles"]["D03"]["files"][0], {DECO_BSSID})
    d02.assign(label=0).to_csv(DEV_OUT / "d02_legitimate_calibration.csv", index=False)
    d03.assign(label=0).to_csv(DEV_OUT / "d03_development_validation.csv", index=False)

    comparison, artifact = train_select(
        parts["train"], parts["validation"], MODEL_DIR, seed=SEED,
        objective="balanced_accuracy",
    )
    comparison.to_csv(DEV_OUT / "awid3_validation_model_comparison.csv", index=False)

    pipeline = artifact["pipeline"]
    awid3_val_prob = pipeline.predict_proba(parts["validation"])[:, 1]
    d02_prob = pipeline.predict_proba(d02)[:, 1]
    threshold_rows = []
    combined_y = np.concatenate([parts["validation"].label.astype(int), np.zeros(len(d02), dtype=int)])
    combined_prob = np.concatenate([awid3_val_prob, d02_prob])
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        metrics = evaluate_predictions(combined_y, combined_prob, threshold)
        threshold_rows.append({
            "threshold": threshold,
            "selection_balanced_accuracy": metrics["balanced_accuracy"],
            "selection_f1": metrics["f1"],
            "awid3_validation_f1": evaluate_predictions(
                parts["validation"].label, awid3_val_prob, threshold
            )["f1"],
            "d02_legitimate_fpr": benign_positive_rate(d02_prob, threshold),
        })
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["selection_balanced_accuracy", "d02_legitimate_fpr", "threshold"],
        ascending=[False, True, True],
    )
    selected_threshold = float(thresholds.iloc[0].threshold)
    thresholds.to_csv(DEV_OUT / "threshold_selection.csv", index=False)

    artifact["threshold"] = selected_threshold
    artifact["clean_experiment_id"] = "clean_rebuild_v1"
    artifact["development_datasets"] = ["AWID3_EVIL_TWIN", "D01", "D02", "D03"]
    artifact["d05_accessed_during_development"] = False
    model_path = MODEL_DIR / "selected_model.joblib"
    joblib.dump(artifact, model_path)

    d03_prob = pipeline.predict_proba(d03)[:, 1]
    test_prob = pipeline.predict_proba(parts["test"])[:, 1]
    development_summary = {
        "experiment_id": "clean_rebuild_v1",
        "model": artifact["model"],
        "params": artifact["params"],
        "threshold": selected_threshold,
        "feature_selection": "PREDECLARED_ALL_ELIGIBLE_FEATURES_NO_D05_SELECTION",
        "awid3_internal_test": evaluate_predictions(parts["test"].label, test_prob, selected_threshold),
        "d01_reference_windows": int(len(d01)),
        "d02_calibration_windows": int(len(d02)),
        "d02_legitimate_fpr": benign_positive_rate(d02_prob, selected_threshold),
        "d03_validation_windows": int(len(d03)),
        "d03_legitimate_fpr": benign_positive_rate(d03_prob, selected_threshold),
        "d05_accessed": False,
    }
    (DEV_OUT / "development_summary.json").write_text(json.dumps(development_summary, indent=2) + "\n")
    metadata = {key: value for key, value in artifact.items() if key != "pipeline"}
    metadata.update({
        "sha256": sha256_file(model_path),
        "training_dataset": "AWID3 7. Evil_Twin temporal train block",
        "model_selection_dataset": "AWID3 temporal validation block",
        "threshold_selection_datasets": ["AWID3 temporal validation block", "D02 legitimate-only"],
        "development_validation_dataset": "D03 legitimate-only",
        "d05_used": False,
    })
    (MODEL_DIR / "selected_model.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(development_summary, indent=2))


def evaluate_d05() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert_stage_allowed(protocol, "D05", "post_freeze_evaluation")
    lock_path = ROOT / protocol["freeze_policy"]["lock_path"]
    if not lock_path.is_file():
        raise RuntimeError("D05 access blocked: freeze manifest does not exist")
    manifest = json.loads(lock_path.read_text())
    lock_errors = verify_freeze_manifest(manifest, ROOT)
    if lock_errors:
        raise RuntimeError("D05 access blocked: " + "; ".join(lock_errors))
    if D05_OUT.exists():
        raise FileExistsError("D05 clean evaluation already exists; refusing to overwrite or rerun")

    artifact = joblib.load(MODEL_DIR / "selected_model.joblib")
    threshold = float(artifact["threshold"])
    pipeline = artifact["pipeline"]
    rows = []
    capture_summaries = []
    for capture_id, filename in (
        ("D05-A", "D05A_same_ssid_different_bssid_ch6.pcap"),
        ("D05-A-R2", "D05A_R2_wpa2_matched_ch6.pcap"),
    ):
        windows = target_windows(filename, {DECO_BSSID, ZYXEL_BSSID})
        probabilities = pipeline.predict_proba(windows)[:, 1]
        labels = windows.bssid.astype(str).str.lower().eq(ZYXEL_BSSID).astype(int).to_numpy()
        predictions = (probabilities >= threshold).astype(int)
        capture_summaries.append({
            "capture_id": capture_id,
            **evaluate_predictions(labels, probabilities, threshold),
        })
        for (_, window), truth, probability, prediction in zip(
            windows.iterrows(), labels, probabilities, predictions
        ):
            rows.append({
                "capture_id": capture_id,
                "source_file": filename,
                "bssid": window.bssid,
                "ground_truth": int(truth),
                "window_start": float(window.window_start),
                "window_end": float(window.window_end),
                "frame_count": int(window.frame_count),
                "attack_probability": float(probability),
                "prediction": int(prediction),
            })

    predictions = pd.DataFrame(rows)
    overall = evaluate_predictions(
        predictions.ground_truth.to_numpy(), predictions.attack_probability.to_numpy(), threshold
    )
    result = {
        "experiment_id": "clean_rebuild_v1",
        "evaluation_status": "PREVIOUSLY_INSPECTED_EXTERNAL_TEST",
        "independent_test_claim": False,
        "model_sha256": sha256_file(MODEL_DIR / "selected_model.joblib"),
        "freeze_lock_sha256": manifest["lock_sha256"],
        "threshold": threshold,
        "model_or_preprocessor_refit": False,
        "recalibrated_on_d05": False,
        "feature_or_threshold_selection_on_d05": False,
        "overall": overall,
        "captures": capture_summaries,
    }
    D05_OUT.mkdir(parents=True)
    predictions.to_csv(D05_OUT / "per_window_predictions.csv", index=False)
    (D05_OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(lock_path, D05_OUT / "freeze_manifest_used.json")
    report = f"""# Clean Rebuild v1 — D05 Evaluation

Status: **Previously Inspected External Test** (not a pristine independent test)

The detector and threshold were frozen before this evaluation. D05 was not used
for training, calibration, model/feature selection, or threshold selection.

- F1: `{overall['f1']:.6f}`
- Precision: `{overall['precision']:.6f}`
- Recall: `{overall['recall']:.6f}`
- Balanced accuracy: `{overall['balanced_accuracy']:.6f}`
- Confusion matrix `[TN, FP; FN, TP]`: `{overall['confusion_matrix']}`
- Frozen model SHA-256: `{result['model_sha256']}`
- Freeze lock SHA-256: `{manifest['lock_sha256']}`
"""
    (D05_OUT / "report.md").write_text(report)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "evaluate-d05"))
    args = parser.parse_args()
    if args.command == "develop":
        develop()
    else:
        evaluate_d05()


if __name__ == "__main__":
    main()
