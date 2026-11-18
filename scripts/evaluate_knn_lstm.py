"""Validation-only KNN/LSTM selection and frozen external diagnostics."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evil_twin.ml.evaluate import evaluate_predictions
from evil_twin.ml.sequence import LSTMConfig, build_causal_sequences, make_lstm
from evil_twin.data.real_capture_scope import is_in_scope, target_identity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "ml_knn_lstm"
ARTIFACTS = OUT / "artifacts"
SEED = 20261117
SEQUENCE_LENGTH = 5
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
FROZEN_RF_SHA256 = "8a1e1717f77ded865336c7d4d9c3db81559bb1300cb979e73c803f02f26ed145"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_row(model: str, split: str, scope: str, metrics: dict) -> dict:
    cm = metrics.get("confusion_matrix")
    return {
        "model": model,
        "split": split,
        "scope": scope,
        "n": metrics["n"],
        "tn": cm[0][0] if cm else None,
        "fp": cm[0][1] if cm else None,
        "fn": cm[1][0] if cm else None,
        "tp": cm[1][1] if cm else None,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "legitimate_fpr": metrics.get("legitimate_fpr"),
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
    }


def selection_key(metrics: dict, threshold: float) -> tuple[float, ...]:
    return (
        float(metrics.get("balanced_accuracy") or -1),
        -float(metrics.get("legitimate_fpr") or 0),
        float(metrics.get("f1") or -1),
        -abs(threshold - 0.5),
    )


def attach_external_ground_truth(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame.apply(lambda row: is_in_scope(row["session"], row["bssid"]), axis=1)].copy()
    result["ap_identity"] = result["bssid"].map(target_identity)
    result["target_eligible"] = True
    result["label"] = 0
    attack = result["session"].isin(["D05-A", "D05-A-R2"]) & result["ap_identity"].eq("Zyxel")
    result.loc[attack, "label"] = 1
    return result


def torch_probabilities(model, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    import torch

    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).float()
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0)


def train_lstm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    hidden_size: int,
    learning_rate: float,
    epochs: int = 80,
):
    import torch
    from torch import nn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = make_lstm(LSTMConfig(input_size=train_x.shape[2], hidden_size=hidden_size))
    positive = max(int(train_y.sum()), 1)
    negative = max(len(train_y) - positive, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = torch.from_numpy(train_x).float()
    targets = torch.from_numpy(train_y.astype(np.float32))
    model.train()
    final_loss = np.nan
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return model, final_loss


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / "models" / "clean_rebuild_v1" / "selected_model.joblib"
    if sha256(frozen_path) != FROZEN_RF_SHA256:
        raise RuntimeError("Frozen Random Forest changed before KNN/LSTM experiment")

    train = pd.read_csv(RESULTS / "awid3_evil_twin_train_w20.csv", low_memory=False)
    validation = pd.read_csv(RESULTS / "awid3_evil_twin_validation_w20.csv", low_memory=False)
    test = pd.read_csv(RESULTS / "awid3_evil_twin_test_w20.csv", low_memory=False)
    external_raw = pd.read_csv(
        RESULTS / "external_real_capture_evaluation" / "external_window_features_target_only.csv", low_memory=False
    )
    external = attach_external_ground_truth(external_raw)

    schema = json.loads((RESULTS / "feature_schema.json").read_text())
    numeric_features = [name for name in schema["numeric"] if name in external.columns]
    preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    train_x = preprocessor.fit_transform(train[numeric_features]).astype(np.float32)
    validation_x = preprocessor.transform(validation[numeric_features]).astype(np.float32)
    test_x = preprocessor.transform(test[numeric_features]).astype(np.float32)
    external_x = preprocessor.transform(external[numeric_features]).astype(np.float32)
    train_y = train["label"].to_numpy(dtype=int)
    validation_y = validation["label"].to_numpy(dtype=int)
    test_y = test["label"].to_numpy(dtype=int)

    selection_rows: list[dict] = []
    best_knn = None
    for neighbors in (3, 5, 7, 11, 15):
        for weights in ("uniform", "distance"):
            model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights, p=2, n_jobs=-1)
            model.fit(train_x, train_y)
            probabilities = model.predict_proba(validation_x)[:, 1]
            for threshold in THRESHOLDS:
                metrics = evaluate_predictions(validation_y, probabilities, threshold)
                row = {
                    "model": "knn",
                    "params": json.dumps({"n_neighbors": neighbors, "weights": weights, "p": 2}),
                    "threshold": threshold,
                    **metrics_row("knn", "validation", "all_windows", metrics),
                }
                row.pop("model", None)
                selection_rows.append({"model": "knn", **row})
                candidate = (selection_key(metrics, threshold), model, neighbors, weights, threshold)
                if best_knn is None or candidate[0] > best_knn[0]:
                    best_knn = candidate

    assert best_knn is not None
    _, knn, knn_neighbors, knn_weights, knn_threshold = best_knn
    joblib.dump(
        {
            "model": knn,
            "preprocessor": preprocessor,
            "features": numeric_features,
            "threshold": knn_threshold,
            "selection_split": "AWID3 validation only",
        },
        ARTIFACTS / "knn.joblib",
    )

    train_seq_x, train_seq_y, _ = build_causal_sequences(train_x, train, SEQUENCE_LENGTH)
    val_seq_x, val_seq_y, val_seq_meta = build_causal_sequences(validation_x, validation, SEQUENCE_LENGTH)
    test_seq_x, test_seq_y, test_seq_meta = build_causal_sequences(test_x, test, SEQUENCE_LENGTH)
    external_seq_x, external_seq_y, external_seq_meta = build_causal_sequences(
        external_x, external, SEQUENCE_LENGTH
    )

    best_lstm = None
    training_log = []
    for hidden_size in (8, 16):
        for learning_rate in (0.001, 0.005):
            started = time.perf_counter()
            model, loss = train_lstm(
                train_seq_x,
                train_seq_y,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
            )
            elapsed = time.perf_counter() - started
            probabilities = torch_probabilities(model, val_seq_x)
            training_log.append(
                {
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "epochs": 80,
                    "final_train_loss": loss,
                    "training_seconds": elapsed,
                }
            )
            for threshold in THRESHOLDS:
                metrics = evaluate_predictions(val_seq_y, probabilities, threshold)
                selection_rows.append(
                    {
                        "model": "lstm",
                        "params": json.dumps(
                            {"hidden_size": hidden_size, "learning_rate": learning_rate, "epochs": 80}
                        ),
                        "threshold": threshold,
                        **{k: v for k, v in metrics_row("lstm", "validation", "causal_seq5", metrics).items() if k != "model"},
                    }
                )
                candidate = (
                    selection_key(metrics, threshold),
                    model,
                    hidden_size,
                    learning_rate,
                    threshold,
                )
                if best_lstm is None or candidate[0] > best_lstm[0]:
                    best_lstm = candidate

    assert best_lstm is not None
    _, lstm, lstm_hidden, lstm_lr, lstm_threshold = best_lstm
    import torch

    torch.save(
        {
            "state_dict": lstm.state_dict(),
            "config": {
                "input_size": len(numeric_features),
                "hidden_size": lstm_hidden,
                "layers": 1,
                "sequence_length": SEQUENCE_LENGTH,
                "learning_rate": lstm_lr,
                "epochs": 80,
                "threshold": lstm_threshold,
            },
            "features": numeric_features,
            "selection_split": "AWID3 validation only",
        },
        ARTIFACTS / "lstm.pt",
    )
    joblib.dump(preprocessor, ARTIFACTS / "numeric_preprocessor.joblib")

    pd.DataFrame(selection_rows).to_csv(OUT / "model_selection.csv", index=False)
    pd.DataFrame(training_log).to_csv(OUT / "lstm_training_log.csv", index=False)

    awid_rows = []
    knn_test_prob = knn.predict_proba(test_x)[:, 1]
    awid_rows.append(
        metrics_row("knn", "AWID3_test", "all_windows", evaluate_predictions(test_y, knn_test_prob, knn_threshold))
    )
    lstm_test_prob = torch_probabilities(lstm, test_seq_x)
    awid_rows.append(
        metrics_row(
            "lstm",
            "AWID3_test",
            "causal_seq5",
            evaluate_predictions(test_seq_y, lstm_test_prob, lstm_threshold),
        )
    )
    pd.DataFrame(awid_rows).to_csv(OUT / "awid3_test_metrics.csv", index=False)

    prediction_frames = []
    external_rows = []
    knn_external_prob = knn.predict_proba(external_x)[:, 1]
    knn_pred = external[["session", "source_file", "bssid", "window_start", "ap_identity", "target_eligible", "label"]].copy()
    knn_pred.insert(0, "model", "knn")
    knn_pred["attack_probability"] = knn_external_prob
    knn_pred["prediction"] = (knn_external_prob >= knn_threshold).astype(int)
    prediction_frames.append(knn_pred)
    eligible = knn_pred["target_eligible"].to_numpy(dtype=bool)
    external_rows.append(
        metrics_row(
            "knn",
            "external",
            "target_AP_all_windows",
            evaluate_predictions(knn_pred.loc[eligible, "label"], knn_external_prob[eligible], knn_threshold),
        )
    )

    lstm_external_prob = torch_probabilities(lstm, external_seq_x)
    lstm_pred = external_seq_meta[
        ["session", "source_file", "bssid", "window_start", "ap_identity", "target_eligible", "label"]
    ].copy()
    lstm_pred.insert(0, "model", "lstm")
    lstm_pred["attack_probability"] = lstm_external_prob
    lstm_pred["prediction"] = (lstm_external_prob >= lstm_threshold).astype(int)
    prediction_frames.append(lstm_pred)
    eligible_seq = lstm_pred["target_eligible"].to_numpy(dtype=bool)
    external_rows.append(
        metrics_row(
            "lstm",
            "external",
            "target_AP_causal_seq5",
            evaluate_predictions(
                lstm_pred.loc[eligible_seq, "label"], lstm_external_prob[eligible_seq], lstm_threshold
            ),
        )
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "external_predictions.csv", index=False)
    pd.DataFrame(external_rows).to_csv(OUT / "external_metrics.csv", index=False)

    scenario_rows = []
    for predictions in prediction_frames:
        target = predictions[predictions["target_eligible"]]
        for (session, ap_identity), group in target.groupby(["session", "ap_identity"], sort=False):
            scenario_rows.append(
                {
                    "model": group.model.iloc[0],
                    "session": session,
                    "ap_identity": ap_identity,
                    "n": len(group),
                    "ground_truth": "evil_twin" if group.label.iloc[0] else "legitimate",
                    "positive_rate": float(group["prediction"].mean()),
                    "score_min": float(group["attack_probability"].min()),
                    "score_median": float(group["attack_probability"].median()),
                    "score_max": float(group["attack_probability"].max()),
                }
            )
    pd.DataFrame(scenario_rows).to_csv(OUT / "per_scenario_metrics.csv", index=False)

    import sklearn

    manifest = {
        "protocol": "KNN and causal LSTM; selection and threshold tuning on AWID3 validation only",
        "seed": SEED,
        "features": numeric_features,
        "preprocessing_fit": "AWID3 train only",
        "knn": {"n_neighbors": knn_neighbors, "weights": knn_weights, "p": 2, "threshold": knn_threshold},
        "lstm": {
            "hidden_size": lstm_hidden,
            "learning_rate": lstm_lr,
            "epochs": 80,
            "sequence_length": SEQUENCE_LENGTH,
            "threshold": lstm_threshold,
        },
        "external_used_for_selection": False,
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "external_ground_truth": {
            "D01-D04_target_AP": "legitimate",
            "D05_Deco": "legitimate",
            "D05_Zyxel": "evil_twin",
            "unlisted_bssids": "excluded before prediction and metrics",
        },
        "frozen_random_forest_sha256_before": FROZEN_RF_SHA256,
        "frozen_random_forest_sha256_after": sha256(frozen_path),
        "frozen_random_forest_unchanged": sha256(frozen_path) == FROZEN_RF_SHA256,
        "artifacts": {
            p.name: sha256(p) for p in sorted(ARTIFACTS.iterdir()) if p.is_file()
        },
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2))

    awid_df = pd.DataFrame(awid_rows).set_index("model")
    ext_df = pd.DataFrame(external_rows).set_index("model")
    report = f"""# KNN และ LSTM: AWID3 → Real Captures

การเลือก hyperparameters และ threshold ใช้เฉพาะ AWID3 validation หลัง fit preprocessing บน AWID3 train เท่านั้น Real Captures ถูกเปิดประเมินภายหลังและไม่ได้ใช้เลือกโมเดล

## ผล AWID3 temporal test

- KNN: F1 `{awid_df.loc['knn','f1']:.4f}`, recall `{awid_df.loc['knn','recall']:.4f}`, FPR `{awid_df.loc['knn','legitimate_fpr']:.4f}` (ทุก W20 window)
- LSTM: F1 `{awid_df.loc['lstm','f1']:.4f}`, recall `{awid_df.loc['lstm','recall']:.4f}`, FPR `{awid_df.loc['lstm','legitimate_fpr']:.4f}` (causal sequence 5 windows)

## ผล External target AP

- KNN: F1 `{ext_df.loc['knn','f1']:.4f}`, recall `{ext_df.loc['knn','recall']:.4f}`, FPR `{ext_df.loc['knn','legitimate_fpr']:.4f}`
- LSTM: F1 `{ext_df.loc['lstm','f1']:.4f}`, recall `{ext_df.loc['lstm','recall']:.4f}`, FPR `{ext_df.loc['lstm','legitimate_fpr']:.4f}`

Ground truth ใช้ exact scope เท่านั้น: D00–D03 Deco `:1c`, D04 Zyxel `:28`, และ D05 ใช้ทั้งสองตัว โดย D05 Zyxel เป็น Evil Twin; BSSID อื่นถูกตัดก่อน prediction และ metric. LSTM ไม่ pad sequence จึงตัด 4 windows แรกของแต่ละ capture+BSSID และมี support ต่างจาก KNN

## ข้อจำกัด

AWID3 train/validation/test เป็น temporal blocks จาก capture และ device profiles เดียวกัน ดังนั้นคะแนน AWID3 สูงไม่ใช่หลักฐาน independent-device generalization. ผล external เป็น diagnostics แบบ AP-role/window และห้ามนำกลับไปเปลี่ยน candidate จากผลชุดนี้
"""
    (OUT / "report_th.md").write_text(report)

    print(json.dumps({"awid3": awid_rows, "external": external_rows, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
