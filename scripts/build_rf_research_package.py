#!/usr/bin/env python3
"""Build publication tables and figures for the frozen RF-only detector.

This script performs no training, threshold tuning, or model selection.  It
consolidates already frozen/evaluated artifacts and clearly marks comparisons
that do not share an identical protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from evil_twin.ml.evaluate import evaluate_predictions
from evil_twin.ml.preprocessing import build_preprocessor, infer_schema


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rf_final_research_evaluation"
FIG = OUT / "figures"
RF_MODEL = ROOT / "models/clean_rebuild_v1/selected_model.joblib"
RF_SHA256 = "8a1e1717f77ded865336c7d4d9c3db81559bb1300cb979e73c803f02f26ed145"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_row(model: str, scenario: str, role: str, metrics: dict, comparability: str) -> dict:
    cm = metrics["confusion_matrix"]
    return {
        "model": model,
        "scenario": scenario,
        "evaluation_role": role,
        "n": metrics["n"],
        "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        "precision": metrics.get("precision"), "recall": metrics.get("recall"),
        "f1": metrics.get("f1"), "balanced_accuracy": metrics.get("balanced_accuracy"),
        "false_positive_rate": metrics.get("legitimate_fpr"),
        "comparison_status": comparability,
    }


def rf_performance() -> pd.DataFrame:
    dev = json.loads((ROOT / "results/clean_rebuild_v1/development/development_summary.json").read_text())
    d05 = json.loads((ROOT / "results/clean_rebuild_v1/d05_previously_inspected_test/evaluation.json").read_text())
    rows = [metric_row("Frozen RF", "AWID3 temporal test", "same-capture temporal test",
                       dev["awid3_internal_test"], "INTERNAL_ONLY")]
    r2 = next(capture for capture in d05["captures"] if capture["capture_id"] == "D05-A-R2")
    rows.append(metric_row("Frozen RF", "D05-A-R2", "previously-inspected cross-dataset test",
                           r2, "PRIMARY_CROSS_DATASET_RESULT"))
    return pd.DataFrame(rows)


def baseline_comparison(rf: pd.DataFrame) -> pd.DataFrame:
    awid = pd.read_csv(ROOT / "results/ml_knn_lstm/awid3_test_metrics.csv")
    predictions = pd.read_csv(ROOT / "results/ml_knn_lstm/external_predictions_d05_ar2.csv")
    rows = []
    rf_awid = rf[rf.scenario.eq("AWID3 temporal test")].iloc[0].to_dict()
    rf_awid["comparison_status"] = "SAME_W20_AS_KNN; LSTM_USES_SEQ5"
    rows.append(rf_awid)
    for _, row in awid.iterrows():
        rows.append({
            "model": row.model.upper(), "scenario": "AWID3 temporal test",
            "evaluation_role": row.scope, "n": int(row.n), "tn": int(row.tn), "fp": int(row.fp),
            "fn": int(row.fn), "tp": int(row.tp), "precision": row.precision, "recall": row.recall,
            "f1": row.f1, "balanced_accuracy": row.balanced_accuracy,
            "false_positive_rate": row.legitimate_fpr,
            "comparison_status": "SAME_W20" if row.model == "knn" else "CAUSAL_SEQ5_DIFFERENT_SUPPORT",
        })
    rf_d05 = rf[rf.scenario.eq("D05-A-R2")].iloc[0].to_dict()
    rf_d05["comparison_status"] = "SAME_W20_AS_KNN"
    rows.append(rf_d05)
    for model, group in predictions[predictions.session.eq("D05-A-R2")].groupby("model"):
        metrics = evaluate_predictions(group.label, group.attack_probability, 0.5)
        rows.append(metric_row(model.upper(), "D05-A-R2", group.iloc[0].model + " frozen external",
                               metrics, "SAME_W20" if model == "knn" else "CAUSAL_SEQ5_DIFFERENT_SUPPORT"))
    rows.extend(classical_baselines())
    return pd.DataFrame(rows)


def temporal_parts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    start, end = float(frame.window_start.min()), float(frame.window_end.max())
    train_end = start + 0.60 * (end - start)
    validation_end = start + 0.80 * (end - start)
    return {
        "train": frame[frame.window_end.le(train_end)].copy(),
        "test": frame[frame.window_start.gt(validation_end)].copy(),
    }


def classical_baselines() -> list[dict]:
    """Reproduce validation-selected CART/LR on frozen AWID3 and D05-A-R2 scopes."""
    awid = pd.read_csv(ROOT / "results/awid3_evil_twin_profile_w20_labeled.csv", low_memory=False)
    parts = temporal_parts(awid)
    train, test = parts["train"], parts["test"]
    external = pd.read_csv(ROOT / "results/d05_ar2_w20_features.csv", low_memory=False)
    external["label"] = (
        external.session.eq("D05-A-R2")
        & external.bssid.astype(str).str.lower().eq("28:28:5d:a6:1c:28")
    ).astype(int)
    schema = infer_schema(train.drop(columns=["label"]))
    candidates = [
        ("CART", {"max_depth": 3, "min_samples_leaf": 5, "threshold": 0.3},
         DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=20261116), False),
        ("Logistic Regression", {"C": 0.1, "threshold": 0.3},
         LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=20261116), True),
    ]
    rows = []
    for name, params, classifier, scale in candidates:
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(schema, scale=scale)),
            ("classifier", classifier),
        ])
        pipeline.fit(train.drop(columns=["label"]), train.label.astype(int))
        threshold = params["threshold"]
        scopes = [
            ("AWID3 temporal test", test, "internal temporal test"),
            ("D05-A-R2", external, "cross-dataset"),
        ]
        for scenario, frame, role in scopes:
            probabilities = pipeline.predict_proba(frame)[:, 1]
            predictions_now = (probabilities >= threshold).astype(int)
            if frame.label.nunique() == 1:
                fp = int(predictions_now.sum()); tn = int(len(frame) - fp)
                row = {"model": name, "scenario": scenario, "evaluation_role": role, "n": len(frame),
                       "tn": tn, "fp": fp, "fn": 0, "tp": 0, "precision": np.nan,
                       "recall": np.nan, "f1": np.nan, "balanced_accuracy": np.nan,
                       "false_positive_rate": fp / len(frame)}
            else:
                row = metric_row(name, scenario, role,
                                 evaluate_predictions(frame.label, probabilities, threshold), "")
            row["comparison_status"] = "SAME_W20_FEATURE_SCHEMA; VALIDATION_SELECTED_NO_D05_TUNING"
            row["configuration"] = json.dumps(params)
            rows.append(row)
    return rows


def lightweight() -> pd.DataFrame:
    latency = pd.read_csv(ROOT / "results/phase4_2_lightweight_benchmark/latency_summary.csv")
    memory = pd.read_csv(ROOT / "results/phase4_2_lightweight_benchmark/memory_summary.csv")
    rf_l = latency[(latency.method == "RF") & (latency.unit == "W20")].iloc[0]
    rf_m = memory[(memory.method == "RF") & (memory.unit == "W20")].iloc[0]
    return pd.DataFrame([{
        "model": "Frozen RF", "prediction_unit": "W20", "window_size_beacons": 20,
        "model_artifact_bytes": RF_MODEL.stat().st_size,
        "latency_p50_ms_single_window": rf_l.latency_p50_us / 1000,
        "latency_p95_ms_single_window": rf_l.latency_p95_us / 1000,
        "latency_p99_ms_single_window": rf_l.latency_p99_us / 1000,
        "throughput_windows_per_second_single_window_calls": rf_l.throughput_units_per_second,
        "cpu_time_ms_per_window": rf_l.cpu_time_us_per_unit / 1000,
        "python_incremental_peak_mib": rf_m.python_incremental_peak_bytes / 2**20,
        "isolated_process_peak_rss_mib": rf_m.isolated_process_peak_rss_bytes / 2**20,
        "interpretation": "small model artifact; Python process RSS is not embedded-class lightweight",
    }])


def literature() -> pd.DataFrame:
    return pd.DataFrame([
        {"work": "This work (Frozen RF)", "year": 2026, "method": "RF, Beacon W20",
         "data": "AWID3 train; D05-A-R2 cross-dataset", "reported_result": "D05-A-R2 F1=0.8892",
         "directly_comparable": "YES only to local baselines on matched W20", "source": "local experiment"},
        {"work": "AWID3 dataset", "year": 2021, "method": "Public Wi-Fi attack corpus",
         "data": "WPA2-Enterprise, 802.11w, Wi-Fi 5", "reported_result": "dataset/provenance reference",
         "directly_comparable": "NO", "source": "https://doi.org/10.1109/ACCESS.2021.3061609"},
        {"work": "ETGuard", "year": 2019, "method": "Beacon fingerprinting, client-server",
         "data": "12 deployment attack scenarios", "reported_result": "1 FP scenario; 0 FN",
         "directly_comparable": "NO: different data, unit, and deployment", "source": "https://doi.org/10.1016/j.cose.2019.02.014"},
        {"work": "Lu et al. / ET-spotter", "year": 2018, "method": "Statistical 802.11 data-frame forwarding",
         "data": "Authors' client-side experiment", "reported_result": "96% accuracy",
         "directly_comparable": "NO: data frames and protocol differ", "source": "https://doi.org/10.1587/transinf.2018EDP7030"},
        {"work": "Seo et al.", "year": 2020, "method": "RF; clock skew/RSS/channel/duration",
         "data": "10 APs; 10-minute samples", "reported_result": "99.51–100% accuracy; ~1% FP noted",
         "directly_comparable": "NO: own data and 300–500 Beacon decision unit", "source": "https://doi.org/10.3745/JIPS.03.0137"},
        {"work": "Wi-Fi Frame Classification", "year": 2020, "method": "Feature selection + J48",
         "data": "Authors' legitimate/rogue captures", "reported_result": "91.2367% accuracy",
         "directly_comparable": "NO: different capture and metric", "source": "https://doi.org/10.1109/TENSYMP50017.2020.9231042"},
    ])


def plots(rf: pd.DataFrame, baselines: pd.DataFrame, ablation: pd.DataFrame, light: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # Generalization profile
    ordered = ["AWID3 temporal test", "D05-A-R2"]
    plot = rf[rf.scenario.isin(ordered)].set_index("scenario").reindex(ordered)
    fig, ax = plt.subplots(figsize=(10, 5.5)); bars = ax.bar(plot.index, plot.f1, color=["#2374AB", "#2A9D8F", "#E76F51", "#2A9D8F", "#8D6EAF"])
    ax.bar_label(bars, fmt="%.3f", padding=3); ax.set_ylim(0, 1.08); ax.set_ylabel("F1 score")
    ax.set_title("Frozen RF: Internal Performance vs Cross-Dataset Generalization")
    ax.tick_params(axis="x", rotation=18); fig.tight_layout(); fig.savefig(FIG / "rf_generalization_f1.png", dpi=180); plt.close(fig)

    # Matched D05 baseline comparison
    order = ["Frozen RF", "CART", "Logistic Regression", "KNN", "LSTM"]
    plot = baselines[baselines.scenario.eq("D05-A-R2")].set_index("model").reindex(order).reset_index()
    colors = ["#2374AB", "#5B8E7D", "#F2A65A", "#E76F51", "#8D6EAF"]
    fig, ax = plt.subplots(figsize=(9, 5)); bars = ax.bar(plot.model, plot.f1, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3); ax.set_ylim(0, 1.02); ax.set_ylabel("F1 score")
    ax.set_title("D05-A-R2 Cross-Dataset Performance (LSTM uses causal Seq5)")
    fig.tight_layout(); fig.savefig(FIG / "model_comparison_d05ar2.png", dpi=180); plt.close(fig)

    # Lightweight summary
    values = [light.iloc[0].latency_p50_ms_single_window, light.iloc[0].latency_p95_ms_single_window,
              light.iloc[0].latency_p99_ms_single_window]
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); bars = ax.bar(["P50", "P95", "P99"], values, color="#2A9D8F")
    ax.bar_label(bars, fmt="%.2f ms", padding=3); ax.set_ylabel("Latency per W20 decision (ms)")
    ax.set_title("Frozen RF Detection-Only Single-Window Latency")
    fig.tight_layout(); fig.savefig(FIG / "rf_latency.png", dpi=180); plt.close(fig)

    # Confusion matrices for the principal labeled RF scenarios.
    scenarios = ["AWID3 temporal test", "D05-A-R2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, scenario in zip(axes, scenarios):
        row = rf[rf.scenario.eq(scenario)].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["Normal", "Evil Twin"])
        ax.set_yticks([0, 1], ["Normal", "Evil Twin"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title(scenario)
    fig.suptitle("Frozen RF Confusion Matrices", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "rf_confusion_matrices.png", dpi=180); plt.close(fig)


def main() -> None:
    if sha256(RF_MODEL) != RF_SHA256:
        raise RuntimeError("Frozen RF hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    for obsolete in (
        OUT / "feature_ablation_external.csv",
        FIG / "model_comparison_d05.png",
        FIG / "model_comparison_d04_fpr.png",
        FIG / "rf_feature_ablation_external.png",
    ):
        obsolete.unlink(missing_ok=True)
    rf = rf_performance(); baselines = baseline_comparison(rf); light = lightweight()
    window = pd.read_csv(ROOT / "results/window_ablation.csv")
    papers = literature()
    rf.to_csv(OUT / "rf_performance_by_scenario.csv", index=False)
    baselines.to_csv(OUT / "model_comparison.csv", index=False)
    light.to_csv(OUT / "lightweight_summary.csv", index=False)
    window.to_csv(OUT / "window_size_ablation_awid3.csv", index=False)
    papers.to_csv(OUT / "literature_comparison.csv", index=False)
    plots(rf, baselines, pd.DataFrame(), light)

    report = f"""# RF-only Research Evaluation

## ตารางผลหลัก

| Scenario | TN | FP | FN | TP | Precision | Recall | F1 | FPR | บทบาท |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
"""
    for _, row in rf.iterrows():
        val = lambda name: "—" if pd.isna(row[name]) else f"{row[name]:.4f}"
        report += f"| {row.scenario} | {row.tn} | {row.fp} | {row.fn} | {row.tp} | {val('precision')} | {val('recall')} | {val('f1')} | {val('false_positive_rate')} | {row.evaluation_role} |\n"
    report += """

## Baseline บน W20 เดียวกัน

| Scenario | Model | N | TN | FP | FN | TP | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    order = ["AWID3 temporal test", "D05-A-R2"]
    names = ["Frozen RF", "CART", "Logistic Regression", "KNN", "LSTM"]
    baseline_view = baselines.assign(
        scenario_order=baselines.scenario.map({v: i for i, v in enumerate(order)}),
        model_order=baselines.model.map({v: i for i, v in enumerate(names)}),
    ).sort_values(["scenario_order", "model_order"])
    for _, row in baseline_view.iterrows():
        f1 = "—" if pd.isna(row.f1) else f"{row.f1:.4f}"
        fpr = "—" if pd.isna(row.false_positive_rate) else f"{row.false_positive_rate:.4f}"
        report += f"| {row.scenario} | {row.model} | {row.n} | {row.tn} | {row.fp} | {row.fn} | {row.tp} | {f1} | {fpr} |\n"
    report += f"""

## Lightweight

- Frozen artifact: **{light.iloc[0].model_artifact_bytes / 1024:.1f} KiB**
- Detection-only latency/W20: P50 **{light.iloc[0].latency_p50_ms_single_window:.2f} ms**, P95 **{light.iloc[0].latency_p95_ms_single_window:.2f} ms**, P99 **{light.iloc[0].latency_p99_ms_single_window:.2f} ms**
- Single-window-call throughput: **{light.iloc[0].throughput_windows_per_second_single_window_calls:.1f} W20/s**
- Python incremental allocation: **{light.iloc[0].python_incremental_peak_mib:.2f} MiB**; isolated process peak RSS: **{light.iloc[0].isolated_process_peak_rss_mib:.1f} MiB**

สรุป: เรียกว่า lightweight ได้เฉพาะในความหมายว่า artifact เล็กและ inference ทันอัตรา Beacon ทั่วไปบนเครื่องทดสอบ แต่ยังไม่ควรอ้างว่า memory-light สำหรับ embedded deployment เพราะ Python process RSS สูง และยังไม่มี benchmark บนอุปกรณ์ปลายทางจริง

## การเปรียบเทียบโมเดล

- RF, CART, Logistic Regression และ KNN ใช้ W20 เดียวกันใน AWID3 test และ D05-A-R2; LSTM ใช้ causal sequence 5 จึง support ต่างกันเล็กน้อย
- AWID3 temporal test: RF, Logistic Regression, KNN และ LSTM ได้ F1=1.0 แต่ CART ได้ 0 แสดงว่าผล internal ยังไวต่อชนิดโมเดลและ temporal block
- D05-A-R2: RF F1=0.8892 และเป็น baseline ที่ดีที่สุดในชุดเปรียบเทียบนี้
- จึงสรุปได้ว่า RF generalize จาก AWID3 ไปยัง D05-A-R2 ได้ดีกว่า baseline ที่ทดลอง โดยยังต้องเปิดเผยว่า D05-A-R2 เป็น previously inspected test

## Window-size ablation

AWID3 W10, W20 และ W50 ได้ F1=1.0 ทั้งหมดในผลเดิม จึงเลือก W20 ตาม protocol ที่ freeze ไว้ ไม่ได้เลือกย้อนหลังจาก D05-A-R2

## การเทียบกับงานวิจัย

ETGuard, ET-spotter, Seo et al. และ Wi-Fi Frame Classification ควรใช้ใน Related Work เท่านั้น ตัวเลขห้ามเทียบเป็น ranking โดยตรง เนื่องจาก dataset, attack construction, observation time และ prediction unit ไม่เหมือนกัน รายละเอียดอยู่ใน `literature_comparison.csv`.

## กราฟ

- `figures/rf_generalization_f1.png`
- `figures/model_comparison_d05ar2.png`
- `figures/rf_latency.png`
- `figures/rf_confusion_matrices.png`

## ข้อจำกัดการอ้างผล

D05-A-R2 เป็น previously inspected cross-dataset test; การเลือก RF เป็น final model หลังเห็นผลนี้ต้องเปิดเผย และไม่ควรเรียกว่า untouched independent model-selection evidence
"""
    (OUT / "report_th.md").write_text(report)
    manifest = {
        "experiment": "rf_final_research_evaluation",
        "model_sha256": sha256(RF_MODEL), "threshold": 0.3,
        "proposed_rf_retrained": False, "proposed_rf_threshold_tuned": False,
        "baseline_reconstruction_training_performed": True,
        "baseline_selection_source": "AWID3 validation only; no D04/D05 tuning",
        "sources": ["clean_rebuild_v1", "ml_knn_lstm", "phase4_2_lightweight_benchmark"],
        "outputs": {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "reproducibility_manifest.json"},
        "command": ".venv/bin/python scripts/build_rf_research_package.py",
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(rf.to_string(index=False)); print(light.to_string(index=False))


if __name__ == "__main__":
    main()
