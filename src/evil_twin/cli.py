"""Command-line interface for the RF-only detector reported in the paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .inference.predict import extract_windows, predict_pcap
from .ml.evaluate import evaluate_predictions
from .ml.train import train_select


DEFAULT_MODEL = Path("models/clean_rebuild_v1/selected_model.joblib")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _model_info(path: Path) -> dict:
    artifact = joblib.load(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": artifact.get("model"),
        "params": artifact.get("params"),
        "threshold": artifact.get("threshold"),
        "seed": artifact.get("seed"),
        "selection_split": artifact.get("selection_split"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evil-twin-rf",
        description="Frozen Random Forest detector for IEEE 802.11 Beacon PCAPs",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="extract causal W20 features from a PCAP")
    extract.add_argument("pcap", type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--size", type=int, default=20)
    extract.add_argument("--stride", type=int, default=20)
    extract.add_argument("--limit", type=int)

    predict = commands.add_parser("predict", help="run the frozen RF on a PCAP")
    predict.add_argument("pcap", type=Path)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--size", type=int, default=20)

    evaluate = commands.add_parser("evaluate", help="evaluate a labeled W20 CSV")
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--output", required=True, type=Path)

    train = commands.add_parser("train", help="validation-only model selection")
    train.add_argument("--train", required=True, type=Path)
    train.add_argument("--validation", required=True, type=Path)
    train.add_argument("--models", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--seed", type=int, default=20261116)

    verify = commands.add_parser("verify-model", help="print model metadata and SHA-256")
    verify.add_argument("--model", type=Path, default=DEFAULT_MODEL)

    args = parser.parse_args(argv)
    if args.command == "extract":
        table = extract_windows(args.pcap, args.size, args.stride, "profile", limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    elif args.command == "predict":
        table = predict_pcap(args.pcap, args.model, args.size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    elif args.command == "evaluate":
        data = pd.read_csv(args.data, low_memory=False)
        if "label" not in data:
            parser.error("evaluate requires a label column")
        artifact = joblib.load(args.model)
        probabilities = artifact["pipeline"].predict_proba(data.drop(columns=["label"]))[:, 1]
        _write_json(
            args.output,
            evaluate_predictions(data.label.astype(int), probabilities, artifact["threshold"]),
        )
    elif args.command == "train":
        comparison, _ = train_select(
            pd.read_csv(args.train, low_memory=False),
            pd.read_csv(args.validation, low_memory=False),
            args.models,
            seed=args.seed,
            objective="balanced_accuracy",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
    elif args.command == "verify-model":
        print(json.dumps(_model_info(args.model), indent=2))


if __name__ == "__main__":
    main()
