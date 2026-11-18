"""Clean-rebuild dataset firewall and immutable-style artifact lock."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEVELOPMENT_STAGES = {
    "training",
    "calibration",
    "feature_selection",
    "hyperparameter_selection",
    "threshold_selection",
    "model_selection",
}
REQUIRED_DATASETS = {"AWID3_EVIL_TWIN", "D01", "D02", "D03", "D05"}


def load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("clean experiment protocol must be a mapping")
    return payload


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    roles = protocol.get("dataset_roles", {})
    missing = REQUIRED_DATASETS - set(roles)
    if missing:
        errors.append(f"missing dataset roles: {sorted(missing)}")

    d05 = roles.get("D05", {})
    if d05.get("role") != "PREVIOUSLY_INSPECTED_EXTERNAL_TEST":
        errors.append("D05 must be PREVIOUSLY_INSPECTED_EXTERNAL_TEST")
    if set(d05.get("allowed_stages", [])) != {"post_freeze_evaluation"}:
        errors.append("D05 may only be used for post_freeze_evaluation")
    missing_prohibitions = DEVELOPMENT_STAGES - set(d05.get("prohibited_stages", []))
    if missing_prohibitions:
        errors.append(f"D05 missing prohibited stages: {sorted(missing_prohibitions)}")
    if not d05.get("disclosure"):
        errors.append("D05 previously-inspected disclosure is required")

    for dataset_id in ("AWID3_EVIL_TWIN", "D01", "D02", "D03"):
        allowed = set(roles.get(dataset_id, {}).get("allowed_stages", []))
        if "post_freeze_evaluation" in allowed:
            errors.append(f"{dataset_id} cannot be the D05 post-freeze test")

    excluded = protocol.get("excluded_from_clean_rebuild", {})
    if "AWID3_ROGUE_AP" not in excluded:
        errors.append("AWID3 Rogue AP exclusion must be explicit")
    if protocol.get("freeze_policy", {}).get("freeze_before_dataset") != "D05":
        errors.append("freeze policy must name D05")
    return errors


def assert_stage_allowed(protocol: dict[str, Any], dataset_id: str, stage: str) -> None:
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("invalid clean experiment protocol: " + "; ".join(errors))
    role = protocol["dataset_roles"].get(dataset_id)
    if role is None:
        raise ValueError(f"dataset is not declared in clean protocol: {dataset_id}")
    if stage not in role.get("allowed_stages", []):
        raise PermissionError(f"{dataset_id} is forbidden at stage {stage}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_freeze_manifest(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    project_root = Path(project_root).resolve()
    protocol = load_protocol(protocol_path)
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("; ".join(errors))

    required = protocol["freeze_policy"].get("required_hash_inputs", [])
    if not required:
        raise ValueError("required_hash_inputs cannot be empty")
    missing = [relative for relative in required if not (project_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("cannot freeze; missing inputs: " + ", ".join(missing))

    files = {relative: file_sha256(project_root / relative) for relative in required}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "lock_status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_firewall": {
            "development": ["AWID3_EVIL_TWIN", "D01", "D02", "D03"],
            "post_freeze_only": ["D05"],
            "d05_disclosure": "PREVIOUSLY_INSPECTED_EXTERNAL_TEST",
        },
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["lock_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_freeze_manifest(manifest: dict[str, Any], project_root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("lock_status") != "FROZEN":
        errors.append("lock status is not FROZEN")
    expected_lock = manifest.get("lock_sha256")
    body = {key: value for key, value in manifest.items() if key != "lock_sha256"}
    actual_lock = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_lock != actual_lock:
        errors.append("freeze manifest content hash mismatch")

    project_root = Path(project_root).resolve()
    for relative, expected in manifest.get("files", {}).items():
        path = project_root / relative
        if not path.is_file():
            errors.append(f"frozen input missing: {relative}")
        elif file_sha256(path) != expected:
            errors.append(f"frozen input changed: {relative}")
    return errors
