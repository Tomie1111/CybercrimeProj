"""Ground-truth policy: filename-derived or predicted labels are never verified labels."""
from __future__ import annotations

VERIFIED = {"verified", "independently_verified"}


def is_supervised_eligible(record: dict) -> bool:
    return (bool(record.get("supervised")) and record.get("label_status") in VERIFIED
            and (record.get("original_label") in {"benign", "attack", "mixed"})
            and bool(record.get("ground_truth_rule") or record.get("original_label") in {"benign", "attack"}))


def apply_ground_truth(frame: dict, rule: dict) -> dict:
    """Apply a declared source rule; never infer labels from model output."""
    if rule.get("type") != "awid3_official_filter":
        raise ValueError(f"Unsupported ground-truth rule: {rule.get('type')}")
    attack = (int(rule["frame_start"]) <= int(frame["frame_number"]) <= int(rule["frame_end"])
              and int(frame["packet_length"]) < int(rule["max_frame_length_exclusive"]))
    if rule.get("bssid"):
        attack = attack and str(frame.get("bssid", "")).lower() == str(rule["bssid"]).lower()
    return frame | {"label": int(attack), "class_name": rule["attack_class"] if attack else "normal",
                    "label_provenance": "awid3_official_filter"}


def audit_label(record: dict) -> dict:
    eligible = is_supervised_eligible(record)
    return {"filename": record["file"], "original_label": record.get("original_label"),
            "label_status": record.get("label_status", "unknown"), "label_provenance": record.get("label_provenance", ""),
            "supervised_eligible": eligible,
            "exclusion_reason": "" if eligible else "label_not_independently_verified"}
