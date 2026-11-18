"""Capture inventory without copying source PCAPs."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
import yaml
from .labels import audit_label


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def inventory(config_path, output_dir, scan=False):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    root = (config_path.parent / cfg["dataset_root"]).resolve()
    records, provenance = [], []
    for source in cfg.get("sources", []) + cfg.get("external_sources", []):
        path = root / source["file"]
        row = {"filename": source["file"], "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
               "dataset": source.get("dataset"), "capture_campaign": source.get("campaign"),
               "session": source.get("session"), "physical_device": source.get("physical_device"),
               "bssid": source.get("bssid"), "ssid": source.get("ssid"), "attack_scenario": source.get("scenario"),
               "original_class_label": source.get("original_label"), "label_provenance": source.get("label_provenance"),
               "capture_start": None, "capture_end": None, "beacon_frame_count": None,
               "phy_capability": None, "security_configuration": None, "radiotap_metadata": None,
               "inventory_status": "metadata_only"}
        if scan:
            row.update(scan_capture(path))
        records.append(row)
        if "label_status" in source: provenance.append(audit_label(source))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir/"dataset_inventory.csv", index=False)
    pd.DataFrame(provenance).to_csv(output_dir/"label_provenance.csv", index=False)
    return records


def scan_capture(path: Path) -> dict:
    from scapy.utils import PcapReader
    from scapy.layers.dot11 import Dot11Beacon, RadioTap
    count, first, last, radio = 0, None, None, False
    with PcapReader(str(path)) as reader:
        for packet in reader:
            ts = float(packet.time); first = ts if first is None else first; last = ts
            if packet.haslayer(Dot11Beacon): count += 1
            radio = radio or packet.haslayer(RadioTap)
    return {"capture_start": first, "capture_end": last, "beacon_frame_count": count,
            "radiotap_metadata": radio, "inventory_status": "scanned"}
