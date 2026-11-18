"""PCAP extraction and inference using the frozen training artifact."""
from __future__ import annotations
from pathlib import Path
import joblib, pandas as pd
from scapy.utils import PcapReader
from scapy.utils import RawPcapReader
from scapy.layers.dot11 import RadioTap
from ..features.beacon import parse_beacon
from ..features.windows import make_windows
from ..data.labels import apply_ground_truth


def extract_frames(path, label=None, limit=None):
    rows=[]
    with PcapReader(str(path)) as reader:
        for number,packet in enumerate(reader,1):
            row=parse_beacon(packet,Path(path).name,number)
            if row is not None:
                row["label"]=label; rows.append(row)
                if limit and len(rows)>=limit: break
    return rows


def extract_labeled_frames(path, ground_truth_rule, limit=None):
    """Fast raw-PCAP pass: decode only Beacon frames and retain original frame numbers."""
    rows=[]
    reader=RawPcapReader(str(path))
    try:
        for number,(raw,meta) in enumerate(reader,1):
            if len(raw)<5:
                continue
            radiotap_length=int.from_bytes(raw[2:4],"little")
            if radiotap_length>=len(raw) or (raw[radiotap_length] & 0xFC) != 0x80:
                continue
            try:
                packet=RadioTap(raw)
                packet.time=meta.sec + meta.usec/1_000_000
                row=parse_beacon(packet,Path(path).name,number,meta.wirelen,getattr(meta,"caplen",len(raw)))
            except Exception:
                row=None
            if row is not None:
                rows.append(apply_ground_truth(row,ground_truth_rule))
                if limit and len(rows)>=limit:
                    break
    finally:
        reader.close()
    return rows


def extract_frames_fast(path,limit=None):
    """Decode only Beacon frames while preserving original PCAP frame numbers."""
    rows=[]; reader=RawPcapReader(str(path))
    try:
        for number,(raw,meta) in enumerate(reader,1):
            if len(raw)<5:
                continue
            radiotap_length=int.from_bytes(raw[2:4],"little")
            if radiotap_length>=len(raw) or (raw[radiotap_length] & 0xFC) != 0x80:
                continue
            try:
                packet=RadioTap(raw); packet.time=meta.sec + meta.usec/1_000_000
                row=parse_beacon(packet,Path(path).name,number,meta.wirelen,getattr(meta,"caplen",len(raw)))
            except Exception:
                row=None
            if row is not None:
                row["label"]=None; rows.append(row)
                if limit and len(rows)>=limit: break
    finally:
        reader.close()
    return rows


def extract_labeled_windows(path,ground_truth_rule,size=20,stride=None,view="profile",partial="drop"):
    frames=extract_labeled_frames(path,ground_truth_rule)
    return pd.DataFrame(make_windows(frames,size,stride,view,partial))


def extract_windows(path,size=20,stride=None,view="profile",label=None,limit=None):
    return pd.DataFrame(make_windows(extract_frames(path,label,limit),size,stride,view))


def predict_pcap(path,artifact_path,size=20):
    artifact=joblib.load(artifact_path); windows=extract_windows(path,size=size)
    if windows.empty: return windows
    probs=artifact["pipeline"].predict_proba(windows)[:,1]
    windows["attack_probability"]=probs
    windows["prediction"]=(probs>=artifact["threshold"]).astype(int)
    return windows
