"""Non-overlapping/strided causal Beacon windows with explicit membership."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
from .relational import conflict_features
from .signal import signal_features
from .temporal import temporal_features

PROFILE_KEYS = ("bssid", "security_profile", "phy_family")


def _mode(values):
    return max(set(values), key=values.count) if values else None


def aggregate(rows, view="profile") -> dict:
    numeric = ["capability","ess","privacy","short_preamble","short_slot",
               "ie_count","unique_ie_count","duplicate_ie_count","ie_length_mean","rate_count","basic_rate_count",
               "beacon_interval","rsn_present","pmf_capable","pmf_required","rsnxe_present","vht_present",
               "vendor_ie_count","vendor_length_mean","vendor_length_min","vendor_length_max",
               "vendor_oui_diversity","vendor_type_diversity","wps_present","advertised_channel"]
    out = {"view": view, "bssid": rows[0]["bssid"], "source_file": rows[0]["source_file"],
           "window_start": rows[0]["timestamp"], "window_end": rows[-1]["timestamp"], "frame_count": len(rows),
           "frame_membership": "|".join(str(x["frame_number"]) for x in rows),
           "membership_hash": hashlib.sha256("|".join(f'{x["source_file"]}:{x["frame_number"]}' for x in rows).encode()).hexdigest(),
           "phy_family": _mode([x["phy_family"] for x in rows]), "label": _mode([x.get("label") for x in rows]),
           "label_purity": max([x.get("label") for x in rows].count(v) for v in set(x.get("label") for x in rows))/len(rows)}
    for key in numeric:
        values = [float(x[key]) for x in rows if x.get(key) is not None]
        out[key+"_mean"] = float(np.mean(values)) if values else None
        out[key+"_stability"] = 1.0/(1.0+float(np.std(values))) if values else None
    for key in ["group_cipher","pairwise_ciphers","akm_suites","he_state","eht_state",
                "security_profile", "vendor_profile", "ie_structure"]:
        vals = [x.get(key) for x in rows]
        out[key+"_change_count"] = sum(a != b for a,b in zip(vals, vals[1:]))
        out[key+"_stability"] = max(vals.count(v) for v in set(vals))/len(vals)
    out.update(temporal_features(rows)); out.update(signal_features(rows))
    if view == "identity": out.update(conflict_features(rows))
    return out


def make_windows(frames, size=20, stride=None, view="profile", partial="drop"):
    stride = stride or size
    grouped = defaultdict(list)
    for row in frames:
        key = tuple(row.get(k) for k in (PROFILE_KEYS if view=="profile" else ("bssid",)))
        grouped[key].append(row)
    result = []
    for _, rows in grouped.items():
        rows.sort(key=lambda x:(x["timestamp"], x["frame_number"] or -1))
        for start in range(0, len(rows), stride):
            chunk = rows[start:start+size]
            if len(chunk)<size and partial=="drop": break
            result.append(aggregate(chunk, view))
    return result
