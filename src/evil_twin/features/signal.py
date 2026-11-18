"""Receiver-dependent radio aggregation."""
from __future__ import annotations
import numpy as np


def signal_features(rows) -> dict:
    def vals(k): return [float(x[k]) for x in rows if x.get(k) is not None]
    rssi, noise, snr = vals("rssi"), vals("noise"), vals("snr")
    advertised = [x.get("advertised_channel") for x in rows if x.get("advertised_channel") is not None]
    observed = [x.get("observed_frequency") for x in rows if x.get("observed_frequency") is not None]
    return {
        "rssi_mean": float(np.mean(rssi)) if rssi else None, "rssi_median": float(np.median(rssi)) if rssi else None,
        "rssi_std": float(np.std(rssi)) if rssi else None, "rssi_range": float(np.ptp(rssi)) if rssi else None,
        "noise_mean": float(np.mean(noise)) if noise else None, "snr_mean": float(np.mean(snr)) if snr else None,
        "observed_frequency": float(np.median(observed)) if observed else None,
        "advertised_channel_stability": len(set(advertised)) == 1 if advertised else None,
        "rssi_availability": len(rssi)/len(rows), "radiotap_availability": np.mean([x.get("radiotap_available",0) for x in rows]),
    }
