"""Wrap-aware, causal temporal aggregations."""
from __future__ import annotations

import numpy as np


def wrapped_deltas(values, modulus: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.int64)
    if len(a) < 2:
        return np.asarray([], dtype=np.int64)
    raw = np.diff(a)
    return np.where(raw < -(modulus // 2), raw + modulus, raw)


def temporal_features(rows) -> dict:
    times = np.asarray([float(x["timestamp"]) for x in rows])
    tsfs = np.asarray([int(x["tsf"]) for x in rows], dtype=np.uint64)
    seq = wrapped_deltas([int(x["sequence"]) for x in rows], 4096)
    dt = np.diff(times)
    tsf_raw = np.diff(tsfs.astype(object)) if len(tsfs) > 1 else np.asarray([])
    expected = max(len(seq), 1)
    return {
        "window_duration": float(times[-1]-times[0]) if len(times)>1 else 0.0,
        "interarrival_mean": float(np.mean(dt)) if len(dt) else None,
        "interarrival_var": float(np.var(dt)) if len(dt) else None,
        "interarrival_cv": float(np.std(dt)/np.mean(dt)) if len(dt) and np.mean(dt)>0 else None,
        "tsf_backward_jumps": int(sum(x < 0 for x in tsf_raw)),
        "tsf_reset_candidates": int(sum(x < -1_000_000 for x in tsf_raw)),
        "sequence_discontinuity_rate": float(sum(x != 1 for x in seq)/expected),
        "duplicate_sequence_count": int(sum(x == 0 for x in seq)),
        "missing_beacon_estimate": int(sum(max(int(x)-1, 0) for x in seq)),
        "retry_rate": float(np.mean([x.get("retry", 0) for x in rows])),
        "timing_valid": int(len(dt)>0 and np.all(dt>=0)),
    }
