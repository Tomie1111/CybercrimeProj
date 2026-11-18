"""Causal sequence helpers and a small deterministic LSTM classifier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_causal_sequences(
    values: np.ndarray,
    frame: pd.DataFrame,
    sequence_length: int,
    *,
    group_columns: tuple[str, ...] = ("source_file", "bssid"),
    time_column: str = "window_start",
    label_column: str = "label",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Make past-to-current sequences without crossing capture/BSSID boundaries.

    The label and metadata belong to the last (current) window in each sequence.
    No padding is used, so the first ``sequence_length - 1`` windows of every
    group are intentionally ineligible.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) != len(frame):
        raise ValueError("values and frame must contain the same number of rows")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    required = {*group_columns, time_column, label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sequence columns: {missing}")

    work = frame.reset_index(drop=True).copy()
    work["_value_position"] = np.arange(len(work))
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict] = []
    for _, group in work.groupby(list(group_columns), sort=False, dropna=False):
        ordered = group.sort_values(time_column, kind="stable")
        positions = ordered["_value_position"].to_numpy(dtype=int)
        for end in range(sequence_length - 1, len(ordered)):
            positions_now = positions[end - sequence_length + 1 : end + 1]
            sequences.append(values[positions_now])
            row = ordered.iloc[end].drop(labels=["_value_position"])
            labels.append(int(row[label_column]))
            metadata.append(row.to_dict())

    feature_count = values.shape[1] if values.ndim == 2 else 0
    if not sequences:
        return (
            np.empty((0, sequence_length, feature_count), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            pd.DataFrame(columns=frame.columns),
        )
    return np.stack(sequences), np.asarray(labels, dtype=np.int64), pd.DataFrame(metadata)


@dataclass(frozen=True)
class LSTMConfig:
    input_size: int
    hidden_size: int = 16
    layers: int = 1
    dropout: float = 0.0


def make_lstm(config: LSTMConfig):
    """Construct the torch module lazily so the core package stays lightweight."""
    import torch
    from torch import nn

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.layers,
                dropout=config.dropout if config.layers > 1 else 0.0,
                batch_first=True,
            )
            self.output = nn.Linear(config.hidden_size, 1)

        def forward(self, inputs):
            encoded, _ = self.lstm(inputs)
            return self.output(encoded[:, -1, :]).squeeze(-1)

    return LSTMClassifier()
