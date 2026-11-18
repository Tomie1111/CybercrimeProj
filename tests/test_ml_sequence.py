import numpy as np
import pandas as pd

from evil_twin.ml.sequence import LSTMConfig, build_causal_sequences, make_lstm


def test_causal_sequences_do_not_cross_capture_or_bssid():
    frame = pd.DataFrame(
        {
            "source_file": ["a", "a", "a", "a", "a", "b", "b"],
            "bssid": ["x", "x", "x", "y", "y", "x", "x"],
            "window_start": [3, 1, 2, 1, 2, 1, 2],
            "label": [1, 0, 1, 0, 0, 1, 1],
        }
    )
    values = np.arange(len(frame), dtype=np.float32)[:, None]
    sequences, labels, metadata = build_causal_sequences(values, frame, 2)
    assert sequences[:, :, 0].tolist() == [[1, 2], [2, 0], [3, 4], [5, 6]]
    assert labels.tolist() == [1, 1, 0, 1]
    assert metadata[["source_file", "bssid"]].values.tolist() == [
        ["a", "x"], ["a", "x"], ["a", "y"], ["b", "x"]
    ]


def test_lstm_forward_returns_one_logit_per_sequence():
    import torch

    model = make_lstm(LSTMConfig(input_size=3, hidden_size=4))
    output = model(torch.zeros((2, 5, 3), dtype=torch.float32))
    assert tuple(output.shape) == (2,)
