from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import FEATURE_COLUMNS, add_technical_indicators, make_sequences


def _synthetic_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": base + rng.normal(0, 0.3, n),
            "High": base + np.abs(rng.normal(0, 0.5, n)),
            "Low": base - np.abs(rng.normal(0, 0.5, n)),
            "Close": base,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_indicators_have_expected_columns():
    df = _synthetic_ohlcv()
    out = add_technical_indicators(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"Coluna {col} ausente"


def test_indicators_drop_initial_nans():
    df = _synthetic_ohlcv()
    out = add_technical_indicators(df)
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_make_sequences_shapes():
    df = _synthetic_ohlcv()
    out = add_technical_indicators(df)
    data = out[FEATURE_COLUMNS].values.astype("float32")
    target_idx = FEATURE_COLUMNS.index("Close")

    X, y = make_sequences(data, target_idx=target_idx, sequence_length=30)
    assert X.shape[1] == 30
    assert X.shape[2] == len(FEATURE_COLUMNS)
    assert X.shape[0] == len(data) - 30
    assert y.shape[0] == X.shape[0]


def test_make_sequences_target_matches():
    df = _synthetic_ohlcv()
    out = add_technical_indicators(df)
    data = out[FEATURE_COLUMNS].values.astype("float32")
    target_idx = FEATURE_COLUMNS.index("Close")

    seq_len = 20
    X, y = make_sequences(data, target_idx=target_idx, sequence_length=seq_len)
    np.testing.assert_allclose(y[0], data[seq_len, target_idx])
    np.testing.assert_allclose(y[-1], data[-1, target_idx])
