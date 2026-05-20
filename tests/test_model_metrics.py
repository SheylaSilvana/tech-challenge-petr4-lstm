from __future__ import annotations

import numpy as np

from app.ml.model import build_lstm, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = regression_metrics(y, y)
    assert out["mae"] == 0.0
    assert out["rmse"] == 0.0


def test_regression_metrics_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == 7 / 3
    np.testing.assert_allclose(out["rmse"], np.sqrt(17 / 3))


def test_build_lstm_shapes():
    model = build_lstm(sequence_length=30, n_features=13)
    assert model.input_shape == (None, 30, 13)
    assert model.output_shape == (None, 1)
