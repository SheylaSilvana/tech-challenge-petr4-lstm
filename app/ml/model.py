"""Modelo LSTM e métricas de regressão."""
from __future__ import annotations

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm(
    sequence_length: int,
    n_features: int,
    units: tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(units[0], return_sequences=True),
            Dropout(dropout),
            LSTM(units[1]),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def default_callbacks() -> list:
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}
