"""Wrapper de inferência."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN, add_technical_indicators

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_close: float
    last_close: float
    delta: float
    delta_pct: float
    horizon_days: int
    model_version: str


class StockPredictor:
    def __init__(self, model_path: Path, scaler_path: Path, meta_path: Path) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.meta_path = Path(meta_path)

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.sequence_length: int = self.meta["sequence_length"]
        self.feature_columns: list[str] = self.meta.get("feature_columns", FEATURE_COLUMNS)
        self.target_idx: int = self.feature_columns.index(TARGET_COLUMN)
        self.model_version: str = self.meta.get("model_version", "1.0.0")

        logger.info(
            "Predictor inicializado",
            extra={"model_version": self.model_version, "sequence_length": self.sequence_length},
        )

    def _prepare_sequence(self, ohlcv: pd.DataFrame) -> np.ndarray:
        if len(ohlcv) < self.sequence_length + 30:
            raise ValueError(
                f"Histórico insuficiente: precisa de pelo menos {self.sequence_length + 30} dias "
                f"(recebidos {len(ohlcv)}). Indicadores técnicos exigem janela extra."
            )

        with_features = add_technical_indicators(ohlcv)
        if len(with_features) < self.sequence_length:
            raise ValueError(
                f"Após cálculo de indicadores restaram {len(with_features)} dias úteis; "
                f"necessários {self.sequence_length}."
            )

        window = with_features[self.feature_columns].iloc[-self.sequence_length:].values
        scaled = self.scaler.transform(window)
        return scaled.reshape(1, self.sequence_length, len(self.feature_columns))

    def _inverse_close(self, scaled_close: float) -> float:
        data_min = self.scaler.data_min_[self.target_idx]
        data_max = self.scaler.data_max_[self.target_idx]
        return float(scaled_close * (data_max - data_min) + data_min)

    def predict_next_close(self, ohlcv: pd.DataFrame) -> PredictionResult:
        x = self._prepare_sequence(ohlcv)
        scaled_pred = float(self.model.predict(x, verbose=0)[0, 0])
        predicted = self._inverse_close(scaled_pred)
        last_close = float(ohlcv["Close"].iloc[-1])
        delta = predicted - last_close
        delta_pct = (delta / last_close) * 100 if last_close else 0.0

        return PredictionResult(
            predicted_close=predicted,
            last_close=last_close,
            delta=delta,
            delta_pct=delta_pct,
            horizon_days=1,
            model_version=self.model_version,
        )
