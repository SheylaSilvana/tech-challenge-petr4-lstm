"""Feature engineering para séries temporais de ações."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "Return_1d",
    "SMA_7",
    "SMA_21",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "Volatility_10",
]

TARGET_COLUMN = "Close"


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["Return_1d"] = close.pct_change()
    out["SMA_7"] = close.rolling(window=7).mean()
    out["SMA_21"] = close.rolling(window=21).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    out["RSI_14"] = _rsi(close, window=14)

    macd, signal = _macd(close)
    out["MACD"] = macd
    out["MACD_Signal"] = signal

    upper, lower = _bollinger(close)
    out["BB_Upper"] = upper
    out["BB_Lower"] = lower

    out["Volatility_10"] = close.pct_change().rolling(window=10).std()

    return out.dropna()


def make_sequences(
    data: np.ndarray,
    target_idx: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes (X, y): X são `sequence_length` dias, y é o `Close` do dia seguinte."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])
        y.append(data[i, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
