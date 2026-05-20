"""Coleta de dados e separação temporal."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def download_history(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    max_attempts: int = 4,
    backoff_seconds: float = 1.5,
) -> pd.DataFrame:
    # auto_adjust=True remove descontinuidades de dividendos/splits que confundem o LSTM.
    # Retry com backoff porque o Yahoo Finance ocasionalmente retorna vazio para
    # tickers .SA quando consultado de IPs cloud (Render, Heroku, etc).
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("yfinance erro tentativa %d/%d para %s: %s", attempt, max_attempts, ticker, exc)
        else:
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                return df.sort_index()
            logger.warning("yfinance retornou vazio tentativa %d/%d para %s", attempt, max_attempts, ticker)

        if attempt < max_attempts:
            time.sleep(backoff_seconds * attempt)

    if last_exc is not None:
        raise ValueError(f"Sem dados retornados para {ticker} após {max_attempts} tentativas: {last_exc}")
    raise ValueError(f"Sem dados retornados para {ticker} após {max_attempts} tentativas")


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> TemporalSplit:
    """Split cronológico estrito: treino → validação → teste, sem vazamento futuro."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return TemporalSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )
