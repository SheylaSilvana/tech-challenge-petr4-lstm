from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.ml.data import download_history
from app.monitoring.metrics import (
    prediction_counter,
    prediction_latency,
    predicted_price_gauge,
)
from app.schemas.prediction import (
    HealthResponse,
    PredictByTickerRequest,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute `python -m ml_training.train` antes.",
        )
    return predictor


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)
    meta = predictor.meta
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=meta.get("model_version"),
        ticker=meta.get("ticker"),
        trained_at=meta.get("trained_at"),
        test_metrics=meta.get("test_metrics"),
    )


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Prevê o fechamento do próximo dia usando histórico OHLCV enviado no body."""
    predictor = _get_predictor(request)

    rows = [p.model_dump() for p in payload.history]
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    start = time.perf_counter()
    try:
        result = predictor.predict_next_close(df)
    except ValueError as exc:
        prediction_counter.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_s = time.perf_counter() - start
    elapsed_ms = elapsed_s * 1000

    ticker_label = predictor.meta.get("ticker", settings.ticker)
    prediction_counter.labels(endpoint="predict", status="success").inc()
    prediction_latency.labels(endpoint="predict").observe(elapsed_s)
    predicted_price_gauge.labels(ticker=ticker_label).set(result.predicted_close)

    logger.info(
        "Predição realizada",
        extra={
            "predicted_close": result.predicted_close,
            "last_close": result.last_close,
            "inference_ms": round(elapsed_ms, 2),
        },
    )

    return PredictResponse(
        ticker=ticker_label,
        predicted_close=round(result.predicted_close, 4),
        last_close=round(result.last_close, 4),
        delta=round(result.delta, 4),
        delta_pct=round(result.delta_pct, 4),
        direction="up" if result.delta >= 0 else "down",
        horizon_days=result.horizon_days,
        model_version=result.model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.post("/predict/by-ticker", response_model=PredictResponse, tags=["prediction"])
def predict_by_ticker(payload: PredictByTickerRequest, request: Request) -> PredictResponse:
    """Predição buscando histórico automaticamente do yfinance."""
    predictor = _get_predictor(request)
    ticker = payload.ticker or predictor.meta.get("ticker", settings.ticker)

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(payload.days_back * 1.6) + 30)

    try:
        df = download_history(ticker, start=start.isoformat(), end=end.isoformat())
    except Exception as exc:
        prediction_counter.labels(endpoint="predict_by_ticker", status="error").inc()
        raise HTTPException(status_code=502, detail=f"Falha ao baixar dados: {exc}") from exc

    t0 = time.perf_counter()
    try:
        result = predictor.predict_next_close(df)
    except ValueError as exc:
        prediction_counter.labels(endpoint="predict_by_ticker", status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_s = time.perf_counter() - t0
    elapsed_ms = elapsed_s * 1000

    prediction_counter.labels(endpoint="predict_by_ticker", status="success").inc()
    prediction_latency.labels(endpoint="predict_by_ticker").observe(elapsed_s)
    predicted_price_gauge.labels(ticker=ticker).set(result.predicted_close)

    return PredictResponse(
        ticker=ticker,
        predicted_close=round(result.predicted_close, 4),
        last_close=round(result.last_close, 4),
        delta=round(result.delta, 4),
        delta_pct=round(result.delta_pct, 4),
        direction="up" if result.delta >= 0 else "down",
        horizon_days=result.horizon_days,
        model_version=result.model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )
