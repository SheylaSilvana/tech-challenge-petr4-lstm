from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OHLCVPoint(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "date": "2024-07-19",
            "open": 35.20,
            "high": 35.80,
            "low": 35.10,
            "close": 35.50,
            "volume": 45000000,
        }
    })

    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("high")
    @classmethod
    def _high_ge_low(cls, v: float, info) -> float:
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("high deve ser >= low")
        return v


class PredictRequest(BaseModel):
    history: list[OHLCVPoint] = Field(
        ...,
        min_length=90,
        description="Histórico OHLCV ordenado por data (mais antigo primeiro). Mínimo 90 dias para acomodar lookback + indicadores.",
    )


class PredictByTickerRequest(BaseModel):
    ticker: Optional[str] = Field(default=None, description="Se omitido, usa o ticker do modelo treinado.")
    days_back: int = Field(default=180, ge=90, le=1825)


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    predicted_close: float
    last_close: float
    delta: float
    delta_pct: float
    direction: str
    horizon_days: int
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    ticker: Optional[str] = None
    trained_at: Optional[str] = None
    test_metrics: Optional[dict] = None


class ErrorResponse(BaseModel):
    detail: str
