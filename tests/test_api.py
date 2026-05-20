"""Testes da API com predictor mockado (evita carregar modelo real)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass
class _StubResult:
    predicted_close: float = 36.5
    last_close: float = 35.5
    delta: float = 1.0
    delta_pct: float = 2.82
    horizon_days: int = 1
    model_version: str = "test"


class _StubPredictor:
    meta = {
        "ticker": "PETR4.SA",
        "model_version": "test",
        "trained_at": "2024-01-01T00:00:00Z",
        "test_metrics": {"mae": 0.5, "rmse": 0.8, "mape": 1.2},
    }

    def predict_next_close(self, df):
        return _StubResult()


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        # Stub injetado pós-lifespan para evitar carga do modelo real
        app.state.predictor = _StubPredictor()
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "lstm-stock-predictor"


def test_health_with_model(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["ticker"] == "PETR4.SA"


def test_predict_with_minimum_history(client):
    history = [
        {
            "date": f"2024-01-{(i % 28) + 1:02d}",
            "open": 30.0 + i * 0.01,
            "high": 30.5 + i * 0.01,
            "low": 29.5 + i * 0.01,
            "close": 30.2 + i * 0.01,
            "volume": 1_000_000,
        }
        for i in range(100)
    ]
    r = client.post("/api/v1/predict", json={"history": history})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "PETR4.SA"
    assert body["predicted_close"] == 36.5
    assert body["direction"] == "up"
    assert "inference_time_ms" in body


def test_predict_rejects_short_history(client):
    history = [
        {
            "date": f"2024-01-{(i % 28) + 1:02d}",
            "open": 30.0,
            "high": 30.5,
            "low": 29.5,
            "close": 30.2,
            "volume": 1_000_000,
        }
        for i in range(10)
    ]
    r = client.post("/api/v1/predict", json={"history": history})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "lstm_model_loaded" in r.text or "python_info" in r.text
