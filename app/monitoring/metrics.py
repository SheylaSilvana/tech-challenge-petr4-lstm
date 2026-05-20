"""Métricas Prometheus customizadas."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

prediction_counter = Counter(
    "lstm_predictions_total",
    "Total de predições realizadas",
    labelnames=("endpoint", "status"),
)

prediction_latency = Histogram(
    "lstm_prediction_latency_seconds",
    "Latência de inferência do LSTM",
    labelnames=("endpoint",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

predicted_price_gauge = Gauge(
    "lstm_last_predicted_close",
    "Último preço de fechamento previsto",
    labelnames=("ticker",),
)

model_loaded_gauge = Gauge(
    "lstm_model_loaded",
    "1 se o modelo está carregado, 0 caso contrário",
)
