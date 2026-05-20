from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.monitoring.metrics import model_loaded_gauge

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)

    try:
        from app.ml.predictor import StockPredictor

        app.state.predictor = StockPredictor(
            model_path=settings.model_path,
            scaler_path=settings.scaler_path,
            meta_path=settings.meta_path,
        )
        model_loaded_gauge.set(1)
        logger.info("Modelo carregado com sucesso")
    except FileNotFoundError as exc:
        app.state.predictor = None
        model_loaded_gauge.set(0)
        logger.warning(
            "Modelo não encontrado — API em modo degradado",
            extra={"missing_path": str(exc)},
        )
    except Exception:
        app.state.predictor = None
        model_loaded_gauge.set(0)
        logger.exception("Falha ao carregar modelo")

    yield


app = FastAPI(
    title="LSTM Stock Predictor API",
    description="API para previsão de preço de fechamento de ações usando LSTM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "lstm-stock-predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "metrics": "/metrics",
    }
