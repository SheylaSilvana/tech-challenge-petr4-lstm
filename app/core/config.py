"""Configurações via variáveis de ambiente."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    ticker: str = "PETR4.SA"

    model_path: Path = BASE_DIR / "models" / "lstm_petr4.keras"
    scaler_path: Path = BASE_DIR / "models" / "scaler_petr4.joblib"
    meta_path: Path = BASE_DIR / "models" / "meta_petr4.json"

    sequence_length: int = 60

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    enable_metrics: bool = True


settings = Settings()
