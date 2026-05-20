"""Pipeline de treinamento do LSTM.

Uso: python -m ml_training.train --ticker PETR4.SA --epochs 50
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from app.core.config import BASE_DIR
from app.core.logging import configure_logging
from app.ml.data import download_history, temporal_split
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN, add_technical_indicators, make_sequences
from app.ml.model import build_lstm, default_callbacks, regression_metrics

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina LSTM para previsão de preço de ações")
    parser.add_argument("--ticker", default="PETR4.SA")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--model-name", default="lstm_petr4")
    return parser.parse_args()


def main() -> None:
    configure_logging("INFO")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando histórico", extra={"ticker": args.ticker, "start": args.start})
    raw = download_history(args.ticker, start=args.start, end=args.end)
    logger.info("Dados baixados", extra={"rows": len(raw)})

    enriched = add_technical_indicators(raw)
    logger.info("Features adicionadas", extra={"rows": len(enriched), "features": len(FEATURE_COLUMNS)})

    split = temporal_split(enriched, train_ratio=0.7, val_ratio=0.15)
    logger.info(
        "Split temporal aplicado",
        extra={"train": len(split.train), "val": len(split.val), "test": len(split.test)},
    )

    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(split.train[FEATURE_COLUMNS])
    val_scaled = scaler.transform(split.val[FEATURE_COLUMNS])
    test_scaled = scaler.transform(split.test[FEATURE_COLUMNS])

    X_train, y_train = make_sequences(train_scaled, target_idx, args.sequence_length)
    X_val, y_val = make_sequences(val_scaled, target_idx, args.sequence_length)
    X_test, y_test = make_sequences(test_scaled, target_idx, args.sequence_length)

    logger.info(
        "Sequências construídas",
        extra={"X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape},
    )

    model = build_lstm(
        sequence_length=args.sequence_length,
        n_features=len(FEATURE_COLUMNS),
    )
    model.summary(print_fn=lambda s: logger.info(s))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=default_callbacks(),
        verbose=2,
    )

    def _inverse_close(scaled: np.ndarray) -> np.ndarray:
        data_min = scaler.data_min_[target_idx]
        data_max = scaler.data_max_[target_idx]
        return scaled * (data_max - data_min) + data_min

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = _inverse_close(y_pred_scaled)
    y_true = _inverse_close(y_test)
    test_metrics = regression_metrics(y_true, y_pred)

    y_val_pred = _inverse_close(model.predict(X_val, verbose=0).flatten())
    y_val_true = _inverse_close(y_val)
    val_metrics = regression_metrics(y_val_true, y_val_pred)

    logger.info("Métricas validação", extra=val_metrics)
    logger.info("Métricas teste", extra=test_metrics)

    model_path = output_dir / f"{args.model_name}.keras"
    scaler_path = output_dir / f"scaler_{args.model_name.replace('lstm_', '')}.joblib"
    meta_path = output_dir / f"meta_{args.model_name.replace('lstm_', '')}.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "ticker": args.ticker,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
        "sequence_length": args.sequence_length,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "data_range": {
            "start": str(enriched.index.min().date()),
            "end": str(enriched.index.max().date()),
        },
        "split": {
            "train_rows": len(split.train),
            "val_rows": len(split.val),
            "test_rows": len(split.test),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_history": {
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Treino concluído",
        extra={"model_path": str(model_path), "meta_path": str(meta_path)},
    )


if __name__ == "__main__":
    main()
