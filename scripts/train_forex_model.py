"""Utilities for training and exporting the forex LSTM artefacts.

This module provides a deterministic training routine for the EUR to target
currency exchange-rate LSTM that powers ``forex_forecast.py``.  The helper
function :func:`train_and_export_model` is importable so application code can
trigger a retrain if artefacts are missing, and the CLI offers a repeatable
workflow for contributors:

.. code-block:: bash

    $ python scripts/train_forex_model.py --currency USD

Running the command above will re-create ``models/forex_lstm.keras`` and
``models/forex_scaler.pkl`` using the historical daily rates in
``data/daily_forex_rates.csv``.
"""

from __future__ import annotations

import argparse
import os
import random
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

DEFAULT_DATA_PATH = Path("data/daily_forex_rates.csv")
DEFAULT_MODEL_PATH = Path("models/forex_lstm.keras")
DEFAULT_SCALER_PATH = Path("models/forex_scaler.pkl")
DEFAULT_BASE_CURRENCY = "EUR"
DEFAULT_LOOKBACK = 30
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_SEED = 42


def _set_global_determinism(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for deterministic behaviour."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:  # TensorFlow < 2.10 does not provide the deterministic helper
        tf.config.experimental.enable_op_determinism()  # type: ignore[attr-defined]
    except Exception:
        pass


def _load_currency_series(
    data_path: Path, currency: str, base_currency: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load a currency time series sorted by date from the historical dataset."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find forex history at {data_path}. Did you download the dataset?"
        )

    df = pd.read_csv(data_path, parse_dates=["date"])
    filtered = df[(df["currency"] == currency) & (df["base_currency"] == base_currency)]
    if filtered.empty:
        raise ValueError(
            f"No rows found for {currency}/{base_currency} in {data_path}. "
            "Update the dataset or choose another currency."
        )

    filtered = filtered.sort_values("date")
    values = filtered["exchange_rate"].astype("float32").to_numpy().reshape(-1, 1)
    return filtered, values


def _build_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling window sequences for the LSTM."""

    if len(values) <= lookback:
        raise ValueError(
            "Not enough data points to build sequences. "
            f"Need more than {lookback} rows, received {len(values)}."
        )

    x_samples = []
    y_targets = []
    for idx in range(lookback, len(values)):
        x_samples.append(values[idx - lookback : idx])
        y_targets.append(values[idx])

    return np.array(x_samples, dtype="float32"), np.array(y_targets, dtype="float32")


def _build_model(lookback: int) -> tf.keras.Model:
    """Instantiate the LSTM network used for forecasting."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(64, activation="tanh"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def train_and_export_model(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    currency: str = "USD",
    base_currency: str = DEFAULT_BASE_CURRENCY,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    seed: int = DEFAULT_SEED,
) -> None:
    """Train the forex LSTM and export the model and scaler artefacts."""

    _set_global_determinism(seed)

    series_frame, raw_values = _load_currency_series(data_path, currency, base_currency)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(raw_values)
    x_all, y_all = _build_sequences(scaled_values, lookback)

    split_idx = int(len(x_all) * train_split)
    if split_idx == 0:
        raise ValueError(
            "Training split produced no training samples. Increase the train_split or "
            "extend the historical dataset."
        )

    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    if len(x_val) == 0:
        x_val, y_val = x_train, y_train

    model = _build_model(lookback)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
        shuffle=False,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path, include_optimizer=False)
    with scaler_path.open("wb") as handle:
        pickle.dump(
            {
                "scaler": scaler,
                "lookback": lookback,
                "currency": currency,
                "base_currency": base_currency,
                "training_rows": len(series_frame),
            },
            handle,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the forex LSTM model and export artefacts to the models/ directory. "
            "Use this command whenever the saved model or scaler is missing."
        )
    )
    parser.add_argument(
        "--currency",
        default="USD",
        help="Target currency to forecast relative to the EUR base (default: USD).",
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the historical forex CSV (default: data/daily_forex_rates.csv).",
    )
    parser.add_argument(
        "--base-currency",
        default=DEFAULT_BASE_CURRENCY,
        help="Base currency column to condition on (default: EUR).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Destination for the exported Keras model (default: models/forex_lstm.keras).",
    )
    parser.add_argument(
        "--scaler-path",
        default=str(DEFAULT_SCALER_PATH),
        help="Destination for the exported scaler pickle (default: models/forex_scaler.pkl).",
    )
    parser.add_argument(
        "--lookback", type=int, default=DEFAULT_LOOKBACK, help="Number of timesteps per sample."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size."
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Fraction of samples to use for training (remainder is validation).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic runs."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_and_export_model(
        data_path=Path(args.data_path),
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
        currency=args.currency,
        base_currency=args.base_currency,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
