"""Forex forecasting helpers that load the trained LSTM artefacts.

The module checks for the presence of ``models/forex_lstm.keras`` and
``models/forex_scaler.pkl`` before attempting to load them.  When the files are
missing you will receive an actionable error instructing you to run the
training routine.  Passing ``auto_train=True`` to :func:`load_model_bundle` or
:func:`forecast_next_rate` will automatically invoke
``scripts.train_forex_model.train_and_export_model`` using the repository's
historical dataset.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

DEFAULT_MODEL_PATH = Path("models/forex_lstm.keras")
DEFAULT_SCALER_PATH = Path("models/forex_scaler.pkl")
DEFAULT_DATA_PATH = Path("data/daily_forex_rates.csv")
DEFAULT_CURRENCY = "USD"
DEFAULT_BASE_CURRENCY = "EUR"


class ArtefactMissingError(FileNotFoundError):
    """Raised when the persisted forecasting artefacts are missing."""


def ensure_artefacts(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    auto_train: bool = False,
    training_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Verify that the forex forecasting artefacts exist.

    When ``auto_train`` is ``True`` the training routine from
    :mod:`scripts.train_forex_model` is executed to regenerate the artefacts.
    Otherwise an :class:`ArtefactMissingError` is raised with actionable
    guidance.
    """

    missing = [path for path in (model_path, scaler_path) if not path.exists()]
    if not missing:
        return model_path, scaler_path

    guidance = (
        "Missing forex forecasting artefacts: "
        + ", ".join(str(path) for path in missing)
        + "\nRun `python scripts/train_forex_model.py --currency USD` to regenerate "
        "them using the bundled historical data."
    )

    if auto_train:
        kwargs = dict(training_kwargs or {})
        kwargs.setdefault("model_path", model_path)
        kwargs.setdefault("scaler_path", scaler_path)
        kwargs.setdefault("data_path", DEFAULT_DATA_PATH)
        kwargs.setdefault("currency", DEFAULT_CURRENCY)
        kwargs.setdefault("base_currency", DEFAULT_BASE_CURRENCY)
        from scripts.train_forex_model import train_and_export_model

        train_and_export_model(**kwargs)
        return model_path, scaler_path

    raise ArtefactMissingError(guidance)


def _load_scaler_metadata(scaler_path: Path) -> Tuple[MinMaxScaler, Dict[str, Any]]:
    with scaler_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "scaler" in payload:
        return payload["scaler"], payload

    raise ValueError(
        "The scaler file does not contain the expected metadata dictionary. "
        "Re-run `python scripts/train_forex_model.py` to regenerate it."
    )


def load_model_bundle(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    auto_train: bool = False,
    training_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[keras.Model, MinMaxScaler, Dict[str, Any]]:
    """Load the trained model, scaler, and metadata required for forecasting."""

    model_path, scaler_path = ensure_artefacts(
        model_path=model_path,
        scaler_path=scaler_path,
        auto_train=auto_train,
        training_kwargs=training_kwargs,
    )

    model = keras.models.load_model(model_path)
    scaler, metadata = _load_scaler_metadata(scaler_path)
    return model, scaler, metadata


def _latest_sequence(
    *,
    data_path: Path,
    currency: str,
    base_currency: str,
    lookback: int,
) -> Tuple[np.ndarray, pd.Timestamp]:
    frame = pd.read_csv(data_path, parse_dates=["date"])
    filtered = frame[(frame["currency"] == currency) & (frame["base_currency"] == base_currency)]
    if len(filtered) < lookback:
        raise ValueError(
            "Not enough historical rows to build a lookback window. "
            f"Need {lookback}, found {len(filtered)}."
        )

    filtered = filtered.sort_values("date")
    tail = filtered.tail(lookback)
    values = tail["exchange_rate"].astype("float32").to_numpy().reshape(-1, 1)
    return values, tail["date"].iloc[-1]


def forecast_next_rate(
    *,
    steps: int = 1,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    auto_train: bool = False,
    training_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forecast the next ``steps`` exchange-rate values."""

    model, scaler, metadata = load_model_bundle(
        model_path=model_path,
        scaler_path=scaler_path,
        auto_train=auto_train,
        training_kwargs=training_kwargs,
    )

    lookback = metadata.get("lookback", 30)
    currency = metadata.get("currency", DEFAULT_CURRENCY)
    base_currency = metadata.get("base_currency", DEFAULT_BASE_CURRENCY)

    raw_values, last_date = _latest_sequence(
        data_path=data_path,
        currency=currency,
        base_currency=base_currency,
        lookback=lookback,
    )

    scaled_window = scaler.transform(raw_values).reshape(1, lookback, 1)

    predictions = []
    rolling_window = scaled_window.copy()
    for _ in range(steps):
        scaled_pred = model.predict(rolling_window, verbose=0)[0, 0]
        unscaled_pred = scaler.inverse_transform([[scaled_pred]])[0, 0]
        predictions.append(float(unscaled_pred))
        rolling_window = np.concatenate(
            [rolling_window[:, 1:, :], np.array([[[scaled_pred]]], dtype="float32")],
            axis=1,
        )

    return {
        "currency": currency,
        "base_currency": base_currency,
        "steps": steps,
        "predictions": predictions,
        "last_observation": float(raw_values[-1, 0]),
        "last_observation_date": last_date.date().isoformat(),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
