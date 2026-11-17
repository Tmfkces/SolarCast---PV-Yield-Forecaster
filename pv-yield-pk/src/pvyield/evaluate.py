"""Metrics and fair comparison utilities for PV yield forecasts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cities import CITIES
from .data import fetch_daily
from .ml import (
    FEATURE_COLUMNS,
    TrainedForecaster,
    climatology_predictions,
    persistence_predictions,
    train_forecaster,
)

METRIC_COLUMNS = ["rmse", "mae", "mape", "bias", "r2", "skill_score"]
MAPE_MIN_KWH = 0.5
ORACLE_ROW = "Physics oracle (true next-day weather)"


def metrics(actual: pd.Series, predicted: pd.Series, baseline_rmse: float | None = None) -> dict[str, float]:
    """Calculate RMSE, MAE, MAPE, mean bias, R², and RMSE skill versus a baseline.

    * MAPE ignores days whose actual value is below 0.5 kWh.
    * ``bias`` is mean(predicted − actual); positive means over-forecasting.
    * ``skill_score`` is ``1 − RMSE / baseline_RMSE`` (NaN without a baseline).
    """
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if aligned.empty:
        return {name: float("nan") for name in METRIC_COLUMNS}
    error = aligned["predicted"] - aligned["actual"]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mask = aligned["actual"] >= MAPE_MIN_KWH
    mape = float(np.mean(np.abs(error[mask] / aligned.loc[mask, "actual"])) * 100) if mask.any() else float("nan")
    denominator = float(np.sum((aligned["actual"] - aligned["actual"].mean()) ** 2))
    r2 = float(1 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
    skill = float(1 - rmse / baseline_rmse) if baseline_rmse is not None and baseline_rmse > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "bias": float(error.mean()), "r2": r2, "skill_score": skill}


@dataclass
class Evaluation:
    """Test-window comparison of the ML model against baselines."""

    trained: TrainedForecaster
    table: pd.DataFrame
    predictions: pd.DataFrame  # columns: actual, persistence, climatology, <model_name>


def evaluate_forecaster(
    df: pd.DataFrame,
    lat: float,
    system_kw: float = 5.0,
    include_oracle: bool = True,
    **physics_kwargs: Any,
) -> Evaluation:
    """Train once and score persistence, climatology, and ML on the untouched test window.

    The "physics oracle" row is given tomorrow's true weather, so it matches
    the proxy target exactly by construction. It is a sanity check, not a
    forecast, and is excluded unless ``include_oracle`` is true.
    """
    trained = train_forecaster(df, lat=lat, system_kw=system_kw, **physics_kwargs)
    test = trained.split.test
    actual = test["target"].rename("actual")
    persistence = persistence_predictions(df, lat=lat, system_kw=system_kw, **physics_kwargs).reindex(actual.index)
    climatology = climatology_predictions(trained.split, actual.index)
    model_prediction = trained.predict(test[FEATURE_COLUMNS])

    baseline_rmse = metrics(actual, persistence)["rmse"]
    candidates = [("Persistence", persistence), ("Climatology (monthly mean)", climatology)]
    if include_oracle:
        candidates.append((ORACLE_ROW, actual))
    candidates.append((trained.model_name, model_prediction))

    rows = []
    for name, prediction in candidates:
        row = metrics(actual, prediction, baseline_rmse=baseline_rmse)
        row["model"] = name
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")[METRIC_COLUMNS]
    predictions = pd.concat(
        [actual, persistence.rename("persistence"), climatology, model_prediction.rename(trained.model_name)],
        axis=1,
    )
    return Evaluation(trained=trained, table=table, predictions=predictions)


def compare(city: str, start: str, end: str, system_kw: float = 5.0, **physics_kwargs: Any) -> pd.DataFrame:
    """Fetch NASA POWER data for ``city`` and return the test-window metric table."""
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}")
    df = fetch_daily(city, start, end)
    return evaluate_forecaster(df, lat=CITIES[city]["lat"], system_kw=system_kw, **physics_kwargs).table


def benchmark(
    cities: Iterable[str],
    start: str,
    end: str,
    system_kw: float = 5.0,
    **physics_kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`compare` for several cities and stack the results."""
    tables = {city: compare(city, start, end, system_kw, **physics_kwargs) for city in cities}
    return pd.concat(tables, names=["city", "model"])


def feature_importance(trained: TrainedForecaster | dict) -> pd.Series:
    """Return normalised importances (sum to 1): gain for LightGBM, impurity for the fallback."""
    model = trained["model"]
    if hasattr(model, "booster_"):
        values = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    else:
        values = np.asarray(model.feature_importances_, dtype=float)
    total = values.sum()
    if total > 0:
        values = values / total
    return pd.Series(values, index=FEATURE_COLUMNS, name="importance").sort_values(ascending=False)
