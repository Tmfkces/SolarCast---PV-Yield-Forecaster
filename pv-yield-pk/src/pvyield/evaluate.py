"""Metrics and fair comparison utilities for PV yield forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cities import CITIES
from .data import fetch_daily
from .ml import FEATURE_COLUMNS, persistence_predictions, train_forecaster
from .physics import physics_yield


def metrics(actual: pd.Series, predicted: pd.Series, baseline_rmse: float | None = None) -> dict[str, float]:
    """Calculate RMSE, MAE, MAPE, R², and skill versus a baseline."""
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if aligned.empty:
        return {name: float("nan") for name in ["rmse", "mae", "mape", "r2", "skill_score"]}
    error = aligned["predicted"] - aligned["actual"]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mask = aligned["actual"] >= 0.5
    mape = float(np.mean(np.abs(error[mask] / aligned.loc[mask, "actual"])) * 100) if mask.any() else float("nan")
    actual_mean = float(aligned["actual"].mean())
    denominator = float(np.sum((aligned["actual"] - actual_mean) ** 2))
    r2 = float(1 - np.sum(error**2) / denominator) if denominator > 0 else 0.0
    skill = float(1 - rmse / baseline_rmse) if baseline_rmse and baseline_rmse > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "skill_score": skill}


def compare(city: str, start: str, end: str, system_kw: float = 5.0) -> pd.DataFrame:
    """Compare persistence, physics upper bound, and ML on the test window."""
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}")
    df = fetch_daily(city, start, end)
    location = CITIES[city]
    trained = train_forecaster(df, lat=location["lat"], system_kw=system_kw, city=city)
    split = trained["split"]
    actual = split.test["target"].rename("actual")
    persistence_all = persistence_predictions(df, lat=location["lat"], system_kw=system_kw)
    persistence = persistence_all.reindex(actual.index).rename("persistence")
    model_prediction = pd.Series(
        trained["model"].predict(split.test[FEATURE_COLUMNS]),
        index=actual.index,
        name="LightGBM",
    ).clip(lower=0)
    upper_bound = actual.rename("physics upper bound")
    baseline_rmse = metrics(actual, persistence)["rmse"]
    rows = []
    for name, prediction in [
        ("Persistence", persistence),
        ("Physics upper bound", upper_bound),
        (trained["model_name"], model_prediction),
    ]:
        row = metrics(actual, prediction, baseline_rmse=baseline_rmse)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")[["rmse", "mae", "mape", "r2", "skill_score"]]


def feature_importance(trained: dict) -> pd.Series:
    """Return gain importances for LightGBM or feature importances for fallback."""
    model = trained["model"]
    if hasattr(model, "booster_"):
        values = model.booster_.feature_importance(importance_type="gain")
    else:
        values = model.feature_importances_
    return pd.Series(values, index=FEATURE_COLUMNS, name="importance").sort_values(ascending=False)