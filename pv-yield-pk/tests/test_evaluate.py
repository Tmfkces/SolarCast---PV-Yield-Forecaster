import numpy as np
import pandas as pd
import pytest

import pvyield.evaluate as evaluate
from pvyield.evaluate import METRIC_COLUMNS, ORACLE_ROW, evaluate_forecaster, feature_importance, metrics


def series(values):
    return pd.Series(values, index=pd.date_range("2023-01-01", periods=len(values), freq="D"), dtype=float)


def test_metrics_match_hand_calculation():
    actual = series([2.0, 4.0, 6.0, 8.0])
    predicted = series([3.0, 3.0, 7.0, 7.0])
    result = metrics(actual, predicted, baseline_rmse=2.0)
    assert result["rmse"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["mape"] == pytest.approx(np.mean([1 / 2, 1 / 4, 1 / 6, 1 / 8]) * 100)
    assert result["r2"] == pytest.approx(1 - 4 / 20)
    assert result["skill_score"] == pytest.approx(0.5)


def test_metrics_perfect_prediction_and_missing_baseline():
    actual = series([1.0, 2.0, 3.0])
    result = metrics(actual, actual)
    assert result["rmse"] == 0 and result["r2"] == 1
    assert np.isnan(result["skill_score"])


def test_mape_ignores_near_zero_days_and_aligns_on_index():
    actual = series([0.1, 5.0])
    predicted = series([1.0, 5.5])
    assert metrics(actual, predicted)["mape"] == pytest.approx(10.0)
    shifted = predicted.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert metrics(actual, shifted)["rmse"] == pytest.approx(4.0)  # only 2023-01-02 overlaps: 1.0 vs 5.0


def test_metrics_on_empty_overlap_are_nan():
    result = metrics(series([1.0]), pd.Series(dtype=float))
    assert set(result) == set(METRIC_COLUMNS)
    assert all(np.isnan(v) for v in result.values())


def test_evaluate_forecaster_scores_all_baselines(weather):
    evaluation = evaluate_forecaster(weather, lat=31.45, system_kw=5.0)
    table = evaluation.table
    model_name = evaluation.trained.model_name
    assert list(table.columns) == METRIC_COLUMNS
    assert {"Persistence", "Climatology (monthly mean)", ORACLE_ROW, model_name} == set(table.index)
    assert table.loc["Persistence", "skill_score"] == pytest.approx(0.0)
    assert table.loc[ORACLE_ROW, "rmse"] == pytest.approx(0.0)
    assert set(evaluation.predictions.columns) == {"actual", "persistence", "climatology", model_name}
    assert evaluation.predictions.index.equals(evaluation.trained.split.test.index)


def test_evaluate_without_oracle(weather):
    table = evaluate_forecaster(weather, lat=31.45, include_oracle=False).table
    assert ORACLE_ROW not in table.index


def test_compare_and_benchmark_use_fetched_weather(monkeypatch, weather):
    monkeypatch.setattr(evaluate, "fetch_daily", lambda city, start, end: weather)
    table = evaluate.compare("Lahore", "2022-01-01", "2023-02-04", 3.0)
    assert "Persistence" in table.index
    stacked = evaluate.benchmark(["Lahore", "Karachi"], "2022-01-01", "2023-02-04")
    assert stacked.index.names == ["city", "model"]
    assert set(stacked.index.get_level_values("city")) == {"Lahore", "Karachi"}
    with pytest.raises(ValueError):
        evaluate.compare("Nowhere", "2022-01-01", "2023-01-01")


def test_feature_importance_is_normalised(weather):
    trained = evaluate_forecaster(weather, lat=31.45).trained
    importance = feature_importance(trained)
    assert importance.sum() == pytest.approx(1.0)
    assert importance.is_monotonic_decreasing
