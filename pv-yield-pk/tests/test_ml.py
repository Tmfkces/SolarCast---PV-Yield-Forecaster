import numpy as np
import pandas as pd
import pytest
from conftest import make_weather

from pvyield.ml import (
    FEATURE_COLUMNS,
    build_features,
    build_supervised_frame,
    climatology_predictions,
    load_forecaster,
    persistence_predictions,
    save_forecaster,
    split_time_ordered,
    train_forecaster,
)
from pvyield.physics import physics_yield

LAT = 31.45


def test_split_is_chronological_and_has_no_overlap(weather):
    features, target = build_supervised_frame(weather, lat=LAT)
    split = split_time_ordered(features, target)
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(features)


def test_split_rejects_bad_inputs(weather):
    features, target = build_supervised_frame(weather, lat=LAT)
    with pytest.raises(ValueError):
        split_time_ordered(features.iloc[:5], target.iloc[:5])
    with pytest.raises(ValueError):
        split_time_ordered(features, target.iloc[:-1])
    with pytest.raises(ValueError):
        split_time_ordered(features, target, train_fraction=0.9, validation_fraction=0.2)


def test_target_is_next_calendar_day_physics_yield(weather):
    features, target = build_supervised_frame(weather, lat=LAT, system_kw=3.0)
    physics = physics_yield(weather, lat=LAT, system_kw=3.0)
    for day in target.index[:20]:
        assert target[day] == pytest.approx(physics[day + pd.Timedelta(days=1)])


def test_gaps_never_pair_a_day_with_a_distant_target_or_lag():
    weather = make_weather(200)
    gap = pd.date_range("2022-03-01", "2022-03-15")
    gapped = weather.drop(gap)
    features, target = build_supervised_frame(gapped, lat=LAT)
    physics = physics_yield(weather, lat=LAT)

    # The day before the gap has no next-day target; days after the gap need 7 days of history.
    assert pd.Timestamp("2022-02-28") not in target.index
    assert features.index.intersection(pd.date_range("2022-03-01", "2022-03-22")).empty
    for day in target.index:
        assert target[day] == pytest.approx(physics[day + pd.Timedelta(days=1)])
        assert features.loc[day, "ghi_lag_1"] == pytest.approx(weather.loc[day - pd.Timedelta(days=1), "ghi"])


def test_features_have_expected_columns_and_no_nans(weather):
    features = build_features(weather)
    assert list(features.columns) == FEATURE_COLUMNS
    assert not features.isna().any().any()
    assert features.index.min() == weather.index.min() + pd.Timedelta(days=7)


def test_build_features_requires_all_weather_columns(weather):
    with pytest.raises(ValueError, match="missing feature columns"):
        build_features(weather.drop(columns=["rh"]))


def test_training_uses_validation_and_predicts_finite_values(weather, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trained = train_forecaster(weather, lat=LAT)
    predictions = trained.predict(trained.split.test)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert trained.best_iteration >= 1
    assert np.isfinite(trained.validation_rmse)
    assert trained["model"] is trained.model  # v1 dict-style access still works
    assert not list(tmp_path.rglob("*.pkl"))  # training has no disk side effects


def test_model_beats_persistence_on_learnable_signal():
    weather = make_weather(900, seed=3)
    trained = train_forecaster(weather, lat=LAT)
    test = trained.split.test
    model_rmse = np.sqrt(np.mean((trained.predict(test) - test["target"]) ** 2))
    persistence = persistence_predictions(weather, lat=LAT).reindex(test.index)
    persistence_rmse = np.sqrt(np.mean((persistence - test["target"]) ** 2))
    assert model_rmse < persistence_rmse


def test_physics_kwargs_change_the_target(weather):
    _, south = build_supervised_frame(weather, lat=LAT, soiling=0.0)
    _, dusty = build_supervised_frame(weather, lat=LAT, soiling=0.2)
    assert np.allclose(dusty, south * 0.8)


def test_climatology_uses_training_history_only(weather):
    features, target = build_supervised_frame(weather, lat=LAT)
    split = split_time_ordered(features, target)
    prediction = climatology_predictions(split, split.test.index)
    assert prediction.index.equals(split.test.index)
    assert np.isfinite(prediction).all()


def test_save_and_load_roundtrip(weather, tmp_path):
    trained = train_forecaster(weather, lat=LAT)
    path = save_forecaster(trained, tmp_path / "models" / "Faisalabad.joblib", city="Faisalabad")
    bundle = load_forecaster(path)
    assert bundle["city"] == "Faisalabad"
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    reloaded = bundle["model"].predict(trained.split.test[FEATURE_COLUMNS])
    np.testing.assert_allclose(reloaded, trained.model.predict(trained.split.test[FEATURE_COLUMNS]))


def test_load_rejects_foreign_files(tmp_path):
    import joblib

    path = tmp_path / "old.pkl"
    joblib.dump({"model": None}, path)
    with pytest.raises(ValueError, match="Unsupported model file"):
        load_forecaster(path)
