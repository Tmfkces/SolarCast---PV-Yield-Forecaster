import numpy as np
import pandas as pd

from pvyield.ml import FEATURE_COLUMNS, build_supervised_frame, split_time_ordered, train_forecaster


def synthetic_weather(days=400):
    index = pd.date_range("2022-01-01", periods=days, freq="D")
    seasonal = 5.5 + 1.8 * np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.25)
    return pd.DataFrame(
        {
            "ghi": seasonal + 0.2 * np.sin(np.arange(days) / 5),
            "ghi_clear": seasonal + 1.0,
            "t_amb": 25 + 12 * np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.25),
            "t_max": 31 + 12 * np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.25),
            "t_min": 19 + 12 * np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.25),
            "rh": np.full(days, 55.0),
            "wind": np.full(days, 2.2),
            "precip": np.zeros(days),
        },
        index=index,
    )


def test_training_runs_and_split_has_no_date_leak():
    weather = synthetic_weather()
    features, target = build_supervised_frame(weather, lat=31.45)
    split = split_time_ordered(features, target)
    assert split.train.index.max() < split.test.index.min()
    trained = train_forecaster(weather, lat=31.45)
    predictions = trained["model"].predict(trained["split"].test[FEATURE_COLUMNS])
    assert np.isfinite(predictions).all()