import numpy as np
import pandas as pd

from pvyield.data import clean_daily_frame


def test_fill_value_is_nan_and_short_gaps_are_forward_filled():
    index = pd.date_range("2023-01-01", periods=6, freq="D")
    raw = pd.DataFrame(
        {
            "ghi": [4.0, -999, -999, 5.0, -999, 7.0],
            "ghi_clear": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            "t_amb": [20.0, 20.0, 20.0, 21.0, 21.0, 22.0],
            "t_max": [25.0] * 6,
            "t_min": [15.0] * 6,
            "rh": [50.0] * 6,
            "wind": [2.0] * 6,
            "precip": [0.0] * 6,
        },
        index=index,
    )
    cleaned = clean_daily_frame(raw)
    assert cleaned.loc[index[1], "ghi"] == 4.0
    assert cleaned.loc[index[2], "ghi"] == 4.0
    assert cleaned.loc[index[4], "ghi"] == 5.0
    assert not cleaned.isna().any().any()


def test_long_gap_is_dropped():
    index = pd.date_range("2023-01-01", periods=8, freq="D")
    raw = pd.DataFrame(
        {
            "ghi": [4.0, -999, -999, -999, -999, 5.0, 5.0, 5.0],
            "ghi_clear": [6.0] * 8,
            "t_amb": [20.0] * 8,
            "t_max": [25.0] * 8,
            "t_min": [15.0] * 8,
            "rh": [50.0] * 8,
            "wind": [2.0] * 8,
            "precip": [0.0] * 8,
        },
        index=index,
    )
    cleaned = clean_daily_frame(raw)
    assert not cleaned.index.intersection(index[1:5]).tolist()
    assert index[4] not in cleaned.index