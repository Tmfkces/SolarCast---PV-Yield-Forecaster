from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import pvyield.data as data
from pvyield.data import _as_daily_frame, clean_daily_frame, fetch_daily, parse_dates


def frame_with_ghi(values, start="2023-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    n = len(values)
    return pd.DataFrame(
        {
            "ghi": values,
            "ghi_clear": [6.0] * n,
            "t_amb": [20.0] * n,
            "t_max": [25.0] * n,
            "t_min": [15.0] * n,
            "rh": [50.0] * n,
            "wind": [2.0] * n,
            "precip": [0.0] * n,
        },
        index=index,
    )


def nasa_payload(start: str, days: int, fill_index: int | None = None) -> dict:
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(start, periods=days, freq="D")]

    def series(value):
        values = {d: value for d in dates}
        if fill_index is not None:
            values[dates[fill_index]] = -999.0
        return values

    return {
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": series(5.1),
                "CLRSKY_SFC_SW_DWN": series(6.3),
                "T2M": series(22.0),
                "T2M_MAX": series(28.0),
                "T2M_MIN": series(16.0),
                "RH2M": series(45.0),
                "WS2M": series(2.4),
                "PRECTOTCORR": series(0.0),
            }
        }
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload, calls):
        self.payload = payload
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResponse(self.payload)


@pytest.fixture
def fake_nasa(monkeypatch):
    calls: list[dict] = []
    state = {"payload": nasa_payload("2023-01-01", 31)}
    monkeypatch.setattr(data, "_session", lambda: FakeSession(state["payload"], calls))
    return state, calls


# --- cleaning ---------------------------------------------------------------


def test_fill_value_is_nan_and_short_gaps_are_forward_filled():
    cleaned = clean_daily_frame(frame_with_ghi([4.0, -999, -999, 5.0, -999, 7.0]))
    index = cleaned.index
    assert cleaned.loc[index[1], "ghi"] == 4.0
    assert cleaned.loc[index[2], "ghi"] == 4.0
    assert cleaned.loc[index[4], "ghi"] == 5.0
    assert not cleaned.isna().any().any()


def test_long_gap_is_dropped():
    raw = frame_with_ghi([4.0, -999, -999, -999, -999, 5.0, 5.0, 5.0])
    cleaned = clean_daily_frame(raw)
    assert not cleaned.index.intersection(raw.index[1:5]).tolist()
    assert len(cleaned) == 4


def test_missing_calendar_days_count_as_gaps():
    raw = frame_with_ghi([4.0] * 20).drop(pd.date_range("2023-01-05", "2023-01-12"))
    cleaned = clean_daily_frame(raw)
    assert cleaned.index.intersection(pd.date_range("2023-01-05", "2023-01-12")).empty
    short = frame_with_ghi([4.0] * 10).drop(pd.Timestamp("2023-01-05"))
    filled = clean_daily_frame(short)
    assert pd.Timestamp("2023-01-05") in filled.index


def test_duplicate_dates_keep_last_value():
    raw = pd.concat([frame_with_ghi([1.0, 2.0]), frame_with_ghi([9.0])])
    cleaned = clean_daily_frame(raw)
    assert cleaned.loc["2023-01-01", "ghi"] == 9.0
    assert cleaned.index.is_unique


def test_empty_frame_is_returned_unchanged():
    assert clean_daily_frame(frame_with_ghi([])).empty


def test_parse_nasa_response_shape():
    frame = _as_daily_frame(nasa_payload("2023-03-01", 5)["properties"]["parameter"])
    assert list(frame.columns) == data.OUTPUT_COLUMNS
    assert frame.index[0] == pd.Timestamp("2023-03-01")
    assert len(frame) == 5


# --- date validation ------------------------------------------------------


def test_parse_dates_validates_ranges():
    assert parse_dates("2023-01-01", "2023-12-31") == (date(2023, 1, 1), date(2023, 12, 31))
    with pytest.raises(ValueError, match="on or after"):
        parse_dates("2023-02-01", "2023-01-01")
    with pytest.raises(ValueError, match="1981"):
        parse_dates("1970-01-01", "1971-01-01")
    with pytest.raises(ValueError, match="future"):
        parse_dates("2023-01-01", date.today() + timedelta(days=5))
    with pytest.raises(ValueError, match="ISO"):
        parse_dates("not a date", "2023-01-01")


# --- fetching and caching -----------------------------------------------------


def test_unknown_city_is_rejected():
    with pytest.raises(ValueError, match="Unknown city"):
        fetch_daily("Atlantis", "2023-01-01", "2023-01-31")


def test_fetch_parses_cleans_and_caches(fake_nasa, isolated_cache):
    state, calls = fake_nasa
    state["payload"] = nasa_payload("2023-01-01", 31, fill_index=10)
    first = fetch_daily("Faisalabad", "2023-01-01", "2023-01-31")
    assert len(first) == 31
    assert first.loc["2023-01-11", "ghi"] == pytest.approx(5.1)  # fill value forward-filled
    assert calls[0]["latitude"] == pytest.approx(31.4504)
    assert calls[0]["start"] == "20230101"
    cache_files = list(isolated_cache.glob("*.csv"))
    assert [p.name for p in cache_files] == ["Faisalabad_2023-01-01_2023-01-31.csv"]

    # Equivalent date spelling hits the same cache entry: no second request.
    second = fetch_daily("Faisalabad", "2023-1-1", "2023-01-31")
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_use_cache_false_always_requests(fake_nasa, isolated_cache):
    _, calls = fake_nasa
    fetch_daily("Lahore", "2023-01-01", "2023-01-31", use_cache=False)
    fetch_daily("Lahore", "2023-01-01", "2023-01-31", use_cache=False)
    assert len(calls) == 2
    assert not isolated_cache.exists() or not list(isolated_cache.glob("*.csv"))


def test_recent_ranges_are_not_cached(fake_nasa, isolated_cache):
    state, _ = fake_nasa
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=20)
    state["payload"] = nasa_payload(start.isoformat(), 21)
    fetch_daily("Karachi", start, end)
    assert not isolated_cache.exists() or not list(isolated_cache.glob("*.csv"))


def test_unexpected_response_shape_raises(fake_nasa, isolated_cache):
    state, _ = fake_nasa
    state["payload"] = {"messages": ["bad request"]}
    with pytest.raises(RuntimeError, match="unexpected response"):
        fetch_daily("Quetta", "2023-01-01", "2023-01-31")


def test_all_missing_response_returns_empty_and_is_not_cached(fake_nasa, isolated_cache):
    state, _ = fake_nasa
    payload = nasa_payload("2023-01-01", 10)
    for values in payload["properties"]["parameter"].values():
        for key in values:
            values[key] = -999.0
    state["payload"] = payload
    result = fetch_daily("Skardu", "2023-01-01", "2023-01-10")
    assert result.empty
    assert not isolated_cache.exists() or not list(isolated_cache.glob("*.csv"))


@pytest.mark.network
def test_live_nasa_power_request(isolated_cache):
    frame = fetch_daily("Faisalabad", "2023-06-01", "2023-06-30")
    assert len(frame) >= 25
    assert frame["ghi"].between(0, 12).all()
    assert np.isfinite(frame.to_numpy()).all()
