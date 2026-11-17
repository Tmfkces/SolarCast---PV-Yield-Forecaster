import numpy as np
import pandas as pd
import pytest
from conftest import make_weather

import pvyield.forecast as forecast
from pvyield.forecast import (
    advise,
    best_window,
    build_outlook,
    clock,
    parse_forecast,
    typical_daily_yield,
    weather_icon,
)

LAT = 31.45


def open_meteo_payload(start="2026-01-10", days=7, radiation_mj=(12.0, 13.0, 6.0, 3.0, 12.5, 14.0, 12.0), codes=(0, 1, 3, 61, 2, 0, 0)):
    dates = pd.date_range(start, periods=days, freq="D")
    hours = pd.date_range(start, periods=days * 24, freq="h")
    # Bell-shaped irradiance between 07:00 and 17:00, peaking at noon, scaled per day.
    hour_of_day = hours.hour.to_numpy()
    shape = np.clip(np.sin(np.pi * (hour_of_day - 7) / 10), 0, None)
    scale = np.repeat(np.asarray(radiation_mj[:days]) / 12.0, 24)
    return {
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in dates],
            "weather_code": list(codes[:days]),
            "temperature_2m_max": [22.0] * days,
            "temperature_2m_min": [9.0] * days,
            "shortwave_radiation_sum": list(radiation_mj[:days]),
            "precipitation_sum": [0.0] * days,
            "precipitation_probability_max": [5] * days,
            "wind_speed_10m_max": [8.0] * days,
            "sunrise": [f"{d:%Y-%m-%d}T07:10" for d in dates],
            "sunset": [f"{d:%Y-%m-%d}T17:20" for d in dates],
        },
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            "shortwave_radiation": list(600 * shape * scale),
            "temperature_2m": [15.0] * len(hours),
            "cloud_cover": [10] * len(hours),
        },
    }


def test_parse_forecast_converts_units():
    daily, hourly = parse_forecast(open_meteo_payload())
    assert daily["ghi"].iloc[0] == pytest.approx(12.0 / 3.6)
    assert daily["t_amb"].iloc[0] == pytest.approx(15.5)
    assert "ghi_w_m2" in hourly
    assert len(hourly) == 7 * 24


def test_parse_forecast_rejects_bad_payload():
    with pytest.raises(RuntimeError):
        parse_forecast({"error": True})


def test_weather_icons():
    assert weather_icon(0) == ("☀️", "Clear")
    assert weather_icon(63)[1] == "Rain"
    assert weather_icon(96)[1] == "Thunderstorm"
    assert weather_icon(None)[1] == "Unknown"


def test_best_window_finds_peak_block():
    index = pd.date_range("2026-01-10 01:00", periods=24, freq="h")  # hour-end stamps
    energy = pd.Series(0.0, index=index)
    energy.loc["2026-01-10 11:00":"2026-01-10 14:00"] = [2.0, 3.0, 3.0, 2.5]
    start, end = best_window(energy, hours=3)
    assert (start, end) == (pd.Timestamp("2026-01-10 11:00"), pd.Timestamp("2026-01-10 14:00"))  # 3+3+2.5 > 2+3+3
    assert clock(start) == "11 AM" and clock(end) == "2 PM"
    assert best_window(pd.Series(0.0, index=index)) is None


@pytest.mark.parametrize(
    ("forecast_kwh", "level", "phrase"),
    [(13.0, "high", "heavy loads"), (10.0, "normal", "Normal"), (7.5, "low", "Save electricity"), (4.0, "very_low", "Avoid heavy")],
)
def test_advice_levels(forecast_kwh, level, phrase):
    window = (pd.Timestamp("2026-01-10 10:00"), pd.Timestamp("2026-01-10 13:00"))
    advice = advise(forecast_kwh, typical_kwh=10.0, window=window)
    assert advice.level == level
    assert phrase in advice.detail
    if level in {"high", "normal", "low"}:
        assert "10 AM–1 PM" in advice.detail


def test_typical_daily_yield_covers_all_months():
    typical = typical_daily_yield(make_weather(400), lat=LAT, system_kw=5)
    assert list(typical.index) == list(range(1, 13))
    assert (typical > 0).all()


def test_build_outlook_ranks_days_and_distributes_hourly_energy():
    daily, hourly = parse_forecast(open_meteo_payload())
    typical = pd.Series(12.0, index=range(1, 13))
    outlook = build_outlook(daily, hourly, typical, lat=LAT, system_kw=5)
    days = outlook.days
    assert len(days) == 7
    assert days["yield_kwh"].idxmax() == pd.Timestamp("2026-01-15")  # sunniest day
    assert days.loc["2026-01-13", "level"] in {"low", "very_low"}  # rainy day
    assert days.loc["2026-01-13", "icon"] == "🌧️"
    # Hourly energy for each day adds up to that day's forecast yield.
    hour_day = (outlook.hourly.index - pd.Timedelta(minutes=1)).normalize()
    per_day = outlook.hourly["energy_kwh"].groupby(hour_day).sum().reindex(days.index)
    np.testing.assert_allclose(per_day, days["yield_kwh"], rtol=1e-6)
    # Best window sits around midday.
    assert 9 <= days.iloc[0]["window_start"].hour <= 12


def test_fetch_forecast_uses_city_coordinates(monkeypatch, isolated_cache):
    calls = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return open_meteo_payload()

    def fake_get(url, params=None, timeout=None):
        calls.update(params)
        return Response()

    monkeypatch.setattr(forecast.requests, "get", fake_get)
    daily, _ = forecast.fetch_forecast("Karachi")
    assert calls["latitude"] == pytest.approx(24.8607)
    assert calls["timezone"] == "Asia/Karachi"
    assert len(daily) == 7
    with pytest.raises(ValueError):
        forecast.fetch_forecast("Atlantis")


@pytest.mark.network
def test_live_open_meteo_forecast():
    daily, hourly = forecast.fetch_forecast("Lahore")
    assert len(daily) == 7
    assert daily["ghi"].between(0, 12).all()
    assert hourly["ghi_w_m2"].max() > 100


def _day_energy(peak=3.0):
    index = pd.date_range("2026-01-10 01:00", periods=24, freq="h")
    shape = np.clip(np.sin(np.pi * (index.hour.to_numpy() - 7) / 10), 0, None)
    return pd.Series(peak * shape, index=index)


def test_plan_fits_small_load_near_midday():
    plan = forecast.plan_appliances(["Washing machine", "Iron"], _day_energy())
    assert plan.fits
    assert plan.need_kwh == pytest.approx(0.5 * 1.0 + 1.0 * 0.5)
    assert 10 <= plan.start.hour <= 12 and plan.end > plan.start


def test_plan_reports_partial_coverage_for_big_loads():
    plan = forecast.plan_appliances(list(forecast.APPLIANCES), _day_energy(peak=1.0))
    assert not plan.fits
    assert 0 < plan.coverage < 1
    assert plan.start.hour >= 7 and plan.end.hour <= 18


def test_plan_handles_no_selection_and_no_sun():
    assert forecast.plan_appliances([], _day_energy()).fits
    dark = forecast.plan_appliances(["Iron"], _day_energy(peak=0.1))
    assert dark.start is None and not dark.fits


def test_fetch_forecast_saves_an_offline_copy(monkeypatch, isolated_cache):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return open_meteo_payload()

    monkeypatch.setattr(forecast.requests, "get", lambda *a, **k: Response())
    assert forecast.load_saved_forecast("Lahore") is None
    forecast.fetch_forecast("Lahore")
    saved = forecast.load_saved_forecast("Lahore")
    assert saved is not None
    daily, hourly, saved_at = saved
    assert len(daily) == 7 and len(hourly) == 7 * 24
    assert saved_at.tzinfo is not None


def test_history_uses_same_source_and_is_cached(monkeypatch, isolated_cache):
    calls = []
    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "daily": {
                    "time": [d.strftime("%Y-%m-%d") for d in dates],
                    "shortwave_radiation_sum": [18.0] * len(dates),
                    "temperature_2m_max": [30.0] * len(dates),
                    "temperature_2m_min": [20.0] * len(dates),
                }
            }

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return Response()

    monkeypatch.setattr(forecast.requests, "get", fake_get)
    history = forecast.fetch_forecast_history("Multan" if "Multan" in forecast.CITIES else "Lahore")
    assert "historical-forecast" in calls[0]
    assert history["ghi"].iloc[0] == pytest.approx(5.0)
    assert history["t_amb"].iloc[0] == pytest.approx(25.0)
    forecast.fetch_forecast_history("Multan" if "Multan" in forecast.CITIES else "Lahore")
    assert len(calls) == 1  # second call served from the disk cache


def test_usual_day_is_the_median_not_the_mean():
    index = pd.date_range("2024-06-01", periods=30, freq="D")
    ghi = np.full(30, 7.0)
    ghi[:5] = 1.0  # five stormy days drag the mean down but not the median
    history = pd.DataFrame({"ghi": ghi, "t_amb": 30.0}, index=index)
    median = typical_daily_yield(history, lat=LAT, system_kw=5)[6]
    mean = typical_daily_yield(history, lat=LAT, system_kw=5, statistic="mean")[6]
    assert median > mean


def test_peak_hours_and_setup_tips():
    assert forecast.peak_label(1) == "5 PM–9 PM"
    assert forecast.peak_label(7) == "7 PM–11 PM"
    window = (pd.Timestamp("2026-01-10 10:00"), pd.Timestamp("2026-01-10 13:00"))
    sunny_battery = advise(13.0, 10.0, window, setup="battery", month=1)
    assert "battery fill" in sunny_battery.detail and "5 PM–9 PM" in sunny_battery.detail
    cloudy_battery = advise(4.0, 10.0, window, setup="battery", month=1)
    assert "Keep battery charge" in cloudy_battery.detail
    assert "buyback" in advise(13.0, 10.0, window, setup="net_metering").detail


def test_estimate_value_by_setup():
    net = forecast.estimate_value(20.0, price=55, setup="net_metering", export_rate=10, daily_use_kwh=12)
    assert net["self_kwh"] == pytest.approx(6.0)  # half of 12 kWh used in daylight
    assert net["export_kwh"] == pytest.approx(14.0)
    assert net["rupees"] == pytest.approx(6 * 55 + 14 * 10)
    grid = forecast.estimate_value(20.0, price=55, setup="grid_only", daily_use_kwh=12)
    assert grid["export_kwh"] == 0 and grid["unused_kwh"] == pytest.approx(14.0)
    battery = forecast.estimate_value(20.0, price=55, setup="battery", daily_use_kwh=12, battery_kwh=5)
    assert battery["self_kwh"] == pytest.approx(11.0)
    small = forecast.estimate_value(3.0, price=55, setup="net_metering", daily_use_kwh=12)
    assert small["self_kwh"] == pytest.approx(3.0) and small["export_kwh"] == 0
    with pytest.raises(ValueError):
        forecast.estimate_value(1.0, price=55, setup="solar_farm")


def test_theme_css_covers_device_and_live_theme():
    from pvyield.ui import DARK, LIGHT, THEME_WATCHER, theme_css

    assert set(LIGHT) == set(DARK)  # every colour has a dark counterpart
    css = theme_css()
    assert "prefers-color-scheme: dark" in css  # device setting before the page script runs
    assert 'html[data-pc-theme="dark"] [data-testid="stAppViewContainer"] { --pc-page: #0F1020' in css
    assert 'html[data-pc-theme="light"] [data-testid="stAppViewContainer"] { --pc-page: #F6F4FD' in css
    assert "data-pc-theme" in THEME_WATCHER and "MutationObserver" in THEME_WATCHER
