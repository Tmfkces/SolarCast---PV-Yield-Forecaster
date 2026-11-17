import numpy as np
import pandas as pd
import pytest

from pvyield.physics import (
    DEG_TO_RAD,
    _declination,
    beam_tilt_factor,
    diffuse_fraction_erbs,
    extraterrestrial_daily,
    physics_yield,
    plane_of_array_irradiance,
    sunset_hour_angle,
)


def weather_frame(ghi=5.0, temperature=25.0, days=3, start="2023-01-01"):
    index = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"ghi": ghi, "ghi_clear": 6.0, "t_amb": temperature}, index=index)


def klein_rb_south(lat, tilt, doy):
    """Closed-form daily Rb for a south-facing surface (Duffie & Beckman eq. 2.19.3a)."""
    decl = _declination(np.array([doy], dtype=float))[0] * DEG_TO_RAD
    phi = lat * DEG_TO_RAD
    beta = tilt * DEG_TO_RAD
    ws = np.arccos(-np.tan(phi) * np.tan(decl))
    ws_tilted = min(ws, np.arccos(np.clip(-np.tan(phi - beta) * np.tan(decl), -1, 1)))
    numerator = np.cos(phi - beta) * np.cos(decl) * np.sin(ws_tilted) + ws_tilted * np.sin(phi - beta) * np.sin(decl)
    denominator = np.cos(phi) * np.cos(decl) * np.sin(ws) + ws * np.sin(phi) * np.sin(decl)
    return numerator / denominator


# --- energy model -----------------------------------------------------------


def test_zero_ghi_produces_zero_energy():
    result = physics_yield(weather_frame(ghi=0), lat=31.45)
    assert np.allclose(result, 0.0)


def test_missing_ghi_is_nan_not_zero():
    frame = weather_frame(days=3)
    frame.loc[frame.index[1], "ghi"] = np.nan
    result = physics_yield(frame, lat=31.45)
    assert np.isnan(result.iloc[1])
    assert result.drop(frame.index[1]).gt(0).all()


def test_higher_ambient_temperature_reduces_energy_for_same_ghi():
    cool = physics_yield(weather_frame(temperature=10), lat=31.45)
    hot = physics_yield(weather_frame(temperature=45), lat=31.45)
    assert (hot < cool).all()


def test_energy_scales_linearly_with_system_size():
    one_kw = physics_yield(weather_frame(), lat=31.45, system_kw=1)
    five_kw = physics_yield(weather_frame(), lat=31.45, system_kw=5)
    assert np.allclose(five_kw, one_kw * 5)


def test_losses_reduce_energy_proportionally():
    base = physics_yield(weather_frame(), lat=31.45, pr=0.8, soiling=0.0)
    dusty = physics_yield(weather_frame(), lat=31.45, pr=0.8, soiling=0.10)
    assert np.allclose(dusty, base * 0.9)


def test_latitude_tilt_beats_flat_panel_on_winter_day():
    frame = weather_frame(ghi=4.0, days=1)
    tilted = physics_yield(frame, lat=31.45, tilt=31.45)
    flat = physics_yield(frame, lat=31.45, tilt=0)
    assert tilted.iloc[0] > flat.iloc[0]


def test_specific_yield_is_plausible_for_faisalabad():
    # Typical NASA POWER climatology: ~5.2 kWh/m²/day GHI averaged over a year.
    index = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    doy = index.day_of_year.to_numpy()
    ghi = 5.2 + 1.8 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    frame = pd.DataFrame({"ghi": ghi, "t_amb": 25 + 9 * np.sin(2 * np.pi * (doy - 110) / 365.25)}, index=index)
    specific = physics_yield(frame, lat=31.45, system_kw=1).sum()
    assert 1_300 < specific < 1_800  # kWh/kWp per year, typical for Punjab


@pytest.mark.parametrize(
    "kwargs",
    [{"system_kw": -1}, {"pr": 1.2}, {"soiling": -0.1}, {"tilt": 95}, {"azimuth": 400}, {"noct": 15}],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        physics_yield(weather_frame(), lat=31.45, **kwargs)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        physics_yield(pd.DataFrame({"ghi": [1.0]}, index=pd.date_range("2023-01-01", periods=1)), lat=30)


# --- solar geometry ---------------------------------------------------------


def test_extraterrestrial_matches_duffie_beckman_example():
    # Duffie & Beckman Example 1.10.1: φ = 43°, April 15 (n = 105) → H0 = 33.8 MJ/m².
    h0 = extraterrestrial_daily(43.0, np.array([105.0]))[0]
    assert h0 == pytest.approx(33.8 / 3.6, rel=0.01)


def test_daylight_at_equinox_is_about_twelve_hours():
    assert sunset_hour_angle(31.45, np.array([80.0]))[0] == pytest.approx(90.0, abs=1.0)


def test_erbs_diffuse_fraction_is_bounded_and_decreasing():
    kt = np.linspace(0, 1, 101)
    fraction = diffuse_fraction_erbs(kt, np.full_like(kt, 75.0))
    assert fraction.min() >= 0 and fraction.max() <= 1
    assert fraction[0] == pytest.approx(1.0)
    assert np.all(np.diff(fraction[:72]) <= 1e-9)


@pytest.mark.parametrize("doy", [15, 80, 172, 266, 355])
@pytest.mark.parametrize("tilt", [0, 15, 31.45, 60])
def test_numerical_rb_matches_klein_closed_form(doy, tilt):
    numeric = beam_tilt_factor(31.45, tilt, 180.0, np.array([doy], dtype=float))[0]
    assert numeric == pytest.approx(klein_rb_south(31.45, tilt, doy), rel=0.01)


def test_flat_panel_poa_equals_ghi():
    ghi = weather_frame(ghi=5.0, days=30)["ghi"]
    poa = plane_of_array_irradiance(ghi, lat=31.45, tilt=0)
    assert np.allclose(poa, ghi)


def test_east_and_west_facing_are_symmetric():
    ghi = weather_frame(ghi=5.0, days=10, start="2023-06-01")["ghi"]
    east = plane_of_array_irradiance(ghi, lat=31.45, tilt=30, azimuth=90)
    west = plane_of_array_irradiance(ghi, lat=31.45, tilt=30, azimuth=270)
    assert np.allclose(east, west, rtol=1e-3)


def test_south_beats_north_in_winter():
    ghi = weather_frame(ghi=4.5, days=10, start="2023-12-10")["ghi"]
    south = plane_of_array_irradiance(ghi, lat=31.45, tilt=30, azimuth=180)
    north = plane_of_array_irradiance(ghi, lat=31.45, tilt=30, azimuth=0)
    assert (south > north * 1.5).all()


def test_overcast_day_gains_little_from_tilt():
    ghi = weather_frame(ghi=1.0, days=1, start="2023-01-01")["ghi"]  # Kt ≈ 0.2 → mostly diffuse
    tilted = plane_of_array_irradiance(ghi, lat=31.45, tilt=31.45).iloc[0]
    clear = weather_frame(ghi=4.8, days=1, start="2023-01-01")["ghi"]
    clear_gain = plane_of_array_irradiance(clear, lat=31.45, tilt=31.45).iloc[0] / 4.8
    assert tilted / 1.0 < clear_gain


def test_legacy_geometric_transposition_is_available():
    ghi = weather_frame(ghi=5.0, days=5)["ghi"]
    poa = plane_of_array_irradiance(ghi, lat=31.45, tilt=31.45, transposition="geometric")
    assert (poa > ghi).all()
    with pytest.raises(ValueError):
        plane_of_array_irradiance(ghi, lat=31.45, transposition="perez")
