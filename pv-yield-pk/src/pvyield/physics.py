"""Transparent first-order physics model for daily PV energy yield."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEG_TO_RAD = np.pi / 180.0
SOLAR_CONSTANT = 1_367.0
DEFAULT_TILT = "latitude"
DEFAULT_NOCT_C = 45.0
DEFAULT_GAMMA_PER_C = -0.0040
DEFAULT_PERFORMANCE_RATIO = 0.80
DEFAULT_SOILING = 0.03


def _declination(day_of_year: np.ndarray) -> np.ndarray:
    """Approximate solar declination in degrees."""
    return 23.45 * np.sin(DEG_TO_RAD * (360.0 * (284.0 + day_of_year) / 365.0))


def daylight_hours(lat: float, day_of_year: np.ndarray) -> np.ndarray:
    """Return astronomical daylight duration in hours."""
    decl = _declination(day_of_year)
    latitude = lat * DEG_TO_RAD
    declination = decl * DEG_TO_RAD
    hour_angle = np.arccos(
        np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
    )
    return 24.0 * hour_angle / np.pi


def plane_of_array_irradiance(
    ghi: pd.Series,
    lat: float,
    tilt: float | None = None,
) -> pd.Series:
    """Estimate daily POA irradiation with a first-order geometric transposition.

    This approximation assumes a south-facing panel and does not split direct
    and diffuse irradiance. It is intentionally transparent, not bankable.
    """
    panel_tilt = float(lat if tilt is None else tilt)
    day_of_year = ghi.index.dayofyear.to_numpy(dtype=float)
    decl = _declination(day_of_year)
    numerator = np.cos(DEG_TO_RAD * (lat - panel_tilt - decl))
    denominator = np.maximum(np.cos(DEG_TO_RAD * (lat - decl)), 0.05)
    factor = np.maximum(numerator, 0.05) / denominator
    return ghi.astype(float).clip(lower=0).mul(factor)


def clear_sky_index(df: pd.DataFrame) -> pd.Series:
    """Return GHI divided by clear-sky GHI, clipped to a physical range."""
    result = df["ghi"].div(df["ghi_clear"].replace(0, np.nan))
    return result.clip(lower=0, upper=1.2).fillna(0.0)


def physics_yield(
    df: pd.DataFrame,
    lat: float,
    system_kw: float = 5.0,
    tilt: float | None = None,
    azimuth: float = 180,
    gamma: float = DEFAULT_GAMMA_PER_C,
    noct: float = DEFAULT_NOCT_C,
    pr: float = DEFAULT_PERFORMANCE_RATIO,
    soiling: float = DEFAULT_SOILING,
) -> pd.Series:
    """Estimate daily AC energy in kWh from daily weather.

    Equations:
      POA = GHI * max(cos(lat - tilt - declination), 0.05)
            / max(cos(lat - declination), 0.05)
      T_cell = T_amb + (NOCT - 20) / 800 * G_avg
      temperature_factor = 1 + gamma * (T_cell - 25)
      E = P_stc * POA * temperature_factor * PR * (1 - soiling)

    ``pr`` covers wiring, inverter, mismatch, and other system losses. The
    default 3% soiling penalty is a modest allowance for dust in Pakistan;
    users can change it for cleaning schedules or site conditions. ``azimuth``
    is exposed for a future orientation model; the current first-order model
    assumes a south-facing panel (180 degrees).
    """
    del azimuth  # The requested first-order equation is south-facing only.
    if system_kw < 0:
        raise ValueError("system_kw must be non-negative")
    if not 0 <= pr <= 1:
        raise ValueError("pr must be between 0 and 1")
    if not 0 <= soiling <= 1:
        raise ValueError("soiling must be between 0 and 1")
    required = {"ghi", "t_amb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weather data is missing required columns: {sorted(missing)}")

    ghi = df["ghi"].astype(float).clip(lower=0).fillna(0)
    poa = plane_of_array_irradiance(ghi, lat=lat, tilt=tilt)
    day = poa.index.dayofyear.to_numpy(dtype=float)
    daylight = pd.Series(daylight_hours(lat, day), index=poa.index).clip(lower=1.0)
    daytime_average = poa.mul(1000).div(daylight)
    t_cell = df["t_amb"].astype(float).fillna(25).add((noct - 20.0) / 800.0 * daytime_average)
    temperature_factor = (1.0 + gamma * (t_cell - 25.0)).clip(lower=0)
    energy = poa.mul(float(system_kw)).mul(temperature_factor).mul(float(pr) * (1.0 - soiling))
    return energy.where(ghi > 0, 0.0).rename("physics_yield")