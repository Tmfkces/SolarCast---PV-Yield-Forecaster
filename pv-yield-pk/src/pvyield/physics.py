"""Transparent daily physics model for PV energy yield.

The model works on *daily* weather (NASA POWER) and follows the classic
textbook chain used for daily solar-resource studies (Duffie & Beckman,
*Solar Engineering of Thermal Processes*):

1. Extraterrestrial daily irradiation ``H0`` from latitude and day of year.
2. Daily clearness index ``Kt = H / H0``.
3. Diffuse fraction ``Hd / H`` from the Erbs et al. (1982) daily correlation.
4. Plane-of-array (POA) irradiation with the Liu-Jordan isotropic-sky model::

       H_T = Hb * Rb + Hd * (1 + cos β) / 2 + H * ρ * (1 - cos β) / 2

   where ``Rb`` is the daily beam tilt factor, integrated numerically over the
   hour angle so that *any* surface azimuth is supported (Klein 1977 gives the
   closed form for south-facing surfaces; the tests check the two agree).
5. Cell temperature from NOCT, a linear temperature coefficient, and a
   performance ratio plus soiling allowance.

A legacy ``"geometric"`` transposition (noon beam ratio applied to all of GHI)
is kept for comparison with earlier versions of this project.

Every quantity is a daily total or a daytime average; this is a first-order,
inspectable model, not a bankable hourly simulation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

DEG_TO_RAD = np.pi / 180.0
SOLAR_CONSTANT_W_M2 = 1_367.0
DEFAULT_NOCT_C = 45.0
DEFAULT_GAMMA_PER_C = -0.0040
DEFAULT_PERFORMANCE_RATIO = 0.80
DEFAULT_SOILING = 0.03
DEFAULT_ALBEDO = 0.20
DEFAULT_AZIMUTH_DEG = 180.0  # compass convention: 180 = due south
MAX_BEAM_RATIO = 10.0

Transposition = Literal["isotropic", "geometric"]

# Hour-angle grid (degrees) used to integrate the daily beam tilt factor.
_HOUR_ANGLES = np.linspace(-180.0, 180.0, 721) * DEG_TO_RAD


def _as_doy(day_of_year: np.ndarray | pd.Index | float) -> np.ndarray:
    return np.asarray(day_of_year, dtype=float)


def _declination(day_of_year: np.ndarray) -> np.ndarray:
    """Approximate solar declination in degrees (Cooper 1969)."""
    return 23.45 * np.sin(DEG_TO_RAD * (360.0 * (284.0 + _as_doy(day_of_year)) / 365.0))


def sunset_hour_angle(lat: float, day_of_year: np.ndarray) -> np.ndarray:
    """Return the sunset hour angle in degrees for a horizontal surface."""
    latitude = lat * DEG_TO_RAD
    declination = _declination(day_of_year) * DEG_TO_RAD
    return np.degrees(np.arccos(np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)))


def daylight_hours(lat: float, day_of_year: np.ndarray) -> np.ndarray:
    """Return astronomical daylight duration in hours."""
    return 2.0 * sunset_hour_angle(lat, day_of_year) / 15.0


def extraterrestrial_daily(lat: float, day_of_year: np.ndarray) -> np.ndarray:
    """Daily extraterrestrial irradiation on a horizontal plane, kWh/m²/day.

    Duffie & Beckman eq. 1.10.3.
    """
    doy = _as_doy(day_of_year)
    latitude = lat * DEG_TO_RAD
    declination = _declination(doy) * DEG_TO_RAD
    ws = sunset_hour_angle(lat, doy) * DEG_TO_RAD
    eccentricity = 1.0 + 0.033 * np.cos(DEG_TO_RAD * 360.0 * doy / 365.0)
    geometry = (
        np.cos(latitude) * np.cos(declination) * np.sin(ws)
        + ws * np.sin(latitude) * np.sin(declination)
    )
    wh = 24.0 / np.pi * SOLAR_CONSTANT_W_M2 * eccentricity * geometry
    return np.maximum(wh, 0.0) / 1000.0


def diffuse_fraction_erbs(clearness_index: np.ndarray, sunset_angle_deg: np.ndarray) -> np.ndarray:
    """Daily diffuse fraction ``Hd/H`` from the Erbs et al. (1982) correlation."""
    kt = np.clip(np.asarray(clearness_index, dtype=float), 0.0, 1.0)
    ws = np.broadcast_to(np.asarray(sunset_angle_deg, dtype=float), kt.shape)
    short_day = 1.0 - 0.2727 * kt + 2.4495 * kt**2 - 11.9514 * kt**3 + 9.3879 * kt**4
    short_day = np.where(kt < 0.715, short_day, 0.143)
    long_day = 1.0 + 0.2832 * kt - 2.5557 * kt**2 + 0.8448 * kt**3
    long_day = np.where(kt < 0.722, long_day, 0.175)
    fraction = np.where(ws <= 81.4, short_day, long_day)
    return np.clip(fraction, 0.0, 1.0)


def _cos_incidence(
    lat_rad: float,
    decl_rad: np.ndarray,
    omega: np.ndarray,
    tilt_rad: float,
    gamma_rad: float,
) -> np.ndarray:
    """Cosine of the angle of incidence (Duffie & Beckman eq. 1.6.2).

    ``gamma_rad`` is the surface azimuth measured from south, east negative.
    """
    sin_d, cos_d = np.sin(decl_rad), np.cos(decl_rad)
    sin_p, cos_p = np.sin(lat_rad), np.cos(lat_rad)
    sin_b, cos_b = np.sin(tilt_rad), np.cos(tilt_rad)
    return (
        sin_d * sin_p * cos_b
        - sin_d * cos_p * sin_b * np.cos(gamma_rad)
        + cos_d * cos_p * cos_b * np.cos(omega)
        + cos_d * sin_p * sin_b * np.cos(gamma_rad) * np.cos(omega)
        + cos_d * sin_b * np.sin(gamma_rad) * np.sin(omega)
    )


def beam_tilt_factor(
    lat: float,
    tilt: float,
    azimuth: float,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """Daily beam tilt factor ``Rb`` for any tilt and compass azimuth.

    ``Rb`` is the ratio of daily extraterrestrial beam irradiation on the tilted
    plane to that on a horizontal plane, integrated over the hour angle while
    the sun is above both the horizon and the plane of the array.
    """
    doy = _as_doy(day_of_year)
    decl = (_declination(doy) * DEG_TO_RAD)[:, None]
    omega = _HOUR_ANGLES[None, :]
    lat_rad = lat * DEG_TO_RAD
    horizontal = _cos_incidence(lat_rad, decl, omega, 0.0, 0.0)
    tilted = _cos_incidence(lat_rad, decl, omega, tilt * DEG_TO_RAD, (azimuth - 180.0) * DEG_TO_RAD)
    sun_up = horizontal > 0
    numerator = np.where(sun_up & (tilted > 0), tilted, 0.0).sum(axis=1)
    denominator = np.where(sun_up, horizontal, 0.0).sum(axis=1)
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9)
    return np.clip(ratio, 0.0, MAX_BEAM_RATIO)


def _validate_geometry(lat: float, tilt: float, azimuth: float) -> None:
    if not -66.0 <= lat <= 66.0:
        raise ValueError("lat must be between -66 and 66 degrees (polar days are not modelled)")
    if not 0.0 <= tilt <= 90.0:
        raise ValueError("tilt must be between 0 and 90 degrees")
    if not 0.0 <= azimuth <= 360.0:
        raise ValueError("azimuth must be between 0 and 360 degrees (compass, 180 = south)")


def plane_of_array_irradiance(
    ghi: pd.Series,
    lat: float,
    tilt: float | None = None,
    azimuth: float = DEFAULT_AZIMUTH_DEG,
    albedo: float = DEFAULT_ALBEDO,
    transposition: Transposition = "isotropic",
) -> pd.Series:
    """Estimate daily plane-of-array irradiation (kWh/m²/day) from daily GHI.

    ``tilt`` defaults to the site latitude. ``azimuth`` uses the compass
    convention (180° = south, 90° = east, 270° = west).
    """
    panel_tilt = float(abs(lat) if tilt is None else tilt)
    _validate_geometry(lat, panel_tilt, azimuth)
    if not 0.0 <= albedo <= 1.0:
        raise ValueError("albedo must be between 0 and 1")
    if not isinstance(ghi.index, pd.DatetimeIndex):
        raise TypeError("ghi must have a DatetimeIndex")

    horizontal = ghi.astype(float).clip(lower=0)
    doy = ghi.index.day_of_year.to_numpy(dtype=float)

    if transposition == "geometric":
        decl = _declination(doy)
        numerator = np.cos(DEG_TO_RAD * (lat - panel_tilt - decl))
        denominator = np.maximum(np.cos(DEG_TO_RAD * (lat - decl)), 0.05)
        factor = np.maximum(numerator, 0.05) / denominator
        return horizontal.mul(factor).rename("poa")
    if transposition != "isotropic":
        raise ValueError("transposition must be 'isotropic' or 'geometric'")

    h0 = extraterrestrial_daily(lat, doy)
    kt = np.divide(horizontal.to_numpy(), h0, out=np.zeros(len(h0)), where=h0 > 0)
    diffuse = horizontal.to_numpy() * diffuse_fraction_erbs(kt, sunset_hour_angle(lat, doy))
    beam = horizontal.to_numpy() - diffuse
    rb = beam_tilt_factor(lat, panel_tilt, azimuth, doy)
    cos_tilt = np.cos(panel_tilt * DEG_TO_RAD)
    poa = beam * rb + diffuse * (1.0 + cos_tilt) / 2.0 + horizontal.to_numpy() * albedo * (1.0 - cos_tilt) / 2.0
    return pd.Series(poa, index=ghi.index, name="poa")


def clear_sky_index(df: pd.DataFrame) -> pd.Series:
    """Return GHI divided by clear-sky GHI, clipped to a physical range."""
    result = df["ghi"].div(df["ghi_clear"].replace(0, np.nan))
    return result.clip(lower=0, upper=1.2).fillna(0.0).rename("clear_sky_index")


def physics_yield(
    df: pd.DataFrame,
    lat: float,
    system_kw: float = 5.0,
    tilt: float | None = None,
    azimuth: float = DEFAULT_AZIMUTH_DEG,
    gamma: float = DEFAULT_GAMMA_PER_C,
    noct: float = DEFAULT_NOCT_C,
    pr: float = DEFAULT_PERFORMANCE_RATIO,
    soiling: float = DEFAULT_SOILING,
    albedo: float = DEFAULT_ALBEDO,
    transposition: Transposition = "isotropic",
) -> pd.Series:
    """Estimate daily AC energy in kWh from daily weather.

    Equations::

      POA       = Hb·Rb + Hd·(1 + cos β)/2 + H·ρ·(1 − cos β)/2
      G_avg     = 1000 · POA / daylight_hours                 (W/m²)
      T_cell    = T_amb + (NOCT − 20) / 800 · G_avg
      f_T       = 1 + γ · (T_cell − 25)
      E         = P_stc · POA · f_T · PR · (1 − soiling)

    ``pr`` covers wiring, inverter, mismatch, and other system losses but *not*
    temperature or soiling, which are modelled separately. Days with missing
    GHI or air temperature return ``NaN`` (not zero) so that gaps never look
    like dark days to downstream code.
    """
    if system_kw < 0:
        raise ValueError("system_kw must be non-negative")
    if not 0 <= pr <= 1:
        raise ValueError("pr must be between 0 and 1")
    if not 0 <= soiling <= 1:
        raise ValueError("soiling must be between 0 and 1")
    if noct <= 20:
        raise ValueError("noct must be above 20 °C")
    required = {"ghi", "t_amb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weather data is missing required columns: {sorted(missing)}")

    ghi = df["ghi"].astype(float)
    poa = plane_of_array_irradiance(
        ghi.fillna(0.0), lat=lat, tilt=tilt, azimuth=azimuth, albedo=albedo, transposition=transposition
    )
    day = poa.index.day_of_year.to_numpy(dtype=float)
    daylight = pd.Series(daylight_hours(lat, day), index=poa.index).clip(lower=1.0)
    daytime_average = poa.mul(1000.0).div(daylight)
    t_cell = df["t_amb"].astype(float).add((noct - 20.0) / 800.0 * daytime_average)
    temperature_factor = (1.0 + gamma * (t_cell - 25.0)).clip(lower=0)
    energy = poa.mul(float(system_kw)).mul(temperature_factor).mul(float(pr) * (1.0 - soiling))
    energy = energy.where(ghi > 0, 0.0)
    energy = energy.mask(ghi.isna() | df["t_amb"].isna())
    return energy.rename("physics_yield")
