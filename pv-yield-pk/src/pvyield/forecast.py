"""7-day PV outlook from a weather forecast, with plain-language energy advice.

Historical analysis uses NASA POWER, which has no future data. The outlook
uses the free Open-Meteo forecast API (no key) for daily and hourly solar
radiation, runs the same physics model, and compares each day with the
city's usual yield for that month, computed from Open-Meteo's own archive
of past forecasts so both sides come from the same models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from . import data as _data
from .cities import CITIES
from .physics import physics_yield

OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORY_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Karachi"
MJ_TO_KWH = 1 / 3.6
DAILY_FIELDS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "sunrise",
    "sunset",
]
HOURLY_FIELDS = ["shortwave_radiation", "temperature_2m", "cloud_cover"]

# Ratio of forecast yield to the typical yield for that month.
HIGH_RATIO = 1.10
LOW_RATIO = 0.85
VERY_LOW_RATIO = 0.60
HEAVY_LOAD_WINDOW_HOURS = 3

# WMO weather codes → (icon, short description)
_WEATHER_CODES: list[tuple[range | tuple[int, ...], str, str]] = [
    ((0,), "☀️", "Clear"),
    ((1,), "🌤️", "Mostly sunny"),
    ((2,), "⛅", "Partly cloudy"),
    ((3,), "☁️", "Overcast"),
    ((45, 48), "🌫️", "Fog / haze"),
    (range(51, 58), "🌦️", "Drizzle"),
    (range(61, 68), "🌧️", "Rain"),
    (range(71, 78), "❄️", "Snow"),
    (range(80, 83), "🌦️", "Showers"),
    ((85, 86), "🌨️", "Snow showers"),
    (range(95, 100), "⛈️", "Thunderstorm"),
]


def weather_icon(code: float | int | None) -> tuple[str, str]:
    """Map a WMO weather code to an emoji and a short label."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "❔", "Unknown"
    for codes, icon, label in _WEATHER_CODES:
        if int(code) in codes:
            return icon, label
    return "🌥️", "Cloudy"


def fetch_forecast(city: str, days: int = 7, timeout: float = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(daily, hourly)`` Open-Meteo forecasts in Pakistan local time.

    ``daily`` has ``ghi`` (kWh/m²/day) and ``t_amb`` (°C) ready for
    :func:`pvyield.physics.physics_yield`; ``hourly`` has ``ghi_w_m2`` where
    each timestamp is the *end* of a one-hour averaging period.
    """
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}")
    if not 1 <= days <= 16:
        raise ValueError("days must be between 1 and 16")
    location = CITIES[city]
    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "daily": ",".join(DAILY_FIELDS),
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": TIMEZONE,
        "forecast_days": days,
    }
    response = requests.get(OPEN_METEO_ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    result = parse_forecast(payload)
    _save_forecast(city, payload)
    return result


def _forecast_path(city: str) -> Path:
    return _data._cache_dir() / f"openmeteo_forecast_{re.sub(r'[^A-Za-z0-9_-]+', '_', city)}.json"


def _save_forecast(city: str, payload: dict[str, Any]) -> None:
    """Keep the last good forecast so the app still works offline."""
    try:
        path = _forecast_path(city)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
        path.write_text(json.dumps(stamped), encoding="utf-8")
    except OSError:  # pragma: no cover - read-only disks just skip the offline copy
        pass


def load_saved_forecast(city: str) -> tuple[pd.DataFrame, pd.DataFrame, datetime] | None:
    """Return the last forecast saved by :func:`fetch_forecast`, or ``None``."""
    path = _forecast_path(city)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        daily, hourly = parse_forecast(payload)
        saved_at = datetime.fromisoformat(payload["saved_at"])
    except (OSError, ValueError, KeyError, RuntimeError):
        return None
    return daily, hourly, saved_at


def fetch_forecast_history(city: str, years: int = 4, timeout: float = 120) -> pd.DataFrame:
    """Past daily weather *from the same forecast models* as :func:`fetch_forecast`.

    Comparing a forecast with NASA POWER climatology mixes two datasets whose
    radiation differs by up to ~40 % in some months (e.g. Skardu in winter).
    Open-Meteo's historical-forecast archive removes that bias. Results are
    cached on disk because the request is slow (10–25 s).
    """
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}")
    last_year = date.today().year - 1
    start, end = date(last_year - years + 1, 1, 1), date(last_year, 12, 31)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", city)
    path = _data._cache_dir() / f"openmeteo_history_{safe}_{start}_{end}.csv"
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if not cached.empty:
            return cached
    location = CITIES[city]
    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "shortwave_radiation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": TIMEZONE,
    }
    response = requests.get(OPEN_METEO_HISTORY_ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()
    try:
        frame = pd.DataFrame(response.json()["daily"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Open-Meteo history returned an unexpected response shape") from exc
    frame.index = pd.to_datetime(frame.pop("time"))
    history = pd.DataFrame(
        {
            "ghi": pd.to_numeric(frame["shortwave_radiation_sum"], errors="coerce") * MJ_TO_KWH,
            "t_amb": (frame["temperature_2m_max"] + frame["temperature_2m_min"]) / 2,
        }
    ).dropna()
    if not history.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(path, index_label="date")
    return history


def parse_forecast(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        daily = pd.DataFrame(payload["daily"])
        hourly = pd.DataFrame(payload["hourly"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Open-Meteo returned an unexpected response shape") from exc
    daily.index = pd.to_datetime(daily.pop("time"))
    hourly.index = pd.to_datetime(hourly.pop("time"))
    daily["ghi"] = pd.to_numeric(daily["shortwave_radiation_sum"], errors="coerce") * MJ_TO_KWH
    daily["t_amb"] = (daily["temperature_2m_max"] + daily["temperature_2m_min"]) / 2
    hourly = hourly.rename(columns={"shortwave_radiation": "ghi_w_m2", "temperature_2m": "t_air"})
    return daily, hourly


def typical_daily_yield(history: pd.DataFrame, lat: float, statistic: str = "median", **physics_kwargs: Any) -> pd.Series:
    """Usual daily physics yield (kWh) for each calendar month 1–12.

    The median is the default: a "usual day" should not be dragged down by a
    handful of storm days the way a mean is.
    """
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    daily = physics_yield(history, lat=lat, **physics_kwargs).dropna()
    by_month = daily.groupby(daily.index.month).agg(statistic)
    return by_month.reindex(range(1, 13)).interpolate(limit_direction="both")


# Typical DISCO peak-tariff hours in Pakistan by season (start hour, end hour), local time.
# They vary slightly by distribution company; users should confirm on their bill.
PEAK_HOURS_BY_MONTH = {
    12: (17, 21), 1: (17, 21), 2: (17, 21),
    3: (18, 22), 4: (18, 22), 5: (18, 22),
    6: (19, 23), 7: (19, 23), 8: (19, 23),
    9: (18, 22), 10: (18, 22), 11: (18, 22),
}  # fmt: skip
SETUPS = {
    "net_metering": "Net metering (grid-tied)",
    "battery": "Battery / hybrid inverter",
    "grid_only": "On-grid, no net metering",
}


def peak_hours(month: int) -> tuple[int, int]:
    return PEAK_HOURS_BY_MONTH[month]


def peak_label(month: int) -> str:
    start, end = peak_hours(month)
    return f"{clock(pd.Timestamp(2000, 1, 1, start))}–{clock(pd.Timestamp(2000, 1, 1, end % 24))}"


def estimate_value(
    yield_kwh: float,
    price: float,
    setup: str = "net_metering",
    export_rate: float = 10.0,
    daily_use_kwh: float = 12.0,
    battery_kwh: float = 5.0,
    daytime_share: float = 0.5,
) -> dict[str, float]:
    """Split a day's solar into self-used and exported kWh and value it in rupees.

    Self-use is capped by the household's daytime consumption (plus battery
    storage for hybrid systems). Only net-metered homes earn ``export_rate``
    for the rest; other setups waste it.
    """
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {sorted(SETUPS)}")
    capacity = daily_use_kwh * daytime_share
    if setup == "battery":
        capacity = min(daily_use_kwh, capacity + battery_kwh)
    self_kwh = max(0.0, min(yield_kwh, capacity))
    surplus = max(0.0, yield_kwh - self_kwh)
    export_kwh = surplus if setup == "net_metering" else 0.0
    rupees = self_kwh * price + export_kwh * export_rate
    return {"self_kwh": self_kwh, "export_kwh": export_kwh, "unused_kwh": surplus - export_kwh, "rupees": rupees}


@dataclass
class DayAdvice:
    level: str  # "high" | "normal" | "low" | "very_low"
    headline: str
    detail: str
    best_window: tuple[pd.Timestamp, pd.Timestamp] | None


def best_window(hourly_energy: pd.Series, hours: int = HEAVY_LOAD_WINDOW_HOURS) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Start and end clock times of the ``hours``-long block with most solar energy.

    ``hourly_energy`` is indexed by hour *end* times (Open-Meteo convention).
    """
    energy = hourly_energy.dropna()
    if energy.sum() <= 0 or len(energy) < hours:
        return None
    rolling = energy.rolling(hours).sum()
    end = rolling.idxmax()
    return end - pd.Timedelta(hours=hours), end


def clock(timestamp: pd.Timestamp) -> str:
    """12-hour clock label such as ``11 AM`` or ``2 PM``."""
    return timestamp.strftime("%I %p").lstrip("0")


_SETUP_TIPS = {
    ("battery", True): " Let the battery fill by mid-afternoon so it carries the evening.",
    ("battery", False): " Keep battery charge for the evening and load-shedding.",
    ("net_metering", True): " Using solar yourself saves more than exporting it at the lower buyback rate.",
    ("net_metering", False): "",
    ("grid_only", True): " Without net metering, unused solar is wasted, so use it while the sun is up.",
    ("grid_only", False): "",
}


def advise(
    forecast_kwh: float,
    typical_kwh: float,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    setup: str = "net_metering",
    month: int | None = None,
) -> DayAdvice:
    """Turn a day's forecast into a plain-language recommendation.

    The base advice depends on how the day compares with a usual day; the
    home ``setup`` and the month's peak-tariff hours add Pakistan-specific tips.
    """
    advice = _base_advice(forecast_kwh, typical_kwh, window)
    sunny = advice.level in {"high", "normal"}
    advice.detail += _SETUP_TIPS.get((setup, sunny), "")
    if month is not None:
        advice.detail += f" Avoid heavy loads {peak_label(month)}, when peak tariff applies."
    return advice


def _base_advice(forecast_kwh: float, typical_kwh: float, window: tuple[pd.Timestamp, pd.Timestamp] | None) -> DayAdvice:
    ratio = forecast_kwh / typical_kwh if typical_kwh > 0 else 1.0
    when = f"{clock(window[0])}–{clock(window[1])}" if window else "midday"
    percent = abs(ratio - 1) * 100
    if ratio >= HIGH_RATIO:
        return DayAdvice(
            "high",
            f"Sunny surplus: {percent:.0f}% above a typical day",
            f"Good day for heavy loads. Run the washing machine, iron, water pump, water heater or AC between {when}, "
            "and charge batteries or devices then.",
            window,
        )
    if ratio >= LOW_RATIO:
        return DayAdvice(
            "normal",
            "About a typical day",
            f"Normal solar output. Schedule heavy appliances between {when} to use your own power.",
            window,
        )
    if ratio >= VERY_LOW_RATIO:
        return DayAdvice(
            "low",
            f"Below average: {percent:.0f}% less sun than usual",
            f"Save electricity. Run only essential heavy loads, and keep them to {when} when output peaks. "
            "Postpone laundry or ironing to a sunnier day if you can.",
            window,
        )
    return DayAdvice(
        "very_low",
        f"Very low solar: {percent:.0f}% below a typical day",
        "Cloudy or rainy day. Avoid heavy appliances and shift laundry, ironing and pumping to the next sunny day.",
        window,
    )


@dataclass
class Outlook:
    days: pd.DataFrame  # one row per day: yield_kwh, typical_kwh, ratio, icon, label, advice …
    hourly: pd.DataFrame  # hourly ghi_w_m2 and energy_kwh


def build_outlook(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    typical: pd.Series,
    lat: float,
    system_kw: float = 5.0,
    setup: str = "net_metering",
    **physics_kwargs: Any,
) -> Outlook:
    """Combine a forecast with the physics model and usual-month yields."""
    yield_kwh = physics_yield(daily[["ghi", "t_amb"]], lat=lat, system_kw=system_kw, **physics_kwargs)

    hourly = hourly.copy()
    # Hour-end timestamps: 01:00 is the 00:00–01:00 period of that calendar day.
    hour_day = (hourly.index - pd.Timedelta(minutes=1)).normalize()
    ghi = hourly["ghi_w_m2"].clip(lower=0).fillna(0)
    share = ghi / ghi.groupby(hour_day).transform("sum").replace(0, np.nan)
    hourly["energy_kwh"] = (share * yield_kwh.reindex(hour_day).to_numpy()).fillna(0)

    rows = []
    for day, kwh in yield_kwh.items():
        typical_kwh = float(typical.get(day.month, np.nan))
        day_hours = hourly.loc[hour_day == day, "energy_kwh"]
        window = best_window(day_hours)
        icon, label = weather_icon(daily.loc[day, "weather_code"] if "weather_code" in daily else None)
        advice = advise(float(kwh), typical_kwh, window, setup=setup, month=day.month)
        rows.append(
            {
                "date": day,
                "yield_kwh": float(kwh),
                "typical_kwh": typical_kwh,
                "ratio": float(kwh) / typical_kwh if typical_kwh > 0 else np.nan,
                "icon": icon,
                "label": label,
                "t_max": daily.loc[day].get("temperature_2m_max", np.nan),
                "t_min": daily.loc[day].get("temperature_2m_min", np.nan),
                "rain_chance": daily.loc[day].get("precipitation_probability_max", np.nan),
                "level": advice.level,
                "headline": advice.headline,
                "detail": advice.detail,
                "window_start": window[0] if window else pd.NaT,
                "window_end": window[1] if window else pd.NaT,
                "sunrise": pd.to_datetime(daily.loc[day].get("sunrise"), errors="coerce"),
                "sunset": pd.to_datetime(daily.loc[day].get("sunset"), errors="coerce"),
            }
        )
    return Outlook(days=pd.DataFrame(rows).set_index("date"), hourly=hourly)


# Typical Pakistani household loads: (power kW, usual run time in hours, emoji)
APPLIANCES: dict[str, tuple[float, float, str]] = {
    "Washing machine": (0.5, 1.0, "🧺"),
    "Iron": (1.0, 0.5, "👔"),
    "Water pump (1 hp)": (0.75, 1.0, "💧"),
    "Electric geyser": (2.0, 1.0, "🚿"),
    "Inverter AC (1.5 ton)": (1.4, 3.0, "❄️"),
    "Microwave / oven": (1.2, 0.5, "🍲"),
    "Electric kettle": (1.5, 0.2, "☕"),
    "EV / e-bike charging": (1.0, 3.0, "🔋"),
}


@dataclass
class AppliancePlan:
    need_kwh: float
    solar_kwh: float
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    fits: bool

    @property
    def coverage(self) -> float:
        return min(self.solar_kwh / self.need_kwh, 1.0) if self.need_kwh > 0 else 1.0


def plan_appliances(selected: list[str], hourly_energy: pd.Series, base_load_kw: float = 0.3) -> AppliancePlan:
    """Find the shortest run of daylight hours whose spare solar covers the chosen appliances.

    ``hourly_energy`` is one day of hourly PV energy (kWh, hour-end stamps).
    ``base_load_kw`` (fridge, fans, lights) is subtracted from every hour first.
    The window grows outward from the sunniest hour until the need is met or
    the day runs out.
    """
    need = sum(APPLIANCES[name][0] * APPLIANCES[name][1] for name in selected)
    spare = (hourly_energy.dropna() - base_load_kw).clip(lower=0)
    if need <= 0 or spare.sum() <= 0:
        return AppliancePlan(need, 0.0, None, None, need <= 0)
    values = spare.to_numpy()
    lo = hi = int(np.argmax(values))
    total = values[lo]
    while total < need and (lo > 0 or hi < len(values) - 1):
        left = values[lo - 1] if lo > 0 else -1.0
        right = values[hi + 1] if hi < len(values) - 1 else -1.0
        if right >= left:
            hi += 1
            total += right
        else:
            lo -= 1
            total += left
    while lo < hi and values[lo] <= 0:
        lo += 1
    while hi > lo and values[hi] <= 0:
        hi -= 1
    start = spare.index[lo] - pd.Timedelta(hours=1)
    end = spare.index[hi]
    return AppliancePlan(float(need), float(min(total, need)), start, end, bool(total >= need))
