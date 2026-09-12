"""NASA POWER daily weather client and cleaning utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .cities import CITIES

NASA_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = (
    "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,"
    "RH2M,WS2M,PRECTOTCORR"
)
OUTPUT_COLUMNS = ["ghi", "ghi_clear", "t_amb", "t_max", "t_min", "rh", "wind", "precip"]


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cache"


def _safe_cache_name(city: str, start: str, end: str) -> str:
    safe_city = re.sub(r"[^A-Za-z0-9_-]+", "_", city.strip())
    return f"{safe_city}_{start}_{end}.csv"


def _as_daily_frame(parameters: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(parameters, orient="columns")
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="coerce")
    frame = frame.rename(
        columns={
            "ALLSKY_SFC_SW_DWN": "ghi",
            "CLRSKY_SFC_SW_DWN": "ghi_clear",
            "T2M": "t_amb",
            "T2M_MAX": "t_max",
            "T2M_MIN": "t_min",
            "RH2M": "rh",
            "WS2M": "wind",
            "PRECTOTCORR": "precip",
        }
    )
    frame = frame.reindex(columns=OUTPUT_COLUMNS)
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def clean_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace NASA fill values, fill short gaps, and drop longer gaps."""
    cleaned = frame.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.replace(-999, np.nan)
    cleaned = cleaned.replace(-999.0, np.nan)
    cleaned = cleaned.sort_index()
    required_columns = [column for column in OUTPUT_COLUMNS if column in cleaned.columns]
    long_gap = pd.Series(False, index=cleaned.index)
    for column in required_columns:
        missing = cleaned[column].isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(groups).transform("sum")
        long_gap |= missing & run_lengths.gt(3)
    cleaned = cleaned.ffill(limit=3)
    return cleaned.loc[~long_gap].dropna(subset=required_columns)


def fetch_daily(city: str, start: str, end: str) -> pd.DataFrame:
    """Fetch cleaned daily NASA POWER weather data, using a local CSV cache."""
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}. Choose one of: {', '.join(sorted(CITIES))}")
    try:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
    except ValueError as exc:
        raise ValueError("start and end must be ISO dates such as 2023-01-01") from exc
    if end_date < start_date:
        raise ValueError("end must be on or after start")

    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _safe_cache_name(city, start, end)
    if cache_path.exists():
        cached = clean_daily_frame(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        if not cached.empty:
            return cached

    location = CITIES[city]
    params = {
        "parameters": NASA_PARAMETERS,
        "community": "RE",
        "longitude": location["lon"],
        "latitude": location["lat"],
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = requests.get(NASA_ENDPOINT, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    try:
        raw = _as_daily_frame(payload["properties"]["parameter"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("NASA POWER returned an unexpected response shape") from exc
    cleaned = clean_daily_frame(raw)
    cleaned.to_csv(cache_path, index_label="date")
    return cleaned