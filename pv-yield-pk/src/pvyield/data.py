"""NASA POWER daily weather client and cleaning utilities."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cities import CITIES

logger = logging.getLogger(__name__)

NASA_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = (
    "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,"
    "RH2M,WS2M,PRECTOTCORR"
)
NASA_FILL_VALUE = -999.0
NASA_FIRST_DAY = date(1981, 1, 1)
# NASA POWER daily data typically lags real time by a few days; ranges that end
# this close to "today" are not cached because the tail can still be revised.
RECENT_DAYS_NOT_CACHED = 10
MAX_GAP_DAYS = 3
REQUEST_TIMEOUT_S = 45
OUTPUT_COLUMNS = ["ghi", "ghi_clear", "t_amb", "t_max", "t_min", "rh", "wind", "precip"]
COLUMN_MAP = {
    "ALLSKY_SFC_SW_DWN": "ghi",
    "CLRSKY_SFC_SW_DWN": "ghi_clear",
    "T2M": "t_amb",
    "T2M_MAX": "t_max",
    "T2M_MIN": "t_min",
    "RH2M": "rh",
    "WS2M": "wind",
    "PRECTOTCORR": "precip",
}


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cache"


def _safe_cache_name(city: str, start: date, end: date) -> str:
    safe_city = re.sub(r"[^A-Za-z0-9_-]+", "_", city.strip())
    return f"{safe_city}_{start.isoformat()}_{end.isoformat()}.csv"


def _session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def parse_dates(start: str | date, end: str | date) -> tuple[date, date]:
    """Validate and normalise a NASA POWER date range."""
    try:
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
    except (ValueError, TypeError) as exc:
        raise ValueError("start and end must be ISO dates such as 2023-01-01") from exc
    if end_date < start_date:
        raise ValueError("end must be on or after start")
    if start_date < NASA_FIRST_DAY:
        raise ValueError(f"NASA POWER daily data starts on {NASA_FIRST_DAY.isoformat()}")
    if end_date > date.today():
        raise ValueError("end cannot be in the future")
    return start_date, end_date


def _as_daily_frame(parameters: dict[str, Any]) -> pd.DataFrame:
    """Turn NASA's ``{parameter: {YYYYMMDD: value}}`` mapping into a daily frame."""
    frame = pd.DataFrame.from_dict(parameters, orient="columns")
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="coerce")
    frame = frame.loc[frame.index.notna()]
    frame = frame.rename(columns=COLUMN_MAP).reindex(columns=OUTPUT_COLUMNS)
    frame.index.name = "date"
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def clean_daily_frame(frame: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Replace NASA fill values, forward-fill short gaps, and drop longer gaps.

    Missing calendar days are made explicit first, so a run of absent dates is
    treated exactly like a run of ``-999`` values. The returned frame may still
    have calendar holes where long gaps were removed; downstream time-series
    code must re-index to daily frequency before shifting.
    """
    cleaned = frame.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    if cleaned.empty:
        return cleaned
    cleaned = cleaned.asfreq("D")
    cleaned = cleaned.mask(np.isclose(cleaned.to_numpy(dtype=float), NASA_FILL_VALUE))
    required_columns = [column for column in OUTPUT_COLUMNS if column in cleaned.columns]
    long_gap = pd.Series(False, index=cleaned.index)
    for column in required_columns:
        missing = cleaned[column].isna()
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(groups).transform("sum")
        long_gap |= missing & run_lengths.gt(max_gap_days)
    cleaned = cleaned.ffill(limit=max_gap_days)
    cleaned.index.name = "date"
    return cleaned.loc[~long_gap].dropna(subset=required_columns)


def fetch_daily(city: str, start: str | date, end: str | date, use_cache: bool = True) -> pd.DataFrame:
    """Fetch cleaned daily NASA POWER weather data, using a local CSV cache."""
    if city not in CITIES:
        raise ValueError(f"Unknown city {city!r}. Choose one of: {', '.join(sorted(CITIES))}")
    start_date, end_date = parse_dates(start, end)

    cache_dir = _cache_dir()
    cache_path = cache_dir / _safe_cache_name(city, start_date, end_date)
    if use_cache and cache_path.exists():
        cached = clean_daily_frame(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        if not cached.empty:
            logger.debug("Loaded %s from cache", cache_path.name)
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
        "time-standard": "LST",
    }
    with _session() as session:
        response = session.get(NASA_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    try:
        payload = response.json()
        raw = _as_daily_frame(payload["properties"]["parameter"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("NASA POWER returned an unexpected response shape") from exc
    if raw.empty:
        raise RuntimeError("NASA POWER returned no dated rows for this request")

    cleaned = clean_daily_frame(raw)
    recent = end_date > date.today() - timedelta(days=RECENT_DAYS_NOT_CACHED)
    if use_cache and not cleaned.empty and not recent:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(cache_path, index_label="date")
    return cleaned
