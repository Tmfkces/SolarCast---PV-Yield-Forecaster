import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_weather(days: int = 400, start: str = "2022-01-01", seed: int = 0) -> pd.DataFrame:
    """Synthetic but physically plausible daily weather for Faisalabad-like latitudes."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D")
    season = np.sin(2 * np.pi * (index.day_of_year.to_numpy() - 80) / 365.25)
    clear = 5.6 + 2.0 * season
    cloudiness = np.clip(rng.normal(0.85, 0.12, days), 0.2, 1.0)
    ghi = clear * cloudiness
    t_amb = 24 + 10 * season + rng.normal(0, 1.5, days)
    return pd.DataFrame(
        {
            "ghi": ghi,
            "ghi_clear": clear,
            "t_amb": t_amb,
            "t_max": t_amb + 6,
            "t_min": t_amb - 6,
            "rh": np.clip(50 - 20 * season + rng.normal(0, 5, days), 5, 100),
            "wind": np.abs(rng.normal(2.2, 0.6, days)),
            "precip": np.where(cloudiness < 0.6, rng.gamma(2.0, 2.0, days), 0.0),
        },
        index=index,
    )


@pytest.fixture
def weather() -> pd.DataFrame:
    return make_weather()


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    import pvyield.data as data

    monkeypatch.setattr(data, "_cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "cache"
