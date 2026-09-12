import numpy as np
import pandas as pd

from pvyield.physics import physics_yield


def weather_frame(ghi=5.0, temperature=25.0, days=3):
    index = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {"ghi": ghi, "ghi_clear": 6.0, "t_amb": temperature},
        index=index,
    )


def test_zero_ghi_produces_zero_energy():
    result = physics_yield(weather_frame(ghi=0), lat=31.45)
    assert np.allclose(result, 0.0)


def test_higher_ambient_temperature_reduces_energy_for_same_ghi():
    cool = physics_yield(weather_frame(temperature=10), lat=31.45)
    hot = physics_yield(weather_frame(temperature=45), lat=31.45)
    assert (hot < cool).all()


def test_energy_scales_linearly_with_system_size():
    one_kw = physics_yield(weather_frame(), lat=31.45, system_kw=1)
    five_kw = physics_yield(weather_frame(), lat=31.45, system_kw=5)
    assert np.allclose(five_kw, one_kw * 5)


def test_latitude_tilt_beats_flat_panel_on_winter_day():
    frame = weather_frame(ghi=5.0, days=1)
    tilted = physics_yield(frame, lat=31.45, tilt=31.45)
    flat = physics_yield(frame, lat=31.45, tilt=0)
    assert tilted.iloc[0] >= flat.iloc[0]