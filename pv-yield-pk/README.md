# PV Yield Forecaster — Pakistan

## Problem

Rooftop solar is becoming an important way for Pakistani households and small businesses to manage high electricity costs and unreliable supply. A day-ahead estimate helps an owner plan consumption, understand seasonal production, and compare a proposed system size with expected demand. This project provides a transparent daily PV estimate for cities across Pakistan and compares it with a time-ordered machine-learning forecast. It is an analytical tool, not a bankable engineering guarantee.

## Data

The data layer retrieves daily NASA POWER data from the public Renewable Energy endpoint. It uses:

- `ALLSKY_SFC_SW_DWN` as daily global horizontal irradiance (`ghi`)
- `CLRSKY_SFC_SW_DWN` as clear-sky irradiance (`ghi_clear`)
- `T2M`, `T2M_MAX`, `T2M_MIN` for air temperature
- `RH2M`, `WS2M`, and `PRECTOTCORR` for humidity, wind, and precipitation

NASA's `-999` fill values become missing values. Gaps up to three days are forward-filled; longer gaps are discarded. Each `(city, start, end)` request is cached as a CSV under `data/cache/` so repeated analysis does not request the same interval again.

## Method

### Physics model

The physics model is intentionally inspectable and uses daily values. For a south-facing panel, the first-order plane-of-array transposition is:

$$
\mathrm{POA} =
\mathrm{GHI}
\frac{\max(\cos(\phi-\beta-\delta), 0.05)}
{\max(\cos(\phi-\delta), 0.05)}
$$

where $\phi$ is latitude, $\beta$ is panel tilt, and $\delta$ is solar declination. This is a geometric approximation; it does not split direct and diffuse irradiance.

Daytime-average irradiance estimates cell temperature:

$$
T_\mathrm{cell} =
T_\mathrm{amb} + \frac{\mathrm{NOCT}-20}{800}G_\mathrm{avg}
$$

Temperature derating and daily energy are:

$$
f_T = 1 + \gamma(T_\mathrm{cell}-25)
$$

$$
E_\mathrm{kWh} =
P_\mathrm{STC}\cdot\mathrm{POA}\cdot f_T\cdot
\mathrm{PR}\cdot(1-s)
$$

The default performance ratio is 0.80. It represents soiling, wiring, inverter, mismatch, and other system losses. The default extra soiling penalty is 3%, a modest allowance for dust between cleaning events in Pakistan; both are exposed in the UI.

### Machine learning model

The ML forecaster uses daily irradiance, clear-sky index, temperature, humidity, wind, precipitation, seasonal sine/cosine features, four GHI lags, and a seven-day rolling GHI mean. The target is the physics model's yield on the true next-day weather. This is a proxy target because metered plant output is not available: the model learns to forecast tomorrow's physics-based yield from today's weather history.

The default model is `LightGBMRegressor`, with `GradientBoostingRegressor` as a fallback if LightGBM's native runtime is unavailable. Data is split by time only: 70% training, 15% validation, and 15% test. The persistence baseline predicts tomorrow's yield as today's yield. Metrics include RMSE, MAE, MAPE excluding days below 0.5 kWh, R², and skill score against persistence.

## Results

The analysis notebook runs the following comparison end to end:

```python
from pvyield.evaluate import compare

for city in ["Faisalabad", "Karachi", "Quetta", "Skardu"]:
    print(city)
    print(compare(city, "2019-01-01", "2024-12-31", 5.0))
```

The benchmark below was run for a 5 kW system on 2019-01-01 through 2024-12-31. The environment used the GradientBoosting fallback because LightGBM's native runtime was unavailable; when LightGBM loads successfully, the third row is named `LightGBM`.

| City | Model | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² | Skill vs persistence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Faisalabad | Persistence | 4.2080 | 2.8008 | 18.6867 | 0.2775 | 0.0000 |
| Faisalabad | Physics upper bound | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Faisalabad | GradientBoosting | 3.6013 | 2.5617 | 17.9020 | 0.4708 | 0.1442 |
| Karachi | Persistence | 3.0326 | 1.8964 | 11.1145 | 0.4963 | 0.0000 |
| Karachi | Physics upper bound | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Karachi | GradientBoosting | 2.7299 | 1.7209 | 11.1286 | 0.5918 | 0.0998 |
| Quetta | Persistence | 4.3087 | 2.5899 | 12.6929 | 0.0495 | 0.0000 |
| Quetta | Physics upper bound | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Quetta | GradientBoosting | 3.7185 | 2.3551 | 12.5408 | 0.2921 | 0.1370 |
| Skardu | Persistence | 6.0982 | 4.5518 | 24.5327 | 0.1968 | 0.0000 |
| Skardu | Physics upper bound | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Skardu | GradientBoosting | 5.5502 | 4.4602 | 24.2267 | 0.3347 | 0.0899 |

![Faisalabad actual versus predicted](docs/figures/faisalabad_actual_vs_predicted.png)

![Faisalabad feature importance](docs/figures/faisalabad_feature_importance.png)

The ML forecast beats persistence on RMSE and MAE in all four cities, with the largest skill score in Faisalabad and the smallest in Skardu. Skardu also has the highest errors and MAPE, which is consistent with a more variable high-altitude weather signal in this proxy setup. The physics upper bound is zero-error only because it is given the true next-day weather; it is a ceiling, not a deployable forecast. The exact values are reproducible from `docs/benchmark_metrics.csv` and the cached NASA responses, although a fresh NASA request can change if upstream data is revised.

The expected pattern is that the physics-with-true-tomorrow-weather row is an upper bound by construction, while ML performance depends on how persistent the weather is in each city. Clear-sky index and recent irradiance history are especially informative because they capture cloud attenuation and short-term persistence. The comparison should be read as a forecast-proxy benchmark, not validation against a real PV meter.

## Limitations and next steps

- No metered plant data is available, so the ML target is a physics proxy.
- The transposition model is daily and first-order; it does not model hourly sun position, diffuse/direct components, shading, or azimuth.
- The performance ratio is a single configurable factor rather than a component-level loss model.
- Dust events, cleaning schedules, outages, clipping, curtailment, and inverter faults are not directly observed.
- A production study would need inverter AC output, plant configuration, tilt/azimuth, shading, cleaning records, and higher-resolution on-site weather or irradiance measurements.

## How to run

```bash
cd pv-yield-pk
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from pvyield.data import fetch_daily; from pvyield.physics import physics_yield; from pvyield.cities import CITIES; d=fetch_daily('Faisalabad','2023-01-01','2023-12-31'); print(physics_yield(d, CITIES['Faisalabad']['lat']).resample('MS').sum())"
streamlit run app.py --server.port 5000
```

The Streamlit app has Forecast, Physics vs ML, Explain, and Data tabs. It defaults to Faisalabad and lets the user change city, system size, tilt, losses, and date range.

## Author

Tahreem Fatima, BS Physics (GCUF)