# SolarCast — Pakistan PV Yield Forecaster

Python scientific-computing project that estimates daily rooftop PV yield for cities in Pakistan and compares a physics model with a time-ordered ML forecast.

## Run & Operate

- `cd pv-yield-pk && streamlit run app.py --server.port 5000` — run the Streamlit app (the Replit "PV Yield Forecaster" workflow)
- `cd pv-yield-pk && pytest` — offline test suite (`pytest -m network` adds a live NASA POWER check)
- `cd pv-yield-pk && python -m pvyield benchmark --out docs/benchmark_metrics.csv --figures docs/figures` — regenerate the README benchmark
- `python main.py` — launch the app from the repository root

The TypeScript pnpm workspace (`artifacts/`, `lib/`, `scripts/`) is the unused Replit scaffold (Express health-check API, mockup sandbox, empty Drizzle schema). Nothing in the PV app depends on it.

## Stack

- Python 3.11+, pandas, NumPy, scikit-learn, LightGBM (optional), Plotly, Streamlit
- Data: NASA POWER daily point API (no key), CSV cache in `pv-yield-pk/data/cache/`

## Where things live

- `pv-yield-pk/src/pvyield/` — importable city, NASA data, physics, ML, evaluation, and plotting modules.
- `pv-yield-pk/app.py` — Streamlit interface.
- `pv-yield-pk/tests/` — offline pytest suite: solar geometry, physics, cleaning, cached/mocked NASA fetches, leakage across gaps, metrics, CLI, Streamlit app.
- `pv-yield-pk/notebooks/analysis.ipynb` — reproducible report analysis.

## Architecture decisions

- NASA POWER is accessed directly because it is a public, no-key endpoint; city/date responses are cached locally.
- The ML target is the physics model's true next-day weather output, explicitly treated as a proxy rather than metered production.
- The default estimator is LightGBM with a GradientBoosting fallback when native LightGBM support is unavailable. Boosting rounds are chosen on the validation window, then the model is refit on train+validation.

## Product

The Streamlit app forecasts annualized yield, specific yield, capacity factor, monthly seasonality, model metrics, feature importance, and cleaned weather data for 30 Pakistan cities.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Run from `pv-yield-pk`; `pip install -e ".[dev]"` (or `PYTHONPATH=src`) makes the package importable.
- Old `data/models/*.pkl` files are v1 artefacts that nothing loads; `python -m pvyield train` writes v2 `.joblib` bundles.
- NASA responses are cached under `pv-yield-pk/data/cache/`; remove a cache file to force a fresh request.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
