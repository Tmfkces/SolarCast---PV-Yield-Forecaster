# Pakistan PV Yield Forecaster

Python scientific-computing project that estimates daily rooftop PV yield for cities in Pakistan and compares a physics model with a time-ordered ML forecast.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `pv-yield-pk/src/pvyield/` — importable city, NASA data, physics, ML, evaluation, and plotting modules.
- `pv-yield-pk/app.py` — Streamlit interface.
- `pv-yield-pk/tests/` — pytest coverage for cleaning, physics, and time-based ML splitting.
- `pv-yield-pk/notebooks/analysis.ipynb` — reproducible report analysis.

## Architecture decisions

- NASA POWER is accessed directly because it is a public, no-key endpoint; city/date responses are cached locally.
- The ML target is the physics model's true next-day weather output, explicitly treated as a proxy rather than metered production.
- The default estimator is LightGBM with a GradientBoosting fallback when native LightGBM support is unavailable.

## Product

The Streamlit app forecasts annualized yield, specific yield, capacity factor, monthly seasonality, model metrics, feature importance, and cleaned weather data for 30 Pakistan cities.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Run from `pv-yield-pk` with `PYTHONPATH=src` so the scientific package is importable.
- NASA responses are cached under `pv-yield-pk/data/cache/`; remove a cache file to force a fresh request.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
