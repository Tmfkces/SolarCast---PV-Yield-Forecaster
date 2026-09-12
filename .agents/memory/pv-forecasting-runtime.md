---
name: PV forecasting runtime
description: Environment-specific lessons for the Pakistan PV forecasting project.
---

NASA POWER daily responses map parameter names to date-keyed values, so parsing must create rows from the date keys (`orient="columns"` in pandas). An `orient="index"` parse silently produces an empty cleaned dataset.

**Why:** The public endpoint returned HTTP 200 with valid data, but the first parse yielded no rows and created a misleading empty cache.

**How to apply:** When adding or changing NASA POWER parameters, validate one small response and assert the resulting frame has dates and rows before caching it.

Streamlit's first-run email prompt blocks a Replit workflow from opening its port. Project-level Streamlit config must enable headless mode and disable usage-statistics collection; LightGBM may also need a native OpenMP runtime, so the ML module should catch both import and native-load failures and keep the sklearn fallback.

**Why:** The first preview timed out because Streamlit waited for interactive input, and LightGBM initially failed on `libgomp.so.1`.

**How to apply:** Keep `.streamlit/config.toml` noninteractive for workflows and treat LightGBM as optional at runtime even when it is listed in requirements.