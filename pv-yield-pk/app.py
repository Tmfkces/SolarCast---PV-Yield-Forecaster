"""SolarCast: everyday solar outlook and long-term analysis for rooftop PV in Pakistan."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pvyield.cities import CITIES  # noqa: E402
from pvyield.data import NASA_FIRST_DAY, fetch_daily  # noqa: E402
from pvyield.evaluate import ORACLE_ROW, Evaluation, evaluate_forecaster, feature_importance  # noqa: E402
from pvyield.forecast import (  # noqa: E402
    APPLIANCES,
    SETUPS,
    TIMEZONE,
    build_outlook,
    estimate_value,
    fetch_forecast,
    fetch_forecast_history,
    load_saved_forecast,
    peak_label,
    plan_appliances,
    typical_daily_yield,
)
from pvyield.ml import model_family  # noqa: E402
from pvyield.physics import physics_yield  # noqa: E402
from pvyield.plots import (  # noqa: E402
    daily_yield_figure,
    feature_importance_figure,
    monthly_profile,
    monthly_profile_figure,
    monthly_total_figure,
    residuals_by_month_figure,
    scatter_figure,
)
from pvyield.ui import CSS, day_name, hero_html, hourly_html, plan_html, tiles_html, week_html  # noqa: E402

MIN_DAYS_FOR_ML = 120
METRIC_LABELS = {
    "rmse": "RMSE (kWh)",
    "mae": "MAE (kWh)",
    "mape": "MAPE (%)",
    "bias": "Bias (kWh)",
    "r2": "R²",
    "skill_score": "Skill vs persistence",
}
TILT_MODES = ["Latitude", "Flat (0°)", "Custom"]
DEFAULT_APPLIANCES = ["Washing machine", "Iron", "Water pump (1 hp)"]

# Everything the user sets is remembered: in the URL (bookmark / share) and in a
# local settings file (the next visit on this computer starts where you left off).
SETTINGS: dict[str, tuple[object, float | None, float | None]] = {
    # key: (default, min, max)
    "city": ("Faisalabad", None, None),
    "kw": (5.0, 1.0, 100.0),
    "price": (55.0, 0.0, 200.0),
    "setup": ("net_metering", None, None),
    "use": (12.0, 1.0, 200.0),
    "export": (10.0, 0.0, 100.0),
    "batt": (5.0, 0.0, 100.0),
    "tilt_mode": ("Latitude", None, None),
    "tilt": (31.0, 0.0, 90.0),
    "az": (180.0, 90.0, 270.0),
    "pr": (0.80, 0.50, 0.95),
    "soil": (0.03, 0.0, 0.20),
    "apps": (DEFAULT_APPLIANCES, None, None),
}

ASSETS = PROJECT_ROOT / "assets"
st.set_page_config(page_title="SolarCast | Pakistan PV forecast", page_icon=str(ASSETS / "logo_icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")


# --------------------------------------------------------------------------- settings


def settings_file() -> Path:
    return Path(os.environ.get("SOLARCAST_SETTINGS_FILE", PROJECT_ROOT / "data" / "user_settings.json"))


def _coerce(key: str, raw: object) -> object:
    default, low, high = SETTINGS[key]
    try:
        if key == "apps":
            items = raw.split(",") if isinstance(raw, str) else list(raw)
            return [a for a in items if a in APPLIANCES]
        if key == "city":
            return raw if raw in CITIES else default
        if key == "setup":
            return raw if raw in SETUPS else default
        if key == "tilt_mode":
            return raw if raw in TILT_MODES else default
        value = float(raw)
        return min(max(value, low), high)
    except (TypeError, ValueError):
        return default


def load_settings() -> None:
    """Fill session state once per visit: URL first, then the saved file, then defaults."""
    if st.session_state.get("_settings_loaded"):
        return
    saved: dict = {}
    try:
        saved = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    params = st.query_params
    for key, (default, _, _) in SETTINGS.items():
        raw = params.get(key, saved.get(key, default))
        st.session_state[key] = _coerce(key, raw)
    st.session_state["_settings_loaded"] = True


def current_settings() -> dict:
    return {key: st.session_state[key] for key in SETTINGS}


def _encoded(values: dict) -> dict[str, str]:
    return {k: ",".join(v) if isinstance(v, list) else f"{v:g}" if isinstance(v, float) else str(v) for k, v in values.items()}


def save_settings() -> None:
    values = current_settings()
    encoded = _encoded(values)
    for key, text in encoded.items():
        if st.query_params.get(key) != text:
            st.query_params[key] = text
    try:
        path = settings_file()
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != values:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    except (OSError, ValueError):  # pragma: no cover - read-only hosting keeps URL persistence only
        pass


def link_with(**extra: str) -> str:
    """Relative link that keeps every setting and adds ``extra`` query parameters."""
    return "?" + urlencode({**_encoded(current_settings()), **extra})


# --------------------------------------------------------------------------- data


def now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)


def latest_available_day() -> date:
    """NASA POWER daily data lags real time by a few days."""
    return date.today() - timedelta(days=7)


def default_dates() -> tuple[date, date]:
    end_year = date.today().year - 1
    return date(end_year - 4, 1, 1), date(end_year, 12, 31)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_weather(city: str, start: str, end: str) -> pd.DataFrame:
    return fetch_daily(city, start, end)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_evaluation(city: str, start: str, end: str, physics: tuple) -> Evaluation:
    weather = cached_weather(city, start, end)
    return evaluate_forecaster(weather, lat=CITIES[city]["lat"], **dict(physics))


@st.cache_data(show_spinner=False, ttl=3_600, max_entries=32)
def cached_forecast(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_forecast(city)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_typical(city: str, physics: tuple) -> tuple[pd.Series, str]:
    """Usual yield per month, from the same source as the forecast when possible."""
    lat = CITIES[city]["lat"]
    try:
        return typical_daily_yield(fetch_forecast_history(city), lat=lat, **dict(physics)), "Open-Meteo archive"
    except Exception:  # fall back to NASA POWER if the archive is unreachable
        start, end = default_dates()
        history = cached_weather(city, start.isoformat(), end.isoformat())
        return typical_daily_yield(history, lat=lat, **dict(physics)), "NASA POWER"


# --------------------------------------------------------------------------- sidebar


def sidebar_inputs() -> dict:
    load_settings()
    cities = sorted(CITIES)
    sb = st.sidebar
    sb.subheader("Your system")
    city = sb.selectbox("City", cities, key="city", format_func=lambda n: f"{n} — {CITIES[n]['province']}")
    kw = sb.slider("System size (kWp)", min_value=1.0, max_value=100.0, step=0.5, key="kw")
    setup = sb.selectbox("Setup", list(SETUPS), key="setup", format_func=SETUPS.get)

    with sb.expander("Bill & usage"):
        st.number_input(
            "Electricity price (Rs/kWh)", min_value=0.0, max_value=200.0, step=1.0, key="price",
            help="What you pay per unit. Check your latest bill.",
        )
        st.number_input(
            "Daily use (kWh)", min_value=1.0, max_value=200.0, step=1.0, key="use",
            help="Units per day from your bill ÷ 30. About half is assumed to be used in daylight.",
        )
        if setup == "net_metering":
            st.number_input("Export rate (Rs/kWh)", min_value=0.0, max_value=100.0, step=1.0, key="export",
                            help="What the DISCO pays for exported units (buyback rate).")
        if setup == "battery":
            st.number_input("Battery size (kWh)", min_value=0.0, max_value=100.0, step=0.5, key="batt")

    with sb.expander("Panel direction & losses"):
        tilt_mode = st.radio("Panel tilt", TILT_MODES, horizontal=True, key="tilt_mode")
        tilt: float | None = None
        if tilt_mode == "Flat (0°)":
            tilt = 0.0
        elif tilt_mode == "Custom":
            tilt = st.slider("Custom tilt (°)", min_value=0.0, max_value=90.0, step=1.0, key="tilt")
        azimuth = st.slider(
            "Azimuth (°, 180 = south)", min_value=90.0, max_value=270.0, step=5.0, key="az",
            help="Compass direction the panels face: 90 = east, 180 = south, 270 = west.",
        )
        pr = st.slider(
            "Performance ratio", min_value=0.50, max_value=0.95, step=0.01, key="pr",
            help="Inverter, wiring, mismatch and other losses. Temperature and soiling are modelled separately.",
        )
        soiling = st.slider("Soiling loss", min_value=0.0, max_value=0.20, step=0.01, format="%.2f", key="soil")
    sb.caption("Your settings are saved on this computer and in the page link, so you can bookmark it.")
    return {
        "city": city,
        "system_kw": kw,
        "setup": setup,
        "tilt": tilt,
        "azimuth": azimuth,
        "pr": pr,
        "soiling": soiling,
    }


def physics_settings(settings: dict) -> tuple:
    """Hashable physics kwargs shared by the forecast and the ML target."""
    keys = ("system_kw", "tilt", "azimuth", "pr", "soiling")
    return tuple(sorted((k, settings[k]) for k in keys))


# --------------------------------------------------------------------------- outlook


def load_outlook_data(city: str) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Live forecast, or the last saved one when offline. Returns a warning when stale."""
    try:
        daily, hourly = cached_forecast(city)
        return daily, hourly, None
    except Exception as exc:
        saved = load_saved_forecast(city)
        if saved is None:
            raise RuntimeError(f"no internet connection and no saved forecast for {city} ({exc})") from exc
        daily, hourly, saved_at = saved
        local = pd.Timestamp(saved_at).tz_convert(TIMEZONE)
        return daily, hourly, f"Offline: showing the forecast saved {local:%a %d %b, %I:%M %p}."


def render_outlook(settings: dict) -> None:
    city = settings["city"]
    physics = physics_settings(settings)
    try:
        with st.spinner("Loading the 7-day forecast..."):
            daily, hourly, stale = load_outlook_data(city)
            typical, baseline_source = cached_typical(city, physics)
    except Exception as exc:  # network / API errors are shown to the user
        st.error(f"The 7-day outlook is unavailable right now: {exc}")
        return
    if stale:
        st.warning(stale, icon=":material/cloud_off:")

    outlook = build_outlook(
        daily, hourly, typical, lat=CITIES[city]["lat"], setup=settings["setup"], **dict(physics)
    )
    days = outlook.days
    now = now_local()
    today = now.normalize()
    # Morning: today is what matters. Afternoon/evening: plan tomorrow.
    preferred = today if now.hour < 12 else today + pd.Timedelta(days=1)
    default = preferred if preferred in days.index else days.index[0]
    requested = st.query_params.get("day")
    if requested:
        try:
            asked = pd.Timestamp(requested)
            default = asked if asked in days.index else default
        except ValueError:
            pass

    st.markdown(CSS, unsafe_allow_html=True)
    focus = st.segmented_control(
        "Day", options=list(days.index), format_func=day_name, default=default, label_visibility="collapsed"
    )
    focus = focus if focus is not None else default
    if focus != default or requested:
        st.query_params["day"] = focus.date().isoformat()

    hour_day = (outlook.hourly.index - pd.Timedelta(minutes=1)).normalize()
    hours = outlook.hourly.loc[hour_day == focus]
    is_today = focus == today
    left_today = float(hours.loc[hours.index > now, "energy_kwh"].sum()) if is_today else None
    st.markdown(hero_html(outlook, focus, left_today=left_today), unsafe_allow_html=True)

    s = st.session_state
    per_day = [
        estimate_value(kwh, s["price"], settings["setup"], s["export"], s["use"], s["batt"])
        for kwh in days["yield_kwh"]
    ]
    week_value = {k: sum(d[k] for d in per_day) for k in per_day[0]}
    st.markdown(tiles_html(outlook, week_value, s["price"], SETUPS[settings["setup"]]), unsafe_allow_html=True)

    row = days.loc[focus]
    start = row["window_start"] if pd.notna(row["window_start"]) else None
    end = row["window_end"] if pd.notna(row["window_end"]) else None
    st.markdown(
        hourly_html(hours, start, end, now=now if is_today else None, peak=peak_label(focus.month)),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown(week_html(outlook, focus, href=lambda d: link_with(day=d.date().isoformat())), unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown(
            f'<div class="sc"><div class="sc-label">Appliance planner · {day_name(focus)}</div></div>',
            unsafe_allow_html=True,
        )
        selected = st.pills("Appliances", options=list(APPLIANCES), selection_mode="multi", key="apps",
                            label_visibility="collapsed")
        chosen = list(selected or [])
        plan = plan_appliances(chosen, hours["energy_kwh"])
        st.markdown(plan_html(chosen, plan, day_name(focus)), unsafe_allow_html=True)
        st.caption("Assumes about 0.3 kW of always-on load (fridge, fans, lights) is served first.")
    st.caption(
        f"Forecast: Open-Meteo, refreshed hourly. A usual day is the median for this month from the "
        f"{baseline_source} over recent years, with your system settings. Peak-tariff hours are typical "
        "DISCO hours; check your bill."
    )


# --------------------------------------------------------------------------- analysis


def render_forecast(weather: pd.DataFrame, settings: dict) -> None:
    city = settings["city"]
    daily = physics_yield(weather, lat=CITIES[city]["lat"], **dict(physics_settings(settings)))
    annual = float(daily.mean() * 365.25)
    specific = annual / settings["system_kw"]
    capacity_factor = annual / (settings["system_kw"] * 8_760) * 100
    profile = monthly_profile(daily)
    calendar_days = (weather.index.max() - weather.index.min()).days + 1
    coverage = len(weather) / calendar_days * 100

    first, second, third, fourth = st.columns(4)
    first.metric("Annual yield (kWh)", f"{annual:,.0f}")
    second.metric("Specific yield (kWh/kWp)", f"{specific:,.0f}")
    third.metric("Capacity factor", f"{capacity_factor:.1f}%")
    fourth.metric("Best / worst month", f"{profile.idxmax()} / {profile.idxmin()}")
    left, right = st.columns(2)
    left.plotly_chart(monthly_profile_figure(daily), width="stretch")
    right.plotly_chart(monthly_total_figure(daily, title="Monthly yield over the selected period"), width="stretch")
    st.caption(
        f"Annual yield = mean daily yield × 365.25 over {len(weather):,} usable days "
        f"({coverage:.0f}% of the calendar range). This is a first-order estimate, "
        "not a bankable production guarantee."
    )


def render_comparison(evaluation: Evaluation) -> None:
    trained = evaluation.trained
    predictions = evaluation.predictions
    test = trained.split.test
    st.markdown(
        f"**{trained.model_name}** trained on {trained.split.train.index.min():%b %Y} – "
        f"{trained.split.validation.index.max():%b %Y} ({trained.best_iteration} boosting rounds chosen on the "
        f"validation window) and tested on {test.index.min():%b %Y} – {test.index.max():%b %Y} "
        f"({len(test)} days)."
    )
    table = evaluation.table.rename(columns=METRIC_LABELS)
    st.table(table.style.format("{:.3f}", na_rep="—"))
    st.caption(
        f"“{ORACLE_ROW}” uses tomorrow's real weather, so it equals the proxy target by construction. "
        "It is a sanity check, not a forecast. Positive skill means lower RMSE than persistence."
    )
    shown = predictions[["actual", trained.model_name, "persistence"]]
    st.plotly_chart(daily_yield_figure(shown), width="stretch")
    st.plotly_chart(scatter_figure(predictions["actual"], predictions[trained.model_name]), width="stretch")


def render_explanation(evaluation: Evaluation, city: str) -> None:
    trained = evaluation.trained
    importance = feature_importance(trained)
    left, right = st.columns(2)
    left.plotly_chart(feature_importance_figure(importance), width="stretch")
    right.plotly_chart(
        residuals_by_month_figure(evaluation.predictions["actual"], evaluation.predictions[trained.model_name]),
        width="stretch",
    )
    top_feature = importance.index[0].replace("_", " ")
    st.markdown(
        f"""
        In **{city}**, yield is driven first by the amount of sunlight reaching the array
        and by how close the day is to clear-sky conditions. The strongest learned signal
        here is **{top_feature}** ({importance.iloc[0]:.0%} of total importance). Temperature
        matters because PV modules lose efficiency as cell temperature rises; humidity, wind,
        and precipitation add context about clouds and dust. The model's target is a
        physics-model proxy rather than metered plant output, so treat feature importance
        as descriptive evidence for this dataset, not as causal proof.
        """
    )


def render_data(weather: pd.DataFrame, city: str) -> None:
    st.dataframe(weather, width="stretch", height=520)
    st.download_button(
        "Download cleaned weather CSV",
        data=weather.to_csv().encode("utf-8"),
        file_name=f"{city.lower().replace(' ', '_')}_nasa_power_daily.csv",
        mime="text/csv",
    )
    st.caption("Source: NASA POWER daily point API (RE community). Gaps of up to 3 days are forward-filled; longer gaps are removed.")


def render_analysis(settings: dict) -> None:
    start_default, end_default = default_dates()
    with st.container(border=True):
        pick_col, run_col = st.columns([3, 1], vertical_alignment="bottom")
        picked = pick_col.date_input(
            "Historical period (NASA POWER)",
            value=(start_default, end_default),
            min_value=NASA_FIRST_DAY,
            max_value=latest_available_day(),
        )
        run = run_col.button("Run analysis", type="primary", width="stretch", icon=":material/bolt:")
    start, end = picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (None, None)

    if run:
        if start is None:
            st.error("Pick both a start and an end date.")
            return
        try:
            with st.spinner("Fetching NASA POWER weather data..."):
                weather = cached_weather(settings["city"], start.isoformat(), end.isoformat())
        except Exception as exc:  # network / API errors are shown to the user
            st.error(f"Could not load weather data for {settings['city']}: {exc}")
            return
        if weather.empty:
            st.error("NASA POWER returned no usable weather rows for this date range.")
            return
        st.session_state["weather"] = weather
        st.session_state["analysis"] = {**settings, "start": start, "end": end}

    weather = st.session_state.get("weather")
    active = st.session_state.get("analysis")
    if weather is None or active is None:
        st.info("Pick a period and select **Run analysis** for a long-term estimate and the model check.")
        st.markdown(
            "- Typical-year production, specific yield and capacity factor from NASA POWER history\n"
            "- Daily diffuse/beam split (Erbs) and isotropic-sky transposition to any tilt and azimuth\n"
            f"- A time-ordered day-ahead {model_family()} forecast compared with persistence and climatology"
        )
        return

    estimate_tab, compare_tab, explain_tab, data_tab = st.tabs(["Long-term estimate", "Model check", "Explain", "Data"])
    with estimate_tab:
        render_forecast(weather, active)

    evaluation: Evaluation | None = None
    ml_error: str | None = None
    if len(weather) < MIN_DAYS_FOR_ML:
        ml_error = f"Select at least {MIN_DAYS_FOR_ML} days of data to train and test the day-ahead model."
    else:
        try:
            with st.spinner("Training the day-ahead model and scoring the test window..."):
                evaluation = cached_evaluation(
                    active["city"], active["start"].isoformat(), active["end"].isoformat(), physics_settings(active)
                )
        except Exception as exc:
            ml_error = f"Model comparison is unavailable for this date range: {exc}"
    with compare_tab:
        if evaluation is None:
            st.warning(ml_error)
        else:
            render_comparison(evaluation)
    with explain_tab:
        if evaluation is None:
            st.warning(ml_error)
        else:
            render_explanation(evaluation, active["city"])
    with data_tab:
        render_data(weather, active["city"])


# --------------------------------------------------------------------------- page


def main() -> None:
    settings = sidebar_inputs()
    save_settings()
    info = CITIES[settings["city"]]
    st.markdown(
        f'<div style="margin:-8px 0 10px"><div style="font-size:34px;font-weight:700;letter-spacing:-.02em">'
        f'{settings["city"]}</div><div style="opacity:.65;font-size:15px">{info["province"]} · '
        f'{settings["system_kw"]:g} kWp · {SETUPS[settings["setup"]]}</div></div>',
        unsafe_allow_html=True,
    )
    outlook_tab, analysis_tab = st.tabs(["Outlook", "Analysis"])
    with outlook_tab:
        render_outlook(settings)
    with analysis_tab:
        render_analysis(settings)


if __name__ == "__main__":
    main()
