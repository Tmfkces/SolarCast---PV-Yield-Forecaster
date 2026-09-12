"""Streamlit UI for the Pakistan PV Yield Forecaster."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pvyield.cities import CITIES
from pvyield.data import fetch_daily
from pvyield.evaluate import compare, feature_importance
from pvyield.ml import FEATURE_COLUMNS, train_forecaster
from pvyield.physics import physics_yield
from pvyield.plots import (
    daily_yield_figure,
    feature_importance_figure,
    monthly_total_figure,
    scatter_figure,
)

st.set_page_config(page_title="PV Yield Forecaster | Pakistan", layout="wide")


def default_dates() -> tuple[date, date]:
    today = date.today()
    end_year = today.year - 1
    return date(end_year - 4, 1, 1), date(end_year, 12, 31)


@st.cache_data(show_spinner=False)
def cached_weather(city: str, start: str, end: str) -> pd.DataFrame:
    return fetch_daily(city, start, end)


@st.cache_data(show_spinner=False)
def cached_comparison(city: str, start: str, end: str, system_kw: float) -> pd.DataFrame:
    return compare(city, start, end, system_kw)


@st.cache_data(show_spinner=False)
def cached_training(city: str, start: str, end: str, system_kw: float):
    weather = cached_weather(city, start, end)
    return train_forecaster(weather, lat=CITIES[city]["lat"], system_kw=system_kw, city=city)


def sidebar_inputs() -> dict:
    start_default, end_default = default_dates()
    st.sidebar.header("Forecast settings")
    city = st.sidebar.selectbox(
        "City",
        sorted(CITIES),
        index=sorted(CITIES).index("Faisalabad"),
        format_func=lambda name: f"{name} — {CITIES[name]['province']}",
    )
    system_kw = st.sidebar.slider("System size (kW)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
    tilt_choice = st.sidebar.selectbox("Panel tilt", ["latitude", "0° flat", "custom"])
    tilt = None
    if tilt_choice == "0° flat":
        tilt = 0.0
    elif tilt_choice == "custom":
        tilt = st.sidebar.number_input("Custom tilt (degrees)", min_value=0.0, max_value=90.0, value=31.0, step=1.0)
    pr = st.sidebar.slider("Performance ratio", min_value=0.50, max_value=0.95, value=0.80, step=0.01)
    soiling = st.sidebar.slider("Soiling loss", min_value=0.0, max_value=0.20, value=0.03, step=0.01)
    start, end = st.sidebar.date_input("Date range", value=(start_default, end_default))
    run = st.sidebar.button("Run forecast", type="primary", use_container_width=True)
    return {
        "city": city,
        "system_kw": system_kw,
        "tilt": tilt,
        "pr": pr,
        "soiling": soiling,
        "start": start,
        "end": end,
        "run": run,
    }


def render_forecast(weather: pd.DataFrame, settings: dict) -> None:
    city = settings["city"]
    daily = physics_yield(
        weather,
        lat=CITIES[city]["lat"],
        system_kw=settings["system_kw"],
        tilt=settings["tilt"],
        pr=settings["pr"],
        soiling=settings["soiling"],
    )
    years = max((daily.index.max() - daily.index.min()).days / 365.25, 1 / 12)
    annual = daily.sum() / years
    specific = annual / settings["system_kw"]
    capacity_factor = annual / (settings["system_kw"] * 8_760) * 100
    monthly = daily.resample("MS").sum()
    best_month = monthly.idxmax().strftime("%B")
    worst_month = monthly.idxmin().strftime("%B")
    st.subheader(f"{city} forecast")
    first, second, third, fourth = st.columns(4)
    first.metric("Annual yield", f"{annual:,.0f} kWh")
    second.metric("Specific yield", f"{specific:,.0f} kWh/kWp")
    third.metric("Capacity factor", f"{capacity_factor:.1f}%")
    fourth.metric("Best / worst month", f"{best_month} / {worst_month}")
    st.plotly_chart(monthly_total_figure(daily, title="Average monthly yield"), use_container_width=True)
    st.caption("The annual figure is annualized over the selected date range. Results are a first-order estimate, not a bankable production guarantee.")


def render_comparison(weather: pd.DataFrame, settings: dict) -> None:
    city = settings["city"]
    start = settings["start"].isoformat()
    end = settings["end"].isoformat()
    with st.spinner("Training the day-ahead model and calculating test metrics..."):
        table = cached_comparison(city, start, end, settings["system_kw"])
        trained = cached_training(city, start, end, settings["system_kw"])
    split = trained["split"]
    actual = split.test["target"].rename("actual")
    predicted = pd.Series(
        trained["model"].predict(split.test[FEATURE_COLUMNS]),
        index=actual.index,
        name="predicted",
    ).clip(lower=0)
    test_frame = pd.concat([actual, predicted], axis=1)
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    st.plotly_chart(daily_yield_figure(test_frame), use_container_width=True)
    st.plotly_chart(scatter_figure(actual, predicted), use_container_width=True)


def render_explanation(settings: dict) -> None:
    start = settings["start"].isoformat()
    end = settings["end"].isoformat()
    trained = cached_training(settings["city"], start, end, settings["system_kw"])
    importance = feature_importance(trained)
    st.plotly_chart(feature_importance_figure(importance), use_container_width=True)
    top_feature = importance.index[0].replace("_", " ")
    location = CITIES[settings["city"]]
    st.markdown(
        f"""
        In **{settings["city"]}**, yield is driven first by the amount of sunlight reaching
        the array and by how closely the day is to clear-sky conditions. The strongest
        learned signal here is **{top_feature}**. Temperature matters because PV modules
        lose efficiency as cell temperature rises; humidity, wind, and precipitation
        provide additional context for cloud and surface conditions. The model's target is
        a physics-model proxy rather than metered plant output, so treat feature importance
        as explanatory evidence for this dataset, not as causal proof.
        """
    )


def render_data(weather: pd.DataFrame) -> None:
    st.dataframe(weather, use_container_width=True, height=520)
    st.download_button(
        "Download cleaned weather CSV",
        data=weather.to_csv().encode("utf-8"),
        file_name="cleaned_weather.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("PV Yield Forecaster")
    st.write("A transparent solar production estimate for cities across Pakistan.")
    settings = sidebar_inputs()
    if settings["run"]:
        if settings["start"] > settings["end"]:
            st.error("The start date must be on or before the end date.")
            return
        try:
            with st.spinner("Fetching NASA POWER weather data..."):
                weather = cached_weather(
                    settings["city"],
                    settings["start"].isoformat(),
                    settings["end"].isoformat(),
                )
            if weather.empty:
                st.error("NASA POWER returned no usable weather rows for this date range.")
                return
            st.session_state["weather"] = weather
            st.session_state["settings"] = settings
        except Exception as exc:
            st.error(f"Could not load weather data for {settings['city']}. {exc}")
            return

    weather = st.session_state.get("weather")
    active_settings = st.session_state.get("settings")
    if weather is None or active_settings is None:
        st.info("Choose a city and date range, then select Run forecast to load the analysis.")
        st.markdown("#### What the estimate includes")
        st.markdown("Plane-of-array irradiance, temperature derating, system losses, a dust allowance, and a time-ordered day-ahead ML comparison.")
        return

    forecast_tab, compare_tab, explain_tab, data_tab = st.tabs(
        ["Forecast", "Physics vs ML", "Explain", "Data"]
    )
    with forecast_tab:
        render_forecast(weather, active_settings)
    with compare_tab:
        try:
            render_comparison(weather, active_settings)
        except Exception as exc:
            st.error(f"Model comparison is unavailable for this date range. {exc}")
    with explain_tab:
        try:
            render_explanation(active_settings)
        except Exception as exc:
            st.error(f"Feature importance is unavailable for this date range. {exc}")
    with data_tab:
        render_data(weather)


if __name__ == "__main__":
    main()