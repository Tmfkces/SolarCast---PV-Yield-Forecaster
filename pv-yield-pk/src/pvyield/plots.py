"""Plotly figures used by the Streamlit app and technical report."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SUN_SCALE = [[0.0, "#14B8A6"], [0.45, "#FBBF24"], [1.0, "#EA580C"]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SERIES_COLORS = {
    "actual": "#0f766e",
    "predicted": "#f59e0b",
    "persistence": "#64748b",
    "climatology": "#8b5cf6",
}


def daily_yield_figure(frame: pd.DataFrame) -> go.Figure:
    """Line chart of every column in ``frame`` (``actual`` first)."""
    fig = go.Figure()
    for column in frame.columns:
        color = SERIES_COLORS.get(column, SERIES_COLORS["predicted"])
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=str(column).capitalize() if column in SERIES_COLORS else str(column),
                mode="lines",
                line={"color": color, "width": 2 if column == "actual" else 1.5},
            )
        )
    fig.update_layout(
        title="Next-day yield on the test window",
        xaxis_title="",
        yaxis_title="Energy (kWh/day)",
        hovermode="x unified",
    )
    return fig


def monthly_total_figure(series: pd.Series, title: str = "Monthly yield") -> go.Figure:
    """Bar chart of total energy for each month in the series."""
    totals = series.resample("MS").sum(min_count=1).rename("yield").dropna()
    labels = totals.index.strftime("%b %Y")
    fig = px.bar(x=labels, y=totals.to_numpy(), labels={"x": "", "y": "Energy (kWh)"}, title=title)
    fig.update_traces(marker_color=SERIES_COLORS["actual"])
    return fig


def monthly_profile_figure(series: pd.Series, title: str = "Typical monthly yield") -> go.Figure:
    """Average total energy for each calendar month across all years."""
    profile = monthly_profile(series)
    fig = px.bar(x=profile.index, y=profile.to_numpy(), labels={"x": "", "y": "Energy (kWh/month)"}, title=title)
    fig.update_traces(marker_color=SERIES_COLORS["actual"])
    return fig


def monthly_profile(series: pd.Series) -> pd.Series:
    """Mean daily yield per calendar month × days in that month."""
    daily_mean = series.groupby(series.index.month).mean()
    days = pd.Series([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))
    profile = (daily_mean * days.reindex(daily_mean.index)).rename("yield")
    profile.index = [MONTHS[m - 1] for m in profile.index]
    return profile


def scatter_figure(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    frame = pd.concat([actual.rename("Actual"), predicted.rename("Predicted")], axis=1).dropna()
    fig = px.scatter(frame, x="Actual", y="Predicted", title="Predicted vs actual yield", opacity=0.6)
    maximum = max(float(frame.max().max()), 1.0) if not frame.empty else 1.0
    fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], name="1:1", mode="lines", line={"dash": "dash"}))
    fig.update_layout(xaxis_title="Actual (kWh/day)", yaxis_title="Predicted (kWh/day)")
    return fig


def feature_importance_figure(importances: pd.Series) -> go.Figure:
    top = importances.head(12).sort_values()
    fig = px.bar(x=top.to_numpy(), y=top.index, orientation="h", title="Feature importance")
    fig.update_layout(xaxis_title="Share of total importance", yaxis_title="")
    fig.update_xaxes(tickformat=".0%")
    return fig


def hourly_outlook_figure(
    hourly_energy: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    title: str = "Expected solar output by hour",
) -> go.Figure:
    """Bar chart of hourly PV energy (hour-end timestamps) with the best window shaded."""
    energy = hourly_energy[hourly_energy > 0.005]
    centers = energy.index - pd.Timedelta(minutes=30)
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=energy.to_numpy(),
            marker={"color": energy.to_numpy(), "colorscale": SUN_SCALE, "cmin": 0, "line": {"width": 0}},
            width=0.8 * 3_600_000,
            hovertemplate="%{x|%I %p}: %{y:.2f} kWh<extra></extra>",
            name="Solar output",
        )
    )
    if window is not None:
        fig.add_vrect(
            x0=window[0],
            x1=window[1],
            fillcolor=SERIES_COLORS["actual"],
            opacity=0.18,
            line_width=0,
            annotation_text="Best time for heavy appliances",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title, xaxis_title="", yaxis_title="kWh per hour", bargap=0.18, showlegend=False,
        barcornerradius=6, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickformat="%I %p", dtick=2 * 3_600_000)
    if not energy.empty:
        fig.update_xaxes(range=[centers.min() - pd.Timedelta(minutes=45), centers.max() + pd.Timedelta(minutes=45)])
    return fig


def residuals_by_month_figure(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    residuals = (predicted - actual).rename("residual").dropna().to_frame()
    residuals["month"] = residuals.index.month_name().str[:3]
    summary = residuals.groupby("month", sort=False)["residual"].mean().reindex(MONTHS).dropna()
    fig = px.bar(
        x=summary.index,
        y=summary.to_numpy(),
        title="Mean residual by month",
        labels={"x": "", "y": "Predicted − actual (kWh/day)"},
    )
    fig.update_traces(marker_color=SERIES_COLORS["predicted"])
    return fig
