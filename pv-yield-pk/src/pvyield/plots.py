"""Plotly figures used by the Streamlit app and technical report."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def daily_yield_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in [
        ("actual", "Actual", "#0f766e"),
        ("predicted", "Predicted", "#f59e0b"),
    ]:
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=label, mode="lines", line={"color": color}))
    fig.update_layout(title="Next-day yield on the test window", xaxis_title="", yaxis_title="Energy (kWh/day)")
    return fig


def monthly_total_figure(series: pd.Series, title: str = "Monthly yield") -> go.Figure:
    totals = series.resample("MS").sum().rename("yield")
    totals.index = totals.index.strftime("%b %Y")
    fig = px.bar(totals, x=totals.index, y="yield", labels={"x": "", "yield": "Energy (kWh)"}, title=title)
    fig.update_layout(showlegend=False)
    return fig


def scatter_figure(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    frame = pd.concat([actual.rename("Actual"), predicted.rename("Predicted")], axis=1).dropna()
    fig = px.scatter(frame, x="Actual", y="Predicted", title="Predicted vs actual yield")
    maximum = max(float(frame.max().max()), 1.0) if not frame.empty else 1.0
    fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], name="1:1", mode="lines", line={"dash": "dash"}))
    return fig


def feature_importance_figure(importances: pd.Series) -> go.Figure:
    top = importances.head(12).sort_values()
    fig = px.bar(top, x=top.values, y=top.index, orientation="h", title="Feature importance")
    fig.update_layout(xaxis_title="Gain / importance", yaxis_title="")
    return fig


def residuals_by_month_figure(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    residuals = (predicted - actual).rename("residual").to_frame()
    residuals["month"] = residuals.index.month_name().str[:3]
    ordered = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    summary = residuals.groupby("month", sort=False)["residual"].mean().reindex(ordered).dropna()
    return px.bar(summary, x=summary.index, y="residual", title="Mean residual by month", labels={"residual": "Predicted - actual (kWh)"})