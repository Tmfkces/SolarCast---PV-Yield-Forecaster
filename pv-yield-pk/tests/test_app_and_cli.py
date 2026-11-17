import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_weather
from test_forecast import open_meteo_payload

import pvyield.cli as cli
import pvyield.data as data
import pvyield.evaluate as evaluate
import pvyield.forecast as forecast
from pvyield.cities import CITIES
from pvyield.plots import (
    daily_yield_figure,
    feature_importance_figure,
    monthly_profile,
    monthly_profile_figure,
    monthly_total_figure,
    residuals_by_month_figure,
    scatter_figure,
)

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def offline_weather(monkeypatch, tmp_path):
    weather = make_weather(500, start="2021-01-01")
    fake = lambda city, start, end, use_cache=True: weather  # noqa: E731
    monkeypatch.setattr(data, "fetch_daily", fake)
    monkeypatch.setattr(cli, "fetch_daily", fake)
    monkeypatch.setattr(evaluate, "fetch_daily", fake)
    payload = open_meteo_payload(start=date.today().isoformat())
    monkeypatch.setattr(forecast, "fetch_forecast", lambda city, days=7: forecast.parse_forecast(payload))
    monkeypatch.setattr(forecast, "fetch_forecast_history", lambda city, years=4: weather[["ghi", "t_amb"]])
    monkeypatch.setenv("SOLARCAST_SETTINGS_FILE", str(tmp_path / "settings.json"))
    return weather


def test_cities_are_inside_pakistan():
    assert len(CITIES) == 30
    for name, info in CITIES.items():
        assert 23.5 < info["lat"] < 37.5, name
        assert 60.5 < info["lon"] < 77.5, name


def test_plot_helpers_build_figures():
    weather = make_weather(400)
    daily = weather["ghi"] * 4
    frame = pd.DataFrame({"actual": daily, "predicted": daily * 1.05})
    assert monthly_profile(daily).index[0] == "Jan"
    for figure in [
        daily_yield_figure(frame),
        monthly_total_figure(daily),
        monthly_profile_figure(daily),
        scatter_figure(frame["actual"], frame["predicted"]),
        feature_importance_figure(pd.Series({"ghi": 0.7, "rh": 0.3})),
        residuals_by_month_figure(frame["actual"], frame["predicted"]),
    ]:
        assert figure.data


def test_cli_yield_prints_monthly_totals(offline_weather, capsys):
    assert cli.main(["yield", "--city", "Lahore", "--start", "2021-01-01", "--end", "2022-05-15", "--kw", "3"]) == 0
    out = capsys.readouterr().out
    assert "2021-01" in out and "Annualised yield" in out


def test_cli_benchmark_writes_csv_and_figures(offline_weather, tmp_path):
    out = tmp_path / "bench.csv"
    figures = tmp_path / "figs"
    code = cli.main(
        ["benchmark", "--cities", "Karachi", "--start", "2021-01-01", "--end", "2022-05-15",
         "--out", str(out), "--figures", str(figures)]
    )
    assert code == 0
    table = pd.read_csv(out)
    assert {"city", "model", "rmse", "skill_score"} <= set(table.columns)
    assert (figures / "karachi_actual_vs_predicted.png").exists()
    assert (figures / "karachi_feature_importance.png").exists()


def test_cli_train_saves_bundle(offline_weather, tmp_path):
    out = tmp_path / "model.joblib"
    assert cli.main(["train", "--city", "Quetta", "--start", "2021-01-01", "--end", "2022-05-15", "--out", str(out)]) == 0
    assert out.exists()


def test_cli_reports_errors_cleanly(capsys):
    assert cli.main(["yield", "--city", "Lahore", "--start", "2023-05-01", "--end", "2023-01-01"]) == 2
    assert "error:" in capsys.readouterr().err


def test_streamlit_app_end_to_end(offline_weather):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    assert any("Run analysis" in info.value for info in app.info)
    outlook_html = " ".join(md.value for md in app.markdown)
    assert "Best hours for heavy appliances" in outlook_html
    assert "kWh expected" in outlook_html
    assert outlook_html.count('class="sc-row') == 7  # 7-day list
    assert 'class="sc-hour' in outlook_html  # hourly strip
    assert "<svg" in outlook_html  # line icons, sun arc
    assert "Solar can run all of this" in outlook_html or "Solar covers about" in outlook_html  # appliance planner
    assert "Value this week" in outlook_html

    assert "Peak tariff" in outlook_html
    assert "Tap a day to open it" in outlook_html and 'href="?' in outlook_html
    settings_path = Path(os.environ["SOLARCAST_SETTINGS_FILE"])
    assert json.loads(settings_path.read_text())["city"] == "Faisalabad"  # settings remembered
    assert app.query_params["kw"] in ("5", ["5"])  # settings also kept in the page link

    next(b for b in app.button if b.label == "Run analysis").click().run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    labels = [metric.label for metric in app.metric]
    assert {"Annual yield (kWh)", "Specific yield (kWh/kWp)", "Capacity factor"} <= set(labels)
    assert len(app.table) == 1  # metric table
    assert len(app.dataframe) == 1  # weather data


def test_streamlit_app_restores_saved_settings(offline_weather):
    testing = pytest.importorskip("streamlit.testing.v1")
    Path(os.environ["SOLARCAST_SETTINGS_FILE"]).write_text(
        json.dumps({"city": "Lahore", "kw": 8.0, "setup": "battery", "apps": ["Iron"]})
    )
    app = testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception
    assert app.session_state["city"] == "Lahore"
    assert app.session_state["kw"] == 8.0
    assert app.session_state["setup"] == "battery"
    assert app.session_state["apps"] == ["Iron"]
    header = " ".join(md.value for md in app.markdown)
    assert "Battery / hybrid inverter" in header


def test_streamlit_app_url_settings_win_and_bad_values_are_safe(offline_weather):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP), default_timeout=120)
    app.query_params.update({"city": "Karachi", "kw": "999", "setup": "nonsense", "day": "not-a-date"})
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["city"] == "Karachi"
    assert app.session_state["kw"] == 100.0  # clamped to the slider range
    assert app.session_state["setup"] == "net_metering"
