"""Command-line entry points.

Examples (run from ``pv-yield-pk``)::

    python -m pvyield yield --city Lahore --start 2023-01-01 --end 2023-12-31 --kw 5
    python -m pvyield benchmark --start 2019-01-01 --end 2024-12-31 --out docs/benchmark_metrics.csv --figures docs/figures
    python -m pvyield train --city Karachi --start 2019-01-01 --end 2024-12-31 --out data/models/Karachi.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .cities import CITIES
from .data import fetch_daily
from .evaluate import benchmark, evaluate_forecaster, feature_importance
from .ml import save_forecaster, train_forecaster
from .physics import physics_yield

DEFAULT_BENCHMARK_CITIES = ["Faisalabad", "Karachi", "Quetta", "Skardu"]


def _physics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kw", type=float, default=5.0, help="DC system size in kWp (default 5)")
    parser.add_argument("--tilt", type=float, default=None, help="panel tilt in degrees (default: latitude)")
    parser.add_argument("--azimuth", type=float, default=180.0, help="compass azimuth, 180 = south")
    parser.add_argument("--pr", type=float, default=0.80, help="performance ratio (default 0.80)")
    parser.add_argument("--soiling", type=float, default=0.03, help="soiling loss fraction (default 0.03)")


def _physics_kwargs(args: argparse.Namespace) -> dict:
    return {"tilt": args.tilt, "azimuth": args.azimuth, "pr": args.pr, "soiling": args.soiling}


def _cmd_yield(args: argparse.Namespace) -> int:
    weather = fetch_daily(args.city, args.start, args.end)
    daily = physics_yield(weather, lat=CITIES[args.city]["lat"], system_kw=args.kw, **_physics_kwargs(args))
    monthly = daily.resample("MS").sum(min_count=1).dropna().round(1)
    monthly.index = monthly.index.strftime("%Y-%m")
    print(monthly.to_string())
    annual = daily.mean() * 365.25
    print(f"\nAnnualised yield: {annual:,.0f} kWh  |  specific yield: {annual / args.kw:,.0f} kWh/kWp")
    return 0


def _save_figures(city: str, args: argparse.Namespace, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weather = fetch_daily(city, args.start, args.end)
    evaluation = evaluate_forecaster(weather, lat=CITIES[city]["lat"], system_kw=args.kw, **_physics_kwargs(args))
    predictions = evaluation.predictions
    model_name = evaluation.trained.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = city.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(predictions.index, predictions["actual"], label="Physics target (true next-day weather)", color="#0f766e", lw=1.4)
    ax.plot(predictions.index, predictions[model_name], label=f"{model_name} forecast", color="#f59e0b", lw=1.2)
    ax.plot(predictions.index, predictions["persistence"], label="Persistence", color="#64748b", lw=0.8, alpha=0.7)
    ax.set_ylabel("Energy (kWh/day)")
    ax.set_title(f"{city}: next-day yield on the test window ({args.kw:g} kWp)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    importance = feature_importance(evaluation.trained).head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.to_numpy() * 100, color="#0f766e")
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title(f"{city}: {model_name} feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_feature_importance.png", dpi=150)
    plt.close(fig)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    table = benchmark(args.cities, args.start, args.end, args.kw, **_physics_kwargs(args))
    with pd.option_context("display.width", 160, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        print(f"\nWrote {out}")
    if args.figures:
        _save_figures(args.cities[0], args, Path(args.figures))
        print(f"Wrote figures for {args.cities[0]} to {args.figures}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    weather = fetch_daily(args.city, args.start, args.end)
    trained = train_forecaster(weather, lat=CITIES[args.city]["lat"], system_kw=args.kw, **_physics_kwargs(args))
    path = save_forecaster(trained, args.out, city=args.city)
    print(f"Saved {trained.model_name} ({trained.best_iteration} rounds) to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvyield", description="Pakistan PV yield forecaster")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    yield_cmd = sub.add_parser("yield", help="print monthly physics yield for one city")
    yield_cmd.add_argument("--city", required=True, choices=sorted(CITIES))
    yield_cmd.add_argument("--start", required=True)
    yield_cmd.add_argument("--end", required=True)
    _physics_args(yield_cmd)
    yield_cmd.set_defaults(func=_cmd_yield)

    bench = sub.add_parser("benchmark", help="compare persistence, climatology and ML per city")
    bench.add_argument("--cities", nargs="+", default=DEFAULT_BENCHMARK_CITIES, choices=sorted(CITIES), metavar="CITY")
    bench.add_argument("--start", default="2019-01-01")
    bench.add_argument("--end", default="2024-12-31")
    bench.add_argument("--out", help="CSV path for the metric table")
    bench.add_argument("--figures", help="directory for PNG figures of the first city")
    _physics_args(bench)
    bench.set_defaults(func=_cmd_benchmark)

    train = sub.add_parser("train", help="train and save a model bundle for one city")
    train.add_argument("--city", required=True, choices=sorted(CITIES))
    train.add_argument("--start", required=True)
    train.add_argument("--end", required=True)
    train.add_argument("--out", required=True, help="output .joblib path")
    _physics_args(train)
    train.set_defaults(func=_cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
