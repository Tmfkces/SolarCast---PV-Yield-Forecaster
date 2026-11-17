"""HTML/CSS for the outlook tab.

Design references: Apple Weather (sky-coloured hero, large light numerals,
sunrise/sunset arc, 10-day list with range bars) and Google Weather (hourly
strip). Colour is used only where it carries meaning: sky for conditions,
amber for the sun and the best hours, green/amber/red for the advice.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from html import escape

import pandas as pd

from .forecast import AppliancePlan, Outlook, clock

STATUS = {
    "high": ("#16A34A", "Use power"),
    "normal": ("#64748B", "Normal"),
    "low": ("#D97706", "Save power"),
    "very_low": ("#DC2626", "Save power"),
}

SKIES = {
    "clear": "linear-gradient(165deg, #1B4F9C 0%, #2F74C8 45%, #6FA9E6 100%)",
    "partly": "linear-gradient(165deg, #33557F 0%, #5E84B0 55%, #93B1D3 100%)",
    "cloudy": "linear-gradient(165deg, #3E4856 0%, #5E6A79 55%, #8995A3 100%)",
    "wet": "linear-gradient(165deg, #1C2430 0%, #313C4C 55%, #4A5768 100%)",
}

_KIND_BY_LABEL = {
    "Clear": "sun",
    "Mostly sunny": "partly",
    "Partly cloudy": "partly",
    "Overcast": "cloud",
    "Cloudy": "cloud",
    "Fog / haze": "fog",
    "Drizzle": "rain",
    "Rain": "rain",
    "Showers": "rain",
    "Snow": "snow",
    "Snow showers": "snow",
    "Thunderstorm": "storm",
}
_SKY_BY_KIND = {"sun": "clear", "partly": "partly", "cloud": "cloudy", "fog": "cloudy"}

_SUN = "#FDB813"
_CLOUD_FILL = "#CBD5E1"
_CLOUD_LINE = "#94A3B8"
_CLOUD = "M7.5 19h9.2a3.8 3.8 0 0 0 .5-7.57A5.2 5.2 0 0 0 7.3 10.1 4.45 4.45 0 0 0 7.5 19z"


def _sun(cx: float, cy: float, r: float, rays: bool = True) -> str:
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_SUN}"/>']
    if rays:
        for i in range(8):
            a = i * math.pi / 4
            x1, y1 = cx + math.cos(a) * (r + 2), cy + math.sin(a) * (r + 2)
            x2, y2 = cx + math.cos(a) * (r + 4.2), cy + math.sin(a) * (r + 4.2)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{_SUN}" stroke-width="1.6" stroke-linecap="round"/>'
            )
    return "".join(parts)


def icon_svg(kind: str, size: int = 28) -> str:
    """Small flat weather icon (24×24 grid) that reads on light and dark backgrounds."""
    cloud = f'<path d="{_CLOUD}" fill="{_CLOUD_FILL}" stroke="{_CLOUD_LINE}" stroke-width="1"/>'
    body = {
        "sun": _sun(12, 12, 4.6),
        "partly": _sun(9, 8.5, 3.6) + cloud,
        "cloud": cloud,
        "fog": "".join(
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{_CLOUD_LINE}" stroke-width="2" stroke-linecap="round"/>'
            for x0, x1, y in [(4, 20, 8), (6, 18, 12.5), (4, 20, 17)]
        ),
        "rain": f'<g transform="translate(0 -3)">{cloud}</g>'
        + "".join(
            f'<line x1="{x}" y1="18.5" x2="{x - 1.2}" y2="21.5" stroke="#60A5FA" stroke-width="1.8" stroke-linecap="round"/>'
            for x in (9, 12.5, 16)
        ),
        "storm": cloud + '<path d="M12.5 13.5 10 18h2.6l-1.4 4 4.2-5.6h-2.7l1.3-2.9z" fill="#FBBF24"/>',
        "snow": cloud + "".join(f'<circle cx="{x}" cy="21" r="1.1" fill="#E0F2FE"/>' for x in (9, 12.5, 16)),
    }.get(kind, cloud)
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" aria-hidden="true">{body}</svg>'


def kind_for(label: str) -> str:
    return _KIND_BY_LABEL.get(label, "cloud")


def kind_for_cloud_cover(cover: float) -> str:
    if pd.isna(cover) or cover < 25:
        return "sun"
    return "partly" if cover < 70 else "cloud"


def day_name(day: pd.Timestamp) -> str:
    today = pd.Timestamp(date.today())
    if day == today:
        return "Today"
    if day == today + pd.Timedelta(days=1):
        return "Tomorrow"
    return f"{day:%a}"


CSS = """
<style>
.sc, .sc * {box-sizing: border-box;}
.sc {font-variant-numeric: tabular-nums;}
.sc-label {font-size: 11.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6;
  display: flex; align-items: center; gap: 6px; margin-bottom: 10px;}
.sc-card {container-type: inline-size; border-radius: 16px; padding: 14px 16px; background: rgba(127,127,127,.07); border: 1px solid rgba(127,127,127,.16);
  margin-bottom: 14px;}

/* Hero */
.sc-hero {border-radius: 22px; padding: 22px 26px 18px; color: #fff; background: var(--sky); margin: 4px 0 16px;
  position: relative; overflow: hidden; box-shadow: 0 12px 32px rgba(15,23,42,.18);}
.sc-hero.sc-clear::before {content: ""; position: absolute; width: 340px; height: 340px; right: -110px; top: -150px;
  background: radial-gradient(circle, rgba(255,214,102,.45) 0%, rgba(255,214,102,0) 65%);}
.sc-hero-grid {position: relative; display: grid; grid-template-columns: 1fr minmax(220px, 300px); gap: 12px; align-items: center;}
@media (max-width: 760px) {.sc-hero-grid {grid-template-columns: 1fr;}}
.sc-when {font-size: 14px; font-weight: 500; opacity: .9;}
.sc-big {display: flex; align-items: baseline; gap: 0 8px; flex-wrap: wrap; margin: 2px 0 2px;}
@media (max-width: 480px) {.sc-big b {font-size: 60px;} .sc-hero {padding: 18px 18px 14px;}}
.sc-big b {font-size: 76px; font-weight: 300; line-height: 1; letter-spacing: -.03em;}
.sc-big span {font-size: 22px; font-weight: 400; opacity: .9; white-space: nowrap;}
.sc-cond {display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 500;}
.sc-rel {margin-top: 6px; font-size: 14px; opacity: .92;}
.sc-rel strong {font-weight: 700;}
.sc-advice {position: relative; margin-top: 16px; padding: 12px 14px; border-radius: 14px;
  background: rgba(255,255,255,.14); backdrop-filter: blur(6px); display: flex; gap: 12px; align-items: flex-start;}
.sc-dot {flex: 0 0 10px; height: 10px; border-radius: 50%; margin-top: 6px; background: var(--st);
  box-shadow: 0 0 0 3px rgba(255,255,255,.25);}
.sc-advice h4 {margin: 0; padding: 0; font-size: 16px; font-weight: 700; color: #fff;}
.sc-advice p {margin: 2px 0 0; font-size: 14px; opacity: .92; line-height: 1.45;}
.sc-arc text {fill: #fff; font-size: 12px; font-family: inherit;}

/* Hourly strip */
.sc-hours {display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px;}
.sc-hour {flex: 0 0 54px; text-align: center; padding: 8px 2px 6px; border-radius: 12px;}
.sc-hour.sc-best {background: rgba(251,191,36,.16);}
.sc-hour .t {font-size: 12px; opacity: .7;}
.sc-hour.sc-best .t {opacity: 1; color: #D97706; font-weight: 700;}
.sc-hour .v {font-size: 14px; font-weight: 600; margin-top: 2px;}
.sc-hour .bar {height: 44px; display: flex; align-items: flex-end; justify-content: center; margin-top: 6px;}
.sc-hour .bar i {display: block; width: 10px; border-radius: 5px 5px 2px 2px; background: linear-gradient(180deg, #FBBF24, #F59E0B);}
.sc-legend {font-size: 12.5px; opacity: .8; margin-top: 8px; display: flex; align-items: center; gap: 6px;}
.sc-legend i {width: 10px; height: 10px; border-radius: 3px; background: rgba(251,191,36,.45); display: inline-block;}

.sc-hour.sc-now {outline: 2px solid currentColor; outline-offset: -2px;}
.sc-hour.sc-now .t {font-weight: 700; opacity: 1;}
.sc-hour.sc-past {opacity: .45;}
.sc-legend {flex-wrap: wrap; gap: 14px;}
.sc-legend span {display: inline-flex; align-items: center; gap: 6px;}
.sc-legend i.pk {background: rgba(220,38,38,.55);}
a.sc-row {color: inherit !important; text-decoration: none !important; cursor: pointer; transition: background .15s;}
a.sc-row:hover {background: rgba(127,127,127,.12);}

/* 7-day list */
.sc-row {display: grid; grid-template-columns: 82px 30px 38px 1fr 74px 92px; gap: 10px; align-items: center;
  padding: 9px 6px; border-top: 1px solid rgba(127,127,127,.14); border-radius: 10px;}
.sc-row:first-of-type {border-top: none;}
.sc-row.sc-sel {background: rgba(127,127,127,.10);}
.sc-row .d {font-weight: 600; font-size: 15px;}
.sc-row .r {font-size: 12px; font-weight: 600; color: #3B82F6;}
.sc-row .k {text-align: right; font-weight: 600; font-size: 15px;}
.sc-row .k small {font-weight: 500; opacity: .6; font-size: 12px;}
.sc-track {position: relative; height: 6px; border-radius: 999px; background: rgba(127,127,127,.18);}
.sc-track i {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: linear-gradient(90deg, #60A5FA, #FBBF24 70%, #F59E0B);}
.sc-track b {position: absolute; top: -4px; width: 2px; height: 14px; border-radius: 1px; background: currentColor; opacity: .55;}
.sc-status {font-size: 12.5px; font-weight: 600; color: var(--st); display: flex; align-items: center; gap: 6px; justify-content: flex-end;}
.sc-status::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--st);}
@container (max-width: 520px) {.sc-row {grid-template-columns: 74px 26px 1fr 72px; gap: 8px;} .sc-row .r, .sc-status {display: none;}}

/* Tiles */
.sc-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 14px;}
.sc-tile {border-radius: 16px; padding: 12px 14px; background: rgba(127,127,127,.07); border: 1px solid rgba(127,127,127,.16);}
.sc-tile .n {font-size: 24px; font-weight: 600; margin-top: 2px;}
.sc-tile .s {font-size: 12px; opacity: .65;}

/* Planner result */
.sc-plan {margin-top: 6px;}
.sc-plan h4 {margin: 0 0 4px; padding: 0; font-size: 15px; font-weight: 700; color: var(--st);}
.sc-plan p {margin: 0; font-size: 14px; line-height: 1.45;}
.sc-meter {height: 6px; border-radius: 999px; background: rgba(127,127,127,.18); margin: 10px 0 8px; overflow: hidden;}
.sc-meter i {display: block; height: 100%; border-radius: 999px; background: var(--st);}
</style>
"""

def _arc_point(frac: float, cx: float = 150, cy: float = 128, r: float = 112) -> tuple[float, float]:
    theta = math.pi * (1 - min(max(frac, 0.0), 1.0))
    return cx + r * math.cos(theta), cy - r * math.sin(theta)


def sun_arc_svg(sunrise: pd.Timestamp, sunset: pd.Timestamp, start: pd.Timestamp | None, end: pd.Timestamp | None) -> str:
    """Sunrise→sunset arc with the best-hours segment highlighted (Apple-Weather-style)."""
    if pd.isna(sunrise) or pd.isna(sunset) or sunset <= sunrise:
        return ""
    span = (sunset - sunrise).total_seconds()

    def frac(t: pd.Timestamp) -> float:
        return (t - sunrise).total_seconds() / span

    x0, y0 = _arc_point(0)
    x1, y1 = _arc_point(1)
    parts = [
        '<svg class="sc-arc" viewBox="0 0 300 168" width="100%" role="img" aria-label="Sun path and best hours">',
        '<line x1="14" y1="128" x2="286" y2="128" stroke="rgba(255,255,255,.35)" stroke-width="1"/>',
        f'<path d="M{x0:.1f} {y0:.1f} A112 112 0 0 1 {x1:.1f} {y1:.1f}" fill="none" '
        'stroke="rgba(255,255,255,.45)" stroke-width="2" stroke-dasharray="3 5"/>',
    ]
    if start is not None and pd.notna(start):
        a, b = frac(start), frac(end)
        ax, ay = _arc_point(a)
        bx, by = _arc_point(b)
        mx, my = _arc_point((a + b) / 2)
        parts += [
            f'<path d="M{ax:.1f} {ay:.1f} A112 112 0 0 1 {bx:.1f} {by:.1f}" fill="none" stroke="#FBBF24" '
            'stroke-width="7" stroke-linecap="round"/>',
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="16" fill="rgba(253,184,19,.28)"/>',
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="9" fill="#FDB813" stroke="#fff" stroke-width="2"/>',
            f'<text x="150" y="104" text-anchor="middle" font-weight="700" font-size="13">Best {clock(start)} – {clock(end)}</text>',
            '<text x="150" y="120" text-anchor="middle" opacity=".8">for heavy appliances</text>',
        ]
    parts += [
        f'<text x="{x0:.0f}" y="150" text-anchor="middle" opacity=".85">↑ {sunrise.strftime("%I:%M %p").lstrip("0")}</text>',
        f'<text x="{x1:.0f}" y="150" text-anchor="middle" opacity=".85">↓ {sunset.strftime("%I:%M %p").lstrip("0")}</text>',
        "</svg>",
    ]
    return "".join(parts)


def hero_html(outlook: Outlook, focus: pd.Timestamp, left_today: float | None = None) -> str:
    row = outlook.days.loc[focus]
    kind = kind_for(row["label"])
    sky = _SKY_BY_KIND.get(kind, "wet")
    color, _ = STATUS[row["level"]]
    rel = ""
    if pd.notna(row["ratio"]):
        rel = (
            f'<div class="sc-rel"><strong>{row["ratio"] * 100:.0f}%</strong> of a usual {focus:%B} day '
            f'({row["typical_kwh"]:.1f} kWh)</div>'
        )
    left = ""
    if left_today is not None:
        left = f'<div class="sc-rel"><strong>{left_today:.1f} kWh</strong> still to come today</div>'
    start = row["window_start"] if pd.notna(row["window_start"]) else None
    end = row["window_end"] if pd.notna(row["window_end"]) else None
    rain = f" · Rain {row['rain_chance']:.0f}%" if pd.notna(row["rain_chance"]) and row["rain_chance"] >= 20 else ""
    best = (
        f'<p style="margin-top:6px"><b>Best hours for heavy appliances: {clock(start)} – {clock(end)}</b></p>'
        if start is not None
        else ""
    )
    return (
        f'<div class="sc sc-hero sc-{sky}" style="--sky:{SKIES[sky]};--st:{color}">'
        '<div class="sc-hero-grid"><div>'
        f'<div class="sc-when">{day_name(focus)} · {focus:%A, %d %B}</div>'
        f'<div class="sc-big"><b>{row["yield_kwh"]:.1f}</b><span>kWh expected</span></div>'
        f'<div class="sc-cond">{icon_svg(kind, 22)}{escape(row["label"])} · H {row["t_max"]:.0f}° L {row["t_min"]:.0f}°{rain}</div>'
        f"{rel}{left}</div>"
        f'<div>{sun_arc_svg(row.get("sunrise", pd.NaT), row.get("sunset", pd.NaT), start, end)}</div>'
        "</div>"
        f'<div class="sc-advice"><div class="sc-dot"></div><div><h4>{escape(row["headline"])}</h4>'
        f'<p>{escape(row["detail"])}</p>{best}</div></div>'
        "</div>"
    )


def hourly_html(
    hours: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    now: pd.Timestamp | None = None,
    peak: str | None = None,
) -> str:
    """Google-Weather-style hourly strip. ``hours`` has hour-end ``energy_kwh`` and ``cloud_cover``.

    ``now`` (local, naive) marks the current hour and dims hours already past.
    """
    visible = hours[hours["energy_kwh"] >= 0.05]
    if visible.empty:
        return '<div class="sc sc-card"><div class="sc-label">Hourly output</div>No meaningful solar output expected.</div>'
    top = max(visible["energy_kwh"].max(), 1e-9)
    cells = []
    for stamp, row in visible.iterrows():
        hour_start = stamp - pd.Timedelta(hours=1)
        classes = ["sc-hour"]
        label = clock(hour_start)
        if start is not None and start <= hour_start and stamp <= end:
            classes.append("sc-best")
        if now is not None:
            if hour_start <= now < stamp:
                classes.append("sc-now")
                label = "Now"
            elif stamp <= now:
                classes.append("sc-past")
        height = max(row["energy_kwh"] / top * 44, 3)
        cover = row["cloud_cover"] if "cloud_cover" in row else float("nan")
        cells.append(
            f'<div class="{" ".join(classes)}">'
            f'<div class="t">{label}</div>'
            f"<div>{icon_svg(kind_for_cloud_cover(cover), 24)}</div>"
            f'<div class="v">{row["energy_kwh"]:.1f}</div>'
            f'<div class="bar"><i style="height:{height:.0f}px"></i></div>'
            "</div>"
        )
    legend = ""
    if start is not None:
        legend += f'<span><i></i>Best 3 hours for heavy appliances ({clock(start)} – {clock(end)})</span>'
    if peak:
        legend += f'<span><i class="pk"></i>Peak tariff {peak}: avoid heavy loads</span>'
    return (
        '<div class="sc sc-card"><div class="sc-label">Hourly solar output · kWh</div>'
        f'<div class="sc-hours">{"".join(cells)}</div><div class="sc-legend">{legend}</div></div>'
    )


def week_html(outlook: Outlook, focus: pd.Timestamp, href: Callable[[pd.Timestamp], str] | None = None) -> str:
    """Apple-Weather-style list: one row per day with a bar and a tick for the usual yield.

    When ``href`` is given each row is a link (tap a day to open it).
    """
    days = outlook.days
    scale = max(days["yield_kwh"].max(), days["typical_kwh"].max(skipna=True) if days["typical_kwh"].notna().any() else 0) * 1.05
    scale = max(scale, 1e-9)
    rows = []
    for day, row in days.iterrows():
        color, status = STATUS[row["level"]]
        rain = f'{row["rain_chance"]:.0f}%' if pd.notna(row["rain_chance"]) and row["rain_chance"] >= 20 else ""
        tick = ""
        if pd.notna(row["typical_kwh"]):
            tick = f'<b style="left:{row["typical_kwh"] / scale * 100:.1f}%" title="Usual: {row["typical_kwh"]:.1f} kWh"></b>'
        tag, attrs = ("a", f' href="{escape(href(day))}" target="_self"') if href else ("div", "")
        rows.append(
            f'<{tag} class="sc-row{" sc-sel" if day == focus else ""}"{attrs} style="--st:{color}">'
            f'<div class="d">{day_name(day)}</div>'
            f'<div title="{escape(row["label"])}">{icon_svg(kind_for(row["label"]), 26)}</div>'
            f'<div class="r">{rain}</div>'
            f'<div class="sc-track"><i style="width:{row["yield_kwh"] / scale * 100:.1f}%"></i>{tick}</div>'
            f'<div class="k">{row["yield_kwh"]:.1f} <small>kWh</small></div>'
            f'<div class="sc-status">{status}</div>'
            f"</{tag}>"
        )
    return (
        '<div class="sc sc-card"><div class="sc-label">7-day solar outlook</div>'
        f'{"".join(rows)}'
        '<div class="sc-legend" style="margin-top:10px"><span><span style="display:inline-block;width:2px;height:12px;'
        'background:currentColor;opacity:.55"></span>Tick marks a usual day for that month. Tap a day to open it.</span></div></div>'
    )


def tiles_html(outlook: Outlook, value: dict[str, float], price_per_kwh: float, setup_label: str) -> str:
    """Three summary tiles. ``value`` comes from summing :func:`pvyield.forecast.estimate_value` over the week."""
    days = outlook.days
    total = days["yield_kwh"].sum()
    save_days = int(days["level"].isin(["low", "very_low"]).sum())
    best = days["yield_kwh"].idxmax()
    split = f'{value["self_kwh"]:,.0f} kWh used at home'
    if value.get("export_kwh", 0) > 0.5:
        split += f' · {value["export_kwh"]:,.0f} exported'
    elif value.get("unused_kwh", 0) > 0.5:
        split += f' · {value["unused_kwh"]:,.0f} unused'
    tiles = [
        ("This week", f"{total:,.0f} kWh", f"Sunniest: {day_name(best)}"),
        ("Value this week", f'Rs {value["rupees"]:,.0f}', split),
        ("Save-power days", f"{save_days} of {len(days)}", "below a usual day" if save_days else "none this week"),
    ]
    cells = "".join(
        f'<div class="sc-tile"><div class="sc-label" style="margin:0">{k}</div><div class="n">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in tiles
    )
    note = (
        f'<div class="sc-legend" style="grid-column:1/-1;margin-top:-4px">{escape(setup_label)} · '
        f"home use valued at Rs {price_per_kwh:g}/kWh</div>"
    )
    return f'<div class="sc sc-tiles">{cells}{note}</div>'


def plan_html(selected: list[str], plan: AppliancePlan, day_label: str) -> str:
    if not selected:
        return '<div class="sc sc-plan"><p style="opacity:.7">Pick the appliances you want to run.</p></div>'
    if plan.start is None:
        color, title = STATUS["very_low"][0], "Not enough sun for these"
        body = "Expected solar barely covers your basic load. Run them on the next sunny day or from the grid."
    elif plan.fits:
        color, title = STATUS["high"][0], f"Solar can run all of this {day_label.lower()}"
        body = (
            f"Run them between <b>{clock(plan.start)} and {clock(plan.end)}</b>. "
            f"Spare solar then covers the full {plan.need_kwh:.1f} kWh."
        )
    else:
        color, title = STATUS["low"][0], f"Solar covers about {plan.coverage:.0%} of it"
        body = (
            f"Across <b>{clock(plan.start)} – {clock(plan.end)}</b> spare solar gives {plan.solar_kwh:.1f} of the "
            f"{plan.need_kwh:.1f} kWh needed. Pick fewer appliances or split them over two days."
        )
    return (
        f'<div class="sc sc-plan" style="--st:{color}"><h4>{title}</h4>'
        f'<div class="sc-meter"><i style="width:{plan.coverage * 100:.0f}%"></i></div><p>{body}</p></div>'
    )
