"""HTML/CSS for the SolarCast outlook.

Visual direction: soft, light and illustrated, like modern consumer air-quality
and weather apps. A pastel sky scene with a Pakistani skyline (Minar-e-Pakistan,
mosque domes, houses with rooftop solar, Basant kites), a colourful ring gauge,
white rounded cards with coloured underlines, and icon tiles that say what to do.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from html import escape

import pandas as pd

from .forecast import APPLIANCES, AppliancePlan, Outlook, clock

STATUS = {
    "high": ("#16A34A", "Use power"),
    "normal": ("#5B5BD6", "Normal"),
    "low": ("#E8890C", "Save power"),
    "very_low": ("#E5484D", "Save power"),
}
STATUS_BG = {"high": "#E7F8EE", "normal": "#ECECFD", "low": "#FFF2E0", "very_low": "#FDE8E8"}
ACCENTS = ["#5B5BD6", "#F59E0B", "#14B8A6", "#E5484D", "#8B5CF6", "#0EA5E9", "#22C55E", "#F472B6"]

# Sky palettes for the hero illustration: (top, middle, bottom, text colour)
SKIES = {
    "day": ("#7DB7F5", "#A9D0FA", "#E4F1FF", "#1E2A4A"),
    "cloudy": ("#9FB0C8", "#C3CFDF", "#ECF0F6", "#1E2A4A"),
    "rain": ("#7F8BA3", "#A5AFC2", "#E0E4EC", "#1E2A4A"),
    "sunset": ("#F39C86", "#F8BDA9", "#FDE5DA", "#3B1F2B"),
    "night": ("#6253D1", "#8574E4", "#B7ABF2", "#FFFFFF"),
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

_SUN = "#FDB813"
_CLOUD_FILL = "#FFFFFF"
_CLOUD_LINE = "#B8C4D6"
_CLOUD = "M7.5 19h9.2a3.8 3.8 0 0 0 .5-7.57A5.2 5.2 0 0 0 7.3 10.1 4.45 4.45 0 0 0 7.5 19z"


# --------------------------------------------------------------------------- helpers


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


def _sun(cx: float, cy: float, r: float) -> str:
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_SUN}"/>']
    for i in range(8):
        a = i * math.pi / 4
        x1, y1 = cx + math.cos(a) * (r + 2), cy + math.sin(a) * (r + 2)
        x2, y2 = cx + math.cos(a) * (r + 4.2), cy + math.sin(a) * (r + 4.2)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_SUN}" '
            'stroke-width="1.6" stroke-linecap="round"/>'
        )
    return "".join(parts)


def icon_svg(kind: str, size: int = 28) -> str:
    """Small flat weather icon (24×24 grid)."""
    cloud = f'<path d="{_CLOUD}" fill="{_CLOUD_FILL}" stroke="{_CLOUD_LINE}" stroke-width="1.1"/>'
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
        "snow": cloud + "".join(f'<circle cx="{x}" cy="21" r="1.1" fill="#93C5FD"/>' for x in (9, 12.5, 16)),
    }.get(kind, cloud)
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" aria-hidden="true">{body}</svg>'


# Line icons for appliances (24×24, stroke = currentColor)
_APPLIANCE_PATHS = {
    "Washing machine": '<rect x="4" y="3" width="16" height="18" rx="2.5"/><circle cx="12" cy="13" r="4.5"/>'
    '<path d="M9.5 13.5c1-.8 2-.8 3 0s2 .8 3 0"/><line x1="7" y1="6.5" x2="9" y2="6.5"/>',
    "Iron": '<path d="M3 17h15a3 3 0 0 0 3-3v-1a5 5 0 0 0-5-5H9a6 6 0 0 0-6 6z"/><path d="M8 8V6h7"/>'
    '<line x1="7" y1="13" x2="15" y2="13"/>',
    "Water pump (1 hp)": '<path d="M12 3s5 5.6 5 9.5a5 5 0 0 1-10 0C7 8.6 12 3 12 3z"/><path d="M10 13.5a2 2 0 0 0 2 2"/>',
    "Electric geyser": '<rect x="7" y="2.5" width="10" height="15" rx="5"/><path d="M12 17.5V21"/>'
    '<path d="M10 7c.8.8.8 1.7 0 2.5M14 7c.8.8.8 1.7 0 2.5"/>',
    "Inverter AC (1.5 ton)": '<rect x="2.5" y="5" width="19" height="8" rx="2"/><line x1="5" y1="10" x2="19" y2="10"/>'
    '<path d="M8 16v3M12 16v4M16 16v3"/>',
    "Microwave / oven": '<rect x="2.5" y="5" width="19" height="14" rx="2"/><rect x="5" y="8" width="10" height="8" rx="1"/>'
    '<line x1="18" y1="9" x2="18" y2="9.01"/><line x1="18" y1="12.5" x2="18" y2="15"/>',
    "Electric kettle": '<path d="M6 8h10l1.5 11.5a1.5 1.5 0 0 1-1.5 1.5H6a1.5 1.5 0 0 1-1.5-1.5z"/>'
    '<path d="M17 10h1.5a2 2 0 0 1 0 4H17.6"/><path d="M8 8l1.5-3h3L14 8"/>',
    "EV / e-bike charging": '<circle cx="6" cy="17" r="3.5"/><circle cx="18" cy="17" r="3.5"/>'
    '<path d="M6 17l4-7h5l3 7M10 10l-1.5-3H6"/><path d="M13.5 4 12 6.5h2.2L13 9"/>',
}


def appliance_svg(name: str, size: int = 26) -> str:
    body = _APPLIANCE_PATHS.get(name, '<circle cx="12" cy="12" r="8"/>')
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


# --------------------------------------------------------------------------- illustration


def _cloud(x: float, y: float, s: float, opacity: float = 0.95) -> str:
    return (
        f'<g transform="translate({x} {y}) scale({s})" fill="#FFFFFF" opacity="{opacity}">'
        '<ellipse cx="0" cy="8" rx="22" ry="8"/><circle cx="-8" cy="3" r="8"/><circle cx="6" cy="0" r="11"/>'
        '<circle cx="17" cy="5" r="7"/></g>'
    )


def _kite(x: float, y: float, color: str, tail: str) -> str:
    return (
        f'<g transform="translate({x} {y})"><path d="M0 -12 L9 0 L0 14 L-9 0 Z" fill="{color}"/>'
        '<path d="M0 -12 L0 14 M-9 0 L9 0" stroke="#FFFFFF" stroke-width=".8" opacity=".7"/>'
        f'<path d="M0 14 q5 8 -2 14 q-6 6 2 12" fill="none" stroke="{tail}" stroke-width="1.1"/></g>'
    )


def _skyline() -> str:
    """Pakistani skyline silhouette: houses with rooftop solar, mosque domes, Minar-e-Pakistan."""
    panel = "#3B4A8C"
    parts = [
        '<g fill="#FFFFFF">',
        '<rect x="0" y="118" width="400" height="22"/>',
        '<rect x="6" y="92" width="44" height="30" rx="2"/><rect x="54" y="84" width="38" height="38" rx="2"/>',
        '<rect x="104" y="96" width="64" height="26"/><path d="M118 96 a18 16 0 0 1 36 0z"/>'
        '<rect x="134" y="72" width="4" height="10"/><circle cx="136" cy="70" r="2.5"/>'
        '<rect x="98" y="66" width="7" height="56"/><path d="M96 66 h11 l-5.5 -9z"/>'
        '<rect x="167" y="66" width="7" height="56"/><path d="M165 66 h11 l-5.5 -9z"/>',
        '<path d="M206 122 L211 60 L221 60 L226 122 Z"/><rect x="200" y="112" width="32" height="10"/>'
        '<path d="M208 60 q8 -14 16 0z"/><rect x="215" y="40" width="2.4" height="8"/>'
        '<path d="M209 74 h14 M208.5 86 h15" stroke="#E9EEF8" stroke-width="2"/>',
        '<rect x="244" y="88" width="40" height="34" rx="2"/><rect x="290" y="70" width="28" height="52" rx="2"/>'
        '<rect x="324" y="94" width="34" height="28" rx="2"/><rect x="362" y="84" width="34" height="38" rx="2"/>',
        '<circle cx="186" cy="108" r="9"/><circle cx="238" cy="110" r="7"/>',
        "</g>",
        f'<g fill="{panel}" opacity=".85">',
        '<path d="M9 92 l6 -7 h14 l-6 7z"/><path d="M27 92 l6 -7 h14 l-6 7z"/>',
        '<path d="M57 84 l6 -7 h13 l-6 7z"/><path d="M74 84 l6 -7 h12 l-6 7z"/>',
        '<path d="M247 88 l6 -7 h14 l-6 7z"/><path d="M265 88 l6 -7 h14 l-6 7z"/>',
        '<path d="M327 94 l6 -7 h12 l-6 7z"/><path d="M365 84 l6 -7 h13 l-6 7z"/>',
        "</g>",
        '<g fill="#DCE6F5">',
        '<rect x="14" y="100" width="7" height="7" rx="1"/><rect x="34" y="100" width="7" height="7" rx="1"/>'
        '<rect x="62" y="94" width="7" height="7" rx="1"/><rect x="77" y="94" width="7" height="7" rx="1"/>'
        '<rect x="62" y="108" width="7" height="7" rx="1"/><rect x="77" y="108" width="7" height="7" rx="1"/>'
        '<rect x="296" y="78" width="6" height="6"/><rect x="307" y="78" width="6" height="6"/>'
        '<rect x="296" y="90" width="6" height="6"/><rect x="307" y="90" width="6" height="6"/>'
        '<rect x="296" y="102" width="6" height="6"/><rect x="307" y="102" width="6" height="6"/>'
        '<rect x="252" y="98" width="7" height="7" rx="1"/><rect x="268" y="98" width="7" height="7" rx="1"/>'
        '<rect x="370" y="94" width="7" height="7" rx="1"/><rect x="382" y="94" width="7" height="7" rx="1"/>',
        "</g>",
    ]
    return "".join(parts)


def scene_svg(sky: str, kind: str) -> str:
    """Full hero illustration for one sky palette and weather kind."""
    top, mid, bottom, _ = SKIES[sky]
    night = sky == "night"
    parts = [
        '<svg class="pc-scene" viewBox="0 0 560 290" preserveAspectRatio="xMinYMax slice" aria-hidden="true">',
        f'<defs><linearGradient id="sky-{sky}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{top}"/>'
        f'<stop offset=".55" stop-color="{mid}"/><stop offset="1" stop-color="{bottom}"/></linearGradient>'
        '<radialGradient id="sunglow"><stop offset="0" stop-color="#FFE38A"/><stop offset=".6" stop-color="#FFC93C"/>'
        '<stop offset="1" stop-color="#FFB703"/></radialGradient></defs>',
        f'<rect width="560" height="290" fill="url(#sky-{sky})"/>',
    ]
    if night:
        parts.append('<path d="M56 36 a26 26 0 1 0 24 38 a21 21 0 1 1 -24 -38z" fill="#FFFFFF"/>')
        parts += [
            f'<circle cx="{x}" cy="{y}" r="1" fill="#FFFFFF" opacity=".8"/>'
            for x, y in [(130, 30), (170, 60), (210, 24), (150, 110), (110, 140), (60, 120), (190, 150)]
        ]
    elif kind in {"sun", "partly"} or sky == "sunset":
        parts.append('<circle cx="66" cy="62" r="48" fill="#FFF6D6" opacity=".45"/>')
        parts.append('<circle cx="66" cy="62" r="32" fill="url(#sunglow)"/>')
    if kind in {"partly", "cloud", "rain", "storm", "fog", "snow"}:
        parts += [_cloud(120, 96, 1.1), _cloud(230, 70, 0.9), _cloud(330, 130, 0.85, 0.9), _cloud(60, 150, 0.8, 0.9)]
    else:
        parts += [_cloud(160, 70, 0.75, 0.9), _cloud(340, 120, 0.6, 0.85)]
    if kind in {"rain", "storm"}:
        parts += [
            f'<line x1="{x}" y1="{y}" x2="{x - 3}" y2="{y + 8}" stroke="#E0ECFF" stroke-width="1.4" '
            'stroke-linecap="round" opacity=".9"/>'
            for x, y in [(100, 112), (114, 120), (128, 112), (180, 146), (194, 152), (54, 170), (68, 176)]
        ]
    if not night and kind in {"sun", "partly"}:
        parts += [_kite(170, 150, "#F472B6", "#BE185D"), _kite(92, 168, "#8B5CF6", "#5B21B6")]
    bird = "#FFFFFF" if night else "#4B5A7A"
    parts += [
        f'<path d="M{x} {y} q3 -3 6 0 q3 -3 6 0" fill="none" stroke="{bird}" stroke-width="1.1" stroke-linecap="round"/>'
        for x, y in [(210, 40), (226, 34), (240, 44)]
    ]
    skyline = _skyline()
    parts.append(f'<g transform="translate(0 150)">{skyline}</g><g transform="translate(400 150)">{skyline}</g>')
    parts.append("</svg>")
    return "".join(parts)


def sky_for(kind: str, focus: pd.Timestamp, now: pd.Timestamp | None) -> str:
    if kind in {"rain", "storm", "snow"}:
        return "rain"
    if kind in {"cloud", "fog"}:
        return "cloudy"
    if now is not None and focus == now.normalize():
        if now.hour >= 19 or now.hour < 5:
            return "night"
        if now.hour >= 16 or now.hour < 7:
            return "sunset"
    return "day"


# --------------------------------------------------------------------------- styles

BASE_CSS = """
<style>
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% -10%, var(--pc-glow1) 0%, transparent 60%),
              radial-gradient(900px 500px at 100% 0%, var(--pc-glow2) 0%, transparent 55%), var(--pc-page);
}
.pc, .pc * {box-sizing: border-box;}
.pc {font-variant-numeric: tabular-nums; color: var(--pc-ink);}
.pc-card {background: var(--pc-card); border-radius: 22px; padding: 16px 18px; margin-bottom: 16px;
  box-shadow: 0 8px 28px var(--pc-shadow); container-type: inline-size;}
.pc-title {font-size: 15px; font-weight: 600; text-align: center; margin: 2px 0 12px; color: var(--pc-ink2);}
.pc-label {font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--pc-muted);
  margin-bottom: 10px;}

.pc-top {display: flex; align-items: center; gap: 12px; margin: -6px 0 6px;}
.pc-loc {flex: 1; display: flex; align-items: center; gap: 10px; background: var(--pc-card); border-radius: 999px;
  padding: 10px 16px; box-shadow: 0 6px 20px var(--pc-shadow); font-size: 15px; font-weight: 500; flex-wrap: wrap;}
.pc-loc small {color: var(--pc-muted); font-weight: 400; margin-left: auto; font-size: 12.5px;}
.pc-avatar {width: 44px; height: 44px; border-radius: 50%; background: var(--pc-card); display: grid; place-items: center;
  box-shadow: 0 6px 20px var(--pc-shadow); flex: 0 0 44px;}
.pc-live {display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 600; color: var(--pc-ink2);
  margin: 6px 0 2px;}
.pc-live::before {content: ""; width: 9px; height: 9px; border-radius: 50%; background: #22C55E;
  box-shadow: 0 0 0 4px rgba(34,197,94,.18);}

.pc-hero {position: relative; border-radius: 26px; overflow: hidden; height: 300px; margin-bottom: 16px;
  box-shadow: 0 14px 34px var(--pc-shadow); color: var(--ink);}
.pc-scene {position: absolute; inset: 0; width: 100%; height: 100%; filter: var(--pc-scene-filter);}
.pc-hero-top {position: relative; display: flex; justify-content: flex-end; padding: 18px 20px 0;}
.pc-when {font-size: 13px; font-weight: 600; opacity: .9; margin-top: 104px; max-width: 60%; line-height: 1.45;}
.pc-ringwrap {display: flex; flex-direction: column; align-items: center; gap: 6px;}
.pc-ring {position: relative; width: 116px; height: 116px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(#E5484D, #F59E0B 22%, #FACC15 42%, #22C55E 66%, #14B8A6 88%, #E5484D 100%);
  box-shadow: 0 8px 20px rgba(43,42,76,.15);}
.pc-ring .pc-dot {position: absolute; inset: 0; transform: rotate(var(--a)); pointer-events: none;}
.pc-ring .pc-dot::after {content: ""; position: absolute; top: 3px; left: 50%; width: 12px; height: 12px; margin-left: -6px;
  border-radius: 50%; background: #FFFFFF; border: 3px solid #2B2A4C;}
.pc-ring > div {width: 92px; height: 92px; border-radius: 50%; background: var(--pc-card); display: grid; place-items: center;
  text-align: center; color: var(--pc-ink);}
.pc-ring small {display: block; font-size: 9.5px; color: var(--pc-muted); font-weight: 600; letter-spacing: .04em;}
.pc-ring b {display: block; font-size: 28px; font-weight: 700; line-height: 1.05;}
.pc-pin {font-size: 12px; font-weight: 600; text-align: center; line-height: 1.4; max-width: 150px;}

.pc-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px 10px;}
@container (max-width: 420px) {.pc-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.pc-m {text-align: center;}
.pc-m .k {font-size: 11.5px; color: var(--pc-muted);}
.pc-m .v {font-size: 17px; font-weight: 600; margin-top: 2px; white-space: nowrap;}
.pc-m .u {font-size: 10.5px; color: var(--pc-muted);}
.pc-m i {display: block; height: 3px; border-radius: 3px; margin: 6px auto 0; width: 70%; background: var(--c);}

.pc-advice {text-align: center; font-size: 13.5px; color: var(--pc-body); margin: -4px 4px 14px; line-height: 1.55;}
.pc-advice b {color: var(--st);}
.pc-tiles {display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px;}
.pc-tile {text-align: center; border-radius: 16px; padding: 12px 8px; background: var(--bg);}
.pc-tile .ic {width: 46px; height: 46px; margin: 0 auto 6px; border-radius: 50%; background: var(--pc-card); display: grid;
  place-items: center; color: var(--st);}
.pc-tile .n {font-size: 12.5px; font-weight: 600; line-height: 1.25;}
.pc-tile .s {font-size: 11.5px; color: var(--st); font-weight: 600; margin-top: 3px;}

.pc-gauge {text-align: center;}
.pc-gauge svg {width: 100%; max-width: 300px;}
.pc-gauge .lbl {font-size: 13px; color: var(--pc-muted); margin-top: -4px;}
.pc-gauge .st {display: inline-block; margin-top: 8px; font-size: 12.5px; font-weight: 600; color: var(--st);
  background: var(--bg); padding: 4px 12px; border-radius: 999px;}

.pc-wx {position: relative; border-radius: 22px; padding: 18px 20px; margin-bottom: 16px; overflow: hidden; min-height: 130px;
  background: var(--pc-wx); color: #FFFFFF;
  box-shadow: 0 12px 28px var(--pc-shadow);}
.pc-wx .t {font-size: 40px; font-weight: 600; line-height: 1;}
.pc-wx .t sup {font-size: 16px;}
.pc-wx .c {font-size: 18px; font-weight: 600; margin-top: 8px;}
.pc-wx .d {font-size: 12.5px; opacity: .9;}
.pc-wx .pill {display: inline-block; margin-top: 8px; padding: 3px 12px; border-radius: 999px; font-size: 12px;
  font-weight: 600; background: rgba(255,255,255,.25);}
.pc-wx .art {position: absolute; right: 8px; top: 8px; width: 132px; height: 112px;}

.pc-save {display: grid; grid-template-columns: 1fr 120px; align-items: center; gap: 8px; border-radius: 22px;
  padding: 18px 18px 14px; margin-bottom: 16px; background: var(--pc-save);
  box-shadow: 0 8px 24px var(--pc-shadow);}
.pc-save .k {font-size: 12px; font-weight: 600; color: var(--pc-save-k); letter-spacing: .04em; text-transform: uppercase;}
.pc-save .v {font-size: 28px; font-weight: 700; color: var(--pc-save-v); margin: 2px 0;}
.pc-save .s {font-size: 12.5px; color: var(--pc-save-s);}

.pc-hours {display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px;}
.pc-hour {flex: 0 0 58px; text-align: center; padding: 10px 2px 8px; border-radius: 16px; background: var(--pc-soft);}
.pc-hour.best {background: var(--pc-best);}
.pc-hour .t {font-size: 11.5px; color: var(--pc-muted);}
.pc-hour.best .t {color: var(--pc-best-ink); font-weight: 700;}
.pc-hour.now {outline: 2px solid var(--pc-accent); outline-offset: -2px;}
.pc-hour.now .t {color: var(--pc-accent); font-weight: 700;}
.pc-hour.past {opacity: .45;}
.pc-hour .v {font-size: 14px; font-weight: 600; margin-top: 2px;}
.pc-hour .bar {height: 40px; display: flex; align-items: flex-end; justify-content: center; margin-top: 6px;}
.pc-hour .bar i {display: block; width: 10px; border-radius: 6px; background: linear-gradient(180deg, #FFC93C, #F59E0B);}
.pc-legend {display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--pc-muted); margin-top: 10px;}
.pc-legend span {display: inline-flex; align-items: center; gap: 6px;}
.pc-legend i {width: 10px; height: 10px; border-radius: 3px; background: var(--pc-legend-best);}
.pc-legend i.pk {background: var(--pc-legend-pk);}

.pc-row {display: grid; grid-template-columns: 86px 32px 1fr 74px 96px; gap: 10px; align-items: center;
  padding: 10px 8px; border-radius: 14px; color: inherit !important; text-decoration: none !important;}
.pc-row + .pc-row {margin-top: 2px;}
a.pc-row:hover {background: var(--pc-hover);}
.pc-row.sel {background: var(--pc-sel);}
.pc-row .d {font-weight: 600; font-size: 14.5px;}
.pc-row .k {text-align: right; font-weight: 600; font-size: 14.5px;}
.pc-row .k small {font-weight: 400; color: var(--pc-muted); font-size: 11.5px;}
.pc-track {position: relative; height: 7px; border-radius: 999px; background: var(--pc-track);}
.pc-track i {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;
  background: linear-gradient(90deg, #8FB2FF, #FFC93C 70%, #F59E0B);}
.pc-track b {position: absolute; top: -4px; width: 2px; height: 15px; border-radius: 1px; background: var(--pc-ink); opacity: .4;}
.pc-status {justify-self: end; font-size: 11.5px; font-weight: 600; color: var(--st); background: var(--bg);
  padding: 3px 10px; border-radius: 999px; white-space: nowrap;}
@container (max-width: 520px) {.pc-row {grid-template-columns: 70px 28px 1fr 66px;} .pc-status {display: none;}}

.pc-plan {margin-top: 12px; padding: 12px 14px; border-radius: 16px; background: var(--bg);}
.pc-plan h4 {margin: 0 0 4px; padding: 0; font-size: 14.5px; font-weight: 600; color: var(--st);}
.pc-plan p {margin: 0; font-size: 13px; line-height: 1.45; color: var(--pc-body);}
.pc-meter {height: 6px; border-radius: 999px; background: var(--pc-meter); margin: 8px 0; overflow: hidden;}
.pc-meter i {display: block; height: 100%; border-radius: 999px; background: var(--st);}
</style>
"""


LIGHT = {
    "page": "#F6F4FD", "glow1": "#EDE7FF", "glow2": "#FFEDE3",
    "card": "#FFFFFF", "ink": "#2B2A4C", "ink2": "#3A3966", "muted": "#8A89A8", "body": "#55547A",
    "shadow": "rgba(91,91,214,.10)", "soft": "#F7F6FE", "best": "#FFF4D6", "best-ink": "#B7791F",
    "hover": "#F4F2FF", "sel": "#EEEBFF", "track": "#EFEDF9", "accent": "#5B5BD6",
    "st-high": "#16A34A", "st-normal": "#5B5BD6", "st-low": "#D97706", "st-very_low": "#E5484D",
    "bg-high": "#E7F8EE", "bg-normal": "#ECECFD", "bg-low": "#FFF2E0", "bg-very_low": "#FDE8E8",
    "wx": "linear-gradient(135deg, #6E8EF5 0%, #8FB2FF 60%, #B8D2FF 100%)",
    "save": "linear-gradient(135deg, #FFF1E6 0%, #FCE7F3 100%)",
    "save-k": "#A1557A", "save-v": "#3B1F2B", "save-s": "#7A5566",
    "meter": "rgba(255,255,255,.8)", "legend-best": "#FFE3A3", "legend-pk": "#FBC4C6",
    "scene-filter": "none",
}
DARK = {
    "page": "#0F1020", "glow1": "#251E57", "glow2": "#34182A",
    "card": "#1A1B30", "ink": "#ECEBFF", "ink2": "#D9D8F7", "muted": "#9C9BC4", "body": "#BDBCE0",
    "shadow": "rgba(0,0,0,.38)", "soft": "#23243D", "best": "#3B3114", "best-ink": "#FBBF24",
    "hover": "#25274A", "sel": "#2D2E5C", "track": "#2B2C47", "accent": "#9C9CFF",
    "st-high": "#4ADE80", "st-normal": "#A5A5FF", "st-low": "#FBBF24", "st-very_low": "#F87171",
    "bg-high": "#123524", "bg-normal": "#26285A", "bg-low": "#3D2D12", "bg-very_low": "#421C23",
    "wx": "linear-gradient(135deg, #2B3B97 0%, #3C55BE 60%, #5C78D6 100%)",
    "save": "linear-gradient(135deg, #3A2235 0%, #2A2348 100%)",
    "save-k": "#F9A8D4", "save-v": "#FFF1F7", "save-s": "#E2C6D6",
    "meter": "rgba(255,255,255,.14)", "legend-best": "#7A6220", "legend-pk": "#7A2E36",
    "scene-filter": "brightness(.8) saturate(1.05)",
}


def _vars(palette: dict[str, str]) -> str:
    return " ".join(f"--pc-{k}: {v};" for k, v in palette.items())


def theme_css() -> str:
    """CSS for the outlook in both colour schemes.

    The live theme is applied in the browser, without a rerun:

    1. Before anything else loads, the device setting decides via
       ``prefers-color-scheme`` (what Streamlit's "System" theme also uses).
    2. :data:`THEME_WATCHER` then marks ``<html data-pc-theme="light|dark">`` from
       Streamlit's *actual* theme and keeps it in sync when the user changes
       the theme in the Settings menu, so the cards switch instantly.
    """
    app = '[data-testid="stAppViewContainer"]'
    rules = [
        f":root {{ {_vars(LIGHT)} }}",
        f"@media (prefers-color-scheme: dark) {{ :root {{ {_vars(DARK)} }} }}",
        f'html[data-pc-theme="light"] {app} {{ {_vars(LIGHT)} }}',
        f'html[data-pc-theme="dark"] {app} {{ {_vars(DARK)} }}',
    ]
    return BASE_CSS.replace("<style>", "<style>\n" + "\n".join(rules), 1)


CSS = theme_css()

# Tiny script (run inside a zero-height component iframe, which shares the app's
# origin) that mirrors Streamlit's current theme onto <html data-pc-theme>.
# Streamlit repaints its own widgets on a theme change without re-running the
# script, so the cards must follow on the client side.
THEME_WATCHER = """
<script>
(function () {
  var doc = window.parent && window.parent.document;
  if (!doc) return;
  var win = window.parent;
  function scheme() {
    var app = doc.querySelector('[data-testid="stApp"]');
    if (!app) return null;
    var cs = win.getComputedStyle(app);
    if (cs.colorScheme === 'dark' || cs.colorScheme === 'light') return cs.colorScheme;
    var m = (cs.backgroundColor || '').match(/\\d+(\\.\\d+)?/g);
    if (!m) return null;
    var lum = 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
    return lum < 128 ? 'dark' : 'light';
  }
  function sync() {
    var s = scheme();
    if (s && doc.documentElement.getAttribute('data-pc-theme') !== s) {
      doc.documentElement.setAttribute('data-pc-theme', s);
    }
  }
  sync();
  if (win.__pcThemeWatcher) return;
  win.__pcThemeWatcher = true;
  var queued = false;
  function soon() {
    if (queued) return;
    queued = true;
    win.requestAnimationFrame(function () { queued = false; sync(); });
  }
  new win.MutationObserver(soon).observe(doc.body, {attributes: true, subtree: true, attributeFilter: ['class']});
  win.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () { setTimeout(sync, 50); });
  win.setInterval(sync, 1000);
})();
</script>
"""


# --------------------------------------------------------------------------- components


def header_html(city: str, province: str, kw: float, setup_label: str) -> str:
    pin = (
        '<svg width="18" height="18" viewBox="0 0 24 24" style="fill:var(--pc-accent)"><path d="M12 2a7 7 0 0 0-7 7c0 5.2 7 13 7 13s7-7.8 '
        '7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5z"/></svg>'
    )
    return (
        '<div class="pc"><div class="pc-top">'
        f'<div class="pc-loc">{pin}<span>{escape(city)}, {escape(province)}</span>'
        f"<small>{kw:g} kWp · {escape(setup_label)}</small></div>"
        f'<div class="pc-avatar">{icon_svg("sun", 26)}</div>'
        '</div><div class="pc-live">Live solar forecast</div></div>'
    )


def hero_html(
    outlook: Outlook,
    focus: pd.Timestamp,
    city: str = "",
    left_today: float | None = None,
    now: pd.Timestamp | None = None,
) -> str:
    row = outlook.days.loc[focus]
    kind = kind_for(row["label"])
    sky = sky_for(kind, focus, now)
    ink = SKIES[sky][3]
    ratio = row["ratio"] if pd.notna(row["ratio"]) else 0.0
    ring = max(0.0, min(ratio / 1.5, 1.0)) * 100
    pct = f"{ratio * 100:.0f}%" if pd.notna(row["ratio"]) else "—"
    pin = f"● {escape(city)} · {escape(row['label'])}"
    if left_today is not None:
        pin += f"<br>{left_today:.1f} kWh still to come today"
    return (
        f'<div class="pc pc-hero" style="--ink:{ink}">{scene_svg(sky, kind)}'
        '<div class="pc-hero-top"><div class="pc-ringwrap">'
        f'<div class="pc-ring"><i class="pc-dot" style="--a:{ring * 3.6 * 0.88:.0f}deg"></i><div><div><small>OF USUAL</small><b>{pct}</b>'
        f'<small>{row["yield_kwh"]:.1f} kWh</small></div></div></div>'
        f'<div class="pc-pin">{pin}</div>'
        "</div></div></div>"
    )


def metrics_html(row: pd.Series, focus: pd.Timestamp, rupees: float, peak: str) -> str:
    start = row["window_start"] if pd.notna(row["window_start"]) else None
    end = row["window_end"] if pd.notna(row["window_end"]) else None
    rise = row.get("sunrise", pd.NaT)
    sset = row.get("sunset", pd.NaT)
    peak_from, _, peak_to = peak.partition("–")
    items = [
        ("Expected", f'{row["yield_kwh"]:.1f}', "kWh"),
        ("Usual day", f'{row["typical_kwh"]:.1f}' if pd.notna(row["typical_kwh"]) else "—", f"kWh in {focus:%b}"),
        ("Best hours", clock(start) if start is not None else "—", f"to {clock(end)}" if end is not None else ""),
        ("Peak tariff", peak_from, f"to {peak_to}" if peak_to else ""),
        ("Sunrise", rise.strftime("%I:%M").lstrip("0") if pd.notna(rise) else "—", "AM"),
        ("Sunset", sset.strftime("%I:%M").lstrip("0") if pd.notna(sset) else "—", "PM"),
        ("High / low", f'{row["t_max"]:.0f}° / {row["t_min"]:.0f}°', "air temp"),
        ("Worth", f"Rs {rupees:,.0f}", "this day"),
    ]
    cells = "".join(
        f'<div class="pc-m" style="--c:{ACCENTS[i % len(ACCENTS)]}"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="u">{u}</div><i></i></div>'
        for i, (k, v, u) in enumerate(items)
    )
    return (
        f'<div class="pc pc-card"><div class="pc-title">{day_name(focus)} · {focus:%A %d %B} at a glance</div>'
        f'<div class="pc-grid">{cells}</div></div>'
    )


def recommendation_html(
    row: pd.Series,
    focus: pd.Timestamp,
    per_appliance: list[tuple[str, AppliancePlan]],
    plan: AppliancePlan,
    chosen: list[str],
) -> str:
    start = row["window_start"] if pd.notna(row["window_start"]) else None
    end = row["window_end"] if pd.notna(row["window_end"]) else None
    best = f" Best hours for heavy appliances: {clock(start)} – {clock(end)}." if start is not None else ""
    tiles = []
    for name, single in per_appliance:
        if single.fits and single.start is not None:
            lvl, status = "high", f"Run {clock(single.start)}–{clock(single.end)}"
        elif single.start is not None:
            lvl, status = "low", "Only partly solar"
        else:
            lvl, status = "very_low", "Wait for a sunny day"
        tiles.append(
            f'<div class="pc-tile" style="--st:var(--pc-st-{lvl});--bg:var(--pc-bg-{lvl})">'
            f'<div class="ic">{appliance_svg(name)}</div><div class="n">{escape(name)}</div>'
            f'<div class="s">{status}</div></div>'
        )
    tiles_block = f'<div class="pc-tiles">{"".join(tiles)}</div>' if tiles else ""
    return (
        f'<div class="pc pc-card" style="--st:var(--pc-st-{row["level"]})">'
        '<div class="pc-title">Your energy recommendation</div>'
        f'<div class="pc-advice"><b>{escape(row["headline"])}.</b> {escape(row["detail"])}{best}</div>'
        f"{tiles_block}{plan_html(chosen, plan, day_name(focus))}</div>"
    )


def gauge_html(row: pd.Series, system_kw: float) -> str:
    """Half-circle gauge of the day's kWh against a very sunny day for this system."""
    usual = row["typical_kwh"] if pd.notna(row["typical_kwh"]) else 0.0
    top = max(system_kw * 7.0, row["yield_kwh"], usual * 1.3, 1.0)
    frac = max(0.0, min(row["yield_kwh"] / top, 1.0))
    cx, cy, r = 150, 150, 118

    def point(f: float) -> tuple[float, float]:
        a = math.pi * (1 - f)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    colors = ["#E5484D", "#F59E0B", "#FACC15", "#22C55E", "#14B8A6"]
    segs = []
    for i, color in enumerate(colors):
        a, b = point(i / 5 + 0.012), point((i + 1) / 5 - 0.012)
        segs.append(
            f'<path d="M{a[0]:.1f} {a[1]:.1f} A{r} {r} 0 0 1 {b[0]:.1f} {b[1]:.1f}" stroke="{color}" '
            'stroke-width="16" fill="none" stroke-linecap="round"/>'
        )
    mx, my = point(frac)
    tick = ""
    if usual > 0:
        tx, ty = point(min(usual / top, 1.0))
        tick = f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.5" style="fill:var(--pc-ink)" opacity=".4"/>'
    level = row["level"]
    svg = (
        '<svg viewBox="0 0 300 178">' + "".join(segs) + tick
        + f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" style="fill:var(--pc-card);stroke:var(--pc-ink)" stroke-width="3"/>'
        '<text x="150" y="112" text-anchor="middle" font-size="13" style="fill:var(--pc-muted)">Expected</text>'
        f'<text x="150" y="152" text-anchor="middle" font-size="42" font-weight="700" style="fill:var(--pc-ink)">{row["yield_kwh"]:.1f}</text>'
        '<text x="32" y="174" text-anchor="middle" font-size="11" style="fill:var(--pc-muted)">0</text>'
        f'<text x="268" y="174" text-anchor="middle" font-size="11" style="fill:var(--pc-muted)">{top:.0f}</text>'
        "</svg>"
    )
    return (
        f'<div class="pc pc-card pc-gauge" style="--st:var(--pc-st-{level});--bg:var(--pc-bg-{level})">'
        f'<div class="pc-label">Solar output</div>{svg}<div class="lbl">kWh expected · dot marks a usual day</div>'
        f'<div class="st">{STATUS[level][1]}</div></div>'
    )


def _cloud_art(kind: str) -> str:
    sun = '<circle cx="80" cy="38" r="34" fill="#FFE38A" opacity=".35"/><circle cx="80" cy="38" r="26" fill="#FFD25A"/>'
    cloud = (
        '<ellipse cx="66" cy="96" rx="46" ry="6" fill="#4A63C9" opacity=".22"/>'
        '<g fill="#FFFFFF"><ellipse cx="66" cy="80" rx="52" ry="20"/><circle cx="46" cy="66" r="20"/>'
        '<circle cx="74" cy="56" r="26"/><circle cx="100" cy="70" r="16"/></g>'
    )
    rain = "".join(
        f'<line x1="{x}" y1="98" x2="{x - 4}" y2="110" stroke="#E0ECFF" stroke-width="3" stroke-linecap="round"/>'
        for x in (44, 64, 84)
    )
    body = {"sun": sun, "partly": sun + cloud, "rain": cloud + rain, "storm": cloud + rain}.get(kind, cloud)
    return f'<svg class="art" viewBox="0 0 132 112" aria-hidden="true">{body}</svg>'


def weather_card_html(row: pd.Series) -> str:
    kind = kind_for(row["label"])
    rain = f" · Rain {row['rain_chance']:.0f}%" if pd.notna(row["rain_chance"]) else ""
    good = {"high": "Great for solar", "normal": "Good for solar", "low": "Weak sun", "very_low": "Poor for solar"}
    return (
        '<div class="pc pc-wx">'
        f"{_cloud_art(kind)}"
        f'<div class="t">{row["t_max"]:.0f}<sup>°C</sup></div>'
        f'<div class="c">{escape(row["label"])}</div>'
        f'<div class="d">H {row["t_max"]:.0f}° · L {row["t_min"]:.0f}°{rain}</div>'
        f'<div class="pill">{good[row["level"]]}</div></div>'
    )


def savings_html(value: dict[str, float], price: float, setup_label: str, total_kwh: float) -> str:
    split = f'{value["self_kwh"]:,.0f} kWh used at home'
    if value.get("export_kwh", 0) > 0.5:
        split += f' · {value["export_kwh"]:,.0f} kWh exported'
    elif value.get("unused_kwh", 0) > 0.5:
        split += f' · {value["unused_kwh"]:,.0f} kWh unused'
    house = (
        '<svg viewBox="0 0 120 100" aria-hidden="true">'
        '<circle cx="98" cy="16" r="12" fill="#FFC93C"/>'
        '<path d="M14 52 L60 18 L106 52 Z" fill="#3B4A8C"/>'
        '<g stroke="#8FA4E8" stroke-width="1"><path d="M37 36 L83 36 M27 44 L93 44 M48 27 L44 50 M60 19 L60 50 M72 27 L76 50"/></g>'
        '<rect x="22" y="50" width="76" height="40" rx="4" fill="#FFFFFF"/>'
        '<rect x="34" y="60" width="16" height="14" rx="2" fill="#FDE2C8"/>'
        '<rect x="70" y="60" width="16" height="14" rx="2" fill="#FDE2C8"/>'
        '<rect x="53" y="66" width="14" height="24" rx="2" fill="#F472B6"/>'
        '<rect x="8" y="88" width="104" height="5" rx="2.5" fill="#F9C9DD"/></svg>'
    )
    return (
        '<div class="pc pc-save"><div>'
        '<div class="k">Value this week</div>'
        f'<div class="v">Rs {value["rupees"]:,.0f}</div>'
        f'<div class="s">{total_kwh:,.0f} kWh from your panels · {split}</div>'
        f'<div class="s" style="margin-top:6px;opacity:.8">{escape(setup_label)} · home use at Rs {price:g}/kWh</div>'
        f"</div>{house}</div>"
    )


def hourly_html(
    hours: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    now: pd.Timestamp | None = None,
    peak: str | None = None,
) -> str:
    visible = hours[hours["energy_kwh"] >= 0.05]
    if visible.empty:
        return '<div class="pc pc-card"><div class="pc-label">Hourly solar output</div>No meaningful solar output expected.</div>'
    top = max(visible["energy_kwh"].max(), 1e-9)
    cells = []
    for stamp, row in visible.iterrows():
        hour_start = stamp - pd.Timedelta(hours=1)
        classes = ["pc-hour"]
        label = clock(hour_start)
        if start is not None and start <= hour_start and stamp <= end:
            classes.append("best")
        if now is not None:
            if hour_start <= now < stamp:
                classes.append("now")
                label = "Now"
            elif stamp <= now:
                classes.append("past")
        height = max(row["energy_kwh"] / top * 40, 3)
        cover = row["cloud_cover"] if "cloud_cover" in row else float("nan")
        cells.append(
            f'<div class="{" ".join(classes)}"><div class="t">{label}</div>'
            f"<div>{icon_svg(kind_for_cloud_cover(cover), 24)}</div>"
            f'<div class="v">{row["energy_kwh"]:.1f}</div>'
            f'<div class="bar"><i style="height:{height:.0f}px"></i></div></div>'
        )
    legend = ""
    if start is not None:
        legend += f"<span><i></i>Best 3 hours for heavy appliances ({clock(start)} – {clock(end)})</span>"
    if peak:
        legend += f'<span><i class="pk"></i>Peak tariff {peak}: avoid heavy loads</span>'
    return (
        '<div class="pc pc-card"><div class="pc-label">Hourly solar output · kWh</div>'
        f'<div class="pc-hours">{"".join(cells)}</div><div class="pc-legend">{legend}</div></div>'
    )


def week_html(outlook: Outlook, focus: pd.Timestamp, href: Callable[[pd.Timestamp], str] | None = None) -> str:
    days = outlook.days
    usual_max = days["typical_kwh"].max(skipna=True) if days["typical_kwh"].notna().any() else 0
    scale = max(days["yield_kwh"].max(), usual_max, 1e-9) * 1.05
    rows = []
    for day, row in days.iterrows():
        _, status = STATUS[row["level"]]
        tick = ""
        if pd.notna(row["typical_kwh"]):
            tick = f'<b style="left:{row["typical_kwh"] / scale * 100:.1f}%" title="Usual: {row["typical_kwh"]:.1f} kWh"></b>'
        tag, attrs = ("a", f' href="{escape(href(day))}" target="_self"') if href else ("div", "")
        rows.append(
            f'<{tag} class="pc-row{" sel" if day == focus else ""}"{attrs} '
            f'style="--st:var(--pc-st-{row["level"]});--bg:var(--pc-bg-{row["level"]})">'
            f'<div class="d">{day_name(day)}</div>'
            f'<div title="{escape(row["label"])}">{icon_svg(kind_for(row["label"]), 28)}</div>'
            f'<div class="pc-track"><i style="width:{row["yield_kwh"] / scale * 100:.1f}%"></i>{tick}</div>'
            f'<div class="k">{row["yield_kwh"]:.1f} <small>kWh</small></div>'
            f'<div class="pc-status">{status}</div>'
            f"</{tag}>"
        )
    return (
        '<div class="pc pc-card"><div class="pc-label">7-day solar outlook</div>'
        f'{"".join(rows)}'
        '<div class="pc-legend"><span>Grey tick = a usual day for that month. Tap a day to open it.</span></div></div>'
    )


def plan_html(selected: list[str], plan: AppliancePlan, day_label: str) -> str:
    if not selected:
        return ""
    if plan.start is None:
        lvl, title = "very_low", "Not enough sun for these together"
        body = "Expected solar barely covers your basic load. Run them on the next sunny day or from the grid."
    elif plan.fits:
        lvl, title = "high", f"Solar can run all of this {day_label.lower()}"
        body = (
            f"Together they need {plan.need_kwh:.1f} kWh. Run them between <b>{clock(plan.start)} and "
            f"{clock(plan.end)}</b> and spare solar covers all of it."
        )
    else:
        lvl, title = "low", f"Solar covers about {plan.coverage:.0%} of it"
        body = (
            f"Across <b>{clock(plan.start)} – {clock(plan.end)}</b> spare solar gives {plan.solar_kwh:.1f} of the "
            f"{plan.need_kwh:.1f} kWh needed. Pick fewer appliances or split them over two days."
        )
    return (
        f'<div class="pc-plan" style="--st:var(--pc-st-{lvl});--bg:var(--pc-bg-{lvl})"><h4>{title}</h4>'
        f'<div class="pc-meter"><i style="width:{plan.coverage * 100:.0f}%"></i></div><p>{body}</p></div>'
    )


__all__ = [
    "APPLIANCES",
    "CSS",
    "day_name",
    "gauge_html",
    "header_html",
    "hero_html",
    "hourly_html",
    "metrics_html",
    "plan_html",
    "recommendation_html",
    "savings_html",
    "weather_card_html",
    "week_html",
]
