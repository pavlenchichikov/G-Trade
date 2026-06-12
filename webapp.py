"""Веб-интерфейс: радар сигналов, track record, риск.

Читает готовые предсказания из market.db (их пишет predict.py),
модели не загружает — стартует мгновенно.

Запуск:
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import json
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FULL_ASSET_MAP, RADAR_GROUPS
from core import track_record
from risk_manager import RISK_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_STATE_PATH = os.path.join(BASE_DIR, "models", "risk_state.json")

app = FastAPI(title="G-Trade")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _risk_state():
    if os.path.exists(RISK_STATE_PATH):
        try:
            with open(RISK_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _spark(closes, w=110, h=26):
    """Точки для svg-спарклайна по списку closes."""
    if len(closes) < 2:
        return None
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1.0
    step = w / (len(closes) - 1)
    pts = " ".join(
        f"{i * step:.1f},{h - 2 - (c - lo) / span * (h - 4):.1f}"
        for i, c in enumerate(closes)
    )
    chg = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0.0
    return {"points": pts, "w": w, "h": h, "up": closes[-1] >= closes[0], "chg": chg}


def _chart(asset, track, days=90, w=720, h=220):
    """Линия цены + маркеры BUY/SELL для страницы актива."""
    series = track_record.price_series(asset, days=days)
    closes = [p["close"] for p in series]
    if len(closes) < 2:
        return None
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1.0
    pad = 12
    step = (w - pad * 2) / (len(closes) - 1)

    def x(i):
        return pad + i * step

    def y(c):
        return h - pad - (c - lo) / span * (h - pad * 2)

    pts = " ".join(f"{x(i):.1f},{y(c):.1f}" for i, c in enumerate(closes))
    idx = {p["date"]: i for i, p in enumerate(series)}
    markers = []
    for t in track:
        if t["signal"] in ("BUY", "SELL") and t["date"] in idx:
            i = idx[t["date"]]
            markers.append({
                "x": round(x(i), 1), "y": round(y(closes[i]), 1),
                "signal": t["signal"], "date": t["date"],
            })
    area = f"{pad},{h - pad} {pts} {x(len(closes) - 1):.1f},{h - pad}"
    return {
        "points": pts, "area": area, "markers": markers, "w": w, "h": h,
        "lo": lo, "hi": hi, "last": closes[-1],
        "first_date": series[0]["date"], "last_date": series[-1]["date"],
        "up": closes[-1] >= closes[0],
        "chg": (closes[-1] - closes[0]) / closes[0] if closes[0] else 0.0,
    }


def _summary(signals, stale):
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    verified = correct = 0
    last_date = None
    for s in signals:
        counts[s["signal"]] = counts.get(s["signal"], 0) + 1
        verified += s["acc"]["n"]
        correct += s["acc"]["correct"]
        if last_date is None or s["date"] > last_date:
            last_date = s["date"]
    return {
        "total": len(signals),
        "counts": counts,
        "accuracy": (correct / verified) if verified else None,
        "verified": verified,
        "last_date": last_date,
        "stale": len(stale),
    }


def _top_signals(signals, n=5):
    actionable = [s for s in signals if s["signal"] in ("BUY", "SELL")]
    return sorted(actionable, key=lambda s: abs((s["probability"] or 0.5) - 0.5),
                  reverse=True)[:n]


def _grouped_signals(signals):
    sigs = {s["asset"]: s for s in signals}
    groups = []
    for group, members in RADAR_GROUPS.items():
        rows = [sigs[a] for a in members if a in sigs]
        if rows:
            groups.append({"name": group, "rows": rows})
    return groups


@app.get("/", response_class=HTMLResponse)
def radar(request: Request):
    signals = track_record.latest_signals()
    for s in signals:
        closes = [p["close"] for p in track_record.price_series(s["asset"], days=30)]
        s["spark"] = _spark(closes)
    stale = track_record.stale_assets()
    return templates.TemplateResponse(request, "radar.html", {
        "groups": _grouped_signals(signals),
        "summary": _summary(signals, stale),
        "top": _top_signals(signals),
        "stale": stale[:8],
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


@app.get("/asset/{name}", response_class=HTMLResponse)
def asset_page(request: Request, name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    track = track_record.asset_track(name, limit=60)
    acc = track_record.asset_accuracy(name)

    rets = [t["actual_next_ret"] for t in track if t["actual_next_ret"] is not None]
    wins = [t for t in track if t["correct"] == 1]
    losses = [t for t in track if t["correct"] == 0]
    stats = {
        "avg_ret": sum(rets) / len(rets) if rets else None,
        "wins": len(wins),
        "losses": len(losses),
        "outcomes": [t["correct"] for t in track[:15]][::-1],
    }
    group = next((g for g, m in RADAR_GROUPS.items() if name in m), None)
    return templates.TemplateResponse(request, "asset.html", {
        "asset": name,
        "ticker": FULL_ASSET_MAP[name],
        "group": group,
        "track": track,
        "acc": acc,
        "stats": stats,
        "chart": _chart(name, track),
        "current": track[0] if track else None,
    })


@app.get("/risk", response_class=HTMLResponse)
def risk_page(request: Request):
    state = _risk_state()
    dd = None
    if state:
        cap = state.get("current_capital", 0.0)
        peak = state.get("peak_capital", cap) or cap
        dd = (peak - cap) / peak if peak else 0.0
    return templates.TemplateResponse(request, "risk.html", {
        "state": state, "config": RISK_CONFIG, "dd": dd,
    })


@app.get("/api/signals")
def api_signals():
    return track_record.latest_signals()


@app.get("/api/track/{name}")
def api_track(name: str):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    return {
        "asset": name,
        "track": track_record.asset_track(name, limit=60),
        "accuracy": track_record.asset_accuracy(name),
    }


@app.get("/api/risk")
def api_risk():
    return {"state": _risk_state(), "config": RISK_CONFIG}
