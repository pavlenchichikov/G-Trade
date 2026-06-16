"""Веб-интерфейс: радар сигналов, track record, модели, риск.

Читает готовые предсказания из market.db (их пишет predict.py),
модели не загружает - стартует мгновенно.

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
MODEL_DIR = os.path.join(BASE_DIR, "models")
RISK_STATE_PATH = os.path.join(MODEL_DIR, "risk_state.json")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
QUALITY_PATH = os.path.join(MODEL_DIR, "quality_report.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

app = FastAPI(title="G-Trade")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default
    return default


def _risk_state():
    return _load_json(RISK_STATE_PATH, None)


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

    reg = _load_json(REGISTRY_PATH, {}).get(name)
    thr = _load_json(THRESHOLDS_PATH, {}).get(name)
    if thr is None and reg:
        thr = {"buy": reg.get("buy_thr"), "sell": reg.get("sell_thr")}
    quality = next((q for q in _load_json(QUALITY_PATH, [])
                    if q.get("Asset") == name), None)
    group = next((g for g, m in RADAR_GROUPS.items() if name in m), None)

    markers = [{"date": t["date"], "signal": t["signal"]}
               for t in track if t["signal"] in ("BUY", "SELL")]
    return templates.TemplateResponse(request, "asset.html", {
        "asset": name,
        "ticker": FULL_ASSET_MAP[name],
        "group": group,
        "track": track,
        "acc": acc,
        "stats": stats,
        "current": track[0] if track else None,
        "reg": reg,
        "thr": thr,
        "quality": quality,
        "markers_json": json.dumps(markers),
    })


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    quality = _load_json(QUALITY_PATH, [])
    registry = _load_json(REGISTRY_PATH, {})
    sigs = {s["asset"]: s for s in track_record.latest_signals()}

    rows = []
    for q in quality:
        asset = q.get("Asset")
        reg = registry.get(asset, {})
        sig = sigs.get(asset)
        rows.append({
            "asset": asset,
            "score": q.get("Score"),
            "cb_acc": q.get("CB_Acc"),
            "lstm_acc": q.get("LSTM_Acc"),
            "profit": q.get("Profit"),
            "trades": q.get("Trades"),
            "status": q.get("Status"),
            "policy": q.get("Policy"),
            "mode": reg.get("ensemble_mode", "-"),
            "lookback": reg.get("lookback"),
            "updated": str(reg.get("updated_at", ""))[:10],
            "signal": sig["signal"] if sig else None,
            "live_acc": sig["acc"] if sig else None,
        })
    rows.sort(key=lambda r: r["score"] or 0, reverse=True)

    n = len(rows)
    stable = sum(1 for r in rows if r["status"] == "STABLE")
    last_train = max((r["updated"] for r in rows if r["updated"]), default=None)
    summary = {
        "total": n,
        "stable": stable,
        "unstable": n - stable,
        "avg_score": (sum(r["score"] or 0 for r in rows) / n) if n else None,
        "last_train": last_train,
    }
    return templates.TemplateResponse(request, "models.html", {
        "rows": rows, "summary": summary,
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


@app.get("/api/prices/{name}")
def api_prices(name: str, days: int = 90):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    days = 100000 if days <= 0 else max(10, min(days, 100000))
    return {"asset": name, "series": track_record.price_series(name, days=days)}


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
