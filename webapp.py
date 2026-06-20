"""Web interface: signal radar, track record, models, risk.

Reads ready-made predictions from market.db (written by predict.py); it does not
load any models, so it starts instantly.

Run:
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import json
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FULL_ASSET_MAP, RADAR_GROUPS, radar_category
from core import track_record
from core import dashboard
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
    """Points for an svg sparkline from a list of closes."""
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
        for r in rows:
            r["cat"] = radar_category(r["asset"])
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
    regime = dashboard.global_regime()
    score = dashboard.regime_score(regime)
    sentiment = dashboard.market_sentiment()
    return templates.TemplateResponse(request, "radar.html", {
        "groups": _grouped_signals(signals),
        "summary": _summary(signals, stale),
        "top": _top_signals(signals),
        "stale": stale[:8],
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "score": score,
        "zone": dashboard.gauge_zone(score),
        "sentiment": sentiment,
        "sent_zone": dashboard.gauge_zone(sentiment["score"]),
        "breadth": dashboard.market_breadth(),
        "leaderboard": dashboard.top_leaderboard(limit=5),
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
        "cat": radar_category(name),
        "track": track,
        "acc": acc,
        "stats": stats,
        "current": track[0] if track else None,
        "reg": reg,
        "thr": thr,
        "quality": quality,
        "markers_json": json.dumps(markers),
        "guru": dashboard.guru_for_asset(name),
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
        "health": dashboard.models_health(),
        "stale": dashboard.models_stale(),
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


@app.get("/market", response_class=HTMLResponse)
def market_page(request: Request):
    regime = dashboard.global_regime()
    score = dashboard.regime_score(regime)
    sentiment = dashboard.market_sentiment()
    return templates.TemplateResponse(request, "market.html", {
        "regime": regime,
        "score": score,
        "zone": dashboard.gauge_zone(score),
        "sentiment": sentiment,
        "sent_zone": dashboard.gauge_zone(sentiment["score"]),
    })


@app.get("/news", response_class=HTMLResponse)
def news_page(request: Request, lang: str = "all", category: str = "all"):
    items = dashboard.news_digest(lang=lang, category=category)
    return templates.TemplateResponse(request, "news.html", {
        "items": items, "lang": lang, "category": category,
    })


@app.get("/sectors", response_class=HTMLResponse)
def sectors_page(request: Request):
    return templates.TemplateResponse(request, "sectors.html", {
        "momentum": dashboard.sector_momentum(),
        "heatmap": dashboard.sector_heatmap(),
    })


@app.get("/correlations", response_class=HTMLResponse)
def correlations_page(request: Request):
    return templates.TemplateResponse(request, "correlations.html", {
        "stress": dashboard.correlation_stress(),
        "heatmap": dashboard.correlation_heatmap(),
    })


@app.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    return templates.TemplateResponse(request, "performance.html", {
        "series": dashboard.accuracy_timeseries(),
        "leaderboard": dashboard.top_leaderboard(limit=20),
        "version": dashboard.current_model_version(),
    })


@app.get("/guru", response_class=HTMLResponse)
def guru_page(request: Request):
    return templates.TemplateResponse(request, "guru.html", {
        "verdicts": dashboard.guru_latest(),
        "accuracy": dashboard.guru_accuracy(),
    })


@app.get("/api/regime")
def api_regime():
    regime = dashboard.global_regime()
    return {**regime, "score": dashboard.regime_score(regime)}


@app.get("/api/sectors")
def api_sectors():
    return {"momentum": dashboard.sector_momentum(),
            "heatmap": dashboard.sector_heatmap()}


@app.get("/api/correlations")
def api_correlations():
    return {"stress": dashboard.correlation_stress(),
            "heatmap": dashboard.correlation_heatmap()}


@app.get("/api/performance")
def api_performance():
    return {"series": dashboard.accuracy_timeseries(),
            "leaderboard": dashboard.top_leaderboard(limit=20)}


@app.get("/api/guru")
def api_guru():
    return {"verdicts": dashboard.guru_latest(),
            "accuracy": dashboard.guru_accuracy()}


@app.post("/api/guru/{asset}/recalculate")
def api_guru_recalculate(asset: str):
    """Live re-score of one asset, persisted as the new latest guru_log verdict.

    Reuses guru_report.py's fundamentals resolution and core.guru's scoring
    engine - the same logic the console report and app.py use - so this never
    drifts into a second implementation of guru scoring.
    """
    asset = asset.upper()
    if asset not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {asset}")

    import guru_report
    import guru_tracker
    from core.guru import get_guru_analysis

    symbol = FULL_ASSET_MAP[asset]
    smartlab = guru_report.fetch_smartlab_data()
    fund = guru_report.resolve_fundamentals(asset, symbol, smartlab)
    tech = guru_report.technical_context(guru_report.get_technical(asset))
    analysis = get_guru_analysis(fund, tech)

    price = (fund or {}).get('price') or (tech['close'] if tech else 0)
    data_source = analysis['data_source']
    council = analysis['council']
    guru_tracker.log_guru_verdict(
        asset,
        analysis['lynch']['_score'], analysis['buffett']['_score'],
        analysis['graham']['_score'], analysis['munger']['_score'],
        council['pct'], council['verdict'], data_source, price,
    )
    return {
        "asset": asset, "verdict": council['verdict'], "pct": council['pct'],
        "source": data_source, "date": datetime.now().strftime("%Y-%m-%d"),
        "lynch": analysis['lynch'], "buffett": analysis['buffett'],
        "graham": analysis['graham'], "munger": analysis['munger'],
    }


@app.get("/api/news")
def api_news(lang: str = "all", category: str = "all"):
    return dashboard.news_digest(lang=lang, category=category)


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


@app.get("/api/ohlc/{name}")
def api_ohlc(name: str, days: int = 120):
    name = name.upper()
    if name not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {name}")
    days = 100000 if days <= 0 else max(10, min(days, 100000))
    return {"asset": name, "series": track_record.ohlc_series(name, days=days)}


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
