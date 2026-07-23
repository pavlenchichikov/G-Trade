"""Web interface: signal radar, track record, models, risk.

Reads ready-made predictions from market.db (written by predict.py); it does not
load any models, so it starts instantly.

Run:
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import json
import os
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import FULL_ASSET_MAP, RADAR_GROUPS, radar_category
from core import track_record
from core import dashboard
from core import positions as positions_mod
from core import timing_policy
from risk_manager import RISK_CONFIG, RiskManager, save_risk_config_override

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
QUALITY_PATH = os.path.join(MODEL_DIR, "quality_report.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

app = FastAPI(title="Atratus")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

LOOP_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loop_state.json")


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default
    return default


def _risk_snapshot():
    rm = RiskManager()
    halted, halt_reason = rm.is_trading_halted()
    return {
        "state": {
            "current_capital": rm.current_capital,
            "peak_capital": rm.peak_capital,
            "initial_capital": rm.initial_capital,
            "open_positions": rm.open_positions,
        },
        "dd": rm.current_drawdown,
        "halted": halted,
        "halt_reason": halt_reason,
        "manual_halt": rm.manual_halt,
    }


def _latest_price(asset):
    series = track_record.price_series(asset, days=5)
    return series[-1]["close"] if series else None


def _portfolio_snapshot():
    """Portfolio view over the risk-manager positions (the same book as /risk):
    holdings + diversification / sector heat / held-asset correlation / per-
    position warnings (portfolio.py analytics on the current positions)."""
    rm = RiskManager()
    cap = rm.current_capital or 1.0
    pm = dashboard.portfolio_manager()
    positions = rm.open_positions
    # weights as a fraction of capital
    fractions = {a: (p.get("size_usd", 0) / cap) for a, p in positions.items()}
    held = list(positions.keys())

    holdings = []
    for a, p in positions.items():
        price = _latest_price(a)
        size = p.get("size_usd", 0)
        entry = p.get("entry_price")
        direction = p.get("direction", "BUY")
        pnl = None
        if price and entry:
            ret = (price - entry) / entry if direction == "BUY" else (entry - price) / entry
            pnl = ret * size
        corr_open = []
        if pm is not None:
            corr_open = [x for x in pm.get_correlated_assets(a, open_only=held) if x != a]
        holdings.append({
            "asset": a, "direction": direction, "size_usd": size,
            "weight": fractions[a], "sector": pm.get_sector(a) if pm else "OTHER",
            "entry": entry, "price": price, "pnl": pnl,
            "correlated_open": corr_open,
        })
    holdings.sort(key=lambda h: h["weight"], reverse=True)

    heat = []
    diversification = 100.0
    corr_rows = []
    if pm is not None and fractions:
        from portfolio import SECTOR_LIMITS
        diversification = pm.get_diversification_score(fractions)
        for sector, exp in pm.get_portfolio_heat(fractions).items():
            if exp <= 0:
                continue
            limit = SECTOR_LIMITS.get(sector, 0.20)
            heat.append({"sector": sector, "exposure": exp, "limit": limit,
                         "over": exp > limit})
        heat.sort(key=lambda h: h["exposure"], reverse=True)
        corr = pm.get_correlation_matrix()
        if held and not corr.empty:
            for a in held:
                vals = []
                for b in held:
                    c = None
                    if a in corr.index and b in corr.columns:
                        try:
                            c = float(corr.loc[a, b])
                        except Exception:
                            c = None
                    vals.append(c)
                corr_rows.append({"asset": a, "vals": vals})

    return {
        "holdings": holdings,
        "held": held,
        "total_exposure": sum(fractions.values()),
        "diversification": diversification,
        "heat": heat,
        "corr_rows": corr_rows,
        "capital": rm.current_capital,
    }


def _research_snapshot():
    """Findings-journal snapshot for the /research page: cumulative counters plus a
    flattened newest-first list of recent winners. Read-only."""
    from core import ar_memory
    summary = ar_memory.findings_summary()
    recent = ar_memory.findings_recent(15)
    rows = []
    for rec in recent:
        for w in rec.get("winners", []):
            rows.append({
                "ts": rec.get("ts", ""), "mode": rec.get("mode", ""),
                "axis": w.get("axis", ""), "adoptable": bool(w.get("adoptable")),
                "replicated": bool(w.get("replicated")),
                "clears": w.get("clears") or 0, "neural_lift": w.get("neural_lift"),
                "tag": w.get("tag", ""),
            })
    # Two independent caps: findings_recent(15) bounds the RECORDS read; rows[:40]
    # bounds the flattened winner-ROWS shown (one record can hold many winners).
    return {"summary": summary, "rows": rows[:40], "runs": len(recent)}


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


def _timing_badge(row, show_timing):
    """Divergence-only badge string for a radar row, or None. `show_timing` is
    the reversibility guard (spec section 4.3), precomputed once per request."""
    if not show_timing:
        return None
    act = row.get("timing_action")
    if not act:
        return None
    text, is_div = timing_policy.display_label(act, row.get("timing_reason"))
    return text if is_div else None


def _grouped_signals(signals):
    show_timing = timing_policy.timing_on() and timing_policy.load_policy() is not None
    sigs = {s["asset"]: s for s in signals}
    groups = []
    for group, members in RADAR_GROUPS.items():
        rows = [sigs[a] for a in members if a in sigs]
        for r in rows:
            r["cat"] = radar_category(r["asset"])
            r["timing_badge"] = _timing_badge(r, show_timing)
        if rows:
            groups.append({"name": group, "rows": rows})
    return groups


@app.get("/", response_class=HTMLResponse)
def radar(request: Request):
    signals = track_record.latest_signals()
    taleb = dashboard.taleb_index()
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]
    for s in signals:
        closes = [p["close"] for p in track_record.price_series(s["asset"], days=30)]
        s["spark"] = _spark(closes)
        s["taleb"] = taleb.get(s["asset"])
        s["taleb_regime"] = dashboard.taleb_regime(s["taleb"], soft_cap, hard_cap)
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
        "config": RISK_CONFIG,
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

    # Collapse the per-bar signals into positions: enter/exit markers for the
    # chart, a state ribbon, a trade log and the current-position card.
    pos = positions_mod.build_positions(
        [{"date": t["date"], "signal": t["signal"], "ret": t["actual_next_ret"]}
         for t in reversed(track)])
    markers = pos["markers"]
    taleb = dashboard.taleb_for_asset(name)
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]

    # asset_track doesn't carry the live-gate columns; pull the gated display
    # value + reason from latest_gated() so the chip matches the radar page.
    current = dict(track[0]) if track else None
    if current:
        gated = track_record.latest_gated(name)
        if gated:
            current["signal"] = gated["signal"]
            current["signal_raw"] = gated["signal_raw"]
            current["gate_reason"] = gated["gate_reason"]
        else:
            current["signal_raw"] = current.get("signal")
            current["gate_reason"] = None

        current["timing_label"] = None
        if timing_policy.timing_on() and timing_policy.load_policy() is not None:
            act = current.get("timing_action") or (gated or {}).get("timing_action")
            if act:
                text, _div = timing_policy.display_label(
                    act, current.get("timing_reason")
                    or (gated or {}).get("timing_reason"))
                current["timing_label"] = text

    return templates.TemplateResponse(request, "asset.html", {
        "asset": name,
        "ticker": FULL_ASSET_MAP[name],
        "taleb": taleb,
        "taleb_regime": dashboard.taleb_regime(taleb, soft_cap, hard_cap),
        "taleb_soft_cap": soft_cap,
        "taleb_hard_cap": hard_cap,
        "group": group,
        "cat": radar_category(name),
        "track": track,
        "acc": acc,
        "stats": stats,
        "current": current,
        "position": pos["current"],
        "trades": pos["trades"],
        "segments": pos["segments"],
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


def _taleb_top(limit=10):
    """Assets with the highest current Taleb tail-risk index, regime-tagged."""
    soft_cap, hard_cap = RISK_CONFIG["taleb_soft_cap"], RISK_CONFIG["taleb_risk_cap"]
    items = [(a, v) for a, v in dashboard.taleb_index().items() if v is not None]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [{"asset": a, "taleb": v,
             "regime": dashboard.taleb_regime(v, soft_cap, hard_cap)}
            for a, v in items[:limit]]


@app.get("/risk", response_class=HTMLResponse)
def risk_page(request: Request):
    return templates.TemplateResponse(request, "risk.html", {
        **_risk_snapshot(), "config": RISK_CONFIG,
        "full_asset_map": sorted(FULL_ASSET_MAP),
        "taleb_top": _taleb_top(),
    })


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(request: Request):
    return templates.TemplateResponse(request, "portfolio.html", {
        **_portfolio_snapshot(),
        "full_asset_map": sorted(FULL_ASSET_MAP),
    })


@app.get("/api/portfolio")
def api_portfolio():
    return _portfolio_snapshot()


@app.get("/api/ticker")
def api_ticker():
    return {"movers": dashboard.top_movers()}


@app.get("/api/health")
def api_health():
    return dashboard.health()


@app.get("/api/palette")
def api_palette():
    pages = [
        ["Radar", "/"], ["Market", "/market"], ["Sectors", "/sectors"],
        ["Correlations", "/correlations"], ["Accuracy", "/performance"],
        ["News", "/news"], ["Guru", "/guru"], ["Models", "/models"],
        ["Risk", "/risk"], ["Portfolio", "/portfolio"], ["What-If", "/whatif"],
        ["Research", "/research"],
    ]
    return {"pages": pages, "assets": sorted(FULL_ASSET_MAP)}


@app.get("/loop", response_class=HTMLResponse)
def loop_page(request: Request):
    from core import loop_state
    return templates.TemplateResponse(request, "loop.html", loop_state.load_state(LOOP_STATE_PATH))


@app.get("/api/loop")
def api_loop():
    from core import loop_state
    return loop_state.load_state(LOOP_STATE_PATH)


@app.post("/api/loop/approve")
async def api_loop_approve(request: Request):
    from core import loop_state
    try:
        body = await request.json()
    except Exception:
        body = {}
    assets = [str(a).upper() for a in (body.get("assets") or [])]
    return loop_state.approve(LOOP_STATE_PATH, assets)


@app.post("/api/loop/dismiss")
async def api_loop_dismiss(request: Request):
    from core import loop_state
    try:
        body = await request.json()
    except Exception:
        body = {}
    return loop_state.dismiss(LOOP_STATE_PATH, str(body.get("asset", "")).upper())


@app.get("/research", response_class=HTMLResponse)
def research_page(request: Request):
    from core import ar_wiki
    snap = _research_snapshot()
    snap["wiki"] = ar_wiki.wiki_summary() if ar_wiki.wiki_on() else ""
    return templates.TemplateResponse(request, "research.html", snap)


@app.get("/api/research")
def api_research():
    return _research_snapshot()


@app.get("/whatif", response_class=HTMLResponse)
def whatif_page(request: Request):
    return templates.TemplateResponse(request, "whatif.html", {
        "full_asset_map": sorted(FULL_ASSET_MAP),
    })


@app.post("/api/whatif")
async def api_whatif(request: Request):
    """Run a hypothetical CatBoost-signal backtest. The simulation is CPU-bound,
    so it runs in a threadpool to avoid blocking the event loop. Assets are
    capped and days bounded to keep a single request responsive."""
    from starlette.concurrency import run_in_threadpool

    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        capital = max(1.0, float(body.get("capital") or 10000))
        days_back = max(10, min(365, int(body.get("days_back") or 90)))
    except (TypeError, ValueError):
        return {"error": "capital and days must be numbers"}
    strategy = body.get("strategy") if body.get("strategy") in ("equal", "kelly") else "equal"

    import whatif_simulator as wf
    try:
        if body.get("mode") == "top":
            n = max(1, min(12, int(body.get("top_n") or 5)))
            return await run_in_threadpool(
                wf.simulate_top_n, n=n, capital=capital, days_back=days_back)
        assets = [a for a in (body.get("assets") or []) if a in FULL_ASSET_MAP][:12]
        if not assets:
            return {"error": "No valid assets selected"}
        return await run_in_threadpool(
            wf.simulate, assets, capital=capital, days_back=days_back, strategy=strategy)
    except Exception as exc:
        return {"error": "Simulation failed: " + str(exc)[:140]}


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
    import performance_tracker
    try:
        meta_shadow = performance_tracker.meta_shadow_report(days=30)
    except Exception:
        meta_shadow = {"rows": 0}
    return templates.TemplateResponse(request, "performance.html", {
        "series": dashboard.accuracy_timeseries(),
        "leaderboard": dashboard.top_leaderboard(limit=20),
        "version": dashboard.current_model_version(),
        "meta_shadow": meta_shadow,
    })


@app.get("/guru", response_class=HTMLResponse)
def guru_page(request: Request):
    verdicts = dashboard.guru_latest()
    # Overlay the ML signal so it is always visible next to the value verdict;
    # flag divergence (ML bullish vs guru bearish or vice versa) as advisory.
    ml = {s["asset"]: s for s in track_record.latest_signals()}
    for v in verdicts:
        sig = ml.get(v["asset"])
        v["ml_signal"] = sig["signal"] if sig else None
        v["ml_prob"] = sig.get("probability") if sig else None
        v["divergent"] = bool(sig and (
            (sig["signal"] == "BUY" and v["verdict"] == "AVOID") or
            (sig["signal"] == "SELL" and v["verdict"] == "BUY")))
    return templates.TemplateResponse(request, "guru.html", {
        "verdicts": verdicts,
        "accuracy": dashboard.guru_accuracy(),
    })


@app.get("/api/regime")
def api_regime():
    regime = dashboard.global_regime()
    return {**regime, "score": dashboard.regime_score(regime)}


@app.get("/api/sentiment")
def api_sentiment():
    return dashboard.market_sentiment()


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


@app.post("/api/reconcile")
async def api_reconcile():
    """Fill in actual outcomes for pending predictions (the loop's reconcile
    step, on demand). DB-bound, so it runs in a threadpool like /api/whatif."""
    from starlette.concurrency import run_in_threadpool

    import performance_tracker
    try:
        return await run_in_threadpool(performance_tracker.update_actuals)
    except Exception as exc:
        return {"error": "Reconcile failed: " + str(exc)[:140]}


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

    # Guru is a fundamentals-based value verdict. Without real fundamentals
    # (crypto/forex/indices/commodities, or a stock whose data failed to load)
    # the "verdict" would just be a shaved momentum read mislabeled as guru - so
    # report an honest N/A and do NOT pollute the accuracy track record. The ML
    # signal for this asset is shown separately and is unaffected.
    if data_source in ("technical", "backup"):
        return {
            "asset": asset, "verdict": "N/A", "no_fundamentals": True,
            "source": data_source, "date": datetime.now().strftime("%Y-%m-%d"),
        }

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


# Background state for the "recalculate all guru verdicts" batch. The batch
# scrapes Smart-Lab once and hits yfinance per US/EU stock, so it runs for
# minutes - too long for a blocking request. One batch runs at a time; the UI
# polls /status and reloads when it finishes.
_guru_recalc_lock = threading.Lock()
_guru_recalc = {"running": False, "done": 0, "total": 0, "updated": 0,
                "skipped": 0, "errors": 0, "error": None, "finished": None}


def _run_guru_recalc():
    import guru_report
    try:
        def _prog(done, total, _asset):
            with _guru_recalc_lock:
                _guru_recalc["done"] = done
                _guru_recalc["total"] = total
        res = guru_report.recalc_all_stocks(progress=_prog)
        with _guru_recalc_lock:
            _guru_recalc.update(updated=res["updated"], skipped=res["skipped"],
                                errors=res["errors"])
    except Exception as exc:
        with _guru_recalc_lock:
            _guru_recalc["error"] = str(exc)[:200]
    finally:
        with _guru_recalc_lock:
            _guru_recalc["running"] = False
            _guru_recalc["finished"] = datetime.now().strftime("%H:%M:%S")


@app.post("/api/guru/recalculate-all")
def api_guru_recalculate_all():
    """Kick off a background re-score of every stock (Smart-Lab once + yfinance
    per US/EU name). Returns immediately with the initial status; a second call
    while a batch is running is a no-op that returns the live status."""
    with _guru_recalc_lock:
        if _guru_recalc["running"]:
            return dict(_guru_recalc)
        _guru_recalc.update(running=True, done=0, total=0, updated=0,
                            skipped=0, errors=0, error=None, finished=None)
    threading.Thread(target=_run_guru_recalc, daemon=True).start()
    with _guru_recalc_lock:
        return dict(_guru_recalc)


@app.get("/api/guru/recalculate-all/status")
def api_guru_recalculate_status():
    with _guru_recalc_lock:
        return dict(_guru_recalc)


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
    return {**_risk_snapshot(), "config": RISK_CONFIG}


# The Risk Alerts scan reads every asset's price history + RSI (a few seconds), so
# it is fetched lazily by the /risk panel and memoized briefly instead of blocking
# the page or the 20s poll.
_ALERTS_CACHE = {"ts": 0.0, "alerts": None}
_ALERTS_TTL = 180.0


@app.get("/api/risk/alerts")
async def api_risk_alerts(force: bool = False):
    """The HTML report's Risk Alerts (RSI overbought/oversold, fearful VIX, stale
    models), surfaced for the /risk panel. DB-heavy, so it runs in a threadpool and
    is cached for a few minutes (force=1 bypasses the cache for a manual refresh)."""
    import time

    from starlette.concurrency import run_in_threadpool

    now = time.time()
    if not force and _ALERTS_CACHE["alerts"] is not None and now - _ALERTS_CACHE["ts"] < _ALERTS_TTL:
        return {"alerts": _ALERTS_CACHE["alerts"], "cached": True}
    try:
        import performance_report
        alerts = await run_in_threadpool(performance_report.collect_risk_alerts)
        _ALERTS_CACHE.update(ts=now, alerts=alerts)
        return {"alerts": alerts, "cached": False}
    except Exception as exc:
        return {"error": "Risk alerts failed: " + str(exc)[:140], "alerts": []}


@app.post("/api/risk/position")
async def api_risk_open_position(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    asset = str(body.get("asset", "")).upper()
    direction = body.get("direction")
    size_usd = body.get("size_usd")
    entry_price = body.get("entry_price")

    if asset not in FULL_ASSET_MAP:
        raise HTTPException(404, f"Unknown asset: {asset}")
    if direction not in ("BUY", "SELL"):
        raise HTTPException(400, "direction must be BUY or SELL")
    try:
        size_usd = float(size_usd)
    except (TypeError, ValueError):
        raise HTTPException(400, "size_usd must be a number")
    if size_usd <= 0:
        raise HTTPException(400, "size_usd must be > 0")

    if entry_price is None:
        entry_price = _latest_price(asset)
    if entry_price is None:
        raise HTTPException(400, f"No price available for {asset} - supply entry_price")

    rm = RiskManager()
    rm.record_trade(asset, direction, size_usd, float(entry_price))
    return _risk_snapshot()


@app.post("/api/risk/position/{asset}/close")
async def api_risk_close_position(asset: str, request: Request):
    asset = asset.upper()
    try:
        body = await request.json()
    except Exception:
        body = {}
    exit_price = body.get("exit_price")

    rm = RiskManager()
    if asset not in rm.open_positions:
        raise HTTPException(404, f"No open position for {asset}")

    if exit_price is None:
        exit_price = _latest_price(asset)
    if exit_price is None:
        raise HTTPException(400, f"No price available for {asset} - supply exit_price")

    pnl = rm.close_trade(asset, float(exit_price))
    return {"pnl": pnl, **_risk_snapshot()}


@app.post("/api/risk/config")
async def api_risk_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        save_risk_config_override(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"config": RISK_CONFIG}


@app.post("/api/risk/halt")
def api_risk_halt():
    rm = RiskManager()
    rm.set_manual_halt(True)
    return _risk_snapshot()


@app.post("/api/risk/resume")
def api_risk_resume():
    rm = RiskManager()
    rm.set_manual_halt(False)
    return _risk_snapshot()
