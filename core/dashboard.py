"""Cached data accessors for the web dashboard.

webapp.py stays a thin routing layer: every repeated or heavier computation goes
through a TTL-cached accessor here, so pages start instantly and never load ML
models in a request. Light analytics are computed on the fly behind the cache;
heavy or network work reads precomputed artifacts.
"""

import os
import time
from functools import wraps

_CACHE = {}

# status to bull/bear gauge score (0 = max bearish, 100 = max bullish)
_REGIME_SCORE = {
    "CRISIS": 10,
    "RISK-OFF": 30,
    "SIDEWAYS": 50,
    "RISK-ON": 85,
    "UNKNOWN": 50,
}

_REGIME_FALLBACK = {
    "status": "UNKNOWN", "vix_level": "UNKNOWN", "vix_value": None,
    "sp500_trend": "UNKNOWN", "sp500_detail": "No data",
    "dxy_trend": "UNKNOWN", "dxy_label": "", "description": "no data",
}


def ttl_cache(ttl_seconds):
    """Memoize an accessor for ttl_seconds, keyed on (name, args). No deps."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            hit = _CACHE.get(key)
            if hit is not None and now - hit[0] < ttl_seconds:
                return hit[1]
            value = fn(*args, **kwargs)
            _CACHE[key] = (now, value)
            return value
        return wrapper
    return deco


def cache_clear():
    """Drop all cached values (tests, manual refresh)."""
    _CACHE.clear()


@ttl_cache(300)
def global_regime():
    """Global market regime dict; safe default on any failure."""
    try:
        from regime_detector import get_global_regime
        return get_global_regime()
    except Exception:
        return dict(_REGIME_FALLBACK)


def regime_score(regime):
    """Map a regime dict to a 0-100 bull/bear gauge score."""
    return _REGIME_SCORE.get(regime.get("status"), 50)


@ttl_cache(120)
def market_breadth():
    """Counts of BUY/SELL/WAIT across the latest signal per asset."""
    try:
        from core import track_record
        signals = track_record.latest_signals()
    except Exception:
        signals = []
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    for s in signals:
        sig = s.get("signal")
        if sig in counts:
            counts[sig] += 1
    counts["total"] = len(signals)
    counts["actionable"] = counts["BUY"] + counts["SELL"]
    return counts


@ttl_cache(300)
def top_leaderboard(days=30, limit=8):
    """Top assets by live accuracy (records); empty list on no data."""
    try:
        import performance_tracker as pt
        df = pt.get_leaderboard(days=days)
        if df is None or df.empty:
            return []
        return df.head(limit).to_dict("records")
    except Exception:
        return []


# sp500 trend to greed contribution (0 = fear, 100 = greed)
_SP500_TREND_SCORE = {
    "BULLISH": 85, "ABOVE BOTH SMAs": 70, "MIXED": 50,
    "BELOW BOTH SMAs": 30, "BEARISH": 15, "UNKNOWN": 50,
}


def _vix_greed(vix_value):
    """Low VIX = greed, high VIX = fear. Maps VIX ~12 to ~92, ~30 to ~28."""
    if vix_value is None:
        return 50.0
    return max(0.0, min(100.0, 100.0 - (vix_value - 12.0) * 4.0))


def sentiment_score(regime, breadth):
    """Fear/Greed score 0-100 (0 = extreme fear, 100 = extreme greed).

    Blends three cheap, available inputs: VIX (volatility), the S&P 500 trend
    (momentum), and signal breadth (share of BUY vs SELL across the radar).
    """
    vix = _vix_greed(regime.get("vix_value"))
    trend = _SP500_TREND_SCORE.get(regime.get("sp500_trend"), 50)
    buy = breadth.get("BUY", 0)
    sell = breadth.get("SELL", 0)
    breadth_score = (buy / (buy + sell) * 100.0) if (buy + sell) else 50.0
    return int(round(0.40 * vix + 0.35 * trend + 0.25 * breadth_score))


def sentiment_label(score):
    """Fear/Greed band label for a 0-100 score."""
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def market_sentiment():
    """Composite Fear/Greed reading plus the components that drove it."""
    regime = global_regime()
    breadth = market_breadth()
    score = sentiment_score(regime, breadth)
    return {
        "score": score,
        "label": sentiment_label(score),
        "vix": regime.get("vix_value"),
        "vix_level": regime.get("vix_level"),
        "sp500_trend": regime.get("sp500_trend"),
        "buy": breadth.get("BUY", 0),
        "sell": breadth.get("SELL", 0),
    }


def gauge_zone(score):
    """CSS modifier for the semicircle gauge: green / amber / red by score.

    Higher is better for both the bull/bear regime gauge and the greed/fear
    gauge, so the colour bands point the same way on each.
    """
    if score >= 60:
        return ""       # green: bullish / greed
    if score >= 35:
        return "g-warn"  # amber: neutral
    return "g-bad"       # red: bearish / fear


@ttl_cache(900)
def sector_momentum(weeks=4):
    """Per-sector momentum records (score, trend, best/worst); empty on failure."""
    try:
        import sector_rotation as sr
        df = sr.get_sector_momentum(weeks=weeks)
        return [] if df is None or df.empty else df.to_dict("records")
    except Exception:
        return []


@ttl_cache(900)
def sector_heatmap(weeks=8):
    """Sector-by-week return matrix shaped for an ECharts heatmap.

    Returns {xLabels, yLabels, data: [[x, y, value], ...], min, max} with a
    symmetric colour range around zero. Values are weekly returns in percent.
    """
    empty = {"xLabels": [], "yLabels": [], "data": [], "min": -1, "max": 1}
    try:
        import sector_rotation as sr
        df = sr.get_sector_returns(weeks=weeks)
        if df is None or df.empty:
            return empty
        week_cols = [c for c in df.columns if c != "Sector"]
        sectors = df["Sector"].tolist()
        data = []
        vmax = 1.0
        for si in range(len(df)):
            for xi, col in enumerate(week_cols):
                v = df.iloc[si][col]
                if v == v:  # not NaN
                    fv = round(float(v), 2)
                    data.append([xi, si, fv])
                    vmax = max(vmax, abs(fv))
        return {"xLabels": week_cols, "yLabels": sectors, "data": data,
                "min": round(-vmax, 2), "max": round(vmax, 2)}
    except Exception:
        return empty


@ttl_cache(600)
def correlation_stress():
    """Market stress from average pairwise correlation, with a gauge score.

    High average correlation = everything moving together = systemic stress (red);
    low = dispersed = healthy (green), so the gauge colour is inverted relative to
    the bull/bear gauge.
    """
    try:
        import correlation_alert as ca
        s = ca.get_stress_indicator()
        avg = s.get("avg_corr") or 0.0
        score = int(round(max(0.0, min(1.0, avg)) * 100))
        zone = "g-bad" if score >= 60 else "g-warn" if score >= 35 else ""
        return {**s, "score": score, "zone": zone}
    except Exception:
        return {"avg_corr": None, "label": "no data", "score": 0, "zone": ""}


@ttl_cache(600)
def correlation_heatmap():
    """Key-pairs correlation matrix shaped for an ECharts heatmap (range -1..1)."""
    empty = {"xLabels": [], "yLabels": [], "data": [], "min": -1, "max": 1}
    try:
        import correlation_alert as ca
        df = ca.get_key_pairs_matrix()
        if df is None or df.empty:
            return empty
        labels = list(df.columns)
        data = []
        for yi in range(len(labels)):
            for xi in range(len(labels)):
                v = df.iloc[yi, xi]
                if v == v:
                    data.append([xi, yi, round(float(v), 2)])
        return {"xLabels": labels, "yLabels": labels, "data": data, "min": -1, "max": 1}
    except Exception:
        return empty


def current_model_version():
    """Feature-space id of the model generation writing predictions now."""
    from core.features import feature_version
    return feature_version()


@ttl_cache(300)
def accuracy_timeseries(window=7):
    """Rolling accuracy over time for the current model and all generations.

    {"current": [...], "all": [...]} where each list holds
    {date, rolling_acc, predictions_count} records.
    """
    res = {"current": [], "all": []}
    try:
        import performance_tracker as pt
        from core.features import feature_version
        for key, mv in (("current", feature_version()), ("all", None)):
            df = pt.get_accuracy_history(window=window, model_version=mv)
            if df is not None and not df.empty:
                res[key] = df.to_dict("records")
    except Exception:
        pass
    return res


@ttl_cache(300)
def guru_latest(limit=200, db_path=None):
    """Latest Guru Council verdict per asset from guru_log, by council_pct desc.

    Reads guru_log directly (like track_record reads prediction_log); empty list
    if the table is missing.
    """
    import sqlite3
    path = db_path or os.path.join(os.path.dirname(__file__), os.pardir, "market.db")
    try:
        con = sqlite3.connect(path)
        try:
            rows = con.execute(
                "SELECT g.asset, g.date, g.council_verdict, g.council_pct, "
                "g.lynch_score, g.buffett_score, g.graham_score, g.munger_score, "
                "g.data_source, g.correct_5d FROM guru_log g "
                "JOIN (SELECT asset, MAX(date) d FROM guru_log GROUP BY asset) m "
                "ON g.asset = m.asset AND g.date = m.d "
                "ORDER BY g.council_pct DESC"
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return []
    out = []
    for r in rows[:limit]:
        out.append({"asset": r[0], "date": r[1], "verdict": r[2], "pct": r[3],
                    "lynch": r[4], "buffett": r[5], "graham": r[6], "munger": r[7],
                    "source": r[8], "correct_5d": r[9]})
    return out


@ttl_cache(600)
def guru_accuracy(days=30, horizon="5d"):
    """Council + per-guru forward accuracy from the guru_log track record."""
    res = {"council": None, "individual": {}}
    try:
        import guru_tracker as gt
        res["council"] = gt.get_guru_accuracy(days=days, horizon=horizon)
        res["individual"] = gt.get_guru_individual_accuracy(days=days, horizon=horizon)
    except Exception:
        pass
    return res


@ttl_cache(300)
def models_health():
    """Model fleet health summary (counts, average age, best/worst)."""
    try:
        import model_health as mh
        return mh.get_health_summary()
    except Exception:
        return {}


@ttl_cache(300)
def models_stale(max_age_days=30, limit=20):
    """Models flagged for retraining (oldest first); empty on failure."""
    try:
        import model_health as mh
        rows = mh.get_stale_models(max_age_days=max_age_days)
        return rows[:limit] if rows else []
    except Exception:
        return []


@ttl_cache(1200)
def news_digest(lang="all", category="all", limit=40):
    """Ranked authority news digest (console parity); empty list on failure.

    Cached 20 min - each call otherwise refetches every RSS feed. Summaries are
    off (fetch_summaries=False) so a page load does not block on per-article
    requests; the digest itself still carries title, source, sentiment, score.
    """
    try:
        from news_analyzer import fetch_authority_digest
        items = fetch_authority_digest(max_per_source=5, lang_filter=lang,
                                       category_filter=category,
                                       fetch_summaries=False)
        return items[:limit]
    except Exception:
        return []
