# core/live_gate.py
"""Live-accuracy signal gate.

Suppresses BUY/SELL on segments with PROVEN bad live accuracy (verified
outcomes in prediction_log) while the measurement stream keeps flowing:
predict.py logs the RAW signal, so a gated class rehabilitates itself when
fresh statistics improve. Spec: docs/superpowers/specs/2026-07-16-live-gate-design.md

Rules (first match wins; every threshold is env-tunable; GTRADE_LIVE_GATE=0
turns the whole gate off):
  1. class gate  - asset-class live accuracy < 0.45 with n >= 100
  2. asset gate  - per-asset live accuracy   < 0.40 with n >= 20
  3. tail gate   - calibrated prob >= 0.85 or <= 0.15 (anti-calibrated tail:
     the 0.9-1.0 bucket scored 32% over 2026-06-12..07-16)
"""

import os
import sqlite3
import time

import config
from core.logger import get_logger

logger = get_logger("live_gate")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")

_TTL = 600
_CACHE = {"t": 0.0, "stats": None}


def _env_f(name, default):
    try:
        return float((os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default


def _env_i(name, default):
    try:
        return int((os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default


def gate_on():
    return (os.getenv("GTRADE_LIVE_GATE", "1") or "1").strip() not in (
        "0", "false", "False")


# Curated meta-groups in ASSET_TYPES that overlap the real sector classes.
# They must not take part in class attribution: "TOP SIGNALS" lists ETH, TSLA,
# GOLD, VIX, PLTR, IMOEX and comes FIRST in the dict, so a first-match walk
# would steal those assets from their real classes (TSLA never counting toward
# US TECH) and form a mixed pseudo-class that could gate GOLD because ETH+VIX
# dragged the group average down.
META_GROUPS = ("TOP SIGNALS",)


def _asset_class(asset):
    for group, assets in config.ASSET_TYPES.items():
        if group in META_GROUPS:
            continue
        if asset in assets:
            return group
    return None


def verified_stats(days=None, db_path=None):
    """Verified BUY/SELL accuracy of the last N days, grouped two ways:
    {"classes": {group: (n, acc)}, "assets": {asset: (n, acc)}}.

    Cached for 10 minutes (one query per predict run, not one per asset);
    an explicit db_path (tests) bypasses the cache. Any DB problem returns
    empty stats - the class/asset rules then never fire."""
    now = time.time()
    if db_path is None and _CACHE["stats"] is not None and now - _CACHE["t"] < _TTL:
        return _CACHE["stats"]
    days = days or _env_i("GTRADE_LIVE_GATE_DAYS", 45)
    stats = {"classes": {}, "assets": {}}
    try:
        con = sqlite3.connect(db_path or DB_PATH)
        try:
            rows = con.execute(
                "SELECT asset, COUNT(*), AVG(correct) FROM prediction_log "
                "WHERE correct IS NOT NULL AND signal IN ('BUY','SELL') "
                "AND date >= date('now', ?) GROUP BY asset",
                ("-%d days" % days,),
            ).fetchall()
        finally:
            con.close()
        cls = {}
        for asset, n, acc in rows:
            stats["assets"][asset] = (n, acc)
            group = _asset_class(asset)
            if group:
                cn, ck = cls.get(group, (0, 0.0))
                cls[group] = (cn + n, ck + acc * n)
        stats["classes"] = {g: (n, k / n) for g, (n, k) in cls.items() if n}
    except Exception as e:
        logger.debug("live-gate stats unavailable: %s", e)
        stats = {"classes": {}, "assets": {}}
    if db_path is None:
        _CACHE["t"], _CACHE["stats"] = now, stats
    return stats


def gate(asset, prob, sig, db_path=None):
    """(sig_out, reason_or_None). WAIT passes through untouched."""
    if sig == "WAIT" or not gate_on():
        return sig, None
    stats = verified_stats(db_path=db_path)
    group = _asset_class(asset)
    if group and group in stats["classes"]:
        n, acc = stats["classes"][group]
        if n >= _env_i("GTRADE_LIVE_GATE_MIN_N", 100) and \
                acc < _env_f("GTRADE_LIVE_GATE_MIN_ACC", 0.45):
            return "WAIT", "live-gate: %s %.0f%% (n=%d)" % (group, 100 * acc, n)
    if asset in stats["assets"]:
        n, acc = stats["assets"][asset]
        if n >= _env_i("GTRADE_LIVE_GATE_ASSET_N", 20) and \
                acc < _env_f("GTRADE_LIVE_GATE_ASSET_ACC", 0.40):
            return "WAIT", "live-gate: %s %.0f%% (n=%d)" % (asset, 100 * acc, n)
    tail = _env_f("GTRADE_LIVE_GATE_TAIL", 0.85)
    if prob >= tail or prob <= 1.0 - tail:
        return "WAIT", "live-gate: anti-calibrated tail"
    return sig, None
