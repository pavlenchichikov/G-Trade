"""
Model Health Monitor — G-Trade
=================================================
Checks age, quality, and drift of trained models.

CLI usage:
  python model_health.py              -- full health report
  python model_health.py --stale 7    -- show models older than 7 days
  python model_health.py --json       -- JSON output for GUI
"""

import argparse
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import FULL_ASSET_MAP

MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _table_name(asset):
    """Convert asset key to the filename prefix used for model files."""
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _file_age_days(path):
    """Return file age in fractional days, or None if file does not exist."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() / 86400


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_health_summary():
    """Return a dict with overall model health statistics."""
    registry = _load_json(REGISTRY_PATH)
    now = datetime.now()

    cb_count = 0
    lstm_count = 0
    ages = []

    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb_path = os.path.join(MODEL_DIR, f"{tbl}_cb.cbm")
        lstm_path = os.path.join(MODEL_DIR, f"{tbl}_lstm.keras")

        cb_age = _file_age_days(cb_path)
        lstm_age = _file_age_days(lstm_path)

        if cb_age is not None:
            cb_count += 1
            ages.append((asset, cb_age))
        if lstm_age is not None:
            lstm_count += 1
            ages.append((asset, lstm_age))

    avg_age = sum(a for _, a in ages) / len(ages) if ages else 0.0
    oldest = max(ages, key=lambda x: x[1]) if ages else (None, 0)

    scores = []
    for asset, entry in registry.items():
        s = entry.get("score")
        if s is not None:
            scores.append((asset, s))
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "cb_count": cb_count,
        "lstm_count": lstm_count,
        "avg_age_days": round(avg_age, 1),
        "oldest_asset": oldest[0],
        "oldest_age_days": round(oldest[1], 1) if oldest[0] else 0,
        "registry_entries": len(registry),
        "best_score": scores[0] if scores else None,
        "worst_score": scores[-1] if scores else None,
        "timestamp": now.isoformat(),
    }


def get_stale_models(max_age_days=7):
    """Return list of dicts for assets whose models are older than max_age_days."""
    registry = _load_json(REGISTRY_PATH)
    stale = []

    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb_age = _file_age_days(os.path.join(MODEL_DIR, f"{tbl}_cb.cbm"))
        lstm_age = _file_age_days(os.path.join(MODEL_DIR, f"{tbl}_lstm.keras"))

        if cb_age is None and lstm_age is None:
            continue

        max_model_age = max(a for a in [cb_age, lstm_age] if a is not None)
        if max_model_age >= max_age_days:
            entry = registry.get(asset, {})
            stale.append({
                "asset": asset,
                "cb_age_days": round(cb_age, 1) if cb_age is not None else None,
                "lstm_age_days": round(lstm_age, 1) if lstm_age is not None else None,
                "score": entry.get("score"),
                "status": "RETRAIN",
            })

    stale.sort(key=lambda x: max(x["cb_age_days"] or 0, x["lstm_age_days"] or 0), reverse=True)
    return stale


def get_missing_models():
    """Return list of asset names that have no .cbm and no .keras file."""
    missing = []
    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb = os.path.exists(os.path.join(MODEL_DIR, f"{tbl}_cb.cbm"))
        lstm = os.path.exists(os.path.join(MODEL_DIR, f"{tbl}_lstm.keras"))
        if not cb and not lstm:
            missing.append(asset)
    return missing


def get_quality_ranking():
    """Return list of dicts sorted by score descending from champion_registry."""
    registry = _load_json(REGISTRY_PATH)
    ranking = []

    for asset, entry in registry.items():
        score = entry.get("score")
        if score is None:
            continue
        ranking.append({
            "asset": asset,
            "score": round(score, 2),
            "policy": entry.get("policy", "UNKNOWN"),
            "updated_at": entry.get("updated_at", ""),
        })

    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def _score_color(score):
    if score >= 5.0:
        return "\033[92m"   # green
    if score >= 2.0:
        return "\033[93m"   # yellow
    if score >= 0:
        return "\033[33m"   # dark yellow
    return "\033[91m"       # red

_RST = "\033[0m"
_W   = 62


def _print_report(max_age_days=7):
    summary = get_health_summary()
    stale   = get_stale_models(max_age_days)
    missing = get_missing_models()
    ranking = get_quality_ranking()

    now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print()
    print("═" * _W)
    print(f"  MODEL HEALTH  │  {now}")
    print("═" * _W)

    # ── SUMMARY ───────────────────────────────────────────────────
    oldest_lbl = summary['oldest_asset'] or "N/A"
    best  = summary["best_score"]
    worst = summary["worst_score"]
    print()
    print(f"  Models  : {summary['cb_count']} CatBoost + {summary['lstm_count']} LSTM"
          f"   │   Registry: {summary['registry_entries']} entries")
    print(f"  Avg age : {summary['avg_age_days']}d"
          f"   │   Oldest: {oldest_lbl} ({summary['oldest_age_days']}d)")
    if best and worst:
        print(f"  Best    : {best[0]} ({best[1]:+.2f})"
              f"   │   Worst: {worst[0]} ({worst[1]:+.2f})")
    print()

    # ── STALE MODELS ──────────────────────────────────────────────
    tag_stale = f"── STALE  (>{max_age_days}d) "
    print("  " + tag_stale + "─" * max(0, _W - 2 - len(tag_stale)))
    if stale:
        print(f"  {'Asset':<10}  {'CB':<7}  {'LSTM':<7}  {'Score':>6}  Status")
        print("  " + "─" * (_W - 4))
        for s in stale:
            cb_s   = f"{s['cb_age_days']}d"   if s["cb_age_days"]   is not None else "—"
            lstm_s = f"{s['lstm_age_days']}d" if s["lstm_age_days"] is not None else "—"
            sc_s   = f"{s['score']:+.2f}"     if s["score"]         is not None else "N/A"
            print(f"  {s['asset']:<10}  {cb_s:<7}  {lstm_s:<7}  {sc_s:>6}  {s['status']}")
    else:
        print("  All models are fresh.")
    print()

    # ── MISSING MODELS ────────────────────────────────────────────
    print("  ── MISSING ─────────────────────────────────────────────")
    if missing:
        for m in missing:
            print(f"  \033[91m{m}\033[0m  — no .cbm / .keras found")
    else:
        print("  All assets have models.")
    print()

    # ── QUALITY RANKING ───────────────────────────────────────────
    print("  ── QUALITY RANKING ─────────────────────────────────────")
    print(f"  {'Asset':<10}  {'Score':>7}  {'Policy':<12}  Updated")
    print("  " + "─" * (_W - 4))
    for r in ranking:
        updated_s = r["updated_at"][:10] if r["updated_at"] else "N/A"
        clr = _score_color(r["score"])
        score_s = f"{r['score']:+.2f}"
        print(f"  {r['asset']:<10}  {clr}{score_s:>7}{_RST}  {r['policy']:<12}  {updated_s}")
    print()


def _print_json(max_age_days=7):
    data = {
        "summary": get_health_summary(),
        "stale": get_stale_models(max_age_days),
        "missing": get_missing_models(),
        "ranking": get_quality_ranking(),
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model Health Monitor")
    parser.add_argument("--stale", type=int, default=7,
                        help="Flag models older than N days (default: 7)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")
    args = parser.parse_args()

    if args.json:
        _print_json(max_age_days=args.stale)
    else:
        _print_report(max_age_days=args.stale)


if __name__ == "__main__":
    main()
