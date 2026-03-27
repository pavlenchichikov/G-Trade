"""
model_comparison.py — Model quality tracking over time for G-Trade.
Saves daily snapshots of quality_report.json and provides comparison utilities.
"""

import os
import json
from datetime import datetime, date

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HISTORY_PATH = os.path.join(MODEL_DIR, "quality_history.json")
REPORT_PATH = os.path.join(MODEL_DIR, "quality_report.json")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. save_snapshot
# ---------------------------------------------------------------------------

def save_snapshot():
    """
    Read quality_report.json, build a snapshot dict, append to quality_history.json.
    Skips saving if a snapshot for today already exists.
    Returns the snapshot dict that was saved (or None if skipped).
    """
    report = _load_json(REPORT_PATH, None)
    if report is None:
        print(f"[model_comparison] quality_report.json not found at {REPORT_PATH}")
        return None

    today = date.today().isoformat()

    history = _load_json(HISTORY_PATH, [])

    # Check for duplicate date
    if history and history[-1].get("date") == today:
        print(f"[model_comparison] Snapshot for {today} already exists — skipping.")
        return None

    # quality_report.json is expected to be a list of dicts with at least:
    # Asset, CB_Acc, LSTM_Acc, Score
    if isinstance(report, list):
        assets = {}
        for row in report:
            asset = row.get("Asset") or row.get("asset")
            if not asset:
                continue
            assets[asset] = {
                "cb_acc": round(float(row.get("CB_Acc", row.get("cb_acc", 0))), 4),
                "lstm_acc": round(float(row.get("LSTM_Acc", row.get("lstm_acc", 0))), 4),
                "score": round(float(row.get("Score", row.get("score", 0))), 4),
            }
    elif isinstance(report, dict):
        # Already in {asset: {...}} form
        assets = {}
        for asset, metrics in report.items():
            assets[asset] = {
                "cb_acc": round(float(metrics.get("CB_Acc", metrics.get("cb_acc", 0))), 4),
                "lstm_acc": round(float(metrics.get("LSTM_Acc", metrics.get("lstm_acc", 0))), 4),
                "score": round(float(metrics.get("Score", metrics.get("score", 0))), 4),
            }
    else:
        print("[model_comparison] Unexpected format in quality_report.json")
        return None

    snapshot = {"date": today, "assets": assets}
    history.append(snapshot)
    _save_json(HISTORY_PATH, history)
    print(f"[model_comparison] Snapshot saved for {today} ({len(assets)} assets).")
    return snapshot


# ---------------------------------------------------------------------------
# 2. get_history
# ---------------------------------------------------------------------------

def get_history(asset=None, metric="score", last_n=20) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [Date, <asset1>, <asset2>, ...] (or single asset).
    Values are the specified metric (cb_acc | lstm_acc | score) over time.
    """
    history = _load_json(HISTORY_PATH, [])
    if not history:
        return pd.DataFrame()

    # Limit to last_n snapshots
    history = history[-last_n:]

    rows = []
    for snap in history:
        row = {"Date": snap["date"]}
        assets_data = snap.get("assets", {})
        if asset:
            row[asset] = assets_data.get(asset, {}).get(metric)
        else:
            for a, metrics in assets_data.items():
                row[a] = metrics.get(metric)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Date")
    return df


# ---------------------------------------------------------------------------
# 3. get_comparison_table
# ---------------------------------------------------------------------------

def get_comparison_table() -> pd.DataFrame:
    """
    Compares the latest snapshot vs the previous one.
    Columns: Asset, Current_CB, Current_LSTM, Current_Score,
             Prev_CB, Prev_LSTM, Prev_Score, CB_Delta, LSTM_Delta, Score_Delta
    """
    history = _load_json(HISTORY_PATH, [])
    if len(history) < 2:
        # Return current only with NaN deltas if only one snapshot exists
        if len(history) == 1:
            snap = history[0]
            rows = []
            for asset, m in snap["assets"].items():
                rows.append({
                    "Asset": asset,
                    "Current_CB": m.get("cb_acc"),
                    "Current_LSTM": m.get("lstm_acc"),
                    "Current_Score": m.get("score"),
                    "Prev_CB": None,
                    "Prev_LSTM": None,
                    "Prev_Score": None,
                    "CB_Delta": None,
                    "LSTM_Delta": None,
                    "Score_Delta": None,
                })
            return pd.DataFrame(rows)
        return pd.DataFrame()

    latest = history[-1]["assets"]
    prev = history[-2]["assets"]

    all_assets = sorted(set(list(latest.keys()) + list(prev.keys())))
    rows = []
    for asset in all_assets:
        cur = latest.get(asset, {})
        prv = prev.get(asset, {})

        cur_cb = cur.get("cb_acc")
        cur_lstm = cur.get("lstm_acc")
        cur_score = cur.get("score")
        prv_cb = prv.get("cb_acc")
        prv_lstm = prv.get("lstm_acc")
        prv_score = prv.get("score")

        def delta(a, b):
            if a is not None and b is not None:
                return round(a - b, 4)
            return None

        rows.append({
            "Asset": asset,
            "Current_CB": cur_cb,
            "Current_LSTM": cur_lstm,
            "Current_Score": cur_score,
            "Prev_CB": prv_cb,
            "Prev_LSTM": prv_lstm,
            "Prev_Score": prv_score,
            "CB_Delta": delta(cur_cb, prv_cb),
            "LSTM_Delta": delta(cur_lstm, prv_lstm),
            "Score_Delta": delta(cur_score, prv_score),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. get_model_age
# ---------------------------------------------------------------------------

def get_model_age() -> pd.DataFrame:
    """
    Reads updated_at from champion_registry.json.
    Returns DataFrame: Asset, Last_Updated, Days_Ago, Score, Status
    Status: FRESH (<7 days), STALE (7-30 days), CRITICAL (>30 days)
    """
    registry = _load_json(REGISTRY_PATH, {})
    if not registry:
        return pd.DataFrame()

    # Load latest scores from history for reference
    history = _load_json(HISTORY_PATH, [])
    latest_scores = {}
    if history:
        latest_scores = {
            asset: m.get("score")
            for asset, m in history[-1].get("assets", {}).items()
        }

    today = date.today()
    rows = []

    for asset, info in registry.items():
        if not isinstance(info, dict):
            continue

        updated_at_str = info.get("updated_at") or info.get("last_updated")
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(str(updated_at_str)).date()
                days_ago = (today - updated_at).days
            except (ValueError, TypeError):
                updated_at = None
                days_ago = None
        else:
            updated_at = None
            days_ago = None

        if days_ago is None:
            status = "UNKNOWN"
        elif days_ago < 7:
            status = "FRESH"
        elif days_ago <= 30:
            status = "STALE"
        else:
            status = "CRITICAL"

        rows.append({
            "Asset": asset,
            "Last_Updated": str(updated_at) if updated_at else None,
            "Days_Ago": days_ago,
            "Score": latest_scores.get(asset),
            "Status": status,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Days_Ago", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 5. get_best_worst
# ---------------------------------------------------------------------------

def get_best_worst(metric="score", n=5) -> dict:
    """
    Returns {"best": [...], "worst": [...]} based on the latest snapshot.
    Each entry is a dict: {asset, value}.
    """
    history = _load_json(HISTORY_PATH, [])
    if not history:
        return {"best": [], "worst": []}

    latest = history[-1].get("assets", {})
    scored = [
        {"asset": asset, "value": metrics.get(metric)}
        for asset, metrics in latest.items()
        if metrics.get(metric) is not None
    ]
    scored.sort(key=lambda x: x["value"], reverse=True)

    return {
        "best": scored[:n],
        "worst": scored[-n:][::-1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Model Quality Tracker — G-Trade")
    print("=" * 60)

    print("\n[1] Saving snapshot...")
    save_snapshot()

    print("\n[2] Comparison Table (latest vs previous):")
    comp = get_comparison_table()
    if comp.empty:
        print("  Not enough history to compare.")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(comp.to_string(index=False))

    print("\n[3] Model Age Report:")
    age = get_model_age()
    if age.empty:
        print("  champion_registry.json not found or empty.")
    else:
        print(age.to_string(index=False))

    print("\n[4] Best / Worst by Score (top 5):")
    bw = get_best_worst(metric="score", n=5)
    print("  Best:")
    for entry in bw["best"]:
        print(f"    {entry['asset']}: {entry['value']}")
    print("  Worst:")
    for entry in bw["worst"]:
        print(f"    {entry['asset']}: {entry['value']}")
