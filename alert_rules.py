# alert_rules.py
# Custom alert rules for G-Trade.
# Users define conditions; module checks them against market.db and notifies on trigger.

import os
import json
import argparse
from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine

from config import FULL_ASSET_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
ALERTS_PATH = os.path.join(BASE_DIR, "alerts.json")

SUPPORTED_CONDITIONS = [
    "rsi_below",
    "rsi_above",
    "price_above",
    "price_below",
    "price_drop_pct",
    "price_rise_pct",
    "trend_change",
]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def load_rules() -> list:
    """Load alert rules from alerts.json. Returns list of rule dicts."""
    if not os.path.exists(ALERTS_PATH):
        return []
    with open(ALERTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def save_rules(rules: list) -> None:
    """Persist rules list to alerts.json."""
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, indent=2, ensure_ascii=False)


def add_rule(asset: str, condition: str, value: float) -> dict:
    """Add a new alert rule. Returns the created rule dict."""
    asset = asset.upper()
    if asset not in FULL_ASSET_MAP:
        raise ValueError(f"Unknown asset '{asset}'. Must be one of: {sorted(FULL_ASSET_MAP)}")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Supported: {SUPPORTED_CONDITIONS}")

    rules = load_rules()
    new_id = max((r["id"] for r in rules), default=0) + 1
    rule = {
        "id": new_id,
        "asset": asset,
        "condition": condition,
        "value": float(value),
        "enabled": True,
        "last_triggered": None,
    }
    rules.append(rule)
    save_rules(rules)
    return rule


def remove_rule(rule_id: int) -> bool:
    """Remove rule by ID. Returns True if found and removed."""
    rules = load_rules()
    new_rules = [r for r in rules if r["id"] != rule_id]
    if len(new_rules) == len(rules):
        return False
    save_rules(new_rules)
    return True


def toggle_rule(rule_id: int) -> bool:
    """Enable/disable rule by ID. Returns new enabled state, raises if not found."""
    rules = load_rules()
    for rule in rules:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_rules(rules)
            return rule["enabled"]
    raise ValueError(f"Rule ID {rule_id} not found.")


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """Return the most recent RSI value for a price series."""
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _load_asset_data(asset: str) -> pd.DataFrame | None:
    """Load daily OHLCV data for an asset from market.db."""
    table = asset.lower()
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        df = pd.read_sql(f'SELECT * FROM "{table}"', engine, parse_dates=["Date"])
        engine.dispose()
    except Exception:
        return None

    if df.empty:
        return None

    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _detect_trend(closes: pd.Series) -> str:
    """Return 'UPTREND' or 'DOWNTREND' based on SMA20/SMA50 cross."""
    if len(closes) < 50:
        return "UNKNOWN"
    sma20 = closes.rolling(20).mean().iloc[-1]
    sma50 = closes.rolling(50).mean().iloc[-1]
    return "UPTREND" if sma20 > sma50 else "DOWNTREND"


def _detect_trend_previous(closes: pd.Series) -> str:
    """Return trend for the second-to-last bar (to detect a change)."""
    if len(closes) < 51:
        return "UNKNOWN"
    sma20 = closes.rolling(20).mean().iloc[-2]
    sma50 = closes.rolling(50).mean().iloc[-2]
    return "UPTREND" if sma20 > sma50 else "DOWNTREND"


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------

def check_alerts() -> list:
    """
    Evaluate all enabled rules against current market data.

    Returns a list of triggered alert dicts with keys:
        rule, asset, condition, value, current_value, message
    """
    rules = load_rules()
    today_str = date.today().isoformat()

    # Group assets to load each DB table only once
    assets_needed = {r["asset"] for r in rules if r["enabled"]}
    asset_data: dict[str, pd.DataFrame | None] = {
        asset: _load_asset_data(asset) for asset in assets_needed
    }

    triggered = []
    rules_changed = False

    for rule in rules:
        if not rule["enabled"]:
            continue

        # Skip if already triggered today
        if rule.get("last_triggered") == today_str:
            continue

        asset = rule["asset"]
        condition = rule["condition"]
        threshold = rule["value"]
        df = asset_data.get(asset)

        if df is None or len(df) < 2:
            continue

        closes = df["Close"].dropna()
        if closes.empty:
            continue

        current_price = float(closes.iloc[-1])
        prev_price = float(closes.iloc[-2])
        alert_triggered = False
        current_value = None
        message = ""

        if condition == "rsi_below":
            if len(closes) < 15:
                continue
            rsi = _compute_rsi(closes)
            current_value = round(rsi, 2)
            if rsi < threshold:
                alert_triggered = True
                message = f"{asset} RSI(14) = {current_value:.1f} < {threshold} (oversold)"

        elif condition == "rsi_above":
            if len(closes) < 15:
                continue
            rsi = _compute_rsi(closes)
            current_value = round(rsi, 2)
            if rsi > threshold:
                alert_triggered = True
                message = f"{asset} RSI(14) = {current_value:.1f} > {threshold} (overbought)"

        elif condition == "price_above":
            current_value = current_price
            if current_price > threshold:
                alert_triggered = True
                message = f"{asset} price {current_price:.4f} > {threshold}"

        elif condition == "price_below":
            current_value = current_price
            if current_price < threshold:
                alert_triggered = True
                message = f"{asset} price {current_price:.4f} < {threshold}"

        elif condition == "price_drop_pct":
            pct_change = (current_price - prev_price) / prev_price * 100
            current_value = round(pct_change, 2)
            if pct_change < -abs(threshold):
                alert_triggered = True
                message = f"{asset} dropped {current_value:.2f}% today (threshold: -{threshold}%)"

        elif condition == "price_rise_pct":
            pct_change = (current_price - prev_price) / prev_price * 100
            current_value = round(pct_change, 2)
            if pct_change > abs(threshold):
                alert_triggered = True
                message = f"{asset} rose {current_value:.2f}% today (threshold: +{threshold}%)"

        elif condition == "trend_change":
            current_trend = _detect_trend(closes)
            prev_trend = _detect_trend_previous(closes)
            current_value = current_trend
            if current_trend != prev_trend and current_trend != "UNKNOWN" and prev_trend != "UNKNOWN":
                alert_triggered = True
                message = f"{asset} trend changed: {prev_trend} -> {current_trend}"

        if alert_triggered:
            rule["last_triggered"] = today_str
            rules_changed = True
            triggered.append({
                "rule": rule["id"],
                "asset": asset,
                "condition": condition,
                "value": threshold,
                "current_value": current_value,
                "message": message,
            })

    if rules_changed:
        save_rules(rules)

    return triggered


def format_alerts(alerts: list) -> str:
    """Format triggered alerts into a human-readable string."""
    if not alerts:
        return "No alerts triggered."
    lines = [f"=== {len(alerts)} Alert(s) Triggered ==="]
    for a in alerts:
        lines.append(
            f"[Rule #{a['rule']}] {a['message']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(rules: list) -> None:
    if not rules:
        print("No rules defined.")
        return
    print(f"{'ID':>4}  {'Asset':<10} {'Condition':<16} {'Value':>8}  {'Enabled':<8}  Last Triggered")
    print("-" * 70)
    for r in rules:
        print(
            f"{r['id']:>4}  {r['asset']:<10} {r['condition']:<16} {r['value']:>8.2f}"
            f"  {'Yes' if r['enabled'] else 'No':<8}  {r['last_triggered'] or '-'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-Trade — Custom Alert Rules")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--add", nargs=3, metavar=("ASSET", "CONDITION", "VALUE"),
                       help="Add a new rule. E.g.: --add BTC rsi_below 30")
    group.add_argument("--remove", type=int, metavar="ID",
                       help="Remove rule by ID")
    group.add_argument("--toggle", type=int, metavar="ID",
                       help="Enable/disable rule by ID")
    group.add_argument("--list", action="store_true",
                       help="List all rules")
    args = parser.parse_args()

    if args.add:
        asset, condition, raw_value = args.add
        try:
            value = float(raw_value)
            rule = add_rule(asset, condition, value)
            print(f"Added rule #{rule['id']}: {rule['asset']} {rule['condition']} {rule['value']}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.remove is not None:
        if remove_rule(args.remove):
            print(f"Rule #{args.remove} removed.")
        else:
            print(f"Rule #{args.remove} not found.")

    elif args.toggle is not None:
        try:
            state = toggle_rule(args.toggle)
            print(f"Rule #{args.toggle} is now {'enabled' if state else 'disabled'}.")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.list:
        _cmd_list(load_rules())

    else:
        # Default: run check
        alerts = check_alerts()
        print(format_alerts(alerts))
        if alerts:
            for a in alerts:
                print(f"  -> {a}")
