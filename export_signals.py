"""
Export Signals — G-Trade
==========================================
Builds a signal table from champion_registry, tuned_thresholds, and market.db,
then exports to CSV.

CLI usage:
  python export_signals.py                        -- export to exports/signals_YYYYMMDD.csv
  python export_signals.py --output my.csv        -- custom filename
  python export_signals.py --dir C:/tmp/exports   -- custom output directory
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import FULL_ASSET_MAP

MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "market.db")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")
DEFAULT_EXPORT_DIR = os.path.join(BASE_DIR, "exports")


def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _table_name(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _file_age_days(path):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return round((datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() / 86400, 1)


def _compute_rsi(series, period=14):
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_trend(series, short=20, long=50):
    """Simple trend label based on SMA crossover."""
    if len(series) < long:
        return "N/A"
    sma_short = series.rolling(short).mean().iloc[-1]
    sma_long = series.rolling(long).mean().iloc[-1]
    if sma_short > sma_long:
        return "UP"
    elif sma_short < sma_long:
        return "DOWN"
    return "FLAT"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_signal_table():
    """
    Build a DataFrame with one row per asset containing:
    Asset, Price, Date, Score, Buy_Thr, Sell_Thr, Policy, RSI, Trend, Model_Age_Days
    """
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    if not os.path.exists(DB_PATH):
        print(f"  Database not found: {DB_PATH}")
        return pd.DataFrame()

    engine = create_engine(f"sqlite:///{DB_PATH}")
    rows = []

    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        reg = registry.get(asset, {})
        thr = thresholds.get(asset, {})

        # Model age (use CB model file)
        cb_path = os.path.join(MODEL_DIR, f"{tbl}_cb.cbm")
        model_age = _file_age_days(cb_path)

        # Read latest price data from DB
        price = None
        last_date = None
        rsi_val = None
        trend_val = "N/A"

        try:
            df = pd.read_sql(f"SELECT * FROM {tbl}", engine,
                             index_col="Date", parse_dates=["Date"])
            df.index = pd.to_datetime(df.index).normalize()
            df = df[~df.index.duplicated(keep="last")].sort_index()

            if len(df) > 0 and "close" in df.columns:
                price = round(float(df["close"].iloc[-1]), 4)
                last_date = df.index[-1].strftime("%Y-%m-%d")

                if len(df) >= 14:
                    rsi_series = _compute_rsi(df["close"])
                    rsi_val = round(float(rsi_series.iloc[-1]), 1)

                trend_val = _compute_trend(df["close"])
        except Exception:
            pass

        rows.append({
            "Asset": asset,
            "Price": price,
            "Date": last_date,
            "Score": round(reg.get("score", 0), 2) if reg.get("score") is not None else None,
            "Buy_Thr": round(thr.get("buy", reg.get("buy_thr", 0.55)), 4),
            "Sell_Thr": round(thr.get("sell", reg.get("sell_thr", 0.45)), 4),
            "Policy": reg.get("policy", "N/A"),
            "RSI": rsi_val,
            "Trend": trend_val,
            "Model_Age_Days": model_age,
        })

    return pd.DataFrame(rows)


def export_csv(path=None, output_dir=None):
    """
    Generate the signal table and write it to a CSV file.
    Returns the absolute path to the written file.
    """
    df = generate_signal_table()

    if df.empty:
        print("  No data to export.")
        return None

    # Determine output path
    if path is None:
        if output_dir is None:
            output_dir = DEFAULT_EXPORT_DIR
        os.makedirs(output_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        path = os.path.join(output_dir, f"signals_{today}.csv")
    else:
        # If path is just a filename, put it in output_dir (or default)
        parent = os.path.dirname(path)
        if not parent:
            if output_dir is None:
                output_dir = DEFAULT_EXPORT_DIR
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, path)
        else:
            os.makedirs(parent, exist_ok=True)

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export signal summary to CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output filename (e.g. my_signals.csv)")
    parser.add_argument("--dir", type=str, default=None,
                        help="Custom output directory (default: exports/)")
    args = parser.parse_args()

    w = 60
    print("=" * w)
    print("  EXPORT SIGNALS")
    print("=" * w)

    result_path = export_csv(path=args.output, output_dir=args.dir)

    if result_path:
        df = pd.read_csv(result_path)
        print(f"  Exported {len(df)} assets to {os.path.basename(result_path)}")
        print(f"  Path: {result_path}")
    else:
        print("  Export failed -- no data available.")

    print()


if __name__ == "__main__":
    main()
