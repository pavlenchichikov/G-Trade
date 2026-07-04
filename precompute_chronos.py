"""Precompute Chronos zero-shot forecast features and cache them per (asset, date).

Opt-in and heavy (a rolling forecast per bar), so run this once when you want to A/B
Chronos features; training then reads the cache cheaply. Needs requirements-chronos.txt.

    python precompute_chronos.py [--assets SP500,NVDA] [--model amazon/chronos-t5-tiny]
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

from core.chronos_features import forecast_features
from core.features import CHRONOS_CACHE_TABLE
from core.track_record import _table_name

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
SELECTION = "SP500,NVDA,BTC,ETH,EURUSD,GBPJPY,GAS,AAPL,SBER,DAX"


def _cached_dates(engine, table):
    try:
        df = pd.read_sql(
            "SELECT date FROM %s WHERE asset = ?" % CHRONOS_CACHE_TABLE,
            engine, params=(table,))
        return set(df["date"].astype(str))
    except Exception as exc:
        if "no such table" in str(exc).lower():
            return set()
        raise


def precompute_asset(asset, engine, forecaster=None, context=64, horizon=5,
                     model="amazon/chronos-t5-tiny"):
    """Forecast + cache the uncached bars of one asset. Returns rows newly written."""
    table = _table_name(asset)
    # market.db stores OHLCV lower-case (open/close/high/low/volume) with a capital Date.
    prices = pd.read_sql('SELECT Date, close FROM "%s" ORDER BY Date' % table,
                         engine, index_col="Date")
    prices.index = pd.to_datetime(prices.index)
    feats = forecast_features(prices["close"], context=context, horizon=horizon,
                              model=model, forecaster=forecaster).dropna()
    if feats.empty:
        return 0
    already = _cached_dates(engine, table)
    rows = []
    for dt, r in feats.iterrows():
        d = str(pd.Timestamp(dt).date())
        if d in already:
            continue
        rows.append({"asset": table, "date": d, "chronos_ret": r["chronos_ret"],
                     "chronos_spread": r["chronos_spread"], "chronos_dir": r["chronos_dir"]})
    if not rows:
        return 0
    pd.DataFrame(rows).to_sql(CHRONOS_CACHE_TABLE, engine, if_exists="append", index=False)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=SELECTION)
    ap.add_argument("--model", default="amazon/chronos-t5-tiny")
    args = ap.parse_args(argv)
    engine = create_engine("sqlite:///" + DB_PATH)
    total = 0
    for a in [x.strip() for x in args.assets.split(",") if x.strip()]:
        n = precompute_asset(a, engine, model=args.model)
        total += n
        print("[chronos] %s: +%d bars cached" % (a, n))
    print("[chronos] done: %d bars cached across the subset" % total)


if __name__ == "__main__":
    sys.exit(main())
