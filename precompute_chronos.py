"""Precompute Chronos zero-shot forecast features and cache them per (asset, date).

Opt-in and heavy (a rolling forecast per bar), so run this once when you want to A/B
Chronos features; training then reads the cache cheaply. Needs requirements-chronos.txt.

    python precompute_chronos.py [--assets SP500,NVDA|all] [--model tiny|mini|small|base|large]

The base model picks the accuracy/speed trade-off: a bigger Chronos model forecasts
better but is much slower per bar. A short name (tiny/mini/small/base/large) resolves to
the matching amazon/chronos-t5-* checkpoint; a full Hugging Face id is also accepted.
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

# Short names for the Chronos-T5 base models (accuracy/speed trade-off, tiny -> large).
CHRONOS_MODELS = {
    "tiny": "amazon/chronos-t5-tiny",
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


def resolve_model(name):
    """Expand a short base-model name to its Hugging Face id; pass a full id through."""
    return CHRONOS_MODELS.get((name or "").strip().lower(), name)


def resolve_assets(spec):
    """The asset list for --assets: 'all' -> the full 181-asset universe, else the
    comma-separated names given."""
    if (spec or "").strip().lower() == "all":
        from config import FULL_ASSET_MAP
        return list(FULL_ASSET_MAP.keys())
    return [x.strip() for x in spec.split(",") if x.strip()]


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
    # OHLCV column case varies (real market.db is lower-case; some tables use "Close"),
    # so read all columns and normalize to lower-case before selecting close.
    prices = pd.read_sql('SELECT * FROM "%s" ORDER BY Date' % table,
                         engine, index_col="Date")
    prices.columns = [c.lower() for c in prices.columns]
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
    ap.add_argument("--assets", default=SELECTION,
                    help="comma-separated asset names, or 'all' for the full 181-asset "
                         "universe (default: a 10-asset selection)")
    ap.add_argument("--model", default="tiny",
                    help="Chronos base model: %s, or a full Hugging Face id "
                         "(default: tiny)" % "/".join(CHRONOS_MODELS))
    args = ap.parse_args(argv)
    model = resolve_model(args.model)
    assets = resolve_assets(args.assets)
    print("[chronos] model=%s  assets=%d" % (model, len(assets)))
    engine = create_engine("sqlite:///" + DB_PATH)
    total = 0
    for a in assets:
        n = precompute_asset(a, engine, model=model)
        total += n
        print("[chronos] %s: +%d bars cached" % (a, n))
    print("[chronos] done: %d bars cached across %d assets" % (total, len(assets)))


if __name__ == "__main__":
    sys.exit(main())
