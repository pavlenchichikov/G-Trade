"""
Watchlist — управление списками избранных активов с текущими ценами.
  python watchlist.py                  — показать default
  python watchlist.py --list crypto    — конкретный список
  python watchlist.py --add BTC GOLD   — добавить
  python watchlist.py --remove VIX     — убрать
  python watchlist.py --create mylist BTC ETH
  python watchlist.py --lists          — все списки
  python watchlist.py --status         — компактный вывод (для GUI)
"""

import os
import sys
import json
import argparse
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
WL_PATH = os.path.join(BASE_DIR, "watchlist.json")

sys.path.insert(0, BASE_DIR)
try:
    from config import FULL_ASSET_MAP
except ImportError:
    FULL_ASSET_MAP = {}

engine = create_engine(f"sqlite:///{DB_PATH}")

DEFAULT_LISTS = {
    "default": ["BTC", "ETH", "SBER", "NVDA", "GOLD", "SP500", "IMOEX"],
    "crypto": ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "TON"],
    "us_tech": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"],
    "russia": ["SBER", "GAZP", "LKOH", "ROSN", "YNDX", "IMOEX"],
    "macro": ["SP500", "NASDAQ", "DOW", "VIX", "DXY", "GOLD", "OIL"],
}


def _load():
    if os.path.exists(WL_PATH):
        with open(WL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_LISTS)


def _save(data):
    with open(WL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _table_name(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _get_asset_data(asset, rows=50):
    """Читает последние N строк из market.db, возвращает price, chg, rsi, trend."""
    table = _table_name(asset)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {table} ORDER BY Date DESC LIMIT {rows}",
            engine, index_col="Date", parse_dates=["Date"]
        )
        if df.empty:
            return None
        df = df.sort_index()
        df.columns = [c.lower() for c in df.columns]
        price = float(df["close"].iloc[-1])
        chg_1d = float(df["close"].pct_change().iloc[-1]) if len(df) > 1 else 0.0

        # RSI 14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        # Trend vs SMA20/SMA50
        sma20 = float(df["close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else price
        sma50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
        if price > sma20 and price > sma50:
            trend = "UPTREND"
        elif price < sma20 and price < sma50:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"

        return {"price": price, "chg_1d": chg_1d, "rsi": rsi, "trend": trend}
    except Exception:
        return None


def show_watchlist(name="default"):
    data = _load()
    assets = data.get(name, [])
    if not assets:
        print(f"  Список '{name}' пуст или не найден.")
        print(f"  Доступные: {', '.join(data.keys())}")
        return

    print()
    print("=" * 60)
    print(f"  WATCHLIST  |  {name} ({len(assets)} assets)")
    print("=" * 60)
    print()
    print(f"  {'Asset':<10}{'Price':>14}  {'Chg 1d':>8}  {'RSI':>5}  {'Trend':<12}")
    print("  " + "-" * 55)

    for asset in assets:
        info = _get_asset_data(asset)
        if info is None:
            print(f"  {asset:<10}{'N/A':>14}  {'N/A':>8}  {'N/A':>5}  {'N/A':<12}")
            continue
        chg_str = f"{info['chg_1d']:+.1%}"
        print(f"  {asset:<10}{info['price']:>14,.2f}  {chg_str:>8}  {info['rsi']:>5.0f}  {info['trend']:<12}")

    print(f"\n  Lists: {', '.join(f'{k}({len(v)})' for k, v in data.items())}")


def show_status():
    """Компактный вывод для GUI."""
    data = _load()
    assets = data.get("default", [])
    print(f"  WATCHLIST  |  default ({len(assets)} assets)")
    for asset in assets:
        info = _get_asset_data(asset)
        if info:
            print(f"  {asset:<8} {info['price']:>12,.2f}  {info['chg_1d']:+.1%}  RSI:{info['rsi']:.0f}  {info['trend']}")


def add_assets(assets, name="default"):
    data = _load()
    lst = data.setdefault(name, [])
    added = []
    for a in assets:
        a_upper = a.upper()
        if a_upper in FULL_ASSET_MAP and a_upper not in lst:
            lst.append(a_upper)
            added.append(a_upper)
        elif a_upper not in FULL_ASSET_MAP:
            print(f"  [!] {a_upper} not in FULL_ASSET_MAP, skipped")
    _save(data)
    if added:
        print(f"  [OK] Added to '{name}': {', '.join(added)}")


def remove_assets(assets, name="default"):
    data = _load()
    lst = data.get(name, [])
    removed = []
    for a in assets:
        a_upper = a.upper()
        if a_upper in lst:
            lst.remove(a_upper)
            removed.append(a_upper)
    _save(data)
    if removed:
        print(f"  [OK] Removed from '{name}': {', '.join(removed)}")


def create_list(name, assets):
    data = _load()
    valid = [a.upper() for a in assets if a.upper() in FULL_ASSET_MAP]
    data[name] = valid
    _save(data)
    print(f"  [OK] Created list '{name}' with {len(valid)} assets")


def show_lists():
    data = _load()
    print("\n  Available watchlists:")
    for name, assets in data.items():
        print(f"    {name}: {', '.join(assets)}")


def main():
    parser = argparse.ArgumentParser(description="Watchlist manager")
    parser.add_argument("--list", default="default", help="List name")
    parser.add_argument("--add", nargs="+", help="Add assets")
    parser.add_argument("--remove", nargs="+", help="Remove assets")
    parser.add_argument("--create", nargs="+", help="Create list: name asset1 asset2 ...")
    parser.add_argument("--lists", action="store_true", help="Show all lists")
    parser.add_argument("--status", action="store_true", help="Compact output for GUI")
    args = parser.parse_args()

    if not os.path.exists(WL_PATH):
        _save(DEFAULT_LISTS)

    if args.status:
        show_status()
    elif args.lists:
        show_lists()
    elif args.add:
        add_assets(args.add, args.list)
    elif args.remove:
        remove_assets(args.remove, args.list)
    elif args.create:
        if len(args.create) < 2:
            print("  Usage: --create name asset1 asset2 ...")
            return
        create_list(args.create[0], args.create[1:])
    else:
        show_watchlist(args.list)


if __name__ == "__main__":
    main()
