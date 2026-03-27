"""
Correlation Alert — детектор смены корреляций между активами.
  python correlation_alert.py            — показать алерты
  python correlation_alert.py --matrix   — матрица ключевых пар
  python correlation_alert.py --json     — JSON вывод
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

sys.path.insert(0, BASE_DIR)
try:
    from config import FULL_ASSET_MAP, ASSET_TYPES
except ImportError:
    FULL_ASSET_MAP = {}
    ASSET_TYPES = {}

engine = create_engine(f"sqlite:///{DB_PATH}")

KEY_ASSETS = ["BTC", "ETH", "SP500", "NASDAQ", "GOLD", "OIL", "VIX", "DXY",
              "SBER", "IMOEX", "NVDA", "TSLA", "EURUSD", "JPM", "XOM"]

KEY_PAIRS = [
    ("BTC", "ETH"), ("BTC", "SP500"), ("BTC", "GOLD"),
    ("GOLD", "DXY"), ("VIX", "SP500"), ("OIL", "GOLD"),
    ("SBER", "IMOEX"), ("NVDA", "SP500"), ("ETH", "SP500"),
    ("EURUSD", "DXY"), ("NASDAQ", "SP500"), ("TSLA", "NASDAQ"),
    ("JPM", "BAC"), ("XOM", "OIL"), ("AVAX", "SOL"),
    ("EURCHF", "EURUSD"), ("GBPAUD", "AUDUSD"), ("USDTRY", "DXY"),
]

SECTOR_GROUPS = {
    "Crypto": ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "DOT", "LINK"],
    "US Tech": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"],
    "US Finance": ["JPM", "BAC", "GS", "V", "MA", "WFC"],
    "US Consumer": ["WMT", "KO", "DIS", "NFLX", "MCD"],
    "Russia": ["SBER", "GAZP", "LKOH", "ROSN", "IMOEX", "YNDX"],
    "Commodities": ["GOLD", "OIL", "SILVER", "GAS"],
}

SHORT_WINDOW = 20
LONG_WINDOW = 60
ALERT_THRESHOLD = 0.40


def _table_name(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _load_returns(assets, days=120):
    """Загружает дневные доходности для списка активов."""
    returns = {}
    for asset in assets:
        table = _table_name(asset)
        try:
            df = pd.read_sql(
                f"SELECT Date, close FROM {table} ORDER BY Date DESC LIMIT {days}",
                engine, parse_dates=["Date"]
            )
            if len(df) < 30:
                continue
            df = df.sort_values("Date").set_index("Date")
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")]
            returns[asset] = df["close"].pct_change().dropna()
        except Exception:
            continue
    if not returns:
        return pd.DataFrame()
    return pd.DataFrame(returns).dropna(how="all")


def get_correlation_alerts(threshold=ALERT_THRESHOLD):
    """Ищет пары с большим изменением корреляции (short vs long window)."""
    all_assets = list(set(a for p in KEY_PAIRS for a in p))
    ret_df = _load_returns(all_assets)
    if ret_df.empty:
        return []

    alerts = []
    for a1, a2 in KEY_PAIRS:
        if a1 not in ret_df.columns or a2 not in ret_df.columns:
            continue
        s1, s2 = ret_df[a1], ret_df[a2]

        corr_short = s1.tail(SHORT_WINDOW).corr(s2.tail(SHORT_WINDOW))
        corr_long = s1.tail(LONG_WINDOW).corr(s2.tail(LONG_WINDOW))

        if pd.isna(corr_short) or pd.isna(corr_long):
            continue

        change = corr_short - corr_long
        abs_change = abs(change)

        if abs_change >= threshold:
            if change > 0:
                signal = "CONVERGING"
            else:
                signal = "DIVERGING"
        else:
            signal = "STABLE"

        alerts.append({
            "pair": f"{a1}-{a2}",
            "a1": a1, "a2": a2,
            "corr_short": float(corr_short),
            "corr_long": float(corr_long),
            "change": float(change),
            "signal": signal,
        })

    alerts.sort(key=lambda x: abs(x["change"]), reverse=True)
    return alerts


def get_stress_indicator():
    """Средняя попарная корреляция — индикатор рыночного стресса."""
    ret_df = _load_returns(KEY_ASSETS)
    if ret_df.shape[1] < 3:
        return {"avg_corr": 0, "label": "N/A", "min_corr": 0, "max_corr": 0}

    corr = ret_df.tail(SHORT_WINDOW).corr()
    # Верхний треугольник без диагонали
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    vals = corr.where(mask).stack().values

    avg = float(np.nanmean(vals))
    mn = float(np.nanmin(vals))
    mx = float(np.nanmax(vals))

    if avg > 0.6:
        label = "HIGH (crisis-like)"
    elif avg > 0.4:
        label = "ELEVATED"
    elif avg > 0.2:
        label = "NORMAL"
    else:
        label = "LOW (dispersed)"

    return {"avg_corr": avg, "label": label, "min_corr": mn, "max_corr": mx}


def get_key_pairs_matrix():
    """Корреляционная матрица для ключевых активов."""
    ret_df = _load_returns(KEY_ASSETS)
    if ret_df.shape[1] < 2:
        return pd.DataFrame()
    return ret_df.tail(SHORT_WINDOW).corr().round(2)


def get_sector_correlations():
    """Средняя внутренняя корреляция по секторам и между секторами."""
    result = {}
    sector_returns = {}

    for sector, assets in SECTOR_GROUPS.items():
        ret_df = _load_returns(assets, days=60)
        if ret_df.shape[1] < 2:
            result[sector] = {"internal": 0, "label": "N/A"}
            continue

        corr = ret_df.tail(SHORT_WINDOW).corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        vals = corr.where(mask).stack().values
        avg = float(np.nanmean(vals)) if len(vals) > 0 else 0

        if avg > 0.7:
            label = "HIGH"
        elif avg > 0.4:
            label = "MODERATE"
        else:
            label = "LOW"

        result[sector] = {"internal": avg, "label": label}
        # Среднюю доходность сектора для cross-sector
        sector_returns[sector] = ret_df.mean(axis=1)

    # Cross-sector correlations
    if len(sector_returns) >= 2:
        sr_df = pd.DataFrame(sector_returns).tail(SHORT_WINDOW)
        cross = sr_df.corr()
        sectors = list(cross.columns)
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                key = f"{sectors[i]} vs {sectors[j]}"
                val = float(cross.iloc[i, j])
                result[key] = {"cross": val, "label": "HIGH" if val > 0.5 else "MODERATE" if val > 0.2 else "LOW"}

    return result


def print_report():
    print()
    print("=" * 60)
    print("  CORRELATION ALERT  |  Key Pairs")
    print("=" * 60)

    # Stress
    stress = get_stress_indicator()
    print()
    print("  [STRESS INDICATOR]")
    print("  " + "-" * 55)
    print(f"  Avg correlation:    {stress['avg_corr']:.2f} ({stress['label']})")
    print(f"  Range:              {stress['min_corr']:.2f} to {stress['max_corr']:.2f}")

    # Alerts
    alerts = get_correlation_alerts()
    significant = [a for a in alerts if a["signal"] != "STABLE"]
    print()
    print(f"  [ALERTS]  ({len(significant)} significant changes)")
    print("  " + "-" * 55)
    print(f"  {'Pair':<16}{'Short(20d)':>10}  {'Long(60d)':>10}  {'Change':>8}  {'Signal':<14}")
    for a in alerts:
        marker = " *" if a["signal"] != "STABLE" else ""
        print(f"  {a['pair']:<16}{a['corr_short']:>+10.2f}  {a['corr_long']:>+10.2f}  "
              f"{a['change']:>+8.2f}  {a['signal']:<14}{marker}")

    # Matrix
    matrix = get_key_pairs_matrix()
    if not matrix.empty and matrix.shape[0] <= 15:
        print()
        print("  [KEY ASSETS MATRIX]  (20d)")
        print("  " + "-" * 55)
        cols = matrix.columns.tolist()
        header = "  " + " " * 8 + "".join(f"{c:>8}" for c in cols)
        print(header)
        for idx in matrix.index:
            row = f"  {idx:<8}" + "".join(f"{matrix.loc[idx, c]:>8.2f}" for c in cols)
            print(row)

    # Sectors
    sectors = get_sector_correlations()
    print()
    print("  [SECTOR CORRELATIONS]")
    print("  " + "-" * 55)
    for name, info in sectors.items():
        if "internal" in info:
            print(f"  {name:<25} internal: {info['internal']:+.2f} ({info['label']})")
        elif "cross" in info:
            print(f"  {name:<25} cross:    {info['cross']:+.2f} ({info['label']})")


def main():
    parser = argparse.ArgumentParser(description="Correlation Alert")
    parser.add_argument("--matrix", action="store_true", help="Show correlation matrix")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.json:
        out = {
            "stress": get_stress_indicator(),
            "alerts": get_correlation_alerts(),
            "sectors": get_sector_correlations(),
        }
        print(json.dumps(out, indent=2, default=str))
    elif args.matrix:
        matrix = get_key_pairs_matrix()
        if matrix.empty:
            print("  Not enough data for matrix")
        else:
            print(matrix.to_string())
    else:
        print_report()


if __name__ == "__main__":
    main()
