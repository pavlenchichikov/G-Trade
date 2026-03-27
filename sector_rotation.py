"""
sector_rotation.py — Sector rotation analysis for G-Trade.
Analyzes which sectors are growing/declining over recent weeks.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

SECTORS = {
    "Crypto": ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "TON",
               "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR"],
    "US Tech": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "COIN", "MSTR"],
    "US Healthcare": ["JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK"],
    "US Finance": ["JPM", "BAC", "GS", "V", "MA", "WFC"],
    "US Consumer": ["WMT", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX", "SBUX"],
    "US Industrial": ["BA", "CAT", "XOM", "CVX", "COP", "INTC", "QCOM", "AVGO", "MU",
                      "CRM", "ORCL", "ADBE", "UBER", "PYPL"],
    "Indices": ["SP500", "NASDAQ", "DOW"],
    "Commodities": ["GOLD", "SILVER", "OIL", "GAS"],
    "Macro": ["VIX", "DXY", "TNX"],
    "MOEX": [
        "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "PLZL", "SIBN", "MGNT",
        "TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM",
        "YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM", "HHRU", "SOFL", "ASTR", "WUSH",
        "CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP",
        "IRAO", "HYDR", "FLOT", "AFLT", "PIKK", "FEES", "UPRO", "MSNG", "NMTP",
        "PHOR", "SGZH", "FIVE", "FIXP", "LENT", "MVID", "SMLT", "LSRG",
    ],
    "Forex": [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDRUB",
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
        "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADJPY", "CHFJPY", "NZDJPY",
        "USDTRY", "USDMXN", "USDZAR", "USDSGD", "USDNOK", "USDSEK", "USDPLN", "USDCNH",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _engine():
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def _available_tables(engine):
    return set(inspect(engine).get_table_names())


def _load_close(asset: str, engine, days: int = 65) -> pd.Series | None:
    """Load Close series for an asset from market.db. Returns None if missing."""
    table = asset.lower()
    tables = _available_tables(engine)
    if table not in tables:
        return None
    try:
        query = f"SELECT Date, close FROM [{table}] ORDER BY Date DESC LIMIT {days}"
        df = pd.read_sql(query, engine, parse_dates=["Date"])
        df = df.drop_duplicates(subset="Date", keep="last").set_index("Date").sort_index()
        if df.empty or "close" not in df.columns:
            return None
        return df["close"].dropna()
    except Exception:
        return None


def _weekly_returns(series: pd.Series, weeks: int) -> list[float]:
    """
    Compute the last `weeks` weekly percentage returns.
    W1 = most recent completed week, W<weeks> = oldest.
    Returns a list of length `weeks` (NaN where data is missing).
    """
    if series is None or len(series) < 2:
        return [float("nan")] * weeks

    weekly = series.resample("W").last().dropna()
    pct = weekly.pct_change().dropna() * 100  # weekly % returns

    # Build list: most-recent first
    result = []
    for i in range(1, weeks + 1):
        idx = -(i)  # -1 = last week, -2 = two weeks ago …
        if abs(idx) <= len(pct):
            result.append(round(float(pct.iloc[idx]), 4))
        else:
            result.append(float("nan"))
    return result


def _sector_asset_weekly(sector: str, assets: list[str], engine, weeks: int) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by asset with columns W1..W<weeks>.
    Rows for assets with no data are dropped.
    """
    rows = {}
    for asset in assets:
        series = _load_close(asset, engine)
        wr = _weekly_returns(series, weeks)
        rows[asset] = wr
    df = pd.DataFrame(rows, index=[f"W{i}" for i in range(1, weeks + 1)]).T
    df.index.name = "Asset"
    return df.dropna(how="all")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sector_returns(weeks: int = 8) -> pd.DataFrame:
    """
    DataFrame with columns: Sector, W1, W2, ..., W<weeks>
    Values are average weekly returns (%) across all available assets in a sector.
    W1 = most recent week.
    """
    engine = _engine()
    records = []
    for sector, assets in SECTORS.items():
        asset_df = _sector_asset_weekly(sector, assets, engine, weeks)
        if asset_df.empty:
            row = {"Sector": sector, **{f"W{i}": float("nan") for i in range(1, weeks + 1)}}
        else:
            row = {"Sector": sector}
            for i in range(1, weeks + 1):
                col = f"W{i}"
                row[col] = round(float(asset_df[col].mean(skipna=True)), 4)
        records.append(row)

    cols = ["Sector"] + [f"W{i}" for i in range(1, weeks + 1)]
    return pd.DataFrame(records, columns=cols)


def get_sector_momentum(weeks: int = 4) -> pd.DataFrame:
    """
    DataFrame: Sector, Momentum_Score, Trend, Best_Asset, Worst_Asset

    Momentum_Score = average return over last `weeks` weeks (sector average).
    Trend:
      "RISING"  — both W1 and W2 are positive
      "FALLING" — both W1 and W2 are negative
      "MIXED"   — otherwise
    Best/Worst — asset with highest/lowest cumulative return over the period.
    """
    engine = _engine()
    records = []

    for sector, assets in SECTORS.items():
        asset_df = _sector_asset_weekly(sector, assets, engine, weeks)
        if asset_df.empty:
            records.append({
                "Sector": sector,
                "Momentum_Score": float("nan"),
                "Trend": "N/A",
                "Best_Asset": "N/A",
                "Worst_Asset": "N/A",
            })
            continue

        week_cols = [f"W{i}" for i in range(1, weeks + 1)]
        # Sector-level momentum: mean of all asset/week returns
        momentum = float(asset_df[week_cols].mean(skipna=True).mean(skipna=True))

        # Trend: use W1 and W2 of sector averages
        w1 = float(asset_df["W1"].mean(skipna=True)) if "W1" in asset_df.columns else float("nan")
        w2 = float(asset_df["W2"].mean(skipna=True)) if "W2" in asset_df.columns and weeks >= 2 else float("nan")

        if not np.isnan(w1) and not np.isnan(w2):
            if w1 > 0 and w2 > 0:
                trend = "RISING"
            elif w1 < 0 and w2 < 0:
                trend = "FALLING"
            else:
                trend = "MIXED"
        else:
            trend = "N/A"

        # Best / Worst by total return over all weeks
        total = asset_df[week_cols].sum(axis=1, skipna=True)
        best_asset = str(total.idxmax()) if not total.empty else "N/A"
        worst_asset = str(total.idxmin()) if not total.empty else "N/A"

        records.append({
            "Sector": sector,
            "Momentum_Score": round(momentum, 4),
            "Trend": trend,
            "Best_Asset": best_asset,
            "Worst_Asset": worst_asset,
        })

    return pd.DataFrame(records, columns=["Sector", "Momentum_Score", "Trend", "Best_Asset", "Worst_Asset"])


def get_rotation_matrix(weeks: int = 8) -> pd.DataFrame:
    """
    DataFrame ready for heatmap plotting.
    Rows = sectors, Columns = week labels like "Mar 3", "Feb 24".
    Values = sector average weekly returns (%).
    """
    # Build sector returns first
    sr = get_sector_returns(weeks=weeks)

    # Build date labels for columns (most-recent week first)
    today = pd.Timestamp.today().normalize()
    # Find the most recent Monday (start of current week) then go back by weeks
    last_week_end = today - pd.offsets.Week(weekday=6)  # last Sunday
    labels = []
    for i in range(weeks):
        week_end = last_week_end - pd.Timedelta(weeks=i)
        labels.append(week_end.strftime("%b %-d").lstrip("0") if os.name != "nt"
                      else week_end.strftime("%b %d").replace(" 0", " "))

    # Build matrix: rows = sectors, cols = week labels
    matrix = sr.set_index("Sector")
    matrix.columns = labels[: len(matrix.columns)]
    return matrix


def get_asset_returns_by_sector(sector: str, weeks: int = 4) -> pd.DataFrame:
    """
    DataFrame with columns: Asset, W1, W2, ..., W<weeks>, Total
    Individual asset returns within a sector.
    Total = sum of weekly returns over the period.
    """
    assets = SECTORS.get(sector, [])
    if not assets:
        raise ValueError(f"Unknown sector: {sector!r}. Valid sectors: {list(SECTORS)}")

    engine = _engine()
    asset_df = _sector_asset_weekly(sector, assets, engine, weeks)
    if asset_df.empty:
        return pd.DataFrame(columns=["Asset"] + [f"W{i}" for i in range(1, weeks + 1)] + ["Total"])

    week_cols = [f"W{i}" for i in range(1, weeks + 1)]
    asset_df["Total"] = asset_df[week_cols].sum(axis=1, skipna=True).round(4)
    asset_df = asset_df.reset_index()  # Asset becomes a column
    return asset_df[["Asset"] + week_cols + ["Total"]].sort_values("Total", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SECTOR MOMENTUM (last 4 weeks)")
    print("=" * 70)
    momentum_df = get_sector_momentum(weeks=4)
    print(momentum_df.to_string(index=False))

    print()
    print("=" * 70)
    print("SECTOR ROTATION MATRIX (last 8 weeks, avg weekly return %)")
    print("=" * 70)
    matrix_df = get_rotation_matrix(weeks=8)
    print(matrix_df.to_string())
