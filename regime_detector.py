# regime_detector.py
# Market Regime Detector -- classifies global and per-asset market regimes
# using data from SQLite market.db.

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

from config import FULL_ASSET_MAP

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def _read_table(table: str, engine=None, min_rows: int = 1) -> pd.DataFrame | None:
    """Read a table from market.db, return DataFrame indexed by Date or None."""
    if engine is None:
        engine = _get_engine()
    try:
        insp = inspect(engine)
        if table.lower() not in [t.lower() for t in insp.get_table_names()]:
            return None
        df = pd.read_sql(f"SELECT * FROM [{table}]", engine, parse_dates=["Date"])
        if df.empty or len(df) < min_rows:
            return None
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception:
        return None


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Global Regime
# ---------------------------------------------------------------------------

def _classify_vix(value: float) -> str:
    if value < 15:
        return "CALM"
    elif value < 25:
        return "NORMAL"
    elif value < 35:
        return "FEAR"
    else:
        return "PANIC"


def _classify_sp500_trend(df: pd.DataFrame) -> str:
    if df is None or len(df) < 200:
        return "UNKNOWN"
    close = df["close"]
    sma50 = _sma(close, 50).iloc[-1]
    sma200 = _sma(close, 200).iloc[-1]
    price = close.iloc[-1]
    if price > sma50 and price > sma200:
        if sma50 > sma200:
            return "BULLISH"
        return "ABOVE BOTH SMAs"
    elif price < sma50 and price < sma200:
        if sma50 < sma200:
            return "BEARISH"
        return "BELOW BOTH SMAs"
    else:
        return "MIXED"


def _sp500_detail(df: pd.DataFrame) -> str:
    if df is None or len(df) < 200:
        return "Insufficient data"
    close = df["close"]
    price = close.iloc[-1]
    sma50 = _sma(close, 50).iloc[-1]
    sma200 = _sma(close, 200).iloc[-1]
    parts = []
    parts.append("Above SMA50" if price > sma50 else "Below SMA50")
    parts.append("Above SMA200" if price > sma200 else "Below SMA200")
    return ", ".join(parts)


def _classify_dxy_trend(df: pd.DataFrame) -> str:
    if df is None or len(df) < 50:
        return "UNKNOWN"
    close = df["close"]
    sma20 = _sma(close, 20).iloc[-1]
    price = close.iloc[-1]
    if price > sma20:
        return "RISING"
    else:
        return "FALLING"


def _dxy_label(trend: str) -> str:
    if trend == "RISING":
        return "STRONG DOLLAR"
    elif trend == "FALLING":
        return "WEAK DOLLAR"
    return "UNKNOWN"


def _combined_global_status(vix_level: str, sp500_trend: str, dxy_trend: str) -> tuple[str, str]:
    """Return (status, description)."""
    if vix_level in ("PANIC", "FEAR") and sp500_trend in ("BEARISH", "BELOW BOTH SMAs"):
        return "CRISIS", "High volatility with bearish equities -- extreme caution"
    if vix_level == "PANIC":
        return "CRISIS", "VIX in panic territory -- flight to safety"
    if sp500_trend == "BEARISH" and dxy_trend == "RISING":
        return "RISK-OFF", "Bearish equities with strong dollar -- defensive positioning"
    if sp500_trend in ("BEARISH", "BELOW BOTH SMAs"):
        return "RISK-OFF", "Equities trending down -- risk-off environment"
    if vix_level in ("CALM", "NORMAL") and sp500_trend in ("BULLISH", "ABOVE BOTH SMAs"):
        if dxy_trend == "FALLING":
            return "RISK-ON", "Low vol, bullish equities, weak dollar -- full risk-on"
        return "RISK-ON", "Low vol with bullish equities -- favorable conditions"
    if sp500_trend == "MIXED":
        return "SIDEWAYS", "Mixed signals across indicators -- no clear direction"
    return "SIDEWAYS", "No dominant regime detected"


def get_global_regime() -> dict:
    """Return global market regime assessment.

    Keys: status, vix_level, vix_value, sp500_trend, sp500_detail,
          dxy_trend, dxy_label, description
    """
    engine = _get_engine()

    # VIX
    vix_df = _read_table("vix", engine, min_rows=5)
    if vix_df is not None:
        vix_value = float(vix_df["close"].iloc[-1])
        vix_level = _classify_vix(vix_value)
    else:
        vix_value = None
        vix_level = "UNKNOWN"

    # SP500
    sp500_df = _read_table("sp500", engine, min_rows=200)
    sp500_trend = _classify_sp500_trend(sp500_df)
    sp500_det = _sp500_detail(sp500_df) if sp500_df is not None else "No data"

    # DXY
    dxy_df = _read_table("dxy", engine, min_rows=50)
    dxy_trend = _classify_dxy_trend(dxy_df)
    dxy_lab = _dxy_label(dxy_trend)

    status, description = _combined_global_status(vix_level, sp500_trend, dxy_trend)

    return {
        "status": status,
        "vix_level": vix_level,
        "vix_value": vix_value,
        "sp500_trend": sp500_trend,
        "sp500_detail": sp500_det,
        "dxy_trend": dxy_trend,
        "dxy_label": dxy_lab,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Per-Asset Regime
# ---------------------------------------------------------------------------

def _classify_trend(close: pd.Series) -> str:
    if len(close) < 50:
        if len(close) >= 20:
            sma20 = _sma(close, 20).iloc[-1]
            price = close.iloc[-1]
            return "UPTREND" if price > sma20 else "DOWNTREND"
        return "UNKNOWN"
    price = close.iloc[-1]
    sma20 = _sma(close, 20).iloc[-1]
    sma50 = _sma(close, 50).iloc[-1]
    if price > sma20 and price > sma50:
        return "UPTREND"
    elif price < sma20 and price < sma50:
        return "DOWNTREND"
    return "SIDEWAYS"


def _classify_volatility(atr_current: float, atr_avg_90: float) -> str:
    if atr_avg_90 == 0 or np.isnan(atr_avg_90):
        return "UNKNOWN"
    ratio = atr_current / atr_avg_90
    if ratio < 0.7:
        return "LOW_VOL"
    elif ratio < 1.3:
        return "NORMAL"
    elif ratio < 2.0:
        return "HIGH_VOL"
    else:
        return "EXTREME"


def _classify_momentum(rsi_value: float) -> str:
    if np.isnan(rsi_value):
        return "UNKNOWN"
    if rsi_value < 30:
        return "OVERSOLD"
    elif rsi_value > 70:
        return "OVERBOUGHT"
    return "NEUTRAL"


def get_asset_regime(asset: str) -> dict | None:
    """Return regime for a single asset.

    Keys: trend, volatility, momentum, rsi, atr_ratio
    Returns None if data is insufficient.
    """
    engine = _get_engine()
    table = asset.lower()
    df = _read_table(table, engine, min_rows=20)
    if df is None:
        return None

    close = df["close"]
    trend = _classify_trend(close)

    # Volatility via ATR
    atr_series = _atr(df, 14)
    if atr_series.dropna().empty:
        vol_label = "UNKNOWN"
        atr_ratio = None
    else:
        atr_current = atr_series.iloc[-1]
        atr_90 = atr_series.tail(90).mean() if len(atr_series.dropna()) >= 90 else atr_series.dropna().mean()
        vol_label = _classify_volatility(atr_current, atr_90)
        atr_ratio = round(float(atr_current / atr_90), 2) if atr_90 and atr_90 != 0 else None

    # Momentum via RSI
    rsi_series = _rsi(close, 14)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.dropna().empty else float("nan")
    momentum = _classify_momentum(rsi_val)

    return {
        "trend": trend,
        "volatility": vol_label,
        "momentum": momentum,
        "rsi": round(rsi_val, 1) if not np.isnan(rsi_val) else None,
        "atr_ratio": atr_ratio,
    }


# ---------------------------------------------------------------------------
# Market Breadth
# ---------------------------------------------------------------------------

def get_market_breadth() -> dict:
    """Return market breadth statistics.

    Keys: above_sma50_pct, positive_20d_pct, above_sma50_count,
          positive_20d_count, total_assets, score
    """
    engine = _get_engine()
    above_sma50 = 0
    positive_20d = 0
    total = 0

    for asset in FULL_ASSET_MAP:
        table = asset.lower()
        df = _read_table(table, engine, min_rows=50)
        if df is None:
            continue
        close = df["close"]
        total += 1

        # Above SMA50
        sma50 = _sma(close, 50)
        if not sma50.dropna().empty and close.iloc[-1] > sma50.iloc[-1]:
            above_sma50 += 1

        # Positive 20-day return
        if len(close) >= 20:
            ret_20 = (close.iloc[-1] / close.iloc[-20]) - 1
            if ret_20 > 0:
                positive_20d += 1

    if total == 0:
        return {
            "above_sma50_pct": 0.0,
            "positive_20d_pct": 0.0,
            "above_sma50_count": 0,
            "positive_20d_count": 0,
            "total_assets": 0,
            "score": "UNKNOWN",
        }

    above_pct = round(100 * above_sma50 / total, 1)
    pos_pct = round(100 * positive_20d / total, 1)

    avg_pct = (above_pct + pos_pct) / 2
    if avg_pct >= 60:
        score = "STRONG"
    elif avg_pct >= 40:
        score = "MODERATE"
    else:
        score = "WEAK"

    return {
        "above_sma50_pct": above_pct,
        "positive_20d_pct": pos_pct,
        "above_sma50_count": above_sma50,
        "positive_20d_count": positive_20d,
        "total_assets": total,
        "score": score,
    }


# ---------------------------------------------------------------------------
# All Regimes
# ---------------------------------------------------------------------------

def get_all_regimes() -> dict:
    """Return complete regime snapshot: global, breadth, and per-asset."""
    global_regime = get_global_regime()
    breadth = get_market_breadth()

    assets = {}
    for asset in FULL_ASSET_MAP:
        regime = get_asset_regime(asset)
        if regime is not None:
            assets[asset] = regime

    return {
        "global": global_regime,
        "breadth": breadth,
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# Formatted Report
# ---------------------------------------------------------------------------

def _format_report(data: dict) -> str:
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append("  MARKET REGIME DETECTOR")
    lines.append(sep)
    lines.append("")

    # -- Global --
    g = data["global"]
    lines.append("[GLOBAL REGIME]")
    lines.append(f"  Status:     {g['status']}")
    vix_str = f"{g['vix_value']:.1f} ({g['vix_level']})" if g["vix_value"] is not None else "N/A"
    lines.append(f"  VIX:        {vix_str}")
    sp_str = f"{g['sp500_detail']} ({g['sp500_trend']})"
    lines.append(f"  SP500:      {sp_str}")
    dxy_str = f"{g['dxy_trend']} ({g['dxy_label']})"
    lines.append(f"  DXY:        {dxy_str}")
    lines.append(f"  Note:       {g['description']}")
    lines.append("")

    # -- Breadth --
    b = data["breadth"]
    lines.append("[MARKET BREADTH]")
    lines.append(f"  Above SMA50:    {b['above_sma50_pct']:.0f}% ({b['above_sma50_count']}/{b['total_assets']} assets)")
    lines.append(f"  Positive 20d:   {b['positive_20d_pct']:.0f}% ({b['positive_20d_count']}/{b['total_assets']} assets)")
    lines.append(f"  Breadth:        {b['score']}")
    lines.append("")

    # -- Per-Asset --
    assets = data.get("assets", {})
    if assets:
        lines.append("[ASSET REGIMES]")
        header = f"  {'Asset':<12} {'Trend':<12} {'Volatility':<12} {'Momentum':<12} {'RSI':>6}"
        lines.append(header)
        lines.append("  " + "-" * 56)
        for name, r in sorted(assets.items()):
            rsi_str = f"{r['rsi']:.0f}" if r["rsi"] is not None else "N/A"
            line = f"  {name:<12} {r['trend']:<12} {r['volatility']:<12} {r['momentum']:<12} {rsi_str:>6}"
            lines.append(line)
    else:
        lines.append("[ASSET REGIMES]")
        lines.append("  No asset data available.")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Market Regime Detector")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    data = get_all_regimes()

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(_format_report(data))


if __name__ == "__main__":
    main()
