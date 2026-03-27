"""
whatif_simulator.py — G-Trade
================================================
"What if I invested $X in these assets N days ago using the model's signals?"

Quick hypothetical backtest using CatBoost signals only (no LSTM — speed).

Usage:
  python whatif_simulator.py BTC ETH NVDA --capital 10000 --days 90
  python whatif_simulator.py --top 5 --capital 10000 --days 90
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "market.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
QUALITY_PATH = os.path.join(BASE_DIR, "quality_report.json")

COMMISSION = 0.001   # 0.1%
SLIPPAGE = 0.0015    # 0.15%
COST = COMMISSION + SLIPPAGE  # per trade side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_quality_report() -> dict:
    if not os.path.exists(QUALITY_PATH):
        return {}
    with open(QUALITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _asset_to_table(asset: str) -> str:
    """Convert asset name to SQLite table name — must match data_engine.py convention."""
    try:
        from config import FULL_ASSET_MAP
        ticker = FULL_ASSET_MAP.get(asset, asset)
    except ImportError:
        ticker = asset
    # data_engine uses: lower + strip ^ . -  (NOT "=")  — e.g. DX=F → "dx=f"
    return ticker.lower().replace("^", "").replace(".", "").replace("-", "")


def _load_prices(asset: str, days_back: int, engine) -> pd.DataFrame | None:
    """Load last `days_back` daily rows from market.db."""
    table = _asset_to_table(asset)
    try:
        df = pd.read_sql(
            f"SELECT * FROM \"{table}\" ORDER BY Date DESC LIMIT {days_back + 300}",
            engine,
            index_col="Date",
            parse_dates=["Date"],
        )
    except Exception as e:
        print(f"  [SKIP] {asset}: DB read error — {e}")
        return None

    if df.empty:
        print(f"  [SKIP] {asset}: no data in DB")
        return None

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _predict_cb(asset: str, df_full: pd.DataFrame, days_back: int, engine) -> pd.DataFrame | None:
    """
    Engineer features, load CatBoost model, predict probabilities.
    Returns DataFrame with columns [close, prob] for the last `days_back` rows.
    """
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from train_hybrid import engineer_features, add_weekly_features

    table = _asset_to_table(asset)
    model_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
    if not os.path.exists(model_path):
        print(f"  [SKIP] {asset}: model not found at {model_path}")
        return None

    # Feature engineering on full history
    try:
        df_feat = engineer_features(df_full.copy())
        df_feat = add_weekly_features(df_feat, table, engine)
    except Exception as e:
        print(f"  [SKIP] {asset}: feature engineering failed — {e}")
        return None

    # Restore DatetimeIndex
    if "Date" in df_feat.columns:
        df_feat = df_feat.set_index("Date")
    elif "date" in df_feat.columns:
        df_feat = df_feat.set_index("date")
    df_feat.index = pd.to_datetime(df_feat.index)

    # Get feature list from registry
    registry = _load_registry()
    feat_list = None
    if asset in registry:
        feat_list = registry[asset].get("features")
    if not feat_list:
        # Fallback: use all numeric columns except target/next_ret
        drop_cols = {"target", "next_ret", "close", "open", "high", "low", "volume"}
        feat_list = [c for c in df_feat.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])]

    # Keep only available features
    feat_list = [f for f in feat_list if f in df_feat.columns]
    if not feat_list:
        print(f"  [SKIP] {asset}: no usable features")
        return None

    df_feat = df_feat.dropna(subset=feat_list)

    # Train/predict split: scaler fit on first 80%
    n = len(df_feat)
    split = max(int(n * 0.8), n - days_back - 10)
    split = min(split, n - days_back)
    if split <= 0:
        split = max(1, n - days_back)

    X = df_feat[feat_list].values
    X_train = X[:split]
    X_pred = X[split:]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Load and predict
    try:
        cb = CatBoostClassifier()
        cb.load_model(model_path)
        probs = cb.predict_proba(X_pred_scaled)[:, 1]
    except Exception as e:
        print(f"  [SKIP] {asset}: CatBoost predict failed — {e}")
        return None

    result = df_feat.iloc[split:][["close"]].copy()
    result["prob"] = probs

    # Return only the last `days_back` rows
    return result.iloc[-days_back:]


def _run_strategy(prices_probs: pd.DataFrame, capital: float) -> dict:
    """
    Simple long-only strategy:
      prob > 0.55 → buy next day
      prob < 0.45 → sell (flat)
      else        → hold

    Tracks equity, counts trades, computes Sharpe + max drawdown.
    Returns per-asset result dict.
    """
    closes = prices_probs["close"].values
    probs = prices_probs["prob"].values
    dates = prices_probs.index
    n = len(closes)

    equity = capital
    position = 0.0        # units held
    entry_price = 0.0
    trade_count = 0

    equity_curve = [(dates[0].strftime("%Y-%m-%d"), equity)]
    daily_returns = []

    for i in range(n - 1):
        p_today = probs[i]
        close_today = closes[i]
        close_next = closes[i + 1]

        prev_equity = equity

        if p_today > 0.55 and position == 0.0:
            # Enter long at next day open ≈ today close + slippage
            entry_price = close_today * (1 + SLIPPAGE)
            units = (equity * (1 - COMMISSION)) / entry_price
            position = units
            equity -= equity * COMMISSION  # commission on entry
            trade_count += 1

        elif p_today < 0.45 and position > 0.0:
            # Exit long
            exit_price = close_today * (1 - SLIPPAGE)
            equity = position * exit_price * (1 - COMMISSION)
            position = 0.0
            trade_count += 1

        # Mark to market
        if position > 0.0:
            equity = position * close_next * (1 - COMMISSION)

        equity_curve.append((dates[i + 1].strftime("%Y-%m-%d"), round(equity, 4)))
        ret = (equity - prev_equity) / (prev_equity + 1e-9)
        daily_returns.append(ret)

    # Close any open position at end
    if position > 0.0:
        exit_price = closes[-1] * (1 - SLIPPAGE)
        equity = position * exit_price * (1 - COMMISSION)
        equity_curve[-1] = (equity_curve[-1][0], round(equity, 4))

    # Metrics
    rets = np.array(daily_returns)
    sharpe = 0.0
    if rets.std() > 1e-9:
        sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 3)

    eq_vals = np.array([v for _, v in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / (peak + 1e-9)
    max_dd = round(float(dd.max() * 100), 2)

    return_pct = round((equity - capital) / capital * 100, 2)

    return {
        "initial": round(capital, 2),
        "final": round(equity, 2),
        "return_pct": return_pct,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": trade_count,
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    assets: list[str],
    capital: float = 10000.0,
    days_back: int = 90,
    strategy: str = "equal",
) -> dict:
    """
    Run what-if simulation for given assets.

    Parameters
    ----------
    assets     : list of asset names, e.g. ["BTC", "ETH", "NVDA"]
    capital    : total starting capital (USD)
    days_back  : number of trading days to look back
    strategy   : "equal" — split capital equally among assets
                 "kelly"  — weight by Score from quality_report.json

    Returns
    -------
    dict with keys: initial, final, return_pct, max_drawdown, sharpe,
                    trades, equity_curve, per_asset
    """
    if not assets:
        return {"error": "No assets provided"}

    engine = create_engine(f"sqlite:///{DB_PATH}")

    # Determine per-asset capital allocation
    alloc = _get_allocation(assets, capital, strategy)

    per_asset: dict[str, dict] = {}
    portfolio_curves: dict[str, list[float]] = {}  # date → equity contributions

    for asset in assets:
        asset_capital = alloc.get(asset, capital / len(assets))
        print(f"  Simulating {asset} (${asset_capital:,.0f})…")

        df_full = _load_prices(asset, days_back, engine)
        if df_full is None:
            continue

        prices_probs = _predict_cb(asset, df_full, days_back, engine)
        if prices_probs is None or prices_probs.empty:
            continue

        result = _run_strategy(prices_probs, asset_capital)
        per_asset[asset] = {
            "return_pct": result["return_pct"],
            "trades": result["trades"],
            "max_dd": result["max_drawdown"],
        }

        for date_str, eq_val in result["equity_curve"]:
            if date_str not in portfolio_curves:
                portfolio_curves[date_str] = 0.0
            portfolio_curves[date_str] += eq_val

    if not per_asset:
        return {"error": "No assets could be simulated (missing data or models)"}

    # Portfolio aggregates
    sorted_dates = sorted(portfolio_curves.keys())
    equity_curve = [(d, round(portfolio_curves[d], 4)) for d in sorted_dates]

    initial = sum(alloc.get(a, capital / len(assets)) for a in per_asset)
    final = equity_curve[-1][1] if equity_curve else initial

    eq_vals = np.array([v for _, v in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / (peak + 1e-9)
    max_dd = round(float(dd.max() * 100), 2)

    all_rets = np.diff(eq_vals) / (eq_vals[:-1] + 1e-9)
    sharpe = 0.0
    if len(all_rets) > 1 and all_rets.std() > 1e-9:
        sharpe = round(float(all_rets.mean() / all_rets.std() * np.sqrt(252)), 3)

    total_trades = sum(v["trades"] for v in per_asset.values())
    return_pct = round((final - initial) / (initial + 1e-9) * 100, 2)

    return {
        "initial": round(initial, 2),
        "final": round(final, 2),
        "return_pct": return_pct,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": total_trades,
        "equity_curve": equity_curve,
        "per_asset": per_asset,
    }


def simulate_top_n(n: int = 5, capital: float = 10000.0, days_back: int = 90) -> dict:
    """
    Auto-select top N assets by Score from quality_report.json, then simulate.
    Falls back to registry if quality_report is unavailable.
    """
    report = _load_quality_report()
    if report:
        # quality_report: {asset: {Score: float, ...}, ...}
        scored = [(a, v.get("Score", 0.0)) for a, v in report.items() if isinstance(v, dict)]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_assets = [a for a, _ in scored[:n]]
    else:
        # Fallback: take first N from champion registry
        registry = _load_registry()
        top_assets = list(registry.keys())[:n]

    if not top_assets:
        return {"error": "quality_report.json and champion_registry.json are empty or missing"}

    print(f"  Top-{n} assets selected: {top_assets}")
    return simulate(top_assets, capital=capital, days_back=days_back, strategy="equal")


def _get_allocation(assets: list[str], capital: float, strategy: str) -> dict[str, float]:
    """Return {asset: capital_share} based on strategy."""
    if strategy == "kelly":
        report = _load_quality_report()
        if report:
            scores = {a: report.get(a, {}).get("Score", 0.0) for a in assets if isinstance(report.get(a), dict)}
            total_score = sum(scores.values())
            if total_score > 0:
                return {a: capital * (scores.get(a, 0.0) / total_score) for a in assets}
        # Fallback to equal if no scores
    # Equal split
    share = capital / max(len(assets), 1)
    return {a: share for a in assets}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        return

    print("\n" + "=" * 58)
    print("  G-TRADE — WHAT-IF SIMULATOR RESULTS")
    print("=" * 58)
    print(f"  Initial capital : ${result['initial']:>12,.2f}")
    print(f"  Final equity    : ${result['final']:>12,.2f}")
    sign = "+" if result["return_pct"] >= 0 else ""
    print(f"  Return          : {sign}{result['return_pct']:>10.2f}%")
    print(f"  Max drawdown    : {result['max_drawdown']:>10.2f}%")
    print(f"  Sharpe ratio    : {result['sharpe']:>10.3f}")
    print(f"  Total trades    : {result['trades']:>10d}")
    print("-" * 58)
    print(f"  {'Asset':<12} {'Return%':>8}  {'Trades':>6}  {'MaxDD%':>7}")
    print("-" * 58)
    for asset, d in result.get("per_asset", {}).items():
        sign2 = "+" if d["return_pct"] >= 0 else ""
        print(f"  {asset:<12} {sign2}{d['return_pct']:>7.2f}%  {d['trades']:>6}  {d['max_dd']:>6.2f}%")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(
        description="G-Trade — What-If Simulator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "assets",
        nargs="*",
        metavar="ASSET",
        help="Asset names (e.g. BTC ETH NVDA). Ignored if --top is used.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Auto-select top N assets by Score from quality_report.json",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        metavar="USD",
        help="Starting capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        metavar="N",
        help="Number of trading days to simulate (default: 90)",
    )
    parser.add_argument(
        "--strategy",
        choices=["equal", "kelly"],
        default="equal",
        help="Capital allocation strategy (default: equal)",
    )
    args = parser.parse_args()

    print(f"\nG-Trade What-If Simulator")
    print(f"  Capital: ${args.capital:,.0f} | Days back: {args.days} | Strategy: {args.strategy}")

    if args.top > 0:
        print(f"  Mode: Top-{args.top} by quality score")
        result = simulate_top_n(n=args.top, capital=args.capital, days_back=args.days)
    elif args.assets:
        print(f"  Mode: Custom assets — {args.assets}")
        result = simulate(
            assets=args.assets,
            capital=args.capital,
            days_back=args.days,
            strategy=args.strategy,
        )
    else:
        parser.print_help()
        sys.exit(0)

    _print_summary(result)


if __name__ == "__main__":
    main()
