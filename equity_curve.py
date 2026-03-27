"""
Equity Curve — график капитала по результатам Paper Trading / Backtest.
Открывает окно matplotlib с интерактивным графиком.
  python equity_curve.py               — из paper trading
  python equity_curve.py --backtest    — из backtest результатов
"""

import os
import sys
import argparse
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DB = os.path.join(BASE_DIR, "paper.db")
MARKET_DB = os.path.join(BASE_DIR, "market.db")

sys.path.insert(0, BASE_DIR)


def _get_paper_equity():
    """Equity из paper trading (закрытые сделки)."""
    if not os.path.exists(PAPER_DB):
        return None

    conn = sqlite3.connect(PAPER_DB)
    cur = conn.cursor()

    # Проверяем таблицы
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "positions" not in tables:
        conn.close()
        return None

    # Начальный баланс
    initial = 10000.0
    try:
        row = cur.execute(
            "SELECT value FROM portfolio_config WHERE key='initial_balance'"
        ).fetchone()
        if row:
            initial = float(row[0])
    except Exception:
        pass

    # Закрытые сделки по дате
    trades = cur.execute(
        "SELECT exit_date, pnl FROM positions WHERE status='CLOSED' AND exit_date IS NOT NULL ORDER BY exit_date"
    ).fetchall()
    conn.close()

    if not trades:
        return None

    equity = initial
    points = [{"date": "Start", "equity": initial}]
    for date, pnl in trades:
        equity += (pnl or 0)
        points.append({"date": date[:10] if date else "?", "equity": equity})

    return pd.DataFrame(points)


def _get_backtest_equity():
    """Equity из backtest результатов (если есть experiments/backtest_results.json)."""
    # Попробуем прочитать из models/experiments
    exp_dir = os.path.join(BASE_DIR, "models", "experiments")
    results_file = os.path.join(exp_dir, "backtest_results.json")

    if os.path.exists(results_file):
        import json
        with open(results_file, "r") as f:
            data = json.load(f)
        if "equity_curve" in data:
            return pd.DataFrame(data["equity_curve"])

    # Альтернативно — сгенерировать из market.db используя простую стратегию
    # на основе champion_registry scores
    registry_path = os.path.join(BASE_DIR, "models", "champion_registry.json")
    if not os.path.exists(registry_path):
        return None

    import json
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{MARKET_DB}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Берём топ-5 активов по score
    scored = [(k, v.get("score", 0)) for k, v in registry.items() if isinstance(v.get("score"), (int, float))]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_assets = [k for k, _ in scored[:5]]

    if not top_assets:
        return None

    # Простой equity: равные доли, buy-and-hold за последние 120 дней
    initial = 10000.0
    share = initial / len(top_assets)
    all_returns = []

    for asset in top_assets:
        table = asset.lower().replace("^", "").replace(".", "").replace("-", "")
        try:
            df = pd.read_sql(
                f"SELECT Date, close FROM {table} ORDER BY Date DESC LIMIT 120",
                engine, parse_dates=["Date"]
            )
            if len(df) < 10:
                continue
            df = df.sort_values("Date").set_index("Date")
            df = df[~df.index.duplicated(keep="last")]
            rets = df["close"].pct_change().dropna()
            all_returns.append(rets * (share / initial))
        except Exception:
            continue

    if not all_returns:
        return None

    combined = pd.concat(all_returns, axis=1).sum(axis=1)
    equity_series = (1 + combined).cumprod() * initial
    result = pd.DataFrame({
        "date": equity_series.index.astype(str),
        "equity": equity_series.values,
    })
    return result


def show_equity_curve(source="paper"):
    if source == "backtest":
        df = _get_backtest_equity()
        title = "Equity Curve (Backtest / Top-5 Assets)"
    else:
        df = _get_paper_equity()
        title = "Equity Curve (Paper Trading)"

    if df is None or df.empty:
        print(f"  No equity data available for source='{source}'.")
        print("  Paper Trading: make some trades first.")
        print("  Backtest: run backtest or train models first.")
        return

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates  # noqa: F401
    except ImportError:
        # Fallback: text output
        print(f"\n  {title}")
        print(f"  {'='*40}")
        for _, row in df.iterrows():
            bar_len = max(0, int((row["equity"] / df["equity"].iloc[0] - 0.5) * 40))
            print(f"  {row['date']:<12} ${row['equity']:>10,.2f}  {'|' * bar_len}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#111827")

    x = range(len(df))
    y = df["equity"].values
    initial = y[0]

    # Цвет линии зависит от результата
    final_color = "#22c55e" if y[-1] >= initial else "#ef4444"
    ax.plot(x, y, color=final_color, linewidth=2)
    ax.fill_between(x, initial, y, where=(y >= initial), alpha=0.15, color="#22c55e")
    ax.fill_between(x, initial, y, where=(y < initial), alpha=0.15, color="#ef4444")
    ax.axhline(y=initial, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.5)

    # Метки
    ax.set_title(title, color="#e2e8f0", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trades" if source == "paper" else "Days", color="#64748b")
    ax.set_ylabel("Equity ($)", color="#64748b")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    # Аннотации
    ret_pct = (y[-1] / initial - 1) * 100
    max_eq = max(y)
    max_dd = min((y[i] / max(y[:i+1]) - 1) * 100 for i in range(len(y)))
    stats_text = f"Return: {ret_pct:+.1f}%  |  Peak: ${max_eq:,.0f}  |  Max DD: {max_dd:.1f}%"
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
            ha="center", color="#94a3b8", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#334155"))

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Equity Curve Viewer")
    parser.add_argument("--backtest", action="store_true", help="Show backtest equity")
    args = parser.parse_args()
    source = "backtest" if args.backtest else "paper"
    show_equity_curve(source)


if __name__ == "__main__":
    main()
