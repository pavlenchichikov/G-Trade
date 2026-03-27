"""
Signal History Logger -- records all generated signals to SQLite
for tracking accuracy over time.
Database: signal_history.db
"""
import os
import sys
import sqlite3
import argparse
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import FULL_ASSET_MAP

SIGNAL_DB_PATH = os.path.join(BASE_DIR, "signal_history.db")
MARKET_DB_PATH = os.path.join(BASE_DIR, "market.db")
market_engine = create_engine(f"sqlite:///{MARKET_DB_PATH}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_name(asset: str) -> str:
    """Convert asset key to market.db table name."""
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _get_signal_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create the signals table if it does not exist."""
    conn = _get_signal_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            signal TEXT,
            confidence REAL,
            cb_prob REAL,
            lstm_prob REAL,
            price_at_signal REAL,
            rsi REAL,
            regime TEXT,
            outcome TEXT DEFAULT 'PENDING',
            price_after_1d REAL,
            price_after_5d REAL,
            return_1d REAL,
            return_5d REAL
        )
    """)
    conn.commit()
    conn.close()


def get_current_price(asset: str):
    """Read the latest Close price from market.db for *asset*."""
    tbl = _table_name(asset)
    try:
        with market_engine.connect() as conn:
            check = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": tbl},
            )
            if not check.fetchone():
                return None
            row = conn.execute(
                text(f'SELECT "close", "Date" FROM {tbl} ORDER BY "Date" DESC LIMIT 1')
            ).fetchone()
            if row:
                return float(row[0])
    except Exception:
        pass
    return None


def _get_price_on_date(asset: str, target_date: str):
    """Return the Close price for *asset* on or after *target_date*."""
    tbl = _table_name(asset)
    try:
        with market_engine.connect() as conn:
            check = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": tbl},
            )
            if not check.fetchone():
                return None
            row = conn.execute(
                text(
                    f'SELECT "close" FROM {tbl} WHERE "Date" >= :d ORDER BY "Date" ASC LIMIT 1'
                ),
                {"d": target_date},
            ).fetchone()
            if row:
                return float(row[0])
    except Exception:
        pass
    return None


def _latest_date_in_market(asset: str):
    """Return the latest date string available in market.db for the asset."""
    tbl = _table_name(asset)
    try:
        with market_engine.connect() as conn:
            check = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": tbl},
            )
            if not check.fetchone():
                return None
            row = conn.execute(
                text(f'SELECT MAX("Date") FROM {tbl}')
            ).fetchone()
            if row and row[0]:
                return str(row[0])[:10]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def log_signal(
    asset: str,
    signal: str,
    confidence: float,
    cb_prob: float = None,
    lstm_prob: float = None,
    price: float = None,
    rsi: float = None,
    regime: str = None,
):
    """Record a new signal into signal_history.db."""
    _init_db()
    if price is None:
        price = get_current_price(asset)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_signal_conn()
    conn.execute(
        """INSERT INTO signals
           (timestamp, asset, signal, confidence, cb_prob, lstm_prob,
            price_at_signal, rsi, regime, outcome)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')""",
        (ts, asset.upper(), signal.upper(), confidence, cb_prob, lstm_prob, price, rsi, regime),
    )
    conn.commit()
    conn.close()
    print(f"  [LOGGED] {ts}  {asset}  {signal}  conf={confidence:.2f}  price={price}")


def verify_signals():
    """Check pending signals and fill in 1d/5d returns + outcome."""
    _init_db()
    conn = _get_signal_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE outcome = 'PENDING'"
    ).fetchall()
    if not rows:
        print("  No pending signals to verify.")
        conn.close()
        return 0

    updated = 0
    now = datetime.now()
    for r in rows:
        sig_time = datetime.strptime(r["timestamp"][:10], "%Y-%m-%d")
        asset = r["asset"]
        price0 = r["price_at_signal"]
        if price0 is None or price0 == 0:
            continue

        date_1d = (sig_time + timedelta(days=1)).strftime("%Y-%m-%d")
        date_5d = (sig_time + timedelta(days=5)).strftime("%Y-%m-%d")

        latest = _latest_date_in_market(asset)
        if latest is None:
            continue

        p1d = None
        p5d = None
        ret1d = None
        ret5d = None

        # 1-day check
        if latest >= date_1d:
            p1d = _get_price_on_date(asset, date_1d)
            if p1d is not None:
                ret1d = (p1d - price0) / price0

        # 5-day check
        if latest >= date_5d:
            p5d = _get_price_on_date(asset, date_5d)
            if p5d is not None:
                ret5d = (p5d - price0) / price0

        # Determine outcome based on available returns
        outcome = "PENDING"
        sig_dir = r["signal"]  # BUY / SELL / HOLD
        if ret5d is not None:
            if sig_dir == "BUY":
                outcome = "CORRECT" if ret5d > 0 else "INCORRECT"
            elif sig_dir == "SELL":
                outcome = "CORRECT" if ret5d < 0 else "INCORRECT"
            else:
                # HOLD -- correct if abs return < 1%
                outcome = "CORRECT" if abs(ret5d) < 0.01 else "INCORRECT"
        elif ret1d is not None:
            if sig_dir == "BUY":
                outcome = "CORRECT" if ret1d > 0 else "INCORRECT"
            elif sig_dir == "SELL":
                outcome = "CORRECT" if ret1d < 0 else "INCORRECT"
            else:
                outcome = "CORRECT" if abs(ret1d) < 0.01 else "INCORRECT"

        conn.execute(
            """UPDATE signals
               SET price_after_1d = ?, price_after_5d = ?,
                   return_1d = ?, return_5d = ?, outcome = ?
               WHERE id = ?""",
            (p1d, p5d, ret1d, ret5d, outcome, r["id"]),
        )
        if outcome != "PENDING":
            updated += 1

    conn.commit()
    conn.close()
    print(f"  Verified {updated} signal(s).")
    return updated


def accuracy_report():
    """Print accuracy breakdown by asset, regime, etc."""
    _init_db()
    conn = _get_signal_conn()
    rows = conn.execute("SELECT * FROM signals").fetchall()
    conn.close()
    if not rows:
        print("  No signals recorded yet.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    total = len(df)
    verified = df[df["outcome"].isin(["CORRECT", "INCORRECT"])]
    pending = df[df["outcome"] == "PENDING"]
    n_verified = len(verified)

    acc_1d = None
    acc_5d = None
    if n_verified > 0:
        has_1d = verified.dropna(subset=["return_1d"])
        has_5d = verified.dropna(subset=["return_5d"])
        if len(has_1d) > 0:
            # Re-evaluate 1d accuracy
            correct_1d = 0
            for _, row in has_1d.iterrows():
                if row["signal"] == "BUY" and row["return_1d"] > 0:
                    correct_1d += 1
                elif row["signal"] == "SELL" and row["return_1d"] < 0:
                    correct_1d += 1
                elif row["signal"] == "HOLD" and abs(row["return_1d"]) < 0.01:
                    correct_1d += 1
            acc_1d = correct_1d / len(has_1d) * 100
        if len(has_5d) > 0:
            correct_5d = 0
            for _, row in has_5d.iterrows():
                if row["signal"] == "BUY" and row["return_5d"] > 0:
                    correct_5d += 1
                elif row["signal"] == "SELL" and row["return_5d"] < 0:
                    correct_5d += 1
                elif row["signal"] == "HOLD" and abs(row["return_5d"]) < 0.01:
                    correct_5d += 1
            acc_5d = correct_5d / len(has_5d) * 100

    print()
    print("=" * 60)
    print("  SIGNAL LOG  |  History & Accuracy")
    print("=" * 60)
    print()
    print("  [ACCURACY SUMMARY]")
    print(f"  Total signals:  {total}")
    print(f"  Verified:       {n_verified}")
    print(f"  Pending:        {len(pending)}")
    print(f"  Accuracy (1d):  {acc_1d:.1f}%" if acc_1d is not None else "  Accuracy (1d):  N/A")
    print(f"  Accuracy (5d):  {acc_5d:.1f}%" if acc_5d is not None else "  Accuracy (5d):  N/A")
    print()

    # By asset -- top 5
    if n_verified > 0:
        print("  [BY ASSET - TOP 5]")
        print(f"  {'Asset':<10}{'Signals':<10}{'Acc(1d)':<10}{'Acc(5d)':<10}{'Avg Conf':<10}")
        print("  " + "-" * 55)
        asset_groups = df.groupby("asset")
        asset_stats = []
        for asset_name, grp in asset_groups:
            v = grp[grp["outcome"].isin(["CORRECT", "INCORRECT"])]
            if len(v) == 0:
                continue
            h1 = v.dropna(subset=["return_1d"])
            h5 = v.dropna(subset=["return_5d"])
            a1 = None
            a5 = None
            if len(h1) > 0:
                c1 = sum(
                    1 for _, rw in h1.iterrows()
                    if (rw["signal"] == "BUY" and rw["return_1d"] > 0)
                    or (rw["signal"] == "SELL" and rw["return_1d"] < 0)
                    or (rw["signal"] == "HOLD" and abs(rw["return_1d"]) < 0.01)
                )
                a1 = c1 / len(h1) * 100
            if len(h5) > 0:
                c5 = sum(
                    1 for _, rw in h5.iterrows()
                    if (rw["signal"] == "BUY" and rw["return_5d"] > 0)
                    or (rw["signal"] == "SELL" and rw["return_5d"] < 0)
                    or (rw["signal"] == "HOLD" and abs(rw["return_5d"]) < 0.01)
                )
                a5 = c5 / len(h5) * 100
            avg_conf = grp["confidence"].mean()
            asset_stats.append((asset_name, len(grp), a1, a5, avg_conf))

        asset_stats.sort(key=lambda x: len(x[1]) if isinstance(x[1], str) else x[1], reverse=True)
        for name, cnt, a1, a5, ac in asset_stats[:5]:
            s1 = f"{a1:.1f}%" if a1 is not None else "N/A"
            s5 = f"{a5:.1f}%" if a5 is not None else "N/A"
            print(f"  {name:<10}{cnt:<10}{s1:<10}{s5:<10}{ac:.2f}")
        print()

    # By regime
    if n_verified > 0 and "regime" in df.columns:
        regimes = df.dropna(subset=["regime"]).groupby("regime")
        if len(regimes) > 0:
            print("  [BY REGIME]")
            print(f"  {'Regime':<15}{'Signals':<10}{'Verified':<10}{'Acc(5d)':<10}")
            print("  " + "-" * 55)
            for regime_name, grp in regimes:
                v = grp[grp["outcome"].isin(["CORRECT", "INCORRECT"])]
                h5 = v.dropna(subset=["return_5d"])
                a5 = None
                if len(h5) > 0:
                    c5 = sum(
                        1 for _, rw in h5.iterrows()
                        if (rw["signal"] == "BUY" and rw["return_5d"] > 0)
                        or (rw["signal"] == "SELL" and rw["return_5d"] < 0)
                        or (rw["signal"] == "HOLD" and abs(rw["return_5d"]) < 0.01)
                    )
                    a5 = c5 / len(h5) * 100
                s5 = f"{a5:.1f}%" if a5 is not None else "N/A"
                print(f"  {str(regime_name):<15}{len(grp):<10}{len(v):<10}{s5:<10}")
            print()


def show_recent(n: int = 20, asset_filter: str = None):
    """Display the last N signals."""
    _init_db()
    conn = _get_signal_conn()
    if asset_filter:
        rows = conn.execute(
            "SELECT * FROM signals WHERE asset = ? ORDER BY timestamp DESC LIMIT ?",
            (asset_filter.upper(), n),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
    conn.close()

    if not rows:
        print("  No signals found.")
        return

    print()
    print("  [RECENT SIGNALS]")
    header = f"  {'Date':<12}{'Asset':<8}{'Signal':<7}{'Conf':<7}{'Price':<12}{'1d Ret':<9}{'5d Ret':<9}{'OK?':<6}"
    print(header)
    print("  " + "-" * 55)
    for r in rows:
        dt = str(r["timestamp"])[:10]
        asset = r["asset"] or ""
        sig = r["signal"] or ""
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "N/A"
        price = r["price_at_signal"]
        price_str = f"${price:,.2f}" if price is not None else "N/A"
        r1 = r["return_1d"]
        r5 = r["return_5d"]
        ret1_str = f"{r1:+.1%}" if r1 is not None else "PEND"
        ret5_str = f"{r5:+.1%}" if r5 is not None else "PEND"
        outcome = r["outcome"] or "PENDING"
        ok_str = "[OK]" if outcome == "CORRECT" else "[X]" if outcome == "INCORRECT" else "[..]"
        print(f"  {dt:<12}{asset:<8}{sig:<7}{conf:<7}{price_str:<12}{ret1_str:<9}{ret5_str:<9}{ok_str:<6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Signal History Logger")
    parser.add_argument("--verify", action="store_true", help="Verify pending signals")
    parser.add_argument("--asset", type=str, default=None, help="Filter by asset")
    parser.add_argument("--accuracy", action="store_true", help="Show accuracy report")
    parser.add_argument(
        "--log",
        nargs=3,
        metavar=("ASSET", "SIGNAL", "CONFIDENCE"),
        help="Manually log a signal: --log BTC BUY 0.65",
    )
    parser.add_argument("-n", type=int, default=20, help="Number of recent signals to show")
    args = parser.parse_args()

    _init_db()

    if args.log:
        asset, signal, conf = args.log
        if asset.upper() not in FULL_ASSET_MAP:
            print(f"  [!] '{asset}' not in FULL_ASSET_MAP. Logging anyway.")
        log_signal(asset, signal, float(conf))
        return

    if args.verify:
        verify_signals()
        return

    if args.accuracy:
        accuracy_report()
        return

    # Default: accuracy summary + recent signals
    accuracy_report()
    show_recent(n=args.n, asset_filter=args.asset)


if __name__ == "__main__":
    main()
