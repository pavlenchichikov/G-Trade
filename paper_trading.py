"""
Paper Trading -- Virtual Portfolio module.
Manages virtual positions using SQLite (paper.db).
Reads live prices from market.db.
"""

import os
import sys
import sqlite3
import argparse
from datetime import datetime

import sqlalchemy as sa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DB = os.path.join(BASE_DIR, "paper.db")
MARKET_DB = os.path.join(BASE_DIR, "market.db")

sys.path.insert(0, BASE_DIR)
from config import FULL_ASSET_MAP

DEFAULT_BALANCE = 10_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_name(asset: str) -> str:
    """Derive market.db table name from asset key."""
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _get_current_price(asset: str) -> float | None:
    """Read the latest close price for *asset* from market.db."""
    if not os.path.exists(MARKET_DB):
        return None
    table = _table_name(asset)
    engine = sa.create_engine(f"sqlite:///{MARKET_DB}")
    try:
        with engine.connect() as conn:
            # Check table exists
            result = conn.execute(
                sa.text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name=:t"
                ),
                {"t": table},
            )
            if result.fetchone() is None:
                return None
            row = conn.execute(
                sa.text(
                    f'SELECT close FROM "{table}" ORDER BY Date DESC LIMIT 1'
                )
            ).fetchone()
            if row is None:
                return None
            return float(row[0])
    except Exception:
        return None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Paper DB management
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(PAPER_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create schema and seed default balance if needed."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS portfolio_config "
        "(key TEXT PRIMARY KEY, value TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS positions ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  asset TEXT,"
        "  side TEXT,"
        "  entry_price REAL,"
        "  entry_date TEXT,"
        "  quantity REAL,"
        "  status TEXT,"
        "  exit_price REAL,"
        "  exit_date TEXT,"
        "  pnl REAL"
        ")"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS equity_history "
        "(date TEXT PRIMARY KEY, equity REAL)"
    )
    # Seed initial balance if not present
    row = cur.execute(
        "SELECT value FROM portfolio_config WHERE key='initial_balance'"
    ).fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO portfolio_config (key, value) VALUES (?, ?)",
            ("initial_balance", str(DEFAULT_BALANCE)),
        )
        cur.execute(
            "INSERT INTO portfolio_config (key, value) VALUES (?, ?)",
            ("created_at", datetime.now().isoformat()),
        )
    conn.commit()
    conn.close()


def _get_balance() -> float:
    """Return current cash balance (initial minus invested plus realised)."""
    conn = _connect()
    cur = conn.cursor()
    initial = float(
        cur.execute(
            "SELECT value FROM portfolio_config WHERE key='initial_balance'"
        ).fetchone()[0]
    )
    # Sum of money spent opening positions still open
    row = cur.execute(
        "SELECT COALESCE(SUM(entry_price * quantity), 0) "
        "FROM positions WHERE status='OPEN'"
    ).fetchone()
    invested = float(row[0])
    # Sum of realised P&L from closed trades
    row = cur.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM positions WHERE status='CLOSED'"
    ).fetchone()
    realised = float(row[0])
    conn.close()
    return initial - invested + realised


def _open_positions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _closed_positions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='CLOSED' ORDER BY exit_date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def portfolio_summary(quiet: bool = False) -> dict:
    """Compute and optionally print portfolio summary. Returns dict."""
    _init_db()
    balance = _get_balance()
    positions = _open_positions()
    unrealised = 0.0
    enriched = []
    for p in positions:
        price = _get_current_price(p["asset"])
        cur_price = price if price is not None else p["entry_price"]
        upnl = (cur_price - p["entry_price"]) * p["quantity"]
        if p["side"] == "SHORT":
            upnl = -upnl
        enriched.append({**p, "current_price": cur_price, "unrealised_pnl": upnl})
        unrealised += upnl

    equity = balance + sum(
        e["current_price"] * e["quantity"] for e in enriched
    ) + unrealised - sum(
        e["current_price"] * e["quantity"] for e in enriched
    )
    # Equity = cash balance + market value of open positions
    market_value = sum(e["current_price"] * e["quantity"] for e in enriched)
    equity = balance + market_value

    conn = _connect()
    initial = float(
        conn.execute(
            "SELECT value FROM portfolio_config WHERE key='initial_balance'"
        ).fetchone()[0]
    )
    conn.close()
    ret_pct = ((equity - initial) / initial) * 100 if initial else 0.0

    summary = {
        "balance": balance,
        "equity": equity,
        "initial": initial,
        "return_pct": ret_pct,
        "positions": enriched,
    }

    if not quiet:
        _print_summary(summary)
    return summary


def _print_summary(s: dict):
    sign = "+" if s["return_pct"] >= 0 else ""
    print("=" * 60)
    print("  PAPER TRADING -- Virtual Portfolio")
    print("=" * 60)
    print(
        f"  Balance: ${s['balance']:,.2f}    "
        f"Equity: ${s['equity']:,.2f}    "
        f"Return: {sign}{s['return_pct']:.1f}%"
    )
    print()
    if s["positions"]:
        print("  Open Positions:")
        header = (
            f"  {'Asset':<10}{'Side':<7}{'Entry':>10}    "
            f"{'Qty':>10}    {'Current':>10}    {'P&L':>10}"
        )
        print(header)
        print("  " + "-" * 57)
        for p in s["positions"]:
            sign_p = "+" if p["unrealised_pnl"] >= 0 else ""
            print(
                f"  {p['asset']:<10}{p['side']:<7}"
                f"${p['entry_price']:>9,.2f}    "
                f"{p['quantity']:>10.4f}    "
                f"${p['current_price']:>9,.2f}    "
                f"{sign_p}${abs(p['unrealised_pnl']):>8,.2f}"
            )
    else:
        print("  No open positions.")
    print()


def buy_asset(asset: str, amount: float):
    """Open a LONG position for *asset* worth *amount* USD."""
    _init_db()
    asset = asset.upper()
    if asset not in FULL_ASSET_MAP:
        print(f"[ERROR] Asset '{asset}' not found in FULL_ASSET_MAP.")
        return
    price = _get_current_price(asset)
    if price is None:
        print(f"[ERROR] Cannot read price for '{asset}' from market.db.")
        return
    balance = _get_balance()
    if amount > balance:
        print(
            f"[ERROR] Insufficient balance. "
            f"Available: ${balance:,.2f}, requested: ${amount:,.2f}"
        )
        return
    if amount <= 0:
        print("[ERROR] Amount must be positive.")
        return

    quantity = amount / price
    now = datetime.now().isoformat()

    conn = _connect()
    conn.execute(
        "INSERT INTO positions "
        "(asset, side, entry_price, entry_date, quantity, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (asset, "LONG", price, now, quantity, "OPEN"),
    )
    conn.commit()
    conn.close()
    print(
        f"[OK] Bought {quantity:.6f} {asset} at ${price:,.2f} "
        f"for ${amount:,.2f}"
    )


def sell_asset(asset: str):
    """Close the oldest open position for *asset*."""
    _init_db()
    asset = asset.upper()
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM positions WHERE asset=? AND status='OPEN' "
        "ORDER BY entry_date LIMIT 1",
        (asset,),
    ).fetchone()
    if row is None:
        print(f"[ERROR] No open position for '{asset}'.")
        conn.close()
        return
    pos = dict(row)
    price = _get_current_price(asset)
    if price is None:
        print(f"[ERROR] Cannot read current price for '{asset}'.")
        conn.close()
        return

    pnl = (price - pos["entry_price"]) * pos["quantity"]
    if pos["side"] == "SHORT":
        pnl = -pnl
    now = datetime.now().isoformat()

    conn.execute(
        "UPDATE positions SET status='CLOSED', exit_price=?, "
        "exit_date=?, pnl=? WHERE id=?",
        (price, now, pnl, pos["id"]),
    )
    conn.commit()
    conn.close()
    sign = "+" if pnl >= 0 else ""
    print(
        f"[OK] Closed {pos['quantity']:.6f} {asset} at ${price:,.2f}  "
        f"P&L: {sign}${abs(pnl):,.2f}"
    )


def show_history():
    """Print all closed trades."""
    _init_db()
    trades = _closed_positions()
    if not trades:
        print("No closed trades yet.")
        return
    print(
        f"  {'Asset':<8}{'Side':<7}{'Entry':>10}  {'Exit':>10}  "
        f"{'Qty':>10}  {'P&L':>10}  {'Date'}"
    )
    print("  " + "-" * 75)
    total = 0.0
    for t in trades:
        sign = "+" if (t["pnl"] or 0) >= 0 else ""
        total += t["pnl"] or 0
        exit_dt = (t["exit_date"] or "")[:10]
        print(
            f"  {t['asset']:<8}{t['side']:<7}"
            f"${t['entry_price']:>9,.2f}  "
            f"${(t['exit_price'] or 0):>9,.2f}  "
            f"{t['quantity']:>10.4f}  "
            f"{sign}${abs(t['pnl'] or 0):>8,.2f}  "
            f"{exit_dt}"
        )
    sign_t = "+" if total >= 0 else ""
    print(f"\n  Total realised P&L: {sign_t}${abs(total):,.2f}")


def equity_snapshot():
    """Save current equity to equity_history."""
    _init_db()
    s = portfolio_summary(quiet=True)
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO equity_history (date, equity) VALUES (?, ?)",
        (today, s["equity"]),
    )
    conn.commit()
    conn.close()
    print(f"[OK] Equity snapshot saved: {today} -> ${s['equity']:,.2f}")


def reset_portfolio():
    """Delete paper.db and reinitialise."""
    if os.path.exists(PAPER_DB):
        os.remove(PAPER_DB)
    _init_db()
    print("[OK] Portfolio reset. Balance: ${:,.2f}".format(DEFAULT_BALANCE))


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def _interactive_menu():
    _init_db()
    while True:
        portfolio_summary()
        print("  [1] Buy (Open Long)")
        print("  [2] Sell (Close Position)")
        print("  [3] Trade History")
        print("  [4] Equity Snapshot")
        print("  [5] Reset Portfolio")
        print("  [0] Exit")
        print()
        choice = input("  Select: ").strip()

        if choice == "1":
            asset = input("  Asset (e.g. BTC, SBER): ").strip().upper()
            try:
                amount = float(input("  Amount in USD: ").strip())
            except ValueError:
                print("[ERROR] Invalid amount.")
                continue
            buy_asset(asset, amount)

        elif choice == "2":
            positions = _open_positions()
            if not positions:
                print("  No open positions to close.")
                continue
            print("  Open positions:")
            for i, p in enumerate(positions, 1):
                print(
                    f"    [{i}] {p['asset']} {p['side']} "
                    f"qty={p['quantity']:.6f} entry=${p['entry_price']:,.2f}"
                )
            try:
                idx = int(input("  Select position number: ").strip()) - 1
            except ValueError:
                print("[ERROR] Invalid selection.")
                continue
            if 0 <= idx < len(positions):
                sell_asset(positions[idx]["asset"])
            else:
                print("[ERROR] Invalid selection.")

        elif choice == "3":
            show_history()

        elif choice == "4":
            equity_snapshot()

        elif choice == "5":
            confirm = input("  Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                reset_portfolio()

        elif choice == "0":
            break
        else:
            print("  Invalid choice.")

        input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading -- Virtual Portfolio"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show portfolio summary (non-interactive)"
    )
    parser.add_argument("--buy", type=str, metavar="ASSET", help="Buy asset")
    parser.add_argument(
        "--amount", type=float, default=500.0,
        help="USD amount for buy (default: 500)"
    )
    parser.add_argument(
        "--sell", type=str, metavar="ASSET", help="Sell/close position"
    )
    parser.add_argument(
        "--history", action="store_true", help="Show trade history"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset portfolio"
    )

    args = parser.parse_args()

    # If no flags, run interactive menu
    if not any([args.status, args.buy, args.sell, args.history, args.reset]):
        _interactive_menu()
        return

    _init_db()

    if args.status:
        portfolio_summary()
    if args.buy:
        buy_asset(args.buy, args.amount)
    if args.sell:
        sell_asset(args.sell)
    if args.history:
        show_history()
    if args.reset:
        confirm = input("Reset portfolio? (yes/no): ").strip().lower()
        if confirm == "yes":
            reset_portfolio()
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
