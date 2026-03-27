"""
auto_trader.py — Auto-executes ML signals in the paper trading system.
Connects predict.py signals (via signal_dashboard) with paper_trading.py execution.
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "auto_trader_config.json")
LOG_FILE = os.path.join(BASE_DIR, "auto_trader.log")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("auto_trader")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "min_confidence": 0.60,      # minimum confidence to act (abs(prob-0.5)*2)
    "min_kelly": 2.0,            # minimum Kelly % to trade
    "max_positions": 10,         # max simultaneous open positions
    "default_amount": 500,       # default $ per trade
    "use_kelly_sizing": True,    # if True, amount = balance * kelly_pct
    "allowed_signals": ["BUY"],  # which signals to auto-execute
    "blacklist": [],             # assets to never trade
    "whitelist": [],             # if non-empty, only trade these assets
    "dry_run": True,             # if True, only log, don't execute
}

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
try:
    import signal_dashboard
    _HAS_SIGNAL_DASHBOARD = True
except ImportError:
    signal_dashboard = None
    _HAS_SIGNAL_DASHBOARD = False
    logger.warning("signal_dashboard not found — signals unavailable")

try:
    import paper_trading
    _HAS_PAPER_TRADING = True
except ImportError:
    paper_trading = None
    _HAS_PAPER_TRADING = False
    logger.warning("paper_trading not found — execution unavailable")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config from auto_trader_config.json, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            config.update(saved)
        except Exception as e:
            logger.warning(f"Could not read config file: {e}. Using defaults.")
    return config


def save_config(config: dict) -> None:
    """Persist config to auto_trader_config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Could not save config: {e}")


# ---------------------------------------------------------------------------
# Core auto-trade logic
# ---------------------------------------------------------------------------

def run_auto_trade() -> dict:
    """
    Main execution loop.

    Returns:
        dict with keys: executed (list), skipped (list), errors (list)
    """
    config = load_config()
    results = {"executed": [], "skipped": [], "errors": []}

    dry_run = config.get("dry_run", True)
    mode_label = "[DRY-RUN]" if dry_run else "[LIVE]"
    logger.info(f"{mode_label} run_auto_trade() started")

    # --- 1. Fetch signals ---
    if not _HAS_SIGNAL_DASHBOARD:
        msg = "signal_dashboard unavailable — cannot fetch signals"
        logger.error(msg)
        results["errors"].append(msg)
        return results

    try:
        logger.info("Scanning all assets (this takes a few minutes)...")
        signals_df = signal_dashboard.get_all_signals(progress=True)
    except Exception as e:
        msg = f"get_all_signals() failed: {e}"
        logger.error(msg)
        results["errors"].append(msg)
        return results

    if signals_df is None or signals_df.empty:
        logger.info("No signals returned.")
        return results

    # Normalise column names (case-insensitive lookup)
    signals_df.columns = [c.strip() for c in signals_df.columns]
    col_map = {c.lower(): c for c in signals_df.columns}

    def col(name: str):
        return col_map.get(name.lower())

    # --- 2. Fetch current paper-trading state ---
    open_positions: set = set()
    balance: float = 0.0

    if _HAS_PAPER_TRADING:
        try:
            portfolio = paper_trading.get_portfolio()  # expected: dict with 'positions' and 'balance'
            if isinstance(portfolio, dict):
                open_positions = set(portfolio.get("positions", {}).keys())
                balance = float(portfolio.get("balance", 0.0))
            else:
                logger.warning("paper_trading.get_portfolio() returned unexpected type")
        except AttributeError:
            # Fallback: try get_positions / get_balance separately
            try:
                open_positions = set(paper_trading.get_positions().keys())
            except Exception:
                pass
            try:
                balance = float(paper_trading.get_balance())
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not fetch portfolio: {e}")

    logger.info(f"Open positions ({len(open_positions)}): {sorted(open_positions)}")
    logger.info(f"Available balance: ${balance:.2f}")

    # --- 3. Filter & execute ---
    min_confidence = float(config.get("min_confidence", 0.60))
    min_kelly = float(config.get("min_kelly", 2.0))
    max_positions = int(config.get("max_positions", 10))
    default_amount = float(config.get("default_amount", 500))
    use_kelly_sizing = bool(config.get("use_kelly_sizing", True))
    allowed_signals = [s.upper() for s in config.get("allowed_signals", ["BUY"])]
    blacklist = [a.upper() for a in config.get("blacklist", [])]
    whitelist = [a.upper() for a in config.get("whitelist", [])]

    current_positions = len(open_positions)

    for _, row in signals_df.iterrows():
        asset = str(row.get(col("asset") or "Asset", "")).strip()
        if not asset:
            continue

        asset_up = asset.upper()

        # Helper to skip with reason
        def skip(reason: str):
            logger.info(f"SKIP {asset}: {reason}")
            results["skipped"].append({"asset": asset, "reason": reason, "ts": _ts()})

        # Whitelist / blacklist
        if whitelist and asset_up not in whitelist:
            skip("not in whitelist")
            continue
        if asset_up in blacklist:
            skip("blacklisted")
            continue

        # Signal type
        signal_raw = str(row.get(col("signal") or "Signal", "")).strip().upper()
        if signal_raw not in allowed_signals:
            skip(f"signal '{signal_raw}' not in allowed_signals {allowed_signals}")
            continue

        # Confidence
        prob = None
        for cname in ["probability", "prob"]:
            c = col(cname)
            if c:
                try:
                    prob = float(row[c])
                except Exception:
                    pass
                break

        confidence = None
        c = col("confidence")
        if c:
            try:
                confidence = float(row[c])
            except Exception:
                pass

        if confidence is None and prob is not None:
            confidence = abs(prob - 0.5) * 2

        if confidence is not None and confidence < min_confidence:
            skip(f"confidence {confidence:.3f} < {min_confidence}")
            continue

        # Kelly size
        kelly_pct = None
        for cname in ["kelly_size", "kelly"]:
            c = col(cname)
            if c:
                try:
                    kelly_pct = float(row[c])
                except Exception:
                    pass
                break

        if kelly_pct is not None and kelly_pct < min_kelly:
            skip(f"kelly {kelly_pct:.2f}% < {min_kelly}%")
            continue

        # Max positions check (only for BUY)
        if signal_raw == "BUY":
            if asset_up in open_positions:
                skip("position already open")
                continue
            if current_positions >= max_positions:
                skip(f"max_positions ({max_positions}) reached")
                continue

        # Determine trade amount
        if use_kelly_sizing and kelly_pct is not None and balance > 0:
            amount = balance * (kelly_pct / 100.0)
        else:
            amount = default_amount
        amount = round(amount, 2)

        # --- Execute or dry-run ---
        action_info = {
            "asset": asset,
            "signal": signal_raw,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "kelly_pct": round(kelly_pct, 4) if kelly_pct is not None else None,
            "amount": amount,
            "dry_run": dry_run,
            "ts": _ts(),
        }

        if dry_run:
            logger.info(
                f"{mode_label} WOULD {signal_raw} {asset} | amount=${amount:.2f} "
                f"| confidence={confidence:.3f if confidence else 'N/A'} "
                f"| kelly={kelly_pct:.2f if kelly_pct else 'N/A'}%"
            )
            results["executed"].append(action_info)
            if signal_raw == "BUY":
                open_positions.add(asset_up)
                current_positions += 1
        else:
            if not _HAS_PAPER_TRADING:
                msg = f"paper_trading unavailable — cannot execute {signal_raw} {asset}"
                logger.error(msg)
                results["errors"].append({"asset": asset, "error": msg, "ts": _ts()})
                continue
            try:
                if signal_raw == "BUY":
                    paper_trading.buy_asset(asset, amount)
                    open_positions.add(asset_up)
                    current_positions += 1
                elif signal_raw == "SELL":
                    paper_trading.sell_asset(asset)
                logger.info(
                    f"{mode_label} {signal_raw} {asset} | amount=${amount:.2f} "
                    f"| confidence={confidence:.3f if confidence else 'N/A'} "
                    f"| kelly={kelly_pct:.2f if kelly_pct else 'N/A'}%"
                )
                results["executed"].append(action_info)
            except Exception as e:
                err_msg = f"{signal_raw} {asset} failed: {e}"
                logger.error(err_msg)
                results["errors"].append({"asset": asset, "error": err_msg, "ts": _ts()})

    logger.info(
        f"{mode_label} run_auto_trade() done | "
        f"executed={len(results['executed'])} skipped={len(results['skipped'])} "
        f"errors={len(results['errors'])}"
    )
    return results


# ---------------------------------------------------------------------------
# Trade log reader
# ---------------------------------------------------------------------------

def get_trade_log(days: int = 7) -> list:
    """
    Return recent auto-trade log lines from the last `days` days.

    Each entry is a dict with keys: ts, level, message.
    """
    entries = []
    if not os.path.exists(LOG_FILE):
        return entries

    cutoff = datetime.now() - timedelta(days=days)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                # Format: "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message"
                try:
                    ts_str, rest = line[:23], line[24:]
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                    if ts < cutoff:
                        continue
                    bracket_end = rest.index("]")
                    level = rest[1:bracket_end].strip()
                    message = rest[bracket_end + 2:].strip()
                    entries.append({"ts": ts_str, "level": level, "message": message})
                except Exception:
                    entries.append({"ts": "", "level": "RAW", "message": line})
    except Exception as e:
        logger.error(f"Could not read log: {e}")

    return entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cast(value: str):
    """Try to cast a CLI string value to bool / int / float / list / str."""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # JSON list/dict
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except Exception:
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main():
    parser = argparse.ArgumentParser(
        description="auto_trader — auto-execute ML signals in paper trading"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force dry-run mode (no real trades)"
    )
    parser.add_argument(
        "--config", action="store_true",
        help="Print current config and exit"
    )
    parser.add_argument(
        "--set", nargs=2, metavar=("KEY", "VALUE"),
        help="Update a config key and save (e.g. --set min_confidence 0.65)"
    )
    parser.add_argument(
        "--log", nargs="?", const=7, type=int, metavar="DAYS",
        help="Print recent trade log (default last 7 days)"
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_config()
        print(json.dumps(cfg, indent=2))
        return

    if args.set:
        key, raw_value = args.set
        cfg = load_config()
        if key not in DEFAULT_CONFIG:
            print(f"Warning: '{key}' is not a known config key.")
        cfg[key] = _cast(raw_value)
        save_config(cfg)
        print(f"Config updated: {key} = {cfg[key]!r}")
        return

    if args.log is not None:
        entries = get_trade_log(days=args.log)
        if not entries:
            print(f"No log entries in the last {args.log} day(s).")
        for e in entries:
            print(f"{e['ts']} [{e['level']}] {e['message']}")
        return

    if args.dry_run:
        cfg = load_config()
        cfg["dry_run"] = True
        save_config(cfg)
        print("Dry-run mode forced.")

    results = run_auto_trade()
    print(
        f"\nDone — executed: {len(results['executed'])} | "
        f"skipped: {len(results['skipped'])} | "
        f"errors: {len(results['errors'])}"
    )
    if results["errors"]:
        print("Errors:")
        for e in results["errors"]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
