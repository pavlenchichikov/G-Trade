"""Console radar: loads champions and prints BUY/SELL/WAIT for all assets.

The scaler and calibrator are the saved ones; features and thresholds come
from the champion registry.
"""

import json
import os
import sys
import time
import warnings

import pandas as pd
import tensorflow as tf
from datetime import datetime
from sqlalchemy import create_engine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from config import FULL_ASSET_MAP, RADAR_GROUPS
except ImportError:
    sys.exit("config.py not found!")

from core.logger import get_logger
from core.features import engineer_features, add_weekly_features, add_crossasset_features, add_macro_features, add_cross_lag_features
from core.scoring import score_asset

logger = get_logger("predict")

DB_PATH = os.path.join(BASE_DIR, "market.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


_GROUPS = RADAR_GROUPS  # single source: config.ASSET_TYPES

W = 62  # output width

_CLR = {"BUY": "\033[92m", "SELL": "\033[91m", "WAIT": "\033[90m"}
_RST = "\033[0m"


def _fmt_price(p):
    if p >= 10000:
        return f"{p:>12,.0f}"
    if p >= 100:
        return f"{p:>12,.2f}"
    if p >= 1:
        return f"{p:>12.3f}"
    return f"{p:>12.5f}"


def _load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _predict_asset(name, registry, thresholds):
    """Returns full scoring dict or None on failure.

    The dict contains: sig (gated), sig_raw, prob, prob_raw, price, mode,
    meta_prob, cb_prob, lstm_prob, gate_reason.

    sig_raw and prob_raw are the raw measurement stream (before gating);
    sig/prob are the gated values shown to the user.
    """
    table = name.lower().replace("^", "").replace(".", "").replace("-", "")
    cb_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
    if not os.path.exists(cb_path):
        return None

    try:
        df_raw = pd.read_sql(f"SELECT * FROM {table}", engine,
                             index_col="Date", parse_dates=["Date"])
        df_raw.index = pd.to_datetime(df_raw.index).normalize()
        df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
        df = engineer_features(df_raw)
        df = add_weekly_features(df, table, engine)
        df = add_crossasset_features(df, table, engine)
        df = add_macro_features(df, engine)
        df = add_cross_lag_features(df, engine)
        if len(df) < 50:
            return None
    except Exception as e:
        logger.warning("Feature engineering failed for %s: %s", table, e)
        return None

    reg_entry = registry.get(name)
    # Model scoring is shared with alert_bot.py through core.scoring so the two
    # serve paths cannot drift apart (see core/scoring.py).
    res = score_asset(df, name, table, reg_entry, thresholds, MODEL_DIR)
    return res  # full dict (or None); run_radar unpacks what it needs


def run_radar():
    t0 = time.time()
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    # -- Update actuals for previous predictions ------------------
    try:
        from performance_tracker import update_actuals, log_prediction
        update_actuals()
        _do_log = True
    except Exception as e:
        logger.warning("Actuals update failed: %s", e)
        _do_log = False

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print()
    print("=" * W)
    print(f"  REAL-TIME RADAR  |  {now}")
    print("=" * W)

    # -- Scan all assets with inline progress ----------------------
    all_names = list(FULL_ASSET_MAP.keys())
    total = len(all_names)
    results = {}
    logged = 0

    for idx, name in enumerate(all_names, 1):
        sys.stdout.write(f"\r  Scanning {idx}/{total}  {name:<12}")
        sys.stdout.flush()
        res = _predict_asset(name, registry, thresholds)
        if res:
            # display tuple keeps the OLD shape so the print section is untouched;
            # sig is the GATED signal (what the user acts on)
            results[name] = (res["sig"], res["prob"], res["price"], res["mode"],
                             res["meta_prob"], res["cb_prob"], res["lstm_prob"])
            # Log the RAW signal + RAW probability - the measurement stream that
            # rehabilitates a gated class; the gated view goes to sig_shown.
            if _do_log:
                try:
                    log_prediction(name, res["sig_raw"], res["prob_raw"],
                                   cb_prob=res["cb_prob"], lstm_prob=res["lstm_prob"],
                                   meta_prob=res["meta_prob"],
                                   sig_shown=res["sig"], gate_reason=res["gate_reason"],
                                   timing_action=res.get("timing_action"),
                                   timing_reason=res.get("timing_reason"))
                    logged += 1
                except Exception as e:
                    logger.debug("Log prediction failed for %s: %s", name, e)

    sys.stdout.write("\r" + " " * 45 + "\r")
    sys.stdout.flush()

    # -- Print grouped results --------------------------------------
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    col_hdr = f"  {'Asset':<10}  {'Sig':<6}  {'Conf':>7}  {'Price':>12}  Mode"

    for group, members in _GROUPS.items():
        rows = [(n, results[n]) for n in members if n in results]
        if not rows:
            continue

        tag = f"  -- {group} "
        print(tag + "-" * max(0, W - len(tag)))
        print(col_hdr)

        for name, (sig, prob, price, mode, _mp, _cbp, _lstmp) in rows:
            clr = _CLR[sig]
            # Pad signal text first, THEN wrap with color codes so
            # surrounding columns stay aligned regardless of escape chars
            sig_col = f"{clr}{sig:<4}{_RST}"
            print(f"  {name:<10}  {sig_col}  {prob:>6.1%}  {_fmt_price(price)}  {mode}")
            counts[sig] += 1

        print()

    # -- Summary ---------------------------------------------------
    elapsed = time.time() - t0
    scanned = sum(counts.values())
    buy_s  = f"\033[92mBUY  {counts['BUY']:>2}{_RST}"
    sell_s = f"\033[91mSELL {counts['SELL']:>2}{_RST}"
    wait_s = f"\033[90mWAIT {counts['WAIT']:>2}{_RST}"
    print("-" * W)
    log_info = f"  Logged {logged}" if _do_log and logged else ""
    print(f"  {buy_s}   {sell_s}   {wait_s}   Total {scanned}   {elapsed:.1f}s{log_info}")
    print("=" * W)


if __name__ == "__main__":
    run_radar()
