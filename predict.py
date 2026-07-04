"""Console radar: loads champions and prints BUY/SELL/WAIT for all assets.

The scaler and calibrator are the saved ones; features and thresholds come
from the champion registry.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from catboost import CatBoostClassifier
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
from core.features import engineer_features, add_weekly_features, add_crossasset_features, add_macro_features, add_cross_lag_features, active_candidate_features
from core.ensemble import build_stacking_features
from core.scaling import load_or_fit_scaler
from core.calibration import load_calibrator, apply_calibrator
from core.model_io import get_lookback as _get_lookback, load_lstm_model as _load_lstm_model
from core import meta_sizer

logger = get_logger("predict")

DB_PATH = os.path.join(BASE_DIR, "market.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


_GROUPS = RADAR_GROUPS  # single source: config.ASSET_TYPES

# A champion whose walk-forward score is below this cannot be trusted for
# direction (negative score = worse than the baseline, typically all-negative
# recent folds / pos_ratio=0.00). Its BUY/SELL is suppressed to WAIT in the
# radar and tagged "low-q" so the suppression is visible, not silent.
MIN_TRUST_SCORE = 0.0

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
    """Returns (sig, prob, price, mode, meta_prob) or None on failure. meta_prob is
    None unless GTRADE_META_SIZING is on (the SP-6 Phase 2b meta-sizing gate)."""
    table = name.lower().replace("^", "").replace(".", "").replace("-", "")
    cb_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
    lstm_path = os.path.join(MODEL_DIR, f"{table}_lstm.keras")

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
    if reg_entry and 'features' in reg_entry:
        features = [f for f in reg_entry['features'] if f in df.columns]
    else:
        features = [f for f in active_candidate_features() if f in df.columns]

    if len(features) < 3:
        return None

    try:
        curr_price = float(df['close'].iloc[-1])
        n_bars = min(500, len(df))
        x_fit = df[features].iloc[-n_bars:].values
        # scaler from training; fitting on the window is only for old models without one
        scaler, _src = load_or_fit_scaler(MODEL_DIR, table, x_fit)
        if _src == "fit":
            logger.debug("No saved scaler for %s; fitting on recent window (retrain to fix)", table)
        X_all = scaler.transform(x_fit)

        cb = CatBoostClassifier()
        cb.load_model(cb_path)
        cb_prob = float(cb.predict_proba(X_all[-1:])[:, 1][0])

        lstm_prob = None
        mode = "CB"
        lookback = _get_lookback(reg_entry, name)
        if os.path.exists(lstm_path):
            if len(X_all) >= lookback:
                lstm_model, mode, lookback = _load_lstm_model(lstm_path, lookback, len(features))
                if lstm_model is not None:
                    try:
                        X_seq = X_all[-lookback:].reshape(1, lookback, len(features))
                        lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                    except Exception as e:
                        logger.debug("LSTM predict failed: %s", e)
                        lstm_prob = None
                        mode = "CB"

        # -- Transformer (3rd ensemble member) ------------------------------
        tf_path = os.path.join(MODEL_DIR, f"{table}_transformer.keras")
        tf_prob = None
        if os.path.exists(tf_path):
            try:
                tf_model, _, tf_lookback = _load_lstm_model(tf_path, lookback, len(features))
                if tf_model is not None:
                    X_seq = X_all[-tf_lookback:].reshape(1, tf_lookback, len(features))
                    tf_prob = float(tf_model.predict(X_seq.astype('float32'), verbose=0).flatten()[0])
            except Exception as e:
                logger.debug("Transformer predict failed: %s", e)
                tf_prob = None

        # -- TCN (4th ensemble member) ------------------------------------
        tcn_path = os.path.join(MODEL_DIR, f"{table}_tcn.keras")
        tcn_prob = None
        if os.path.exists(tcn_path):
            try:
                tcn_model = tf.keras.models.load_model(tcn_path)
                X_seq = X_all[-lookback:].reshape(1, lookback, len(features))
                tcn_prob = float(tcn_model.predict(X_seq.astype('float32'), verbose=0).flatten()[0])
            except Exception as e:
                logger.debug("TCN predict failed: %s", e)
                tcn_prob = None

        # -- Stacking meta-classifier -------------------------------------
        meta_path = os.path.join(MODEL_DIR, f"{table}_meta.pkl")
        trend_val = float(df['trend_strength'].iloc[-1]) if 'trend_strength' in df.columns else 0.01

        all_probs = [cb_prob, lstm_prob, tf_prob, tcn_prob]
        n_available = sum(1 for p in all_probs if p is not None)

        if n_available >= 3 and os.path.exists(meta_path):
            try:
                meta_clf = joblib.load(meta_path)
                # Fill missing probs with 0.5 (neutral)
                _cb = cb_prob
                _lstm = lstm_prob if lstm_prob is not None else 0.5
                _tf = tf_prob if tf_prob is not None else 0.5
                _tcn = tcn_prob if tcn_prob is not None else 0.5
                X_meta = build_stacking_features(
                    np.array([_cb]), np.array([_lstm]),
                    np.array([_tf]), np.array([_tcn]),
                    np.array([trend_val]))
                prob = float(meta_clf.predict_proba(X_meta)[:, 1][0])
                mode = "STACK"
            except Exception as e:
                logger.debug("Stacking fallback: %s", e)
                prob = np.mean([p for p in all_probs if p is not None])
        elif lstm_prob is not None:
            # Fallback: weighted average of available models
            probs_avail = [cb_prob, lstm_prob]
            if tf_prob is not None:
                probs_avail.append(tf_prob)
            if tcn_prob is not None:
                probs_avail.append(tcn_prob)
            prob = float(np.mean(probs_avail))
        else:
            prob = cb_prob

        # Calibrate the raw ensemble probability into an honest up-move frequency
        # when a calibrator exists for this asset (identity otherwise).
        prob = float(apply_calibrator(load_calibrator(MODEL_DIR, table), np.array([prob]))[0])

        thr = thresholds.get(name, {})
        buy_thr = thr.get('buy', 0.55)
        sell_thr = thr.get('sell', 0.45)

        if prob > buy_thr:
            sig = "BUY"
        elif prob < sell_thr:
            sig = "SELL"
        else:
            sig = "WAIT"

        # Quality gate: do not emit a directional call from an untrustworthy
        # champion (score below MIN_TRUST_SCORE). Surface it as WAIT and tag the
        # mode, so a -5.8-score model can no longer print "BUY 100%".
        score = (reg_entry or {}).get('score')
        if sig != "WAIT" and score is not None and score < MIN_TRUST_SCORE:
            sig = "WAIT"
            mode = f"{mode} low-q"

        meta_p = None
        if meta_sizer.meta_enabled() != "off":
            try:
                model = meta_sizer.load_meta(name)
                if model is not None:
                    mcols = [c for c in (meta_sizer.OPINION_COLS + meta_sizer.REGIME_COLS)
                             if c in df.columns]
                    if len(mcols) >= 3:
                        meta_p = meta_sizer.meta_prob(model, df[mcols].iloc[-1].to_numpy())
                        sig, _info = meta_sizer.gate(sig, meta_p)
            except Exception as e:
                logger.debug("meta-sizing skipped for %s: %s", name, e)
        return sig, prob, curr_price, mode, meta_p

    except Exception as e:
        logger.error("Prediction failed for %s: %s", table, e)
        return None


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
            results[name] = res
            # Log prediction for performance tracking
            if _do_log:
                sig, prob, price, mode, meta_p = res
                try:
                    log_prediction(name, sig, prob, cb_prob=prob, meta_prob=meta_p)
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

        for name, (sig, prob, price, mode, _mp) in rows:
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
