"""
Predict V51 — G-Trade
========================================
Synced with train_hybrid V50:
  * Features from champion_registry.json
  * engineer_features() from train_hybrid
  * Multi-arch LSTM loader (V49 + V50)
  * Tuned thresholds per asset
  * Ensemble with soft gating
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
from sklearn.preprocessing import StandardScaler
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
    from config import FULL_ASSET_MAP
except ImportError:
    sys.exit("config.py not found!")

from core.logger import get_logger
from core.features import engineer_features, add_weekly_features
from core.architectures import ReduceSumLayer, build_lstm_attention
from core.ensemble import ensemble_with_gating, build_stacking_features
from core.profiles import FOREX
from backtest import _load_lstm_model, _get_lookback

logger = get_logger("predict")

DB_PATH = os.path.join(BASE_DIR, "market.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


_GROUPS = {
    "INDICES & MACRO": ['VIX', 'DXY', 'TNX', 'SP500', 'NASDAQ', 'DOW'],
    "COMMODITIES":     ['GOLD', 'SILVER', 'OIL', 'GAS'],
    "US TECH":         ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN',
                        'META', 'AMD', 'PLTR', 'COIN', 'MSTR'],
    "US HEALTHCARE":   ['JNJ', 'UNH', 'PFE', 'LLY', 'ABBV', 'MRK'],
    "US FINANCE":      ['JPM', 'BAC', 'GS', 'V', 'MA', 'WFC'],
    "US CONSUMER":     ['WMT', 'KO', 'PEP', 'MCD', 'NKE', 'DIS', 'NFLX', 'SBUX'],
    "US INDUSTRIAL":   ['BA', 'CAT', 'XOM', 'CVX', 'COP'],
    "US SEMI":         ['INTC', 'QCOM', 'AVGO', 'MU'],
    "US SOFTWARE":     ['CRM', 'ORCL', 'ADBE', 'UBER', 'PYPL'],
    "CRYPTO":          ['BTC', 'ETH', 'SOL', 'XRP', 'TON', 'DOGE', 'BNB',
                        'ADA', 'AVAX', 'DOT', 'LINK', 'SHIB', 'ATOM', 'UNI', 'NEAR'],
    "MOEX":            ['IMOEX', 'SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK',
                        'TATN', 'SNGS', 'PLZL', 'SIBN', 'MGNT', 'TCSG',
                        'VTBR', 'BSPB', 'MOEX_EX', 'CBOM',
                        'YNDX', 'OZON', 'VKCO', 'POSI', 'MTSS', 'RTKM',
                        'HHRU', 'SOFL', 'ASTR', 'WUSH',
                        'CHMF', 'NLMK', 'MAGN', 'RUAL', 'ALRS', 'TRMK', 'MTLR', 'RASP',
                        'IRAO', 'HYDR', 'FLOT', 'AFLT', 'PIKK',
                        'FEES', 'UPRO', 'MSNG', 'NMTP',
                        'PHOR', 'SGZH', 'FIVE', 'FIXP', 'LENT', 'MVID',
                        'SMLT', 'LSRG'],
    "FOREX":           ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDRUB',
                        'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
                        'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
                        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADJPY', 'CHFJPY', 'NZDJPY',
                        'USDTRY', 'USDMXN', 'USDZAR', 'USDSGD', 'USDNOK', 'USDSEK', 'USDPLN', 'USDCNH'],
}

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
    """Returns (sig, prob, price, mode) or None on failure."""
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
        if len(df) < 50:
            return None
    except Exception as e:
        logger.warning("Feature engineering failed for %s: %s", table, e)
        return None

    reg_entry = registry.get(name)
    if reg_entry and 'features' in reg_entry:
        features = [f for f in reg_entry['features'] if f in df.columns]
    else:
        features = ["close", "volume", "vol_z", "taleb_risk", "ret_1",
                    "ret_5", "trend_strength", "rsi", "sma_20", "sma_50"]
        features = [f for f in features if f in df.columns]

    if len(features) < 3:
        return None

    try:
        curr_price = float(df['close'].iloc[-1])
        n_bars = min(500, len(df))
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[features].iloc[-n_bars:].values)

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

        # ── Transformer (3rd ensemble member) ──────────────────────────────
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

        # ── TCN (4th ensemble member) ────────────────────────────────────
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

        # ── Stacking meta-classifier ─────────────────────────────────────
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

        thr = thresholds.get(name, {})
        buy_thr = thr.get('buy', 0.55)
        sell_thr = thr.get('sell', 0.45)

        if prob > buy_thr:
            sig = "BUY"
        elif prob < sell_thr:
            sig = "SELL"
        else:
            sig = "WAIT"

        return sig, prob, curr_price, mode

    except Exception as e:
        logger.error("Prediction failed for %s: %s", table, e)
        return None


def run_radar():
    t0 = time.time()
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    # ── Update actuals for previous predictions ──────────────────
    try:
        from performance_tracker import update_actuals, log_prediction
        update_actuals()
        _do_log = True
    except Exception as e:
        logger.warning("Actuals update failed: %s", e)
        _do_log = False

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print()
    print("═" * W)
    print(f"  REAL-TIME RADAR  │  {now}")
    print("═" * W)

    # ── Scan all assets with inline progress ──────────────────────
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
                sig, prob, price, mode = res
                try:
                    log_prediction(name, sig, prob, cb_prob=prob)
                    logged += 1
                except Exception as e:
                    logger.debug("Log prediction failed for %s: %s", name, e)

    sys.stdout.write("\r" + " " * 45 + "\r")
    sys.stdout.flush()

    # ── Print grouped results ──────────────────────────────────────
    counts = {"BUY": 0, "SELL": 0, "WAIT": 0}
    col_hdr = f"  {'Asset':<10}  {'Sig':<6}  {'Conf':>7}  {'Price':>12}  Mode"

    for group, members in _GROUPS.items():
        rows = [(n, results[n]) for n in members if n in results]
        if not rows:
            continue

        tag = f"  ── {group} "
        print(tag + "─" * max(0, W - len(tag)))
        print(col_hdr)

        for name, (sig, prob, price, mode) in rows:
            clr = _CLR[sig]
            # Pad signal text first, THEN wrap with color codes so
            # surrounding columns stay aligned regardless of escape chars
            sig_col = f"{clr}{sig:<4}{_RST}"
            print(f"  {name:<10}  {sig_col}  {prob:>6.1%}  {_fmt_price(price)}  {mode}")
            counts[sig] += 1

        print()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    scanned = sum(counts.values())
    buy_s  = f"\033[92mBUY  {counts['BUY']:>2}{_RST}"
    sell_s = f"\033[91mSELL {counts['SELL']:>2}{_RST}"
    wait_s = f"\033[90mWAIT {counts['WAIT']:>2}{_RST}"
    print("─" * W)
    log_info = f"  Logged {logged}" if _do_log and logged else ""
    print(f"  {buy_s}   {sell_s}   {wait_s}   Total {scanned}   {elapsed:.1f}s{log_info}")
    print("═" * W)


if __name__ == "__main__":
    run_radar()
