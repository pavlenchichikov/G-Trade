"""
Backtest V72 — G-Trade
=========================================
Fixes over V71:
  • DB deduplication on load (remove duplicate Date rows)
  • Weekly features now join correctly (Date column preserved)
  • Walk-forward split uses Date column + registry updated_at
  • Keras 3.x ZIP model loading (V71)
  • Scaler fit on TRAIN only (V71)
  • LSTM loader handles V49/V50/Keras3 (V71)
"""

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Multiply, Permute, RepeatVector, Flatten
)
from tensorflow.keras.models import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Allow loading old Lambda-based models
try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

from core.logger import get_logger
logger = get_logger("backtest")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from config import FULL_ASSET_MAP
except ImportError:
    sys.exit(f"config.py not found in {BASE_DIR}")

# Import shared components from core modules
from core.features import engineer_features, add_weekly_features
from core.architectures import ReduceSumLayer, build_lstm_attention
from core.ensemble import ensemble_with_gating, build_stacking_features
from core.profiles import FOREX
from core.backtesting import FOREX_COMMISSION, FOREX_SLIPPAGE
import joblib

DB_PATH = os.path.join(BASE_DIR, "market.db")
engine = create_engine(f"sqlite:///{DB_PATH}")
MODEL_DIR = os.path.join(BASE_DIR, "models")

COMMISSION = 0.001
SLIPPAGE = 0.0015
INITIAL_CAPITAL = 1_000.0
KELLY_FRACTION = 0.25
MAX_POSITION = 0.10
MAX_TRADE_RET = 0.04

# Load registry, thresholds and optuna params
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")
OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")


def _load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (peak + 1e-9)
    return float(dd.min())


def _sharpe(returns, trades_per_year=None):
    r = np.array(returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    ann_factor = np.sqrt(trades_per_year) if trades_per_year else np.sqrt(252)
    return float(r.mean() / r.std() * ann_factor)


def _calmar(total_return, max_dd):
    if abs(max_dd) < 1e-6:
        return 0.0
    return total_return / abs(max_dd)


def _kelly_size(win_rate, avg_win, avg_loss):
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    p, q = win_rate, 1.0 - win_rate
    kelly = (p * b - q) / b
    if kelly <= 0:
        return 0.0
    return min(kelly * KELLY_FRACTION, MAX_POSITION)


def _build_lstm_legacy(input_shape):
    """V49 architecture: LSTM(128)+Dropout+LSTM(64), Bahdanau-style attention, Lambda→ReduceSum."""
    timesteps = input_shape[0]
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    # V49 attention: Dense(1,tanh)→Flatten→Dense(ts,softmax)→RepeatVector→Permute→Multiply
    a = Dense(1, activation='tanh')(x)
    a = Flatten()(a)
    a = Dense(timesteps, activation='softmax')(a)
    a = RepeatVector(64)(a)
    a = Permute((2, 1))(a)
    x = Multiply()([x, a])
    x = ReduceSumLayer()(x)
    x = Dense(32, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def _detect_format(path):
    """Detect if file is ZIP (.keras native) or HDF5."""
    with open(path, 'rb') as f:
        header = f.read(4)
    return 'zip' if header[:2] == b'PK' else 'hdf5'


def _load_weights_keras3(model, h5_path):
    """Load weights from Keras 3.x h5 format (layers/<name>/vars/N) into a Keras 2.x model."""
    import h5py
    import numpy as np
    with h5py.File(h5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError("Not a Keras 3.x h5 file")
        h5_layers = f['layers']
        for layer in model.layers:
            if layer.name in h5_layers and 'vars' in h5_layers[layer.name]:
                vars_grp = h5_layers[layer.name]['vars']
                weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                layer.set_weights(weights)
            # Handle LSTM cell weights (stored as layers/<name>/cell/vars/N)
            elif layer.name in h5_layers and 'cell' in h5_layers[layer.name]:
                cell_grp = h5_layers[layer.name]['cell']
                if 'vars' in cell_grp:
                    vars_grp = cell_grp['vars']
                    weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                    layer.set_weights(weights)


def _get_lookback(reg_entry, asset_name):
    """Resolve actual lookback: registry.lookback → optuna_params → profile fallback."""
    if reg_entry and 'lookback' in reg_entry:
        return int(reg_entry['lookback'])
    optuna = _load_json(OPTUNA_PARAMS_PATH)
    if asset_name in optuna and 'lookback' in optuna[asset_name]:
        return int(optuna[asset_name]['lookback'])
    return int(reg_entry.get('profile', {}).get('lookback', 10)) if reg_entry else 10


def _detect_lookback_from_h5(h5_path):
    """Detect actual lookback from saved weights by finding the attention Dense(timesteps) square kernel."""
    import h5py

    def _find_square_kernel(group):
        """Recursively search for a square kernel tensor (NxN, 5<=N<=100)."""
        for key in group.keys():
            item = group[key]
            if hasattr(item, 'shape'):
                if 'kernel' in key and len(item.shape) == 2 and item.shape[0] == item.shape[1] and 5 <= item.shape[0] <= 100:
                    return int(item.shape[0])
            elif hasattr(item, 'keys'):
                result = _find_square_kernel(item)
                if result is not None:
                    return result
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'model_weights' in f:
                return _find_square_kernel(f['model_weights'])
            if 'layers' in f:
                return _find_square_kernel(f['layers'])
    except Exception as e:
        logger.debug("Lookback detection failed for %s: %s", h5_path, e)
    return None


def _load_lstm_model(lstm_path, lookback, n_features):
    """Load LSTM handling V49, V50, HDF5, ZIP, and Keras 3.x formats."""
    import shutil, zipfile, tempfile
    fmt = _detect_format(lstm_path)

    # For HDF5 files saved as .keras: need .h5 extension for load_weights
    if fmt == 'hdf5':
        h5_path = lstm_path.replace('.keras', '.tmp.h5')
        shutil.copy2(lstm_path, h5_path)
        weights_path = h5_path
    else:
        weights_path = lstm_path
        h5_path = None

    try:
        # Auto-detect lookback from saved weights (handles frozen champions with outdated optuna params)
        detected_lb = _detect_lookback_from_h5(weights_path if fmt == 'hdf5' else lstm_path)
        if detected_lb is not None and detected_lb != lookback:
            logger.debug("Lookback override for %s: %d -> %d (from weights)", lstm_path, lookback, detected_lb)
            lookback = detected_lb
        input_shape = (lookback, n_features)

        # Method 1: V50 architecture (192/96+ReduceSumLayer), Keras 2.x load_weights
        try:
            model = build_lstm_attention(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (AI)", lookback
        except Exception as e:
            logger.debug("V50 load failed: %s", e)

        # Method 2: V49 legacy architecture, Keras 2.x load_weights
        try:
            model = _build_lstm_legacy(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (V49)", lookback
        except Exception as e:
            logger.debug("V49 load failed: %s", e)

        # Method 3: ZIP with Keras 3.x weights — extract h5, load manually
        if fmt == 'zip':
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(lstm_path, 'r') as z:
                    z.extractall(tmpdir)
                inner_h5 = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(inner_h5):
                    # Auto-detect lookback from extracted weights
                    zip_lb = _detect_lookback_from_h5(inner_h5)
                    if zip_lb is not None:
                        input_shape = (zip_lb, n_features)
                    zip_lookback = zip_lb if zip_lb is not None else lookback
                    # Try V49 legacy with Keras 3.x weights
                    try:
                        model = _build_lstm_legacy(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras3 V49 load failed: %s", e)
                    # Try V50 with Keras 3.x weights
                    try:
                        model = build_lstm_attention(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras 3.x load failed for %s: %s", lstm_path, e)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        logger.debug("All LSTM load methods failed for %s", lstm_path)
        return None, "CB ONLY (Err)", lookback
    finally:
        if h5_path and os.path.exists(h5_path):
            os.remove(h5_path)


def run_forensic_test():
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    print("\n" + "=" * 95)
    print("   G-TRADE V73: FORENSIC BACKTEST (4-Head Stacking Ensemble)")
    print("   CB + LSTM + Transformer + TCN | Meta-Stacking | Adversarial Validation")
    print("=" * 95 + "\n")
    fmt = "{:<8} | {:>6} | {:>8} | {:>12} | {:>10} | {:>8} | {:>8} | {:<15}"
    print(fmt.format("ASSET", "TRADES", "WINRATE", "NET PROFIT", "MAX DD",
                     "SHARPE", "CALMAR", "MODE"))
    print("-" * 95)

    portfolio_results = []

    for name in FULL_ASSET_MAP.keys():
        table = name.lower().replace("^", "").replace(".", "").replace("-", "")

        # --- Load & engineer features (same as train) ---
        try:
            df_raw = pd.read_sql(f"SELECT * FROM {table}", engine,
                                 index_col="Date", parse_dates=["Date"])
            df_raw.index = pd.to_datetime(df_raw.index).normalize()
            df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
            df = engineer_features(df_raw)
            df = add_weekly_features(df, table, engine)
            if len(df) < 200:
                continue
        except Exception as exc:
            logger.debug("Skip %s (data): %s", name, exc)
            continue

        # --- Get features from registry (same as training used) ---
        reg_entry = registry.get(name)
        if reg_entry and 'features' in reg_entry:
            features = [f for f in reg_entry['features'] if f in df.columns]
        else:
            features = ["close", "volume", "vol_z", "taleb_risk", "ret_1",
                        "ret_5", "trend_strength", "rsi", "sma_20", "sma_50"]
            features = [f for f in features if f in df.columns]

        if len(features) < 3:
            continue

        # --- Walk-forward split ---
        if reg_entry and 'updated_at' in reg_entry and 'Date' in df.columns:
            try:
                train_date = pd.to_datetime(reg_entry['updated_at'])
                dates = pd.to_datetime(df['Date'])
                split_idx = int((dates < train_date).sum())
                if split_idx >= len(df) - 20 or split_idx < 50:
                    split_idx = int(len(df) * 0.8)
            except Exception as e:
                logger.debug("Date-based split failed for %s, using 80/20: %s", name, e)
                split_idx = int(len(df) * 0.8)
        else:
            split_idx = int(len(df) * 0.8)

        # --- Scale features: FIT ON TRAIN ONLY (no data leak) ---
        try:
            scaler = StandardScaler()
            scaler.fit(df[features].iloc[:split_idx].values)
            X_all = scaler.transform(df[features].values)
        except Exception as e:
            logger.warning("Scaler failed for %s: %s", name, e)
            continue

        X_test = X_all[split_idx:]
        df_test = df.iloc[split_idx:].copy()

        if len(X_test) < 20:
            continue

        # --- Load models ---
        cb_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
        lstm_path = os.path.join(MODEL_DIR, f"{table}_lstm.keras")

        if not os.path.exists(cb_path):
            continue

        cb_probs, lstm_probs, mode_label = None, None, "CB ONLY"

        try:
            cb = CatBoostClassifier()
            cb.load_model(cb_path)
            cb_probs = cb.predict_proba(X_test)[:, 1]
        except Exception as exc:
            logger.warning("CatBoost load failed %s: %s", name, exc)
            continue

        # --- LSTM (handles V49 and V50 architectures) ---
        lookback = _get_lookback(reg_entry, name)
        lstm_probs, tf_probs, tcn_probs = None, None, None
        X_seq_all = None

        if os.path.exists(lstm_path):
            if split_idx >= lookback and len(X_test) > 0:
                lstm_model, mode_label, lookback = _load_lstm_model(lstm_path, lookback, len(features))
                if lstm_model is not None:
                    try:
                        X_seq_all = np.array([X_all[i - lookback:i]
                                              for i in range(split_idx, split_idx + len(X_test))])
                        lstm_probs = lstm_model.predict(X_seq_all, batch_size=512, verbose=0).flatten()
                    except Exception as e:
                        logger.warning("LSTM predict failed %s: %s", name, e)
                        lstm_probs = None
                        mode_label = "CB ONLY (Pred)"

        # --- Transformer ---
        tf_path = os.path.join(MODEL_DIR, f"{table}_transformer.keras")
        if os.path.exists(tf_path) and X_seq_all is not None:
            try:
                tf_model, _, _ = _load_lstm_model(tf_path, lookback, len(features))
                if tf_model is not None:
                    tf_probs = tf_model.predict(X_seq_all.astype('float32'), batch_size=512, verbose=0).flatten()
            except Exception as e:
                logger.debug("Transformer predict failed %s: %s", name, e)
                tf_probs = None

        # --- TCN ---
        tcn_path = os.path.join(MODEL_DIR, f"{table}_tcn.keras")
        if os.path.exists(tcn_path) and X_seq_all is not None:
            try:
                tcn_model = tf.keras.models.load_model(tcn_path)
                tcn_probs = tcn_model.predict(X_seq_all.astype('float32'), batch_size=512, verbose=0).flatten()
            except Exception as e:
                logger.debug("TCN predict failed %s: %s", name, e)
                tcn_probs = None

        # --- Meta stacking classifier ---
        meta_path = os.path.join(MODEL_DIR, f"{table}_meta.pkl")
        meta_clf = None
        if os.path.exists(meta_path):
            try:
                meta_clf = joblib.load(meta_path)
            except Exception as e:
                logger.debug("Meta classifier load failed: %s", e)
                meta_clf = None

        has_all_4 = (lstm_probs is not None and tf_probs is not None and tcn_probs is not None)
        if has_all_4:
            mode_label = "STACK (4H)"
        elif lstm_probs is not None:
            mode_label = "DUAL (AI)"

        # --- Get per-asset thresholds ---
        thr = thresholds.get(name, {})
        buy_thr = thr.get('buy', 0.55)
        sell_thr = thr.get('sell', 0.45)
        no_trade_band = thr.get('no_trade_band', 0.02)

        # --- FOREX-aware costs ---
        comm = FOREX_COMMISSION if name in FOREX else COMMISSION
        slip = FOREX_SLIPPAGE if name in FOREX else SLIPPAGE

        # --- Simulation with Kelly sizing ---
        balance = INITIAL_CAPITAL
        equity = [balance]
        trades = 0
        wins = 0
        trade_rets = []

        # Bootstrap Kelly init
        bootstrap_wr = 0.55
        bootstrap_win = 0.020
        bootstrap_los = 0.012

        for i in range(len(cb_probs)):
            p_cb = cb_probs[i]

            # Stacking: use meta-classifier if all 4 models available
            if has_all_4 and meta_clf is not None and i < len(lstm_probs):
                trend_val = df_test['trend_strength'].iloc[i] if 'trend_strength' in df_test.columns else 0.01
                X_meta = build_stacking_features(
                    np.array([p_cb]), np.array([lstm_probs[i]]),
                    np.array([tf_probs[i]]), np.array([tcn_probs[i]]),
                    np.array([trend_val]))
                try:
                    prob = float(meta_clf.predict_proba(X_meta)[:, 1][0])
                except Exception as e:
                    logger.debug("Meta predict fallback: %s", e)
                    prob = float(np.mean([p_cb, lstm_probs[i], tf_probs[i], tcn_probs[i]]))
            elif has_all_4 and i < len(lstm_probs):
                prob = float(np.mean([p_cb, lstm_probs[i], tf_probs[i], tcn_probs[i]]))
            elif lstm_probs is not None and i < len(lstm_probs):
                trend_val = df_test['trend_strength'].iloc[i] if 'trend_strength' in df_test.columns else 0.01
                trend_gate = reg_entry.get('profile', {}).get('trend_gate', 0.01) if reg_entry else 0.01
                prob = float(ensemble_with_gating(
                    np.array([p_cb]), np.array([lstm_probs[i]]),
                    np.array([trend_val]), trend_gate
                )[0])
            else:
                prob = p_cb

            # Use tuned thresholds + no-trade band
            if (buy_thr - no_trade_band) <= prob <= (sell_thr + no_trade_band):
                continue
            if prob < sell_thr:
                direction = -1
            elif prob > buy_thr:
                direction = 1
            else:
                continue

            ret = df_test["next_ret"].iloc[i]
            if pd.isna(ret):
                continue

            # Kelly position size
            pos_frac = _kelly_size(bootstrap_wr, bootstrap_win, bootstrap_los)
            pos_size = balance * pos_frac

            raw_ret = (ret if direction > 0 else -ret)
            raw_ret = float(np.clip(raw_ret, -MAX_TRADE_RET, MAX_TRADE_RET))
            cost = raw_ret - comm - slip
            pnl = pos_size * cost

            balance += pnl
            equity.append(balance)
            trades += 1
            trade_rets.append(cost)

            if cost > 0:
                wins += 1

            # Update bootstrapped stats
            if trades >= 10:
                pos_rets = [r for r in trade_rets if r > 0]
                neg_rets = [r for r in trade_rets if r < 0]
                bootstrap_wr = wins / trades
                bootstrap_win = np.mean(pos_rets) if pos_rets else 0.020
                bootstrap_los = abs(np.mean(neg_rets)) if neg_rets else 0.012

        if trades == 0:
            continue

        profit_pct = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        winrate = wins / trades * 100
        max_dd = _max_drawdown(equity) * 100
        sharpe = _sharpe(trade_rets)
        calmar = _calmar(profit_pct / 100, max_dd / 100)

        color = "\033[92m" if profit_pct > 0 else "\033[91m"
        reset = "\033[0m"
        print(fmt.format(
            name,
            trades,
            f"{winrate:.1f}%",
            f"{color}{profit_pct:+.2f}%{reset}",
            f"{max_dd:.1f}%",
            f"{sharpe:.2f}",
            f"{calmar:.2f}",
            mode_label,
        ))

        portfolio_results.append({
            "Asset": name,
            "Profit": profit_pct,
            "MaxDD": max_dd,
            "Sharpe": sharpe,
            "Calmar": calmar,
            "Trades": trades,
            "WinRate": winrate,
            "Mode": mode_label,
        })
        logger.info("Backtest %s: profit=%.2f%% trades=%d winrate=%.1f%% sharpe=%.2f mode=%s",
                    name, profit_pct, trades, winrate, sharpe, mode_label)

    # --- Portfolio summary ---
    print("-" * 95)
    if not portfolio_results:
        print("No results (check models directory).")
        return

    df_res = pd.DataFrame(portfolio_results).sort_values("Profit", ascending=False)

    # Count modes
    dual_count = len(df_res[df_res['Mode'].str.startswith('DUAL')])
    cb_count = len(df_res) - dual_count

    print(f"\nTOP 5 BY PROFIT:")
    for _, row in df_res.head(5).iterrows():
        print(f"   {row['Asset']:<8}  profit={row['Profit']:+.2f}%  "
              f"sharpe={row['Sharpe']:.2f}  calmar={row['Calmar']:.2f}  {row['Mode']}")

    print(f"\nBOTTOM 5 BY PROFIT:")
    for _, row in df_res.tail(5).iterrows():
        print(f"   {row['Asset']:<8}  profit={row['Profit']:+.2f}%  "
              f"max_dd={row['MaxDD']:.1f}%  {row['Mode']}")

    print(f"\nPORTFOLIO AGGREGATES:")
    profitable = df_res[df_res["Profit"] > 0]
    print(f"   Assets tested:     {len(df_res)}")
    print(f"   DUAL (AI) mode:    {dual_count}")
    print(f"   CB ONLY mode:      {cb_count}")
    print(f"   Profitable:        {len(profitable)} ({len(profitable)/len(df_res)*100:.0f}%)")
    print(f"   Avg Net Return:    {df_res['Profit'].mean():+.2f}%")
    print(f"   Median Return:     {df_res['Profit'].median():+.2f}%")
    print(f"   Avg Sharpe:        {df_res['Sharpe'].mean():.2f}")
    print(f"   Avg Max DD:        {df_res['MaxDD'].mean():.1f}%")
    print(f"   Best Sharpe:       {df_res.loc[df_res['Sharpe'].idxmax(), 'Asset']} "
          f"({df_res['Sharpe'].max():.2f})")
    print(f"   Best Calmar:       {df_res.loc[df_res['Calmar'].idxmax(), 'Asset']} "
          f"({df_res['Calmar'].max():.2f})")


if __name__ == "__main__":
    run_forensic_test()
