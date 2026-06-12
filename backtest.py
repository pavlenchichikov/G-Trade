"""Backtest: evaluate champions on held-out data.

Loads each asset's champion ensemble, scales features fit-on-train (no leak),
and reports PnL, win rate, Sharpe, directional accuracy, Brier, and buy & hold
alpha. The LSTM loader handles the older and newer saved model formats.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

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
from core.features import engineer_features, add_weekly_features, add_crossasset_features
from core.ensemble import ensemble_with_gating, build_stacking_features
from core.model_io import (
    get_lookback as _get_lookback,
    load_json as _load_json,
    load_lstm_model as _load_lstm_model,
)
from core.profiles import FOREX
from core.backtesting import FOREX_COMMISSION, FOREX_SLIPPAGE
from core.calibration import load_calibrator, apply_calibrator
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

# Load registry and thresholds (optuna params читает core.model_io)
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


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


def run_forensic_test():
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    print("\n" + "=" * 95)
    print("   G-TRADE BACKTEST")
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
            df = add_crossasset_features(df, table, engine)
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

        # --- Calibrator (identity when asset has no saved calibrator) ---
        calibrator = load_calibrator(MODEL_DIR, table)

        # --- Buy & hold benchmark over the same test window ---
        bench_ret = 0.0
        if len(df_test) > 1:
            bench_ret = (df_test['close'].iloc[-1] / df_test['close'].iloc[0] - 1.0) * 100

        # --- Simulation with Kelly sizing ---
        balance = INITIAL_CAPITAL
        equity = [balance]
        trades = 0
        wins = 0
        trade_rets = []

        # Raw directional accuracy + calibration, measured over ALL bars
        # (not only the ones that became trades) - the honest model hit rate.
        n_eval = 0
        n_correct = 0
        brier_sum = 0.0

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

            # Calibrate raw ensemble prob (identity if no calibrator saved yet)
            prob = float(apply_calibrator(calibrator, np.array([prob]))[0])

            # --- Directional accuracy + Brier over every bar ---
            _tgt_ret = df_test["next_ret"].iloc[i]
            if not pd.isna(_tgt_ret):
                _tgt = 1 if _tgt_ret > 0 else 0
                n_eval += 1
                if int(prob >= 0.5) == _tgt:
                    n_correct += 1
                brier_sum += (prob - _tgt) ** 2

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
        dir_acc = (n_correct / n_eval * 100) if n_eval else 0.0
        brier = (brier_sum / n_eval) if n_eval else 0.0
        alpha = profit_pct - bench_ret

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
            "DirAcc": dir_acc,
            "Brier": brier,
            "Benchmark": bench_ret,
            "Alpha": alpha,
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

    print("\nTOP 5 BY PROFIT:")
    for _, row in df_res.head(5).iterrows():
        print(f"   {row['Asset']:<8}  profit={row['Profit']:+.2f}%  "
              f"sharpe={row['Sharpe']:.2f}  calmar={row['Calmar']:.2f}  {row['Mode']}")

    print("\nBOTTOM 5 BY PROFIT:")
    for _, row in df_res.tail(5).iterrows():
        print(f"   {row['Asset']:<8}  profit={row['Profit']:+.2f}%  "
              f"max_dd={row['MaxDD']:.1f}%  {row['Mode']}")

    print("\nPORTFOLIO AGGREGATES:")
    profitable = df_res[df_res["Profit"] > 0]
    beat_bench = df_res[df_res["Alpha"] > 0]
    print(f"   Assets tested:     {len(df_res)}")
    print(f"   DUAL (AI) mode:    {dual_count}")
    print(f"   CB ONLY mode:      {cb_count}")
    print(f"   Profitable:        {len(profitable)} ({len(profitable)/len(df_res)*100:.0f}%)")
    print(f"   Beat buy&hold:     {len(beat_bench)} ({len(beat_bench)/len(df_res)*100:.0f}%)")
    print(f"   Avg Net Return:    {df_res['Profit'].mean():+.2f}%")
    print(f"   Median Return:     {df_res['Profit'].median():+.2f}%")
    print(f"   Avg Alpha:         {df_res['Alpha'].mean():+.2f}% (vs buy&hold)")
    print(f"   Avg Dir. Accuracy: {df_res['DirAcc'].mean():.1f}%  (raw direction hit rate, all bars)")
    print(f"   Avg Brier score:   {df_res['Brier'].mean():.4f}  (lower = better calibrated)")
    print(f"   Avg Sharpe:        {df_res['Sharpe'].mean():.2f}")
    print(f"   Avg Max DD:        {df_res['MaxDD'].mean():.1f}%")
    print(f"   Best Sharpe:       {df_res.loc[df_res['Sharpe'].idxmax(), 'Asset']} "
          f"({df_res['Sharpe'].max():.2f})")
    print(f"   Best Calmar:       {df_res.loc[df_res['Calmar'].idxmax(), 'Asset']} "
          f"({df_res['Calmar'].max():.2f})")


if __name__ == "__main__":
    run_forensic_test()
