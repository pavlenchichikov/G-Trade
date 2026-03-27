import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

from config import FULL_ASSET_MAP
from train_hybrid import (
    engineer_features, add_weekly_features,
    ensemble_with_gating, FOREX,
)
from backtest import _load_lstm_model, _get_lookback, _load_json

DB_PATH = os.path.join(BASE_DIR, "market.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

_DEFAULT_BUY_THR = 0.55
_DEFAULT_SELL_THR = 0.45
_KELLY_FRACTION = 0.25
_MAX_POSITION = 0.10


def _table_name(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _calc_trend(df):
    close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    last_close = close.iloc[-1]
    last_sma20 = sma20.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    if pd.isna(last_sma20) or pd.isna(last_sma50):
        return "SIDEWAYS"
    if last_close > last_sma20 and last_sma20 > last_sma50:
        return "UPTREND"
    if last_close < last_sma20 and last_sma20 < last_sma50:
        return "DOWNTREND"
    return "SIDEWAYS"


def _kelly_size(win_rate, avg_win=0.020, avg_loss=0.012):
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return float(np.clip(kelly * _KELLY_FRACTION, 0.0, _MAX_POSITION))


def _load_transformer(path, lookback, n_features):
    """Try loading a transformer model — returns (model, lookback) or (None, lookback)."""
    try:
        from train_hybrid import build_transformer_encoder
        import shutil, zipfile, tempfile
        from backtest import _detect_format, _load_weights_keras3, _detect_lookback_from_h5

        fmt = _detect_format(path)
        if fmt == 'zip':
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall(tmpdir)
                inner_h5 = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(inner_h5):
                    detected = _detect_lookback_from_h5(inner_h5)
                    if detected:
                        lookback = detected
                    model = build_transformer_encoder((lookback, n_features))
                    _load_weights_keras3(model, inner_h5)
                    return model, lookback
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            import shutil
            h5_path = path.replace('.keras', '.tmp_tr.h5')
            shutil.copy2(path, h5_path)
            try:
                model = build_transformer_encoder((lookback, n_features))
                model.load_weights(h5_path)
                return model, lookback
            finally:
                if os.path.exists(h5_path):
                    os.remove(h5_path)
    except Exception:
        pass
    return None, lookback


def get_all_signals(progress=True):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    rows = []
    asset_list = list(FULL_ASSET_MAP.keys())
    total = len(asset_list)

    for idx, name in enumerate(asset_list):
        if progress:
            done = idx + 1
            bar_len = 30
            filled = int(bar_len * done / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {done}/{total}  {name:<12}", end="", flush=True)
        table = _table_name(name)

        # --- Load data ---
        try:
            df_raw = pd.read_sql(
                f"SELECT * FROM {table}", engine,
                index_col="Date", parse_dates=["Date"]
            )
            df_raw.index = pd.to_datetime(df_raw.index).normalize()
            df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
            df = engineer_features(df_raw)
            df = add_weekly_features(df, table, engine)
            if len(df) < 60:
                continue
        except Exception:
            continue

        # --- Price / daily change ---
        try:
            close_col = 'close' if 'close' in df.columns else df.columns[1]
            current_price = float(df[close_col].iloc[-1])
            chg_1d = float(df['ret_1'].iloc[-1]) if 'ret_1' in df.columns else float('nan')
        except Exception:
            current_price = float('nan')
            chg_1d = float('nan')

        # --- RSI ---
        try:
            rsi_val = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else float('nan')
        except Exception:
            rsi_val = float('nan')

        # --- Trend ---
        try:
            trend = _calc_trend(df)
        except Exception:
            trend = "SIDEWAYS"

        # --- Features ---
        reg_entry = registry.get(name)
        if reg_entry and 'features' in reg_entry:
            features = [f for f in reg_entry['features'] if f in df.columns]
        else:
            features = ["close", "volume", "vol_z", "taleb_risk", "ret_1",
                        "ret_5", "trend_strength", "rsi", "sma_20", "sma_50"]
            features = [f for f in features if f in df.columns]

        if len(features) < 3:
            continue

        # --- Scale on last 500 rows ---
        try:
            data_slice = df[features].tail(500)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data_slice.values)
        except Exception:
            continue

        n_features = len(features)

        # --- CatBoost ---
        cb_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
        if not os.path.exists(cb_path):
            continue

        try:
            cb = CatBoostClassifier()
            cb.load_model(cb_path)
            cb_prob = float(cb.predict_proba(X_scaled[-1:])[:, 1][0])
        except Exception:
            continue

        # Proxy CB accuracy as win_rate from registry
        cb_accuracy = 0.55
        if reg_entry:
            cb_accuracy = float(reg_entry.get('cb_accuracy',
                                reg_entry.get('accuracy',
                                reg_entry.get('win_rate', 0.55))))
            cb_accuracy = max(0.01, min(0.99, cb_accuracy))

        # --- LSTM ---
        lstm_prob = None
        lstm_path = os.path.join(MODEL_DIR, f"{table}_lstm.keras")
        if os.path.exists(lstm_path):
            try:
                lookback = _get_lookback(reg_entry, name)
                if len(X_scaled) >= lookback:
                    lstm_model, _, actual_lb = _load_lstm_model(lstm_path, lookback, n_features)
                    if lstm_model is not None:
                        seq = X_scaled[-actual_lb:][np.newaxis, :, :]
                        lstm_prob = float(lstm_model.predict(seq, batch_size=1, verbose=0).flatten()[0])
            except Exception:
                lstm_prob = None

        # --- Transformer ---
        tr_prob = None
        tr_path = os.path.join(MODEL_DIR, f"{table}_transformer.keras")
        if os.path.exists(tr_path):
            try:
                lookback = _get_lookback(reg_entry, name)
                if len(X_scaled) >= lookback:
                    tr_model, actual_lb = _load_transformer(tr_path, lookback, n_features)
                    if tr_model is not None:
                        seq = X_scaled[-actual_lb:][np.newaxis, :, :]
                        tr_prob = float(tr_model.predict(seq, batch_size=1, verbose=0).flatten()[0])
            except Exception:
                tr_prob = None

        # --- Ensemble ---
        trend_strength = float(df['trend_strength'].iloc[-1]) if 'trend_strength' in df.columns else 0.01
        trend_gate = reg_entry.get('profile', {}).get('trend_gate', 0.01) if reg_entry else 0.01

        # Check registry for ensemble weights
        w_lstm_override = None
        if reg_entry:
            w_lstm_override = reg_entry.get('w_lstm', None)

        if lstm_prob is not None and tr_prob is not None:
            # 3-model ensemble: average LSTM and Transformer, then gate with CB
            combined_seq = (lstm_prob + tr_prob) / 2.0
            prob = float(ensemble_with_gating(
                np.array([cb_prob]), np.array([combined_seq]),
                np.array([trend_strength]), trend_gate, w_lstm_override
            )[0])
        elif lstm_prob is not None:
            prob = float(ensemble_with_gating(
                np.array([cb_prob]), np.array([lstm_prob]),
                np.array([trend_strength]), trend_gate, w_lstm_override
            )[0])
        elif tr_prob is not None:
            prob = float(ensemble_with_gating(
                np.array([cb_prob]), np.array([tr_prob]),
                np.array([trend_strength]), trend_gate, w_lstm_override
            )[0])
        else:
            prob = cb_prob

        # --- Thresholds ---
        thr = thresholds.get(name, {})
        buy_thr = thr.get('buy', _DEFAULT_BUY_THR)
        sell_thr = thr.get('sell', _DEFAULT_SELL_THR)

        if prob >= buy_thr:
            signal = "BUY"
        elif prob <= sell_thr:
            signal = "SELL"
        else:
            signal = "WAIT"

        confidence = abs(prob - 0.5) * 2.0
        kelly = _kelly_size(cb_accuracy)

        rows.append({
            "Asset": name,
            "Price": current_price,
            "Chg_1d": chg_1d,
            "Signal": signal,
            "Probability": round(prob, 4),
            "Confidence": round(confidence, 4),
            "RSI": round(rsi_val, 1) if not pd.isna(rsi_val) else float('nan'),
            "Trend": trend,
            "CB_Prob": round(cb_prob, 4),
            "LSTM_Prob": round(lstm_prob, 4) if lstm_prob is not None else float('nan'),
            "Kelly_Size": round(kelly, 4),
        })

    if progress:
        print(f"\r  [{'#' * 30}] {total}/{total}  Done.{' ' * 20}")

    if not rows:
        return pd.DataFrame(columns=[
            "Asset", "Price", "Chg_1d", "Signal", "Probability",
            "Confidence", "RSI", "Trend", "CB_Prob", "LSTM_Prob", "Kelly_Size"
        ])

    df_out = pd.DataFrame(rows)
    df_out['_dist'] = (df_out['Probability'] - 0.5).abs()
    df_out = df_out.sort_values('_dist', ascending=False).drop(columns=['_dist']).reset_index(drop=True)
    return df_out


def get_signal_summary(df=None):
    if df is None:
        df = get_all_signals()

    total_buy = int((df['Signal'] == 'BUY').sum())
    total_sell = int((df['Signal'] == 'SELL').sum())
    total_wait = int((df['Signal'] == 'WAIT').sum())

    buys = df[df['Signal'] == 'BUY']
    sells = df[df['Signal'] == 'SELL']

    strongest_buy = buys.loc[buys['Probability'].idxmax(), 'Asset'] if not buys.empty else None
    strongest_sell = sells.loc[sells['Probability'].idxmin(), 'Asset'] if not sells.empty else None

    return {
        'total_buy': total_buy,
        'total_sell': total_sell,
        'total_wait': total_wait,
        'strongest_buy': strongest_buy,
        'strongest_sell': strongest_sell,
    }


if __name__ == "__main__":
    print("Loading signals for all assets...")
    df = get_all_signals()

    if df.empty:
        print("No signals generated. Check models directory and market.db.")
        sys.exit(0)

    summary = get_signal_summary(df)
    print(f"\nSignal Summary: BUY={summary['total_buy']}  "
          f"SELL={summary['total_sell']}  WAIT={summary['total_wait']}")
    if summary['strongest_buy']:
        print(f"Strongest BUY:  {summary['strongest_buy']}")
    if summary['strongest_sell']:
        print(f"Strongest SELL: {summary['strongest_sell']}")

    print()

    display_cols = ["Asset", "Price", "Chg_1d", "Signal", "Probability",
                    "Confidence", "RSI", "Trend", "CB_Prob", "LSTM_Prob", "Kelly_Size"]

    try:
        from tabulate import tabulate
        print(tabulate(df[display_cols], headers="keys", tablefmt="rounded_outline",
                       floatfmt=".4f", showindex=False))
    except ImportError:
        # Manual fallback
        col_widths = {c: max(len(c), df[c].astype(str).str.len().max()) for c in display_cols}
        header = "  ".join(c.ljust(col_widths[c]) for c in display_cols)
        sep = "  ".join("-" * col_widths[c] for c in display_cols)
        print(header)
        print(sep)
        for _, row in df[display_cols].iterrows():
            line = "  ".join(str(row[c]).ljust(col_widths[c]) for c in display_cols)
            print(line)
