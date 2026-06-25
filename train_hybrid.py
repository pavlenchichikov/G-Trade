import os
import sys

# -- GPU: add the conda environment's Library\bin to PATH before importing TF -
# Needed when launching python.exe directly without `conda activate`
_env_lib_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(_env_lib_bin) and _env_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _env_lib_bin + os.pathsep + os.environ.get("PATH", "")
# -----------------------------------------------------------------------------

import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
import tensorflow as tf
from core.logger import get_logger

logger = get_logger("train_hybrid")
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import time

# --- GPU CONFIG ---
# Detect VRAM and size the TF memory pool to it. No GPU - run on CPU.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Detect available VRAM and reserve ~60% for TF (rest for cuDNN workspaces + OS)
        _vram_mb = None
        try:
            import subprocess as _sp_vram
            _smi_r = _sp_vram.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if _smi_r.returncode == 0:
                _vram_mb = int(_smi_r.stdout.strip().split("\n")[0].strip())
        except Exception as _e:
            logger.debug("VRAM detection failed: %s", _e)

        if _vram_mb and _vram_mb > 0:
            # Reserve 60% of VRAM for TF pool, leave rest for cuDNN + OS
            _tf_pool_mb = int(_vram_mb * 0.60)
            _tf_pool_mb = max(1024, min(_tf_pool_mb, _vram_mb - 1024))  # at least 1GB free
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_pool_mb)]
            )
            print(f"  [GPU] {gpus[0].name}  |  VRAM: {_vram_mb}MB  |  TF pool: {_tf_pool_mb}MB")
        else:
            # VRAM unknown - use memory_growth (safe for any GPU)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  [GPU] {gpus[0].name}  |  memory_growth=True")
    except (RuntimeError, AttributeError):
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  [GPU] {gpus[0].name}  |  memory_growth=True (fallback)")
else:
    print("  [CPU] No GPU detected - running on CPU")

tf.get_logger().setLevel('ERROR')

_HAS_GPU = bool(gpus)


def _env_int(name: str, default: int) -> int:
    """Read a positive int env override, falling back to `default`."""
    try:
        v = int((os.getenv(name) or "").strip())
        return v if v >= 1 else default
    except (TypeError, ValueError):
        return default


# --- PARALLEL LOCKS ---
# Neural block (LSTM+TF+TCN) is gated by a semaphore. GPU: 1 slot (concurrent
# cuDNN fits OOM outside the TF pool). CPU: small batches don't saturate 12 cores,
# so run several models at once, each capped to a few intra-op threads - total
# threads ~= cores. Same models/epochs, scheduling only. Env-tunable below.
_cores = os.cpu_count() or 8
if _HAS_GPU:
    _GPU_SLOTS = _env_int("GTRADE_NEURAL_SLOTS", 1)
    _CB_THREADS = _env_int("GTRADE_CB_THREADS", max(4, _cores))
    _N_WORKERS = _env_int("GTRADE_WORKERS", 10)
else:
    _TF_THREADS = _env_int("GTRADE_TF_THREADS", 3)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(_TF_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except (RuntimeError, ValueError):
        pass
    _GPU_SLOTS = _env_int("GTRADE_NEURAL_SLOTS", max(1, _cores // _TF_THREADS))
    _CB_THREADS = _env_int("GTRADE_CB_THREADS", max(2, _TF_THREADS))  # CB runs alongside nets
    _N_WORKERS = _env_int("GTRADE_WORKERS", _GPU_SLOTS + 2)

_gpu_lock = threading.Semaphore(_GPU_SLOTS)
_print_lock = threading.Lock()

# CatBoost device. CatBoost bundles its own CUDA and runs on native Windows
# (unlike TF, whose Windows wheel is CPU-only), so this is independent of TF's
# _HAS_GPU. Default CPU: on the small per-asset datasets here the host-device
# transfer overhead can make GPU SLOWER than CPU, so benchmark before keeping it
# on. Set GTRADE_CB_DEVICE=GPU to enable. Only the heavy champion fit (700 iters)
# moves to GPU; the tiny helper fits (adversarial validation, feature selection)
# stay on CPU. _cb_gpu_lock serializes GPU fits so a 4GB card doesn't OOM when
# several assets train at once (CPU CatBoost stays fully parallel, no lock).
_CB_TASK_TYPE = "GPU" if (os.getenv("GTRADE_CB_DEVICE") or "").strip().upper() == "GPU" else "CPU"
_cb_gpu_lock = threading.Semaphore(_env_int("GTRADE_CB_GPU_SLOTS", 1))

# Data-adaptive net right-sizing (see docs/.../adaptive-net-rightsizing-design.md).
# OFF by default: the builders use their original flat 192/128/64 sizes, exactly
# reproducing the current models. ON: nets are sized once per asset to its data
# volume (less overfit on small assets, fewer params so faster). Warm-start reuses
# the previous fold's weights (shapes are constant per asset); it defaults to
# following the adaptive flag but can be toggled on its own.
_ADAPTIVE_NETS = (os.getenv("GTRADE_ADAPTIVE_NETS") or "").strip() in ("1", "true", "True")
_warmstart_env = (os.getenv("GTRADE_NET_WARMSTART") or "").strip()
_NET_WARMSTART = (_warmstart_env in ("1", "true", "True")) if _warmstart_env else _ADAPTIVE_NETS

# Capacity / epoch knobs (defaults reproduce the historical config exactly, so
# behaviour is unchanged unless these are set). Lowering GTRADE_NET_CAP shrinks
# the LSTM on data-rich assets, the main speed/RAM lever for the heavy tickers.
_NET_CAP = _env_int("GTRADE_NET_CAP", 128)
_EP_LSTM = _env_int("GTRADE_EPOCHS_LSTM", 160)
_EP_TF = _env_int("GTRADE_EPOCHS_TF", 100)
_EP_TCN = _env_int("GTRADE_EPOCHS_TCN", 80)


def _ws_load(model, key, store):
    """Warm-start: load the previous fold's weights when shapes match (best effort)."""
    if _NET_WARMSTART and store.get(key) is not None:
        try:
            model.set_weights(store[key])
        except Exception:
            pass  # shape changed - fall back to fresh init


def _ws_save(model, key, store):
    if _NET_WARMSTART:
        store[key] = model.get_weights()

# --- SHARED EPOCH STATE (callback - ticker thread) ---
_progress_state = {'label': '-', 'epoch': 0, 'total_ep': 0, 'loss': float('nan')}
_state_lock = threading.Lock()

# fp16 speeds things up on cards with Tensor Cores; older cards stay on fp32
if _HAS_GPU:
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except Exception:
        logger.warning("mixed_float16 not supported on this GPU, using float32")
        print("  [WARN] mixed_float16 not supported on this GPU, using float32")

# -- Imports from core/ modules ------------------------------------------------
from core.architectures import (
    build_transformer_encoder,
    build_lstm_multitask, build_tcn, adaptive_units,
)
from core.profiles import (
    FOREX,
    get_profile,
)
from core.features import engineer_features, add_weekly_features, add_crossasset_features, add_macro_features, add_cross_lag_features, active_candidate_features
from core.backtesting import (
    adaptive_split_params, make_walk_forward_splits,
    pnl_from_signals, max_drawdown_from_returns, score_strategy,
    sharpe_from_returns,
    make_signals, apply_regime_filter,
    COMMISSION, SLIPPAGE, FOREX_COMMISSION, FOREX_SLIPPAGE,
    MAX_TRADE_RET,
)
from core.ensemble import build_stacking_features
from core.ensemble import ensemble_with_gating, tune_ensemble_weights  # noqa: F401 (re-exported for signal_engine.py)
from core.scaling import save_scaler, scaler_path
from core.calibration import fit_calibrator, save_calibrator


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


# Retraining policy (env flags):
#   GTRADE_FORCE_PROMOTE=1   accept newly trained champions regardless of score.
#   GTRADE_RETRAIN_FROZEN=1  train only assets that lack a saved scaler (the ones
#                            that previously froze) AND force-promote them, so a
#                            single run migrates every frozen asset to the new
#                            pipeline (saved scaler + calibrator + live features).
_FORCE_PROMOTE = _env_flag("GTRADE_FORCE_PROMOTE") or _env_flag("GTRADE_RETRAIN_FROZEN")
_ONLY_FROZEN = _env_flag("GTRADE_RETRAIN_FROZEN")


def _asset_table(asset: str) -> str:
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


class EpochStateCallback(Callback):
    """Callback: only writes state to _progress_state, never prints."""
    def __init__(self, asset: str, fold: int, total_epochs: int,
                 label: str = "LSTM", val_metric: str = "val_loss"):
        super().__init__()
        self._val_metric = val_metric
        self._label = f"{label} {asset} f{fold}/{total_epochs}"
        self._total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get(self._val_metric, float('nan'))
        with _state_lock:
            _progress_state['label'] = self._label
            _progress_state['epoch'] = epoch + 1
            _progress_state['total_ep'] = self._total
            _progress_state['loss'] = val_loss

    def on_train_end(self, logs=None):
        pass

# --- PATH SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from config import FULL_ASSET_MAP
except ImportError:
    exit(f"ERROR: config.py not found in {BASE_DIR}!")

DB_PATH = os.path.join(BASE_DIR, "market.db")
engine = create_engine(f'sqlite:///{DB_PATH}')
# GTRADE_MODEL_DIR redirects all training output (champions, registry, quality
# report, thresholds) to an isolated dir - used by ab_validate.py so an A/B run
# never touches the production models/. Default = the real models/ dir.
MODEL_DIR = os.getenv("GTRADE_MODEL_DIR") or os.path.join(BASE_DIR, "models")
EXPERIMENTS_DIR = os.path.join(MODEL_DIR, "experiments")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")

# --- ASSET PROFILES --- (imported from core.profiles above)

# Trading constants imported from core.backtesting above
INDICATOR_ONLY_ASSETS = set()

# --- ARCHITECTURES --- (imported from core.architectures above)


def adversarial_fold_weight(X_train, X_test):
    """Detect distribution shift between train and test via adversarial validation.
    Returns weight in [0.3, 1.0]: 1.0 = same distribution, 0.3 = heavy shift.
    Used to penalize fold scores when train/test distributions diverge."""
    n_tr, n_te = len(X_train), len(X_test)
    if n_tr < 30 or n_te < 20:
        return 1.0
    max_n = 400
    rng = np.random.RandomState(42)
    X_tr_sub = X_train[rng.choice(n_tr, min(n_tr, max_n), replace=False)]
    X_te_sub = X_test[rng.choice(n_te, min(n_te, max_n), replace=False)]
    X = np.vstack([X_tr_sub, X_te_sub])
    y = np.concatenate([np.zeros(len(X_tr_sub)), np.ones(len(X_te_sub))])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = len(X) // 2
    try:
        cb = CatBoostClassifier(iterations=50, depth=3, verbose=0,
                                task_type='CPU', thread_count=_CB_THREADS)
        cb.fit(X[:split], y[:split])
        probs = cb.predict_proba(X[split:])[:, 1]
        auc = roc_auc_score(y[split:], probs)
    except Exception:
        return 1.0
    weight = max(0.3, 2.0 - 2.0 * auc)
    return float(min(1.0, weight))


# -- Functions moved to core/ modules (imported above) ------------------------
# build_stacking_features - core.ensemble
# get_profile - core.profiles
# engineer_features, add_weekly_features, add_crossasset_features - core.features
# adaptive_split_params, make_walk_forward_splits - core.backtesting
# pnl_from_signals, max_drawdown_from_returns, score_strategy - core.backtesting

def build_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def filter_noisy_samples(X_seq, y_seq, y_mag, noise_threshold=0.3):
    """Filter out noisy samples where |next_ret| is too small (random noise).
    Keeps only samples where magnitude > noise_threshold (after normalization).
    Returns filtered arrays + original indices mask."""
    mag_abs = np.abs(y_mag)
    mask = mag_abs > noise_threshold  # ~0.3 of normalized magnitude = ~1.5% real move
    if mask.sum() < max(50, len(mask) * 0.3):
        # If too few samples survive, relax threshold
        sorted_mag = np.sort(mag_abs)
        cutoff_idx = max(50, int(len(sorted_mag) * 0.3))
        adaptive_thr = sorted_mag[-cutoff_idx] if cutoff_idx < len(sorted_mag) else 0.0
        mask = mag_abs > adaptive_thr
    return X_seq[mask], y_seq[mask], y_mag[mask], mask

def derive_feature_set(df, train_idx, candidate_features, top_k):
    X = df.loc[train_idx, candidate_features].values
    y = df.loc[train_idx, 'target'].values
    cb = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05, verbose=0,
                            task_type='CPU', thread_count=_CB_THREADS)
    cb.fit(X, y)
    imps = cb.get_feature_importance()
    ranked = [f for _, f in sorted(zip(imps, candidate_features), reverse=True)]
    return ranked[:min(top_k, len(ranked))]

# ensemble_with_gating, tune_ensemble_weights - core.ensemble (imported above)
# make_signals, apply_regime_filter - core.backtesting (imported above)

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_registry(reg):
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

_stop_requested = False

def _signal_handler(sig, frame):
    global _stop_requested
    if _stop_requested:
        print("\n\n  [FORCE] Second Ctrl+C - force quit.")
        sys.exit(1)
    _stop_requested = True
    print("\n\n" + "=" * 60)
    print("  [STOP] Ctrl+C received. Finishing current assets...")
    print("         Press Ctrl+C again to force quit.")
    print("=" * 60)


def _safe_print(*args, flush=True):
    with _print_lock:
        print(*args, flush=flush)


def _load_optuna_params():
    if os.path.exists(OPTUNA_PARAMS_PATH):
        with open(OPTUNA_PARAMS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

_optuna_params_cache = None
_optuna_params_lock  = threading.Lock()

def _get_optuna_params(asset):
    """Return Optuna-tuned params for asset, or {} if not found."""
    global _optuna_params_cache
    with _optuna_params_lock:
        if _optuna_params_cache is None:
            _optuna_params_cache = _load_optuna_params()
    return _optuna_params_cache.get(asset, {})


import gc as _gc


def _free_keras_from_fold(fold_info):
    """Drop the heavy Keras models from fold_info['models'] to free GPU/RAM.
    Keeps CatBoost (small) and the scaler for stats. Call right after scoring."""
    models = fold_info.get('models')
    if models is None:
        return
    for key in ('lstm', 'tf_enc', 'tcn'):
        m = models.pop(key, None)
        if m is not None:
            try:
                del m
            except Exception as _e:
                logger.debug("Model cleanup failed: %s", _e)
    # meta (LogisticRegression) is small too - keep it
    _gc.collect()


def _train_one_asset(asset, candidate_features, prev_registry_entry):
    """Train a single asset. Runs inside a ThreadPoolExecutor."""
    profile = get_profile(asset)
    table = asset.lower().replace("^", "").replace(".", "").replace("-", "")
    # -- Load Optuna-tuned hyperparams if available ----------------------
    opt = _get_optuna_params(asset)

    pass  # progress shown by live ticker
    try:
        df_raw = pd.read_sql(f"SELECT * FROM {table}", engine, index_col="Date", parse_dates=["Date"])
        df_raw.index = pd.to_datetime(df_raw.index).normalize()
        df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
        df = engineer_features(df_raw)
        df = add_weekly_features(df, table, engine)
        df = add_crossasset_features(df, table, engine)
        df = add_macro_features(df, engine)
        df = add_cross_lag_features(df, engine)
        sp = adaptive_split_params(len(df))
        if sp is None:
            _safe_print(f"  [SKIP] {asset:<12} insufficient rows ({len(df)})")
            return None

        # Embargo = sequence lookback: purges overlap between train/val/test so
        # sequence models cannot peek at neighbouring windows (purged walk-forward).
        _embargo = int(opt.get('lookback', profile['lookback']))
        splits = make_walk_forward_splits(
            len(df),
            min_train=sp['min_train'],
            val_size=sp['val_size'],
            test_size=sp['test_size'],
            step=sp['step'],
            embargo=_embargo,
        )
        if not splits:
            _safe_print(f"  [SKIP] {asset:<12} no walk-forward windows ({len(df)} rows)")
            return None

        # Long histories yield 12-18 folds, each retraining all 3 nets. Champion is
        # the median of folds (stable past ~8-10) and recent windows match live best,
        # so keep only the newest GTRADE_MAX_FOLDS.
        _max_folds = _env_int("GTRADE_MAX_FOLDS", 10)
        if _max_folds and len(splits) > _max_folds:
            _dropped = len(splits) - _max_folds
            splits = splits[-_max_folds:]
            _safe_print(f"  [FOLDS] {asset:<12} {_max_folds + _dropped} - {_max_folds} (recent)")

        # Filter candidate_features to those present in df
        available_features = [f for f in candidate_features if f in df.columns]

        # Feature selection (CatBoost CPU, half cores)
        # Use Optuna features if available, else derive from CatBoost importance
        if opt.get('selected_features'):
            selected = [f for f in opt['selected_features'] if f in available_features]
            if len(selected) < 4:  # fallback if features changed
                selected = derive_feature_set(df, slice(0, sp['min_train']),
                                              available_features, profile['top_k_features'])
        else:
            selected = derive_feature_set(df, slice(0, sp['min_train']),
                                          available_features, profile['top_k_features'])
        # Optuna lookback overrides profile default
        lookback = int(opt.get('lookback', profile['lookback']))

        # --- PRE-COMPUTE: all fold data computed up front ---
        precomputed = []
        for tr, va, te in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(df.loc[tr, selected].values)
            y_tr = df.loc[tr, 'target'].values
            X_va = scaler.transform(df.loc[va, selected].values)
            y_va = df.loc[va, 'target'].values
            X_te = scaler.transform(df.loc[te, selected].values)
            y_te = df.loc[te, 'target'].values

            X_all_seq = np.vstack([X_tr, X_va, X_te])
            y_all_seq = np.concatenate([y_tr, y_va, y_te])
            X_seq, y_seq = build_sequences(X_all_seq, y_all_seq, lookback)

            seq_tr_end = max(0, len(X_tr) - lookback)
            seq_va_end = seq_tr_end + len(X_va)
            seq_te_end = seq_va_end + len(X_te)

            # Magnitude targets for multi-task LSTM
            y_next_all = np.concatenate([
                df.loc[tr, 'next_ret'].values,
                df.loc[va, 'next_ret'].values,
                df.loc[te, 'next_ret'].values,
            ])
            _, y_mag_all_seq = build_sequences(X_all_seq, y_next_all, lookback)

            precomputed.append({
                'scaler': scaler,
                'X_train': X_tr, 'y_train': y_tr,
                'X_val': X_va, 'y_val': y_va,
                'X_test': X_te, 'y_test': y_te,
                'X_seq_train': X_seq[:seq_tr_end], 'y_seq_train': y_seq[:seq_tr_end],
                'X_seq_val': X_seq[seq_tr_end:seq_va_end], 'y_seq_val': y_seq[seq_tr_end:seq_va_end],
                'X_seq_test': X_seq[seq_va_end:seq_te_end], 'y_seq_test': y_seq[seq_va_end:seq_te_end],
                'va': va, 'te': te,
                'y_mag_seq_train': y_mag_all_seq[:seq_tr_end],
                'y_mag_seq_val':   y_mag_all_seq[seq_tr_end:seq_va_end],
            })

        fold_metrics = []
        best_fold = None
        best_fold_score = -1e9

        # Right-size the nets to this asset's data ONCE (from the largest fold),
        # so the architecture is constant across folds and warm-start can reuse
        # weights. Flag off means empty kwargs, so builders keep their flat defaults.
        _max_seq = max((len(p['X_seq_train']) for p in precomputed), default=0)
        if _ADAPTIVE_NETS and _max_seq > 0:
            _u1 = adaptive_units(_max_seq, lo=32, hi=_NET_CAP, divisor=16)
            # No recurrent_dropout: it forces Keras off the fused LSTM kernel and
            # is ~1.4x slower on CPU. L2 + the data-adaptive (smaller) size carry
            # the regularization instead.
            _lstm_kw = dict(units1=_u1, units2=max(16, _u1 // 2),
                            head_dim=max(16, _u1 // 2), l2_reg=1e-5)
            _tf_kw = dict(num_heads=2,
                          ff_dim=adaptive_units(_max_seq, lo=32, hi=96, divisor=24),
                          dropout=0.15)
            _tcn_kw = dict(n_filters=adaptive_units(_max_seq, lo=24, hi=64, divisor=24),
                           dropout=0.20)
        else:
            _lstm_kw, _tf_kw, _tcn_kw = {}, {}, {}
        _warm = {}  # per-asset weight carry across folds (warm-start)

        for k, fold_data in enumerate(precomputed, 1):
            if _stop_requested:
                _safe_print(f"  [STOP] {asset:<12} interrupted on fold {k}/{len(splits)}")
                break

            X_train = fold_data['X_train']; y_train = fold_data['y_train']
            X_val = fold_data['X_val'];     y_val = fold_data['y_val']
            X_test = fold_data['X_test'];   y_test = fold_data['y_test']
            X_seq_train = fold_data['X_seq_train']; y_seq_train = fold_data['y_seq_train']
            X_seq_val = fold_data['X_seq_val'];     y_seq_val = fold_data['y_seq_val']
            X_seq_test = fold_data['X_seq_test'];   y_seq_test = fold_data['y_seq_test']
            va = fold_data['va']; te = fold_data['te']
            scaler = fold_data['scaler']

            # --- CatBoost (CPU, half cores) in a background thread ---
            cb_result = {}

            def _train_catboost(_X_train=X_train, _y_train=y_train,
                                _X_val=X_val, _y_val=y_val,
                                _X_test=X_test, _y_test=y_test,
                                _opt=opt):
                _cb_kwargs = dict(
                    iterations=int(_opt.get('cb_iterations', 700)),
                    depth=int(_opt.get('cb_depth', 8)),
                    learning_rate=float(_opt.get('cb_lr', 0.03)),
                    verbose=0,
                    task_type=_CB_TASK_TYPE,
                    thread_count=_CB_THREADS,
                    early_stopping_rounds=50,
                )
                if _CB_TASK_TYPE == 'GPU':
                    _cb_kwargs['devices'] = '0'
                cb = CatBoostClassifier(**_cb_kwargs)
                if _CB_TASK_TYPE == 'GPU':
                    # 4GB VRAM: only one champion fit on the card at a time.
                    with _cb_gpu_lock:
                        cb.fit(_X_train, _y_train, eval_set=(_X_val, _y_val))
                else:
                    cb.fit(_X_train, _y_train, eval_set=(_X_val, _y_val))
                cb_result['cb'] = cb
                cb_result['cb_val'] = cb.predict_proba(_X_val)[:, 1]
                cb_result['cb_test'] = cb.predict_proba(_X_test)[:, 1]
                cb_result['cb_acc'] = float(((cb.predict_proba(_X_test)[:, 1] >= 0.5).astype(int) == _y_test).mean())

            cb_thread = threading.Thread(target=_train_catboost)
            cb_thread.start()

            # --- LSTM (GPU: build + fit + predict under gpu_lock, one at a time) ---
            BATCH = 256  # default batch size; works well on 4GB+ VRAM GPUs with fp16
            y_mag_train = np.clip(fold_data['y_mag_seq_train'] / 0.05, -1.0, 1.0).astype('float32')
            y_mag_val_d  = np.clip(fold_data['y_mag_seq_val']   / 0.05, -1.0, 1.0).astype('float32')

            # GPU models use ALL training data - noise filter discards ~70% of samples,
            # leaving 150 samples - batch shrinks to 64-75 - GPU busy 5ms/step, idle 50ms - 0% util.
            # LSTM/TF/TCN have Dropout(0.15-0.20) which handles noise natively.
            # CatBoost uses X_train (flat, non-sequential) - unaffected by this change.
            _batch = min(BATCH, max(64, len(X_seq_train) // 4))

            with _gpu_lock:
                _dev = '/GPU:0' if _HAS_GPU else '/CPU:0'
                with tf.device(_dev):
                    if k == 1:
                        _placed = tf.constant(1.0).device
                        try:
                            _mem = tf.config.experimental.get_memory_info('GPU:0')
                            _vram = f"  VRAM={_mem['current']//1024//1024}MB"
                        except Exception:
                            _vram = ""
                        _safe_print(
                            f"  [Device] {asset:<10} - {_placed}"
                            f"  train={len(X_seq_train)} seq  batch={_batch}{_vram}"
                        )
                    # -- Multi-task LSTM --------------------------------------------
                    lstm_mt = build_lstm_multitask((lookback, len(selected)),
                                                   n_train_samples=len(X_seq_train),
                                                   **_lstm_kw)
                    _ws_load(lstm_mt, 'lstm', _warm)
                    train_ds = tf.data.Dataset.from_tensor_slices((
                        X_seq_train.astype('float32'),
                        {'direction': y_seq_train.astype('float32'),
                         'magnitude': y_mag_train}
                    )).shuffle(len(X_seq_train)).batch(_batch).prefetch(tf.data.AUTOTUNE)
                    val_ds = tf.data.Dataset.from_tensor_slices((
                        X_seq_val.astype('float32'),
                        {'direction': y_seq_val.astype('float32'),
                         'magnitude': y_mag_val_d}
                    )).batch(BATCH).prefetch(tf.data.AUTOTUNE)
                    # mode='min' is required on Keras 3 / TF 2.20+: it cannot infer
                    # the direction for a multi-output loss name like
                    # 'val_direction_loss' and raises without an explicit mode.
                    es_lstm = EarlyStopping(monitor='val_direction_loss', patience=10,
                                            mode='min', restore_best_weights=True)
                    _lstm_cb = EpochStateCallback(asset, k, _EP_LSTM,
                                                 label="LSTM", val_metric="val_direction_loss")
                    lstm_mt.fit(train_ds, validation_data=val_ds, epochs=_EP_LSTM,
                                callbacks=[es_lstm, _lstm_cb], verbose=0)
                    _ws_save(lstm_mt, 'lstm', _warm)
                    lstm_test_prob = lstm_mt.predict(
                        X_seq_test.astype('float32'), batch_size=BATCH, verbose=0)[0].flatten()
                    lstm_val_prob  = lstm_mt.predict(
                        X_seq_val.astype('float32'),  batch_size=BATCH, verbose=0)[0].flatten()
                    # Extract direction-only model - compatible with predict.py / backtest.py
                    lstm = Model(inputs=lstm_mt.input,
                                 outputs=lstm_mt.get_layer('direction').output)
                    del lstm_mt  # Python obj freed; shared layers still in lstm - OK

                    # -- Transformer encoder --------------------------------------
                    tf_enc = build_transformer_encoder((lookback, len(selected)),
                                                       n_train_samples=len(X_seq_train),
                                                       **_tf_kw)
                    _ws_load(tf_enc, 'tf', _warm)
                    train_ds_tf = tf.data.Dataset.from_tensor_slices(
                        (X_seq_train.astype('float32'), y_seq_train.astype('float32'))
                    ).shuffle(len(X_seq_train)).batch(_batch).prefetch(tf.data.AUTOTUNE)
                    val_ds_tf = tf.data.Dataset.from_tensor_slices(
                        (X_seq_val.astype('float32'), y_seq_val.astype('float32'))
                    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)
                    es_tf = EarlyStopping(monitor='val_loss', patience=8,
                                          mode='min', restore_best_weights=True)
                    _tf_cb = EpochStateCallback(asset, k, _EP_TF,
                                               label="TF", val_metric="val_loss")
                    tf_enc.fit(train_ds_tf, validation_data=val_ds_tf, epochs=_EP_TF,
                               callbacks=[es_tf, _tf_cb], verbose=0)
                    _ws_save(tf_enc, 'tf', _warm)
                    tf_test_prob = tf_enc.predict(
                        X_seq_test.astype('float32'), batch_size=BATCH, verbose=0).flatten()
                    tf_val_prob  = tf_enc.predict(
                        X_seq_val.astype('float32'),  batch_size=BATCH, verbose=0).flatten()

                    # -- TCN (4th ensemble member) -------------------------------
                    tcn_model = build_tcn((lookback, len(selected)),
                                          n_train_samples=len(X_seq_train),
                                          **_tcn_kw)
                    _ws_load(tcn_model, 'tcn', _warm)
                    train_ds_tcn = tf.data.Dataset.from_tensor_slices(
                        (X_seq_train.astype('float32'), y_seq_train.astype('float32'))
                    ).shuffle(len(X_seq_train)).batch(_batch).prefetch(tf.data.AUTOTUNE)
                    val_ds_tcn = tf.data.Dataset.from_tensor_slices(
                        (X_seq_val.astype('float32'), y_seq_val.astype('float32'))
                    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)
                    es_tcn = EarlyStopping(monitor='val_loss', patience=7,
                                           mode='min', restore_best_weights=True)
                    _tcn_cb = EpochStateCallback(asset, k, _EP_TCN,
                                                 label="TCN", val_metric="val_loss")
                    tcn_model.fit(train_ds_tcn, validation_data=val_ds_tcn, epochs=_EP_TCN,
                                  callbacks=[es_tcn, _tcn_cb], verbose=0)
                    _ws_save(tcn_model, 'tcn', _warm)
                    tcn_test_prob = tcn_model.predict(
                        X_seq_test.astype('float32'), batch_size=BATCH, verbose=0).flatten()
                    tcn_val_prob = tcn_model.predict(
                        X_seq_val.astype('float32'), batch_size=BATCH, verbose=0).flatten()

            lstm_acc = float(((lstm_test_prob >= 0.5).astype(int) == y_seq_test).mean())

            # Wait for CatBoost
            cb_thread.join()
            cb = cb_result['cb']
            cb_val = cb_result['cb_val']
            cb_test = cb_result['cb_test']
            cb_acc = cb_result['cb_acc']

            # align arrays
            val_close = df.loc[va, 'close'].values
            val_sma200 = df.loc[va, 'sma_200'].values
            val_taleb = df.loc[va, 'taleb_risk'].values
            val_ret = df.loc[va, 'next_ret'].values
            val_trend = df.loc[va, 'trend_strength'].values
            cb_val_aligned = cb_val

            test_close = df.loc[te, 'close'].values
            test_sma200 = df.loc[te, 'sma_200'].values
            test_taleb = df.loc[te, 'taleb_risk'].values
            test_ret = df.loc[te, 'next_ret'].values
            test_trend = df.loc[te, 'trend_strength'].values
            cb_test_aligned = cb_test

            n_val = min(len(cb_val_aligned), len(lstm_val_prob), len(val_trend), len(val_close), len(val_sma200), len(val_taleb), len(val_ret))
            cb_val_aligned = cb_val_aligned[:n_val]
            lstm_val_prob = lstm_val_prob[:n_val]
            val_trend = val_trend[:n_val]
            val_close = val_close[:n_val]
            val_sma200 = val_sma200[:n_val]
            val_taleb = val_taleb[:n_val]
            val_ret = val_ret[:n_val]

            n_test = min(len(cb_test_aligned), len(lstm_test_prob), len(test_trend), len(test_close), len(test_sma200), len(test_taleb), len(test_ret))
            cb_test_aligned = cb_test_aligned[:n_test]
            lstm_test_prob = lstm_test_prob[:n_test]
            test_trend = test_trend[:n_test]
            test_close = test_close[:n_test]
            test_sma200 = test_sma200[:n_test]
            test_taleb = test_taleb[:n_test]
            test_ret = test_ret[:n_test]

            # -- Adversarial validation: detect distribution shift ------
            adv_weight = adversarial_fold_weight(X_train, X_test)

            # -- Align all 4 model probs ---------------------------------
            val_target_aligned = df.loc[va, 'target'].values[:n_val]
            tf_val_al   = tf_val_prob[:n_val]
            tf_test_al  = tf_test_prob[:n_test]
            tcn_val_al  = tcn_val_prob[:n_val]
            tcn_test_al = tcn_test_prob[:n_test]

            # -- Stacking meta-classifier --------------------------------
            X_meta_val = build_stacking_features(
                cb_val_aligned, lstm_val_prob[:n_val], tf_val_al, tcn_val_al,
                val_trend)
            X_meta_test = build_stacking_features(
                cb_test_aligned[:n_test], lstm_test_prob[:n_test],
                tf_test_al, tcn_test_al, test_trend)

            try:
                meta_clf = LogisticRegression(C=1.0, max_iter=300, solver='lbfgs')
                meta_clf.fit(X_meta_val, val_target_aligned)
                val_prob = meta_clf.predict_proba(X_meta_val)[:, 1]
                test_prob = meta_clf.predict_proba(X_meta_test)[:, 1]
            except Exception:
                # Fallback: equal-weight average
                meta_clf = None
                val_prob = 0.25 * (cb_val_aligned + lstm_val_prob[:n_val]
                                   + tf_val_al + tcn_val_al)
                test_prob = 0.25 * (cb_test_aligned[:n_test]
                                    + lstm_test_prob[:n_test]
                                    + tf_test_al + tcn_test_al)

            # threshold tuning on validation (top-3 averaging for stability)
            comm = FOREX_COMMISSION if asset in FOREX else COMMISSION
            slip = FOREX_SLIPPAGE if asset in FOREX else SLIPPAGE
            all_configs = []
            for b in profile['thr_buy_grid']:
                for s in profile['thr_sell_grid']:
                    if s >= b:
                        continue
                    sig = make_signals(val_prob, b, s, profile['no_trade_band'])
                    sig = apply_regime_filter(sig, val_close, val_sma200, val_taleb, profile['regime_risk_cap'])
                    ret_stream = [(float(np.clip((r if g > 0 else -r), -MAX_TRADE_RET, MAX_TRADE_RET)) - (comm + slip)) for g, r in zip(sig, val_ret) if g != 0 and not np.isnan(r)]
                    p, t, w = pnl_from_signals(sig, val_ret, commission=comm, slippage=slip)
                    mdd = max_drawdown_from_returns(ret_stream)
                    score = score_strategy(p, mdd, w, t, sharpe=sharpe_from_returns(ret_stream))
                    all_configs.append((b, s, p, t, w, mdd, score))

            # Average top-3 thresholds for stability (reduces overfitting)
            top3 = sorted(all_configs, key=lambda x: x[6], reverse=True)[:3]
            buy_thr = float(np.mean([c[0] for c in top3]))
            sell_thr = float(np.mean([c[1] for c in top3]))
            val_profit = top3[0][2]
            top3[0][3]
            top3[0][4]
            val_mdd = top3[0][5]

            sig_test = make_signals(test_prob, buy_thr, sell_thr, profile['no_trade_band'])
            sig_test = apply_regime_filter(sig_test, test_close, test_sma200, test_taleb, profile['regime_risk_cap'])
            test_returns = [(float(np.clip((r if g > 0 else -r), -MAX_TRADE_RET, MAX_TRADE_RET)) - (comm + slip)) for g, r in zip(sig_test, test_ret) if g != 0 and not np.isnan(r)]
            test_profit, test_trades, test_win = pnl_from_signals(sig_test, test_ret, commission=comm, slippage=slip)
            test_mdd = max_drawdown_from_returns(test_returns)
            test_score = score_strategy(test_profit, test_mdd, test_win, test_trades,
                                        sharpe=sharpe_from_returns(test_returns))

            # Apply adversarial weight to score (penalize distribution-shifted folds)
            test_score_weighted = test_score * adv_weight

            fold_info = {
                'fold': k,
                'features': selected,
                'cb_acc': cb_acc,
                'lstm_acc': lstm_acc,
                'buy_thr': buy_thr,
                'sell_thr': sell_thr,
                'val_profit': val_profit,
                'val_mdd': val_mdd,
                'test_profit': test_profit,
                'test_mdd': test_mdd,
                'test_trades': int(test_trades),
                'test_winrate': test_win,
                'score': test_score_weighted,
                'adv_weight': adv_weight,
                'models': {'cb': cb, 'lstm': lstm, 'tf_enc': tf_enc,
                           'tcn': tcn_model, 'meta': meta_clf, 'scaler': scaler},
                'ensemble_mode': 'stacking' if meta_clf is not None else 'avg',
                # Kept for probability calibration at champion-save time.
                'val_prob': np.asarray(val_prob, dtype=float),
                'val_target': np.asarray(val_target_aligned, dtype=int),
            }
            fold_metrics.append(fold_info)

            # -- EAGER MEMORY MANAGEMENT -------------------------------------
            # Keep Keras models ONLY for the running-best fold (by weighted score).
            # All other folds: strip lstm/tf_enc/tcn immediately to prevent VRAM accumulation.
            # Root cause of OOM crash: 60 folds x 4 workers x 3 models = hundreds of objects.
            if test_score_weighted > best_fold_score:
                # This fold is new best - free previous best's Keras models
                if best_fold is not None:
                    _free_keras_from_fold(best_fold)
                best_fold_score = test_score_weighted
                best_fold = fold_info          # keep this fold's models intact
            else:
                # Not best - free Keras models immediately
                _free_keras_from_fold(fold_info)

        # --- Post-processing ---
        valid_folds = [f for f in fold_metrics if f['score'] > -999 and f['test_trades'] >= 10]
        if not valid_folds:
            _safe_print(f"  [WARN] {asset:<12} no robust folds")
            return None

        pos_ratio = sum(1 for f in valid_folds if f['test_profit'] > 0) / len(valid_folds)
        median_score = float(np.median([f['score'] for f in valid_folds]))
        median_profit = float(np.median([f['test_profit'] for f in valid_folds]))

        if pos_ratio < 0.45:
            _safe_print(f"  [WARN] {asset:<12} unstable folds (pos_ratio={pos_ratio:.2f})")

        best_fold = sorted(valid_folds, key=lambda x: x['score'])[-1]
        best_fold['score'] = median_score
        best_fold['test_profit'] = median_profit

        # Save models
        cb_out = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
        lstm_out = os.path.join(MODEL_DIR, f"{table}_lstm.keras")

        promote = (_FORCE_PROMOTE
                   or prev_registry_entry is None
                   or best_fold['score'] > (prev_registry_entry.get('score', -1e9) + 0.2))

        # If final best_fold differs from running best (edge case: adv_weight divergence),
        # its Keras models may have already been freed - fall back to FROZEN_CHAMPION.
        _bm = best_fold.get('models', {})
        if not _bm.get('lstm') or not _bm.get('tf_enc') or not _bm.get('tcn'):
            promote = False  # models unavailable - keep existing champion

        if promote:
            best_fold['models']['cb'].save_model(cb_out)
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['lstm'].save(lstm_out)
            tf_out = os.path.join(MODEL_DIR, f"{table}_transformer.keras")
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['tf_enc'].save(tf_out)
            tcn_out = os.path.join(MODEL_DIR, f"{table}_tcn.keras")
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['tcn'].save(tcn_out)
            meta_out = os.path.join(MODEL_DIR, f"{table}_meta.pkl")
            if best_fold['models']['meta'] is not None:
                joblib.dump(best_fold['models']['meta'], meta_out)
            # Persist the train-fold scaler so inference reuses identical stats
            # (prevents train/serve skew) and an isotonic calibrator fitted on the
            # champion fold's validation probabilities.
            save_scaler(best_fold['models']['scaler'], MODEL_DIR, table)
            _calib = fit_calibrator(best_fold.get('val_prob'), best_fold.get('val_target'))
            save_calibrator(_calib, MODEL_DIR, table)
            registry_update = {
                'score': best_fold['score'],
                'updated_at': datetime.now().isoformat(),
                'buy_thr': best_fold['buy_thr'],
                'sell_thr': best_fold['sell_thr'],
                'features': best_fold['features'],
                'profile': get_profile(asset),
                'lookback': lookback,
                'ensemble_mode': best_fold.get('ensemble_mode', 'stacking'),
                'policy': 'champion'
            }
            policy_status = "PROMOTED"
        else:
            ch_dir = os.path.join(EXPERIMENTS_DIR, "challengers")
            os.makedirs(ch_dir, exist_ok=True)
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_fold['models']['cb'].save_model(os.path.join(ch_dir, f"{table}_cb_{_ts}.cbm"))
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['lstm'].save(os.path.join(ch_dir, f"{table}_lstm_{_ts}.keras"))
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['tf_enc'].save(os.path.join(ch_dir, f"{table}_transformer_{_ts}.keras"))
            with _gpu_lock, tf.device('/CPU:0'):
                best_fold['models']['tcn'].save(os.path.join(ch_dir, f"{table}_tcn_{_ts}.keras"))
            if best_fold['models']['meta'] is not None:
                joblib.dump(best_fold['models']['meta'], os.path.join(ch_dir, f"{table}_meta_{_ts}.pkl"))
            registry_update = None
            policy_status = "FROZEN_CHAMPION"

        status = "TRUSTED" if (best_fold['score'] > 1.5 and best_fold['test_trades'] >= 20) else "UNSTABLE"
        quality_row = {
            'Asset': asset,
            'CB_Acc': float(best_fold['cb_acc']),
            'LSTM_Acc': float(best_fold['lstm_acc']),
            'Score': float(best_fold['score']),
            'Profit': float(best_fold['test_profit']),
            'Trades': int(best_fold['test_trades']),
            'Status': status,
            'Policy': policy_status,
            'Mode': 'TRADEABLE_ROBUST'
        }

        exp_row = {
            'asset': asset,
            'time': datetime.now().isoformat(),
            'policy': policy_status,
            'best_score': best_fold['score'],
            'folds': [
                {
                    'fold': x['fold'],
                    'cb_acc': x['cb_acc'],
                    'lstm_acc': x['lstm_acc'],
                    'buy_thr': x['buy_thr'],
                    'sell_thr': x['sell_thr'],
                    'test_profit': x['test_profit'],
                    'test_mdd': x['test_mdd'],
                    'test_trades': x['test_trades'],
                    'score': x['score']
                } for x in fold_metrics
            ]
        }

        tuned_threshold = {
            'buy': float(best_fold['buy_thr']),
            'sell': float(best_fold['sell_thr']),
            'no_trade_band': float(get_profile(asset)['no_trade_band'])
        }

        return {
            'asset': asset,
            'quality_row': quality_row,
            'exp_row': exp_row,
            'tuned_threshold': tuned_threshold,
            'registry_update': registry_update,
        }

    except Exception as e:
        _safe_print(f"  [ ERR] {asset:<12} {e}")
        return None


def train_system():
    global _stop_requested
    signal.signal(signal.SIGINT, _signal_handler)

    ensure_dirs()
    registry = load_registry()

    # Asset selection: by default everything; in RETRAIN_FROZEN mode only assets
    # that lack a saved scaler (i.e. never promoted under the new pipeline).
    asset_list = list(FULL_ASSET_MAP.keys())
    if _ONLY_FROZEN:
        asset_list = [a for a in asset_list
                      if not os.path.exists(scaler_path(MODEL_DIR, _asset_table(a)))]
        if not asset_list:
            print("\n  RETRAIN FROZEN: every asset already has a saved scaler - nothing to do.\n")
            return
    # Subset filter for A/B validation: GTRADE_ASSETS=AAPL,SP500,BTC trains only
    # those (comma-separated keys, case-insensitive). Empty means all.
    _subset = [a.strip().upper() for a in (os.getenv("GTRADE_ASSETS") or "").split(",") if a.strip()]
    if _subset:
        asset_list = [a for a in asset_list if a.upper() in _subset]
        if not asset_list:
            print(f"\n  GTRADE_ASSETS={_subset} matched no assets - nothing to do.\n")
            return

    W = 72
    _dev_label = f"GPU: {gpus[0].name}" if _HAS_GPU else "CPU only"
    total_assets = len(asset_list)
    print()
    print("=" * W)
    print("  G-TRADE TRAINER  |  Ensemble (CB+LSTM+TF+TCN)")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    _nets_label = ("adaptive" + (" + warm-start" if _NET_WARMSTART else "")) if _ADAPTIVE_NETS else "flat (default)"
    print(f"  Device : {_dev_label}  |  CatBoost: {_CB_TASK_TYPE}  |  Nets: {_nets_label}")
    print(f"  Workers: {_N_WORKERS} parallel  |  GPU slots: {_GPU_SLOTS}  |  CB threads: {_CB_THREADS}  |  Assets: {total_assets}")
    if _ONLY_FROZEN:
        print(f"  Mode   : RETRAIN FROZEN (only assets without a saved scaler, force-promote) | {total_assets} selected")
    elif _FORCE_PROMOTE:
        print("  Mode   : FORCE-PROMOTE (accept new champions regardless of score)")
    print("  Ctrl+C = safe stop (saves results)")
    print("=" * W)

    # -- GPU DIAGNOSTICS ------------------------------------------------------
    if _HAS_GPU:
        print()
        print("  GPU DIAGNOSTICS")
        print("  " + "-" * 50)

        # 1. Logical devices
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"  Logical GPUs visible to TF : {len(logical_gpus)}")

        # 2. Default tensor placement
        default_dev = tf.constant(1.0).device
        print(f"  Default tensor device      : {default_dev}")

        # 3. Mixed precision policy
        policy = tf.keras.mixed_precision.global_policy()
        print(f"  Precision policy           : {policy.name}  "
              f"(compute={policy.compute_dtype}, var={policy.variable_dtype})")

        # 4. Quick GPU benchmark - 1024x1024 fp16 matmul
        try:
            import time as _btime
            with tf.device('/GPU:0'):
                _w = tf.cast(tf.random.normal((1024, 1024)), tf.float16)
                _ = tf.matmul(_w, _w).numpy()          # warmup
                _t0 = _btime.perf_counter()
                for _ in range(10):
                    tf.matmul(_w, _w).numpy()
                _dt_ms = (_btime.perf_counter() - _t0) / 10 * 1000
            _verdict = "GPU ACTIVE" if _dt_ms < 5.0 else "SLOW - may be CPU !"
            print(f"  fp16 matmul 1024x1024 x10 : {_dt_ms:.2f} ms/call  - {_verdict}")
        except Exception as _be:
            print(f"  Benchmark error            : {_be}")

        # 5. VRAM pool status
        try:
            _mem = tf.config.experimental.get_memory_info('GPU:0')
            print(f"  VRAM current / peak        : "
                  f"{_mem['current']//1024//1024} MB  /  {_mem['peak']//1024//1024} MB")
        except Exception:
            pass

        # 6. nvidia-smi snapshot (if available)
        try:
            import subprocess as _sp
            _smi = _sp.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,"
                 "utilization.gpu,utilization.memory,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if _smi.returncode == 0:
                _fields = [v.strip() for v in _smi.stdout.strip().split(",")]
                _labels = ["Name", "Driver", "TempC", "GPU%", "Mem%", "UsedMB", "FreeMB"]
                for _lbl, _val in zip(_labels, _fields):
                    print(f"  {_lbl:<28}: {_val}")
        except Exception:
            pass

        print("  " + "-" * 50)
    print(flush=True)
    sys.stdout.flush()

    tuned_thresholds = {}
    quality_rows = []
    exp_rows = []
    completed_assets = 0
    _results_lock = threading.Lock()

    candidate_features = active_candidate_features()

    # --- LIVE PROGRESS BAR (direct \r for Tkinter launcher) ---
    t_train = time.time()
    ok_count = 0
    skip_count = 0
    _bar_stop = threading.Event()

    def _make_bar(done, total, elapsed, postfix=""):
        pct = int(done / total * 30) if total else 0
        bar = '#' * pct + '-' * (30 - pct)
        if 0 < done < total:
            eta = elapsed / done * (total - done)
            time_str = f"{elapsed:.0f}s<{eta:.0f}s"
        else:
            time_str = f"{elapsed:.0f}s"
        return f"  [{bar}] {done}/{total}  {time_str}  {postfix}"

    def _write_bar(line):
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()

    def _write_result(msg):
        sys.stdout.write(f"\r{' ' * 100}\r")
        sys.stdout.flush()
        print(msg, flush=True)

    # nvidia-smi stats cached every 3 sec (subprocess in background thread)
    _smi_cache = {
        "gpu_pct": "?", "mem_pct": "?",
        "mem_used": "?", "mem_free": "?", "mem_total": "?",
        "temp": "?", "pwr": "?", "pwr_limit": "?",
        "fan": "?", "clk_gpu": "?", "clk_mem": "?",
        "ts": 0.0,
    }

    def _refresh_smi():
        import subprocess as _sp
        try:
            _r = _sp.run(
                ["nvidia-smi",
                 "--query-gpu="
                 "utilization.gpu,utilization.memory,"
                 "memory.used,memory.free,memory.total,"
                 "temperature.gpu,power.draw,power.limit,"
                 "fan.speed,clocks.gr,clocks.mem",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4
            )
            if _r.returncode == 0:
                p = [v.strip() for v in _r.stdout.strip().split(",")]
                def _g(i): return p[i] if i < len(p) else "?"
                _smi_cache["gpu_pct"]   = _g(0)
                _smi_cache["mem_pct"]   = _g(1)
                _smi_cache["mem_used"]  = _g(2)
                _smi_cache["mem_free"]  = _g(3)
                _smi_cache["mem_total"] = _g(4)
                _smi_cache["temp"]      = _g(5)
                _smi_cache["pwr"]       = _g(6).split(".")[0]
                _smi_cache["pwr_limit"] = _g(7).split(".")[0]
                _smi_cache["fan"]       = _g(8)
                _smi_cache["clk_gpu"]   = _g(9)
                _smi_cache["clk_mem"]   = _g(10)
        except Exception:
            pass
        _smi_cache["ts"] = time.time()

    # Prime the cache before first tick
    _refresh_smi()

    def _ticker():
        """Redraws the progress bar once a second."""
        while not _bar_stop.wait(1.0):
            elapsed = time.time() - t_train
            with _state_lock:
                s = dict(_progress_state)

            # -- Training state -----------------------------------------------
            if s['total_ep'] > 0:
                model_line = f"{s['label']}  ep {s['epoch']}  loss={s['loss']:.4f}"
            else:
                model_line = "preparing..."

            # -- GPU stats (nvidia-smi, refreshed every 3 sec) ----------------
            if _HAS_GPU:
                if time.time() - _smi_cache["ts"] >= 3.0:
                    threading.Thread(target=_refresh_smi, daemon=True).start()

                # TF allocator VRAM (precise, every tick)
                try:
                    _m = tf.config.experimental.get_memory_info('GPU:0')
                    tf_vram = f"{_m['current']//1024//1024}MB"
                except Exception:
                    tf_vram = "?MB"

                c = _smi_cache
                gpu_line = (
                    f"GPU {c['gpu_pct']}% | "
                    f"MEM {c['mem_pct']}% {c['mem_used']}/{c['mem_total']}MB "
                    f"(TF={tf_vram}) | "
                    f"TEMP {c['temp']}C | "
                    f"PWR {c['pwr']}/{c['pwr_limit']}W | "
                    f"FAN {c['fan']}% | "
                    f"CLK {c['clk_gpu']}MHz/{c['clk_mem']}MHz"
                )
            else:
                gpu_line = "CPU only"

            postfix = f"{model_line}  |  {gpu_line}"
            with _print_lock:
                _write_bar(_make_bar(completed_assets, total_assets, elapsed, postfix))

    _ticker_thread = threading.Thread(target=_ticker, daemon=True)
    _ticker_thread.start()

    # --- N ASSETS SIMULTANEOUSLY ---
    with ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
        futures = {}
        for asset in asset_list:
            if _stop_requested:
                break
            future = executor.submit(
                _train_one_asset, asset, candidate_features, registry.get(asset)
            )
            futures[future] = asset

        for future in as_completed(futures):
            asset = futures[future]
            completed_assets += 1
            try:
                result = future.result()
                if result is not None:
                    with _results_lock:
                        quality_rows.append(result['quality_row'])
                        exp_rows.append(result['exp_row'])
                        tuned_thresholds[asset] = result['tuned_threshold']
                        if result['registry_update']:
                            registry[asset] = result['registry_update']
                    ok_count += 1
                    score = result['quality_row'].get('Score', 0)
                    policy = result['quality_row'].get('Policy', '')
                    with _print_lock:
                        _write_result(f"  [{completed_assets:>3}/{total_assets}]  {asset:<12} score={score:+.2f}  {policy}")
                else:
                    skip_count += 1
                    with _print_lock:
                        _write_result(f"  [{completed_assets:>3}/{total_assets}]  {asset:<12} SKIP")
            except Exception as e:
                skip_count += 1
                err_short = str(e).split('\n')[0][:60]
                with _print_lock:
                    _write_result(f"  [{completed_assets:>3}/{total_assets}]  {asset:<12} ERR: {err_short}")

    _bar_stop.set()
    _ticker_thread.join(timeout=2)
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()

    elapsed_train = time.time() - t_train
    print()
    print(f"  Trained {ok_count} assets, skipped {skip_count} in {elapsed_train:.0f}s")
    print()

    # --- SAVE RESULTS ---
    print(f"\n  Saving results ({completed_assets}/{total_assets} assets)...")
    if quality_rows:
        rep_df = pd.DataFrame(quality_rows)
        rep_df.to_json(os.path.join(MODEL_DIR, 'quality_report.json'), orient='records', indent=2)
        rep_sorted = rep_df.sort_values(by='Score', ascending=False)
        W2 = 72
        print()
        print("=" * W2)
        print("  QUALITY REPORT")
        print("=" * W2)
        # Print top assets
        cols = ['Asset', 'Score', 'CB_Acc', 'LSTM_Acc', 'Status']
        cols_present = [c for c in cols if c in rep_sorted.columns]
        if cols_present:
            header = "  " + "  ".join(f"{c:<12}" for c in cols_present)
            print(header)
            print("  " + "-" * (len(header) - 2))
            for _, row in rep_sorted.iterrows():
                line = "  " + "  ".join(f"{str(row.get(c,'')):<12}" for c in cols_present)
                print(line)
        else:
            print(rep_sorted.to_string(index=False))
        print("=" * W2)

    with open(THRESHOLDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(tuned_thresholds, f, ensure_ascii=False, indent=2)

    save_registry(registry)

    if exp_rows:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(EXPERIMENTS_DIR, f'experiment_{stamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(exp_rows, f, ensure_ascii=False, indent=2)

    # elapsed = time.time() - _t_start  # removed: _t_start not defined
    print()
    print("=" * 72)
    print("  DONE")
    print(f"  Trained : {completed_assets}/{total_assets} assets")
    print("  Saved   : thresholds + champion registry")
    if _stop_requested:
        print("  Stopped : Ctrl+C (results saved)")
    print("=" * 72)
    print()


if __name__ == "__main__":
    train_system()
