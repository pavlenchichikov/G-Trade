"""
Optuna Hyperparameter Tuner — G-Trade
========================================================
Tunes per-asset hyperparameters using Bayesian optimization.
Saves best params to models/optuna_params.json.
train_hybrid.py loads these params automatically if the file exists.

Usage:
  python optuna_tune.py                    # tune all assets (fast mode)
  python optuna_tune.py --asset BTC        # tune specific asset
  python optuna_tune.py --trials 30        # more trials per asset
  python optuna_tune.py --fast             # CatBoost only (no LSTM), faster
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    sys.exit("[ERR] optuna not installed. Run: pip install optuna")

from config import FULL_ASSET_MAP
from train_hybrid import (
    engineer_features, add_weekly_features, add_crossasset_features,
    build_sequences, make_walk_forward_splits, adaptive_split_params,
    pnl_from_signals, max_drawdown_from_returns, score_strategy,
    make_signals, apply_regime_filter, get_profile,
    COMMISSION, SLIPPAGE, FOREX_COMMISSION, FOREX_SLIPPAGE, FOREX,
)

DB_PATH    = os.path.join(BASE_DIR, "market.db")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")
engine = create_engine(f"sqlite:///{DB_PATH}")

W = 62  # output width


def _load_params():
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_params(params):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def _load_asset_df(asset):
    """Load and engineer features for one asset. Returns df or None."""
    table = asset.lower().replace("^","").replace(".","").replace("-","")
    try:
        df_raw = pd.read_sql(f"SELECT * FROM {table}", engine,
                             index_col="Date", parse_dates=["Date"])
        df_raw.index = pd.to_datetime(df_raw.index).normalize()
        df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()
        df = engineer_features(df_raw)
        df = add_weekly_features(df, table, engine)
        df = add_crossasset_features(df, table, engine)
        return df
    except Exception:
        return None


def _catboost_objective(trial, df, selected_features, profile, asset):
    """Optuna objective: tune CatBoost hyperparameters."""
    depth      = trial.suggest_int("cb_depth",       4, 10)
    iterations = trial.suggest_int("cb_iterations",  200, 1000, step=50)
    lr         = trial.suggest_float("cb_lr",        0.005, 0.15, log=True)
    lookback   = trial.suggest_int("lookback",       15, 50, step=5)

    sp = adaptive_split_params(len(df))
    if sp is None:
        return -999.0

    splits = make_walk_forward_splits(
        len(df), sp['min_train'], sp['val_size'], sp['test_size'], sp['step'])
    if not splits:
        return -999.0

    comm = FOREX_COMMISSION if asset in FOREX else COMMISSION
    slip = FOREX_SLIPPAGE   if asset in FOREX else SLIPPAGE
    scores = []

    for tr, va, te in splits[:4]:  # use max 4 folds for speed
        try:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(df.loc[tr, selected_features].values)
            y_tr = df.loc[tr, 'target'].values
            X_va = scaler.transform(df.loc[va, selected_features].values)
            y_va = df.loc[va, 'target'].values
            X_te = scaler.transform(df.loc[te, selected_features].values)
            y_te = df.loc[te, 'target'].values

            cb = CatBoostClassifier(
                iterations=iterations, depth=depth, learning_rate=lr,
                verbose=0, task_type='CPU',
                thread_count=max(2, (os.cpu_count() or 4) // 2),
                early_stopping_rounds=40
            )
            cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            prob = cb.predict_proba(X_te)[:, 1]

            # simple threshold tuning
            best_score = -999.0
            for b in profile['thr_buy_grid']:
                for s in profile['thr_sell_grid']:
                    if s >= b:
                        continue
                    sigs = make_signals(prob, b, s, profile['no_trade_band'])
                    if 'close' in df.columns and 'sma_200' in df.columns:
                        sigs = apply_regime_filter(
                            sigs,
                            df.loc[te, 'close'].values,
                            df.loc[te, 'sma_200'].values,
                            df.loc[te, 'taleb_risk'].values,
                            profile['regime_risk_cap']
                        )
                    rets = df.loc[te, 'next_ret'].values
                    p, t, w = pnl_from_signals(sigs, rets, comm, slip)
                    ret_stream = [
                        float(np.clip((r if g > 0 else -r), -0.04, 0.04)) - (comm + slip)
                        for g, r in zip(sigs, rets) if g != 0 and not np.isnan(r)
                    ]
                    mdd = max_drawdown_from_returns(ret_stream)
                    sc = score_strategy(p, mdd, w, t)
                    if sc > best_score:
                        best_score = sc
            scores.append(best_score)
        except Exception:
            scores.append(-999.0)

    valid = [s for s in scores if s > -999.0]
    return float(np.median(valid)) if valid else -999.0


def tune_asset(asset, n_trials=20, fast_mode=True):
    """Run Optuna study for one asset. Returns best params dict."""
    df = _load_asset_df(asset)
    if df is None or len(df) < 300:
        return None

    profile = get_profile(asset)
    sp = adaptive_split_params(len(df))
    if sp is None:
        return None

    # Quick feature selection (CatBoost with 150 iterations)
    candidate = [
        'close', 'volume', 'vol_z', 'taleb_risk', 'ret_1', 'ret_5',
        'ret_10', 'ret_20', 'trend_strength', 'rsi', 'sma_20', 'sma_50',
        'macd_hist', 'bb_pos', 'atr', 'vol_ratio',
        'w_ret', 'w_rsi', 'w_trend',
        'corr_btc', 'corr_sp500', 'corr_dxy',
    ]
    available = [f for f in candidate if f in df.columns]
    if len(available) < 4:
        return None

    try:
        cb_sel = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                     verbose=0, task_type='CPU')
        X_sel = df.loc[slice(0, sp['min_train']), available].values
        y_sel = df.loc[slice(0, sp['min_train']), 'target'].values
        cb_sel.fit(X_sel, y_sel)
        imps = cb_sel.get_feature_importance()
        ranked = [f for _, f in sorted(zip(imps, available), reverse=True)]
        selected = ranked[:min(profile['top_k_features'], len(ranked))]
    except Exception:
        selected = available[:profile['top_k_features']]

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _catboost_objective(trial, df, selected, profile, asset),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    best['selected_features'] = selected
    best['n_trials']   = n_trials
    best['score']      = study.best_value
    best['tuned_at']   = datetime.now().isoformat()
    return best


def run_tuner(assets=None, n_trials=20, fast_mode=True):
    targets = assets or list(FULL_ASSET_MAP.keys())
    total   = len(targets)
    params  = _load_params()

    print()
    print("═" * W)
    print(f"  OPTUNA TUNER  │  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Assets: {total}   Trials/asset: {n_trials}   Mode: {'fast' if fast_mode else 'full'}")
    print("═" * W)

    improved = 0
    for idx, asset in enumerate(targets, 1):
        print(f"  [{idx:>2}/{total}] {asset:<12}", end=' ', flush=True)
        try:
            best = tune_asset(asset, n_trials=n_trials, fast_mode=fast_mode)
            if best is None:
                print("SKIP (insufficient data)")
                continue

            prev_score = params.get(asset, {}).get('score', -999.0)
            if best['score'] > prev_score + 0.1:
                params[asset] = best
                _save_params(params)
                improved += 1
                print(f"score={best['score']:+.2f}  lookback={best.get('lookback','-')}  "
                      f"depth={best.get('cb_depth','-')}  lr={best.get('cb_lr',0):.4f}  [SAVED]")
            else:
                print(f"score={best['score']:+.2f}  (no improvement, kept prev={prev_score:+.2f})")
        except Exception as e:
            print(f"ERR: {e}")

    print()
    print("─" * W)
    print(f"  Done.  Improved: {improved}/{total}  │  Params saved to models/optuna_params.json")
    print("═" * W)
    print()


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuner")
    parser.add_argument("--asset",  type=str, default=None,
                        help="Tune specific asset (e.g. BTC)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Optuna trials per asset (default: 20)")
    parser.add_argument("--fast",   action="store_true",
                        help="Fast mode: CatBoost only, fewer folds")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else None
    run_tuner(assets=assets, n_trials=args.trials, fast_mode=args.fast)


if __name__ == "__main__":
    main()
