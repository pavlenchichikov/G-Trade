"""Fit and gate the Stage-A timing policy (spec 2026-07-18-rl-timing-policy).

Offline pipeline: reconstruct per-asset historical CatBoost-champion
probabilities (the What-If pattern), replay the rules policy on the
position-persistent simulator (core.backtesting.simulate_positions - the
Stage-0 timing environment), fit the 8 rule parameters with the SP-1
separable ES on a global time split, and gate policy-vs-baseline with a
one-sided Wilcoxon before writing timing_policy.json.

Usage:
    python train_timing.py                # fit + gate + report
    python train_timing.py --assets SP500,NVDA,BTC   # subset
    python train_timing.py --budget 400   # ES evaluations

This module (part 1) provides the dataset builder and the policy/baseline
evaluators; the ES fit + gate + save CLI lands in a later part of the same
file.
"""
import argparse
import json
import os

import numpy as np

from core import timing_policy as tp
from core.ar_rl import CmaEmitter
from core.backtesting import (
    COMMISSION, FOREX_COMMISSION, FOREX_SLIPPAGE, SLIPPAGE,
    evaluate_signals_v2, score_strategy,
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DB_PATH = os.path.join(BASE, "market.db")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

MIN_PROB_ROWS = 300


def _costs(series):
    """(commission, slippage) for one asset's series - forex legs are cheaper."""
    if series.get("is_forex"):
        return FOREX_COMMISSION, FOREX_SLIPPAGE
    return COMMISSION, SLIPPAGE


def _run_sides(series, sides):
    comm, slip = _costs(series)
    profit, n_trades, win_rate, max_dd, sharpe = evaluate_signals_v2(
        sides, series["next_ret"], comm, slip)
    score = score_strategy(profit, max_dd, win_rate, n_trades, sharpe,
                           min_trades=5)
    return {"score": score, "profit": profit, "sharpe": sharpe,
            "n_trades": n_trades, "win_rate": win_rate, "max_dd": max_dd}


def eval_policy(series, policy):
    """Run `policy` over one asset's reconstructed series.

    Returns a dict with keys score, profit, sharpe, n_trades, win_rate
    (plus max_dd) from the position-aware v2 objective.
    """
    sides, _actions, _reasons = policy.apply(
        series["probs"], series["buy_thr"], series["sell_thr"],
        series["atr"], series["taleb_hi"], series["risky"],
        next_ret=series["next_ret"])
    return _run_sides(series, sides)


def eval_baseline(series):
    """Same as eval_policy but sides = the raw thresholded signal per bar
    (DEFAULT_PARAMS reproduce this identity), i.e. today's production
    baseline behavior. One known divergence: on a direct one-bar flip
    (prob jumps across the whole neutral band) production reverses the
    position the same bar, while the policy exits flat and re-enters next
    bar; flips are rare and the effect is common-mode across both arms of
    the gate, so the ADOPT/HOLD delta is unaffected."""
    return eval_policy(series, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))


def fitness(per_asset_scores):
    """median - 0.25 * IQR over per-asset scores; -inf for an empty list."""
    if not per_asset_scores:
        return float("-inf")
    a = np.asarray(per_asset_scores, dtype=float)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    return float(np.median(a) - 0.25 * iqr)


def build_asset_series(asset, engine=None):
    """Reconstruct one asset's champion-scorable history for policy fitting.

    Mirrors whatif_simulator._predict_cb's loading pattern (registry lookup,
    champion model path, feature engineering, scaler parity via load_or_fit
    pattern matching production serve) but over the FULL price history instead
    of a `days_back` slice, so the returned arrays cover every bar the
    champion can be scored on.

    Returns a dict with keys probs, next_ret, atr, taleb_hi, buy_thr,
    sell_thr, risky, is_forex, dates (numpy arrays / scalars, oldest-first),
    or None when the asset has no CatBoost champion or too little
    champion-scorable history (< 300 rows).
    """
    import pandas as pd
    from catboost import CatBoostClassifier

    import config
    from core.features import add_weekly_features, compute_taleb_risk, engineer_features
    from core.scaling import load_or_fit_scaler
    from core.track_record import _table_name
    from whatif_simulator import _load_registry

    if engine is None:
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{DB_PATH}")

    table = _table_name(asset)
    model_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
    if not os.path.exists(model_path):
        return None

    try:
        df_raw = pd.read_sql(
            f'SELECT * FROM "{table}"', engine,
            index_col="Date", parse_dates=["Date"])
    except Exception as e:
        print(f"[timing] skip {asset}: db read failed ({e})")
        return None
    if df_raw.empty:
        return None
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()

    try:
        df_feat = engineer_features(df_raw.copy())
        df_feat = add_weekly_features(df_feat, table, engine)
    except Exception as e:
        print(f"[timing] skip {asset}: feature engineering failed ({e})")
        return None

    if "Date" in df_feat.columns:
        df_feat = df_feat.set_index("Date")
    elif "date" in df_feat.columns:
        df_feat = df_feat.set_index("date")
    df_feat.index = pd.to_datetime(df_feat.index)

    registry = _load_registry()
    feat_list = None
    if asset in registry:
        feat_list = registry[asset].get("features")
    if not feat_list:
        drop_cols = {"target", "next_ret", "close", "open", "high", "low", "volume"}
        feat_list = [c for c in df_feat.columns
                     if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])]
    feat_list = [f for f in feat_list if f in df_feat.columns]
    if not feat_list:
        return None

    df_feat = df_feat.dropna(subset=feat_list)
    n = len(df_feat)
    split = int(n * 0.8)
    if split <= 0 or split >= n:
        return None

    X = df_feat[feat_list].values
    # Load saved train-fold scaler (train/serve parity) or fit on first 80%
    # (fallback for legacy models). Matches production pattern in core.scoring.
    scaler, _src = load_or_fit_scaler(MODEL_DIR, table, X[:split])
    X_pred_scaled = scaler.transform(X[split:])

    try:
        cb = CatBoostClassifier()
        cb.load_model(model_path)
        probs = cb.predict_proba(X_pred_scaled)[:, 1]
    except Exception as e:
        print(f"[timing] skip {asset}: prediction failed ({e})")
        return None

    if len(probs) < MIN_PROB_ROWS:
        return None

    # Taleb risk needs the earlier history to warm up its rolling window, so
    # compute it on the full close series and slice afterward (same alignment
    # as atr/close below).
    taleb_full = compute_taleb_risk(df_feat["close"])
    close_slice = df_feat["close"].iloc[split:]
    next_ret = close_slice.pct_change().shift(-1).to_numpy(dtype=float)
    atr = df_feat["atr"].iloc[split:].to_numpy(dtype=float)
    taleb_hi = (taleb_full.iloc[split:] > 0.7).fillna(False).to_numpy()
    dates = df_feat.index[split:].to_numpy()

    thresholds = {}
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, encoding="utf-8") as fh:
            thresholds = json.load(fh)
    thr = thresholds.get(asset, {})
    buy_thr = thr.get("buy", 0.55)
    sell_thr = thr.get("sell", 0.45)

    forex_groups = ("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC")
    is_forex = any(asset in config.ASSET_TYPES.get(g, []) for g in forex_groups)
    risky = is_forex or asset in config.ASSET_TYPES.get("CRYPTO", [])

    return {
        "probs": probs, "next_ret": next_ret, "atr": atr, "taleb_hi": taleb_hi,
        "buy_thr": buy_thr, "sell_thr": sell_thr, "risky": risky,
        "is_forex": is_forex, "dates": dates,
    }


_SLICED = ("probs", "next_ret", "atr", "taleb_hi", "dates")


def split_series(series, train_frac=0.6, val_frac=0.2):
    """Time-ordered (train, val, test) slices of `series`.

    Only the per-bar arrays in `_SLICED` are cut; scalar fields (buy_thr,
    sell_thr, risky, is_forex) are copied unchanged into every slice.
    """
    n = len(series["probs"])
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    out = []
    for lo, hi in ((0, a), (a, b), (b, n)):
        part = dict(series)
        for k in _SLICED:
            part[k] = series[k][lo:hi]
        out.append(part)
    return tuple(out)


class _P:
    """Attribute carrier so CmaEmitter.ask/vector_of/seed_from can read and
    write the 8 timing-policy params via getattr/setattr per PARAM_SPECS."""

    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)


def _params_of(obj):
    return {name: getattr(obj, name) for name, _, _, _ in tp.PARAM_SPECS}


def fit_policy(train_by_asset, budget=300, seed=42, val_by_asset=None):
    """Separable-ES fit of the 8 timing params over `train_by_asset`.

    Each candidate is scored on TRAIN (drives the ES) and on VAL (drives
    model selection, i.e. the returned params are the best-on-VAL vector
    seen across the whole budget, not just the ES's final mean).
    """
    import random as _random
    rng = _random.Random(seed)
    es = CmaEmitter(rng=rng, dims=tp.PARAM_SPECS)
    es.seed_from(_P(dict(tp.DEFAULT_PARAMS)))
    best_params, best_val = dict(tp.DEFAULT_PARAMS), float("-inf")
    val_by_asset = val_by_asset or train_by_asset
    for _ in range(budget):
        cand = es.ask(_P(dict(tp.DEFAULT_PARAMS)))
        params = _params_of(cand)
        pol = tp.RulesPolicy(params)
        train_fit = fitness([eval_policy(s, pol)["score"]
                             for s in train_by_asset.values()])
        es.tell(es.vector_of(cand), train_fit)
        val_fit = fitness([eval_policy(s, pol)["score"]
                           for s in val_by_asset.values()])
        if val_fit > best_val:
            best_val, best_params = val_fit, params
    return best_params


def gate_policy(test_by_asset, params):
    """Policy-vs-baseline verdict on TEST: one-sided Wilcoxon over per-asset
    score deltas. ADOPT requires n >= 8 assets, p < 0.05, and mean_d > 0.5."""
    from scipy.stats import wilcoxon
    pol = tp.RulesPolicy(params)
    per_asset, deltas = {}, []
    for asset, s in test_by_asset.items():
        d = eval_policy(s, pol)["score"] - eval_baseline(s)["score"]
        per_asset[asset] = round(d, 4)
        deltas.append(d)
    n = len(deltas)
    if n >= 8 and any(abs(d) > 1e-12 for d in deltas):
        try:
            p = float(wilcoxon(deltas, alternative="greater").pvalue)
        except ValueError:
            p = 1.0
    else:
        p = 1.0
    mean_d = float(np.mean(deltas)) if deltas else 0.0
    verdict = "ADOPT" if (n >= 8 and p < 0.05 and mean_d > 0.5) else "HOLD"
    return {"verdict": verdict, "p": p, "mean_d": mean_d, "n": n,
            "per_asset": per_asset}


def save_policy(params, gate, path=None):
    """Always write timing_report.json next to `path`; write the adopted
    timing_policy.json itself only when gate["verdict"] == "ADOPT"."""
    from datetime import datetime
    path = path or tp.POLICY_PATH
    report = {"verdict": gate["verdict"], "per_asset": gate.get("per_asset"),
              "p": gate.get("p"), "mean_d": gate.get("mean_d"),
              "params": params, "fitted": datetime.utcnow().isoformat()}
    with open(os.path.join(os.path.dirname(path), "timing_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    if gate["verdict"] == "ADOPT":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "params": params,
                       "fitted": report["fitted"]}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import config
    assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
              or [a for grp in config.ASSET_TYPES.values() for a in grp])
    series = {}
    for a in assets:
        s = build_asset_series(a)
        if s is not None:
            series[a] = s
    print(f"[timing] {len(series)}/{len(assets)} assets with usable history")
    tr = {a: split_series(s)[0] for a, s in series.items()}
    va = {a: split_series(s)[1] for a, s in series.items()}
    te = {a: split_series(s)[2] for a, s in series.items()}
    params = fit_policy(tr, budget=args.budget, seed=args.seed,
                        val_by_asset=va)
    gate = gate_policy(te, params)
    save_policy(params, gate)
    print(f"[timing] params: {params}")
    print(f"[timing] verdict: {gate['verdict']}  p={gate['p']:.4f}  "
          f"mean_d={gate['mean_d']:+.2f}  n={gate['n']}")
    if gate["verdict"] == "ADOPT":
        print("[timing] wrote timing_policy.json - set GTRADE_TIMING_POLICY=1 "
              "to run in shadow mode.")


if __name__ == "__main__":
    main()
