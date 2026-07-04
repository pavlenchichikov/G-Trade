"""Meta-probability sizing gate for the production predictor (SP-6 Phase 2b).

P(CB is correct on this bar), learned from a leakage-free walk-forward CB, used as an
env-gated confidence gate: a weak-meta BUY/SELL is suppressed to WAIT. Default OFF ->
the predictor is byte-identical. Heavy deps (catboost, joblib) are imported lazily."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.getenv("GTRADE_MODEL_DIR") or os.path.join(BASE_DIR, "models")
META_DIR = os.path.join(MODEL_DIR, "meta")


def meta_enabled():
    """GTRADE_META_SIZING: off (default) / shadow / active. Unknown -> off."""
    v = (os.getenv("GTRADE_META_SIZING") or "off").strip().lower()
    return v if v in ("off", "shadow", "active") else "off"


def gate_threshold():
    """GTRADE_META_GATE_THR (default 0.5): the meta_prob below which a directional
    signal is untrusted."""
    try:
        return float(os.getenv("GTRADE_META_GATE_THR") or "0.5")
    except ValueError:
        return 0.5


def gate(signal, meta_prob, mode=None, thr=None):
    """Apply the meta-probability confidence gate. A BUY/SELL whose meta_prob is below
    thr is 'would-gate'; in 'active' it is suppressed to WAIT, in 'shadow' the signal is
    kept but the would-gate flag is still reported (for the journal), and 'off' is a
    no-op. Returns (signal_out, {"meta_prob", "meta_gated"})."""
    mode = meta_enabled() if mode is None else mode
    thr = gate_threshold() if thr is None else thr
    would_gate = (mode != "off" and signal in ("BUY", "SELL") and meta_prob is not None
                  and float(meta_prob) < thr)
    signal_out = "WAIT" if (mode == "active" and would_gate) else signal
    return signal_out, {"meta_prob": meta_prob, "meta_gated": bool(would_gate)}


def build_meta_label(feat, cb_cols, meta_cols, target_col="target", n_blocks=5,
                     cb_params=None):
    """Leakage-free walk-forward CB (expanding window, first block excluded) over cb_cols
    -> OOS direction preds; y = (pred == target); X = feat[meta_cols] on those OOS rows.
    Returns (None, None) when unfittable. Lazy-imports catboost."""
    import numpy as np
    from catboost import CatBoostClassifier
    params = {"iterations": 200, "depth": 4}
    if cb_params:
        params.update(cb_params)
    x = feat[list(cb_cols)].to_numpy(dtype="float32")
    tgt = feat[target_col].to_numpy(dtype="float32")
    n = len(tgt)
    block = n // (n_blocks + 1)
    if block == 0:
        return None, None
    idx_parts, pred_parts = [], []
    for i in range(1, n_blocks + 1):
        tr_end = block * i
        te = np.arange(tr_end, min(tr_end + block, n))
        if len(te) == 0 or len(np.unique(tgt[:tr_end])) < 2:
            continue
        clf = CatBoostClassifier(verbose=False, allow_writing_files=False, **params)
        clf.fit(x[:tr_end], tgt[:tr_end])
        idx_parts.append(te)
        pred_parts.append(clf.predict(x[te]).reshape(-1).astype("float32"))
    if not idx_parts:
        return None, None
    idx = np.concatenate(idx_parts)
    y = (np.concatenate(pred_parts) == tgt[idx]).astype("float32")
    X = feat[list(meta_cols)].to_numpy(dtype="float32")[idx]
    return X, y


def train_meta_model(X, y):
    """A tabular P(CB correct) classifier (CatBoost). None on too-few rows or single
    class. Lazy-imports catboost."""
    import numpy as np
    if X is None or y is None or len(y) < 100 or len(np.unique(y)) < 2:
        return None
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=200, depth=4, verbose=False,
                           allow_writing_files=False)
    m.fit(X, y)
    return m


def save_meta(asset, model):
    """Persist the per-asset meta-model to MODEL_DIR/meta/<asset>.joblib."""
    import joblib
    os.makedirs(META_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(META_DIR, "%s.joblib" % asset.lower()))


def load_meta(asset):
    """Load the per-asset meta-model, or None if absent/unreadable."""
    path = os.path.join(META_DIR, "%s.joblib" % asset.lower())
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def meta_prob(model, row):
    """P(CB correct) for one feature row (1D array of the meta_cols features)."""
    import numpy as np
    p = model.predict_proba(np.asarray(row, dtype="float32").reshape(1, -1))
    return float(p[0, 1])


OPINION_COLS = ["ret_1", "ret_5", "ret_10", "rsi", "macd_hist", "bb_pos",
                "trend_strength", "vol_z"]
REGIME_COLS = ["taleb_risk", "atr"]


def train_and_save(asset, feat):
    """Build the meta-label from feat, train the meta-model, and persist it for asset.
    Uses OPINION_COLS for the label-CB and OPINION_COLS+REGIME_COLS as the meta-model
    inputs (all must be columns of feat). Returns True if a model was saved, else False.
    Never raises - any problem is swallowed so it cannot break champion training."""
    try:
        cb_cols = [c for c in OPINION_COLS if c in feat.columns]
        meta_cols = [c for c in OPINION_COLS + REGIME_COLS if c in feat.columns]
        if len(cb_cols) < 3 or "target" not in feat.columns:
            return False
        X, y = build_meta_label(feat, cb_cols, meta_cols)
        model = train_meta_model(X, y)
        # A single-class y_meta means CB was right (or wrong) on every OOS bar - a
        # degenerate / leak-contaminated asset where an "is-CB-right" sizer is
        # meaningless; train_meta_model returns None and we correctly skip (no meta
        # model saved -> predict never gates that asset). Do NOT fabricate a label.
        if model is None:
            return False
        save_meta(asset, model)
        return True
    except Exception:
        return False
