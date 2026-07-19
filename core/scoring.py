"""Serve-time asset scoring shared by predict.py and alert_bot.py.

Given a DataFrame that has already been feature-engineered (via
core.features.engineer_features + the add_* helpers), this loads the champion's
members - CatBoost + LSTM + Transformer + TCN - combines them through the
stacking meta-classifier (or a documented fallback), calibrates, thresholds and
gates the result. Both the console radar and the Telegram bot call this one
function, so their signals cannot drift apart the way two hand-maintained copies
of the scoring block did.

The caller owns data acquisition and feature engineering (predict.py reads from
market.db; alert_bot.py fetches live quotes); this function owns only the model
scoring, which must be identical everywhere.
"""

import os

import joblib
import numpy as np
from catboost import CatBoostClassifier

from core import live_gate, meta_sizer, timing_policy
from core.calibration import apply_calibrator, apply_live_global, load_calibrator
from core.ensemble import build_stacking_features
from core.features import active_candidate_features
from core.logger import get_logger
from core.model_io import get_lookback, load_keras_native, load_lstm_model
from core.scaling import load_or_fit_scaler

logger = get_logger("scoring")

# A champion whose walk-forward score is below this cannot be trusted for
# direction (a negative score is worse than the baseline). Its BUY/SELL is
# suppressed to WAIT and tagged "low-q" so the suppression is visible.
MIN_TRUST_SCORE = 0.0


def select_features(df, reg_entry):
    """Champion feature list intersected with what the df actually has.

    Uses the registry's stored feature list when present (train/serve parity),
    otherwise the current active candidate set.
    """
    if reg_entry and "features" in reg_entry:
        return [f for f in reg_entry["features"] if f in df.columns]
    return [f for f in active_candidate_features() if f in df.columns]


def score_asset(df, name, table, reg_entry, thresholds, model_dir):
    """Score one asset from an already feature-engineered df.

    Returns a dict, or None when the champion is missing / the data is too short
    / scoring raised:

        {
            "sig":         "BUY" | "SELL" | "WAIT",   # post live-gate
            "prob":        float,   # calibrated + live-global-adjusted probability
            "price":       float,   # last close
            "mode":        str,     # STACK / CB / LSTM (+ " low-q" if suppressed)
            "sig_raw":     "BUY" | "SELL" | "WAIT",   # pre live-gate
            "prob_raw":    float,   # pre live-global-layer probability
            "gate_reason": str | None,   # why sig_raw was gated, None if not gated
            "cb_prob":     float,
            "lstm_prob":   float | None,
            "tf_prob":     float | None,
            "tcn_prob":    float | None,
            "meta_prob":   float | None,   # None unless GTRADE_META_SIZING is on
            "timing_action": str | None,   # shadow-only; None unless GTRADE_TIMING_POLICY=1
            "timing_reason": str | None,   # and timing_policy.json exists
        }
    """
    cb_path = os.path.join(model_dir, f"{table}_cb.cbm")
    if not os.path.exists(cb_path) or len(df) < 50:
        return None

    features = select_features(df, reg_entry)
    if len(features) < 3:
        return None

    try:
        curr_price = float(df["close"].iloc[-1])
        n_bars = min(500, len(df))
        x_fit = df[features].iloc[-n_bars:].values
        # Saved train-fold scaler (train/serve parity); fitting on the window is
        # only a fallback for legacy models without a saved scaler.
        scaler, _src = load_or_fit_scaler(model_dir, table, x_fit)
        if _src == "fit":
            logger.debug("No saved scaler for %s; fitting on recent window (retrain to fix)", table)
        X_all = scaler.transform(x_fit)

        cb = CatBoostClassifier()
        cb.load_model(cb_path)
        cb_prob = float(cb.predict_proba(X_all[-1:])[:, 1][0])

        # -- LSTM (2nd member) --------------------------------------------
        lstm_prob = None
        mode = "CB"
        lookback = get_lookback(reg_entry, name)
        lstm_path = os.path.join(model_dir, f"{table}_lstm.keras")
        if os.path.exists(lstm_path) and len(X_all) >= lookback:
            lstm_model, mode, lookback = load_lstm_model(lstm_path, lookback, len(features))
            if lstm_model is not None:
                try:
                    X_seq = X_all[-lookback:].reshape(1, lookback, len(features))
                    lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                except Exception as e:
                    logger.debug("LSTM predict failed for %s: %s", table, e)
                    lstm_prob = None
                    mode = "CB"

        # -- Transformer (3rd member) -------------------------------------
        tf_prob = None
        tf_path = os.path.join(model_dir, f"{table}_transformer.keras")
        if os.path.exists(tf_path):
            try:
                tf_model, _, tf_lookback = load_lstm_model(tf_path, lookback, len(features))
                if tf_model is not None:
                    X_seq = X_all[-tf_lookback:].reshape(1, tf_lookback, len(features))
                    tf_prob = float(tf_model.predict(X_seq.astype("float32"), verbose=0).flatten()[0])
            except Exception as e:
                logger.debug("Transformer predict failed for %s: %s", table, e)
                tf_prob = None

        # -- TCN (4th member) ---------------------------------------------
        tcn_prob = None
        tcn_path = os.path.join(model_dir, f"{table}_tcn.keras")
        if os.path.exists(tcn_path):
            try:
                # load_keras_native replays the saved cast Lambda; the plain
                # load_model call used to die at predict on every asset
                # ("could not infer the shape of the Lambda's output").
                tcn_model = load_keras_native(tcn_path)
                if tcn_model is not None:
                    tcn_lb = tcn_model.input_shape[1] or lookback
                    X_seq = X_all[-tcn_lb:].reshape(1, tcn_lb, len(features))
                    tcn_prob = float(tcn_model.predict(X_seq.astype("float32"), verbose=0).flatten()[0])
            except Exception as e:
                logger.debug("TCN predict failed for %s: %s", table, e)
                tcn_prob = None

        # -- Stacking meta-classifier (or documented fallback) ------------
        meta_path = os.path.join(model_dir, f"{table}_meta.pkl")
        trend_val = float(df["trend_strength"].iloc[-1]) if "trend_strength" in df.columns else 0.01
        all_probs = [cb_prob, lstm_prob, tf_prob, tcn_prob]
        n_available = sum(1 for p in all_probs if p is not None)

        if n_available >= 3 and os.path.exists(meta_path):
            try:
                meta_clf = joblib.load(meta_path)
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
                logger.debug("Stacking fallback for %s: %s", table, e)
                prob = float(np.mean([p for p in all_probs if p is not None]))
        elif lstm_prob is not None:
            probs_avail = [cb_prob, lstm_prob]
            if tf_prob is not None:
                probs_avail.append(tf_prob)
            if tcn_prob is not None:
                probs_avail.append(tcn_prob)
            prob = float(np.mean(probs_avail))
        else:
            prob = cb_prob

        # Calibrate into an honest up-move frequency (identity when no calibrator),
        # then apply the GLOBAL live-outcome layer (identity until
        # recalibrate_live.py has produced models/live_calib_global.pkl).
        # prob_raw (pre-live-layer) is what performance_tracker logs, so refits
        # always train on a homogeneous raw -> P(up) history.
        prob = float(apply_calibrator(load_calibrator(model_dir, table), np.array([prob]))[0])
        prob_raw = prob
        prob = apply_live_global(prob, model_dir)

        thr = thresholds.get(name, {})
        buy_thr = thr.get("buy", 0.55)
        sell_thr = thr.get("sell", 0.45)
        if prob > buy_thr:
            sig = "BUY"
        elif prob < sell_thr:
            sig = "SELL"
        else:
            sig = "WAIT"

        # Quality gate: an untrustworthy champion (score < MIN_TRUST_SCORE) may
        # not emit a directional call; surface it as WAIT, tagged "low-q".
        score = (reg_entry or {}).get("score")
        if sig != "WAIT" and score is not None and score < MIN_TRUST_SCORE:
            sig = "WAIT"
            mode = f"{mode} low-q"

        # Optional meta-sizing gate (GTRADE_META_SIZING; default off = no-op).
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

        # Live-accuracy gate: the displayed/acted-on signal may be suppressed,
        # but sig_raw keeps flowing into the tracker so a gated segment can
        # rehabilitate itself with fresh statistics.
        sig_raw = sig
        sig, gate_reason = live_gate.gate(name, prob, sig)

        # Timing-policy shadow (GTRADE_TIMING_POLICY; default off = no-op). Never
        # allowed to affect sig/prob/gate_reason above, and any failure here must
        # not break scoring - hence the guard clause plus its own try/except.
        timing_action = timing_reason = None
        if timing_policy.timing_on():
            pol = timing_policy.load_policy()
            if pol is not None:
                try:
                    from performance_tracker import timing_state
                    import config as _cfg
                    risky = any(name in _cfg.ASSET_TYPES.get(g, ())
                                for g in ("CRYPTO", "FOREX MAJORS",
                                          "FOREX CROSSES", "FOREX EXOTIC"))
                    atr_now = float(df["atr"].iloc[-1]) if "atr" in df else 0.0
                    from core.features import latest_taleb_risk
                    taleb_val = latest_taleb_risk(df["close"])
                    taleb_now = taleb_val is not None and taleb_val > 0.7
                    st = timing_state(
                        name, cooldown_days=int(pol.params.get("cooldown_days", 0)))
                    act, reason, st2 = timing_policy.policy_step(
                        pol, prob, buy_thr, sell_thr, atr_now, taleb_now,
                        risky, st)
                    timing_action = (
                        f"ENTER:{'+1' if st2['pos'] == 1 else '-1'}"
                        if act == "ENTER" else act)
                    timing_reason = reason
                except Exception as e:
                    logger.debug("Timing-policy shadow skipped for %s: %s", name, e)
                    timing_action = timing_reason = None

        return {
            "sig": sig, "prob": prob, "price": curr_price, "mode": mode,
            "sig_raw": sig_raw, "prob_raw": prob_raw, "gate_reason": gate_reason,
            "cb_prob": cb_prob, "lstm_prob": lstm_prob,
            "tf_prob": tf_prob, "tcn_prob": tcn_prob, "meta_prob": meta_p,
            "timing_action": timing_action, "timing_reason": timing_reason,
        }

    except Exception as e:
        logger.error("Scoring failed for %s: %s", table, e)
        return None
