"""Isotonic calibration of ensemble probabilities.

Raw probabilities are not comparable across different assets; calibrating
against validation predictions brings them to honest frequencies.

Fit on validation, save next to the champion, apply at inference. When no
calibrator is available the functions degrade to identity, so the pipeline keeps
working on models trained before calibration existed.
"""

import os

import joblib
import numpy as np

MIN_SAMPLES = 50


def calibrator_path(model_dir: str, table: str) -> str:
    return os.path.join(model_dir, f"{table}_calib.pkl")


def fit_calibrator(probs, targets):
    """Fit an isotonic calibrator mapping raw prob - P(target=1).

    Returns the fitted IsotonicRegression, or None when there is not enough data
    or only one class is present (calibration would be undefined).
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    mask = ~np.isnan(probs)
    probs, targets = probs[mask], targets[mask]
    if len(probs) < MIN_SAMPLES or len(np.unique(targets)) < 2:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs, targets)
        return iso
    except Exception:
        return None


def apply_calibrator(calibrator, probs):
    """Map raw probabilities through the calibrator. Identity if calibrator is None."""
    probs = np.asarray(probs, dtype=float)
    if calibrator is None:
        return probs
    try:
        out = calibrator.predict(probs)
        return np.clip(out, 0.0, 1.0)
    except Exception:
        return probs


def save_calibrator(calibrator, model_dir: str, table: str):
    """Persist a calibrator. No-op when calibrator is None. Returns path or None."""
    if calibrator is None:
        return None
    path = calibrator_path(model_dir, table)
    joblib.dump(calibrator, path)
    return path


def load_calibrator(model_dir: str, table: str):
    """Load a saved calibrator, or None when absent/unreadable."""
    path = calibrator_path(model_dir, table)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# --- global LIVE calibration layer (fitted on real outcomes) -----------------
# A second isotonic layer over the per-asset calibrators, fitted by
# recalibrate_live.py on verified prediction_log outcomes. Missing file =
# identity, so nothing changes until the owner runs the CLI. The pkl stores
# {"model": IsotonicRegression, ...metadata...}; a tiny mtime cache avoids a
# disk read per asset per radar run.

LIVE_GLOBAL_FILENAME = "live_calib_global.pkl"
_LIVE_CACHE = {}


def live_global_path(model_dir: str) -> str:
    return os.path.join(model_dir, LIVE_GLOBAL_FILENAME)


def save_live_global(model, meta: dict, model_dir: str) -> str:
    path = live_global_path(model_dir)
    joblib.dump({"model": model, **meta}, path)
    _LIVE_CACHE.pop(path, None)
    return path


def _load_live_global(model_dir: str):
    path = live_global_path(model_dir)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _LIVE_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        entry = joblib.load(path)
    except Exception:
        entry = None
    _LIVE_CACHE[path] = (mtime, entry)
    return entry


def apply_live_global(prob: float, model_dir: str) -> float:
    """Second calibration layer fitted on LIVE outcomes. Identity when the
    pkl is absent or unreadable."""
    entry = _load_live_global(model_dir)
    if not isinstance(entry, dict) or entry.get("model") is None:
        return float(prob)
    return float(apply_calibrator(entry["model"], np.array([prob]))[0])
