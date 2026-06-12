"""Изотоническая калибровка вероятностей ансамбля.

Сырые вероятности у разных активов несравнимы, калибровка по валидационным
прогнозам приводит их к честным частотам.

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
    """Fit an isotonic calibrator mapping raw prob -> P(target=1).

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
