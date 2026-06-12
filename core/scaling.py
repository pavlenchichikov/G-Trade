"""Скейлер общий для трейна и инференса.

Тренировка сохраняет StandardScaler train-фолда рядом с чемпионом, инференс
обязан брать его же — иначе train/serve skew. Если сохранённого нет
(старые модели), фитимся на текущем окне.
"""

import os

import joblib
from sklearn.preprocessing import StandardScaler


def scaler_path(model_dir: str, table: str) -> str:
    return os.path.join(model_dir, f"{table}_scaler.pkl")


def save_scaler(scaler, model_dir: str, table: str) -> str:
    """Persist a fitted scaler next to the champion model. Returns the path."""
    path = scaler_path(model_dir, table)
    joblib.dump(scaler, path)
    return path


def load_or_fit_scaler(model_dir: str, table: str, x_fit):
    """Return (scaler, source).

    source is "saved" when a matching saved scaler was loaded, "fit" when a fresh
    scaler was fitted on x_fit. A saved scaler whose feature count does not match
    x_fit is rejected (returns a freshly fitted one) so dimension drift never
    produces a silent transform error downstream.
    """
    path = scaler_path(model_dir, table)
    n_cols = x_fit.shape[1] if hasattr(x_fit, "shape") and len(x_fit.shape) == 2 else None
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            saved_n = getattr(scaler, "n_features_in_", None)
            if saved_n is None or n_cols is None or saved_n == n_cols:
                return scaler, "saved"
        except Exception:
            pass
    scaler = StandardScaler()
    scaler.fit(x_fit)
    return scaler, "fit"
