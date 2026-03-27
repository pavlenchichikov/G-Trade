"""Ensemble methods: gating, stacking, weight tuning.

Extracted from train_hybrid.py for reusability and testability.
"""

import numpy as np


def ensemble_with_gating(
    cb_prob: np.ndarray,
    lstm_prob: np.ndarray,
    trend_strength: np.ndarray,
    gate: float = 0.01,
    w_lstm_override: float | None = None,
) -> np.ndarray:
    """Soft-gated ensemble of CatBoost and LSTM predictions.

    Uses trend_strength to dynamically weight LSTM vs CatBoost.
    Higher trend → more LSTM weight. Disagreement between models
    pulls the prediction toward 0.5 (reduces confidence).

    Args:
        cb_prob: CatBoost predicted probabilities
        lstm_prob: LSTM predicted probabilities
        trend_strength: abs(close/sma50 - 1)
        gate: trend threshold for gating activation
        w_lstm_override: fixed LSTM weight (skips dynamic gating)

    Returns:
        Blended probability array
    """
    if w_lstm_override is not None:
        w_lstm = np.full_like(cb_prob, w_lstm_override, dtype=float)
    else:
        # Soft sigmoid gating (smooth transition instead of binary)
        w_lstm = 1.0 / (1.0 + np.exp(-10.0 * (trend_strength - gate)))
        w_lstm = 0.4 + 0.3 * w_lstm  # range: 0.4..0.7
    w_cb = 1.0 - w_lstm
    raw = w_lstm * lstm_prob + w_cb * cb_prob

    # Disagreement penalty: if models disagree, pull toward 0.5
    disagree = np.abs(lstm_prob - cb_prob)
    confidence = 1.0 - 0.3 * disagree
    return 0.5 + (raw - 0.5) * confidence


def tune_ensemble_weights(
    cb_val: np.ndarray,
    lstm_val: np.ndarray,
    y_val: np.ndarray,
    trend_val: np.ndarray,
    gate: float,
    lstm_acc: float | None = None,
) -> tuple[float, float]:
    """Grid-search optimal LSTM weight on validation set.

    Returns (best_weight, best_accuracy).
    """
    if lstm_acc is not None and lstm_acc < 0.52:
        max_lstm_w = max(0.15, (lstm_acc - 0.45) * 5.0)
        max_lstm_w = min(max_lstm_w, 0.40)
        min_lstm_w = 0.10
    else:
        max_lstm_w = 0.80
        min_lstm_w = 0.20

    best_w, best_acc = None, -1.0
    for w_int in range(int(min_lstm_w * 100), int(max_lstm_w * 100) + 1, 5):
        w = w_int / 100.0
        prob = ensemble_with_gating(cb_val, lstm_val, trend_val, gate, w_lstm_override=w)
        acc = float(((prob >= 0.5).astype(int) == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc


def build_stacking_features(
    cb_p: np.ndarray,
    lstm_p: np.ndarray,
    tf_p: np.ndarray,
    tcn_p: np.ndarray,
    trend: np.ndarray,
) -> np.ndarray:
    """Build meta-features for stacking classifier from 4 model outputs.

    Returns array with columns: [cb, lstm, tf, tcn, disagreement, mean_prob, trend]
    """
    stack = np.column_stack([cb_p, lstm_p, tf_p, tcn_p])
    disagree = np.std(stack, axis=1, keepdims=True)
    mean_prob = np.mean(stack, axis=1, keepdims=True)
    trend_col = np.array(trend).reshape(-1, 1)
    return np.hstack([stack, disagree, mean_prob, trend_col])
