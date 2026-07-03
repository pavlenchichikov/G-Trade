"""Pure, testable neural-hygiene helpers for the ensemble trainer.

All logic for the SP-2 levers (seed-averaging, uniqueness sample-weights, per-net
calibration + abstention) lives here so train_hybrid.py stays thin and the feature
is testable without real training. Every lever is env-gated and default-OFF; when a
flag is unset the trainer's behavior is byte-identical to before.
"""

import os

import numpy as np


def net_seeds():
    """GTRADE_NET_SEEDS: how many seeds to average each net over (default 1)."""
    try:
        return max(1, int(os.getenv("GTRADE_NET_SEEDS", "1")))
    except ValueError:
        return 1


def uniqueness_on():
    """GTRADE_NET_UNIQUENESS: down-weight overlapping labels in the net fit."""
    return (os.getenv("GTRADE_NET_UNIQUENESS") or "").strip() in ("1", "true", "True")


def calibrate_nets_on():
    """GTRADE_NET_CALIBRATE: calibrate each net's probs before stacking."""
    return (os.getenv("GTRADE_NET_CALIBRATE") or "").strip() in ("1", "true", "True")


def abstain_eps():
    """GTRADE_NET_ABSTAIN_EPS: |p-0.5| below this abstains to 0.5 (default 0.0)."""
    try:
        return max(0.0, float(os.getenv("GTRADE_NET_ABSTAIN_EPS", "0.0")))
    except ValueError:
        return 0.0


def average_probs(prob_list):
    """Elementwise mean over a list of equal-length probability arrays."""
    if not prob_list:
        raise ValueError("average_probs requires at least one array")
    arrs = [np.asarray(p, dtype=float) for p in prob_list]
    return arrs[0] if len(arrs) == 1 else np.mean(np.vstack(arrs), axis=0)


def uniqueness_weights(n, horizon):
    """Lopez de Prado average-uniqueness weights for a fixed forward-window label of
    length `horizon` over `n` samples, normalized to mean 1.0. Sample i occupies bars
    [i, i+horizon); a bar's concurrency is how many samples' windows cover it; sample
    i's weight is the mean of 1/concurrency over its window. horizon <= 1 -> all-ones
    (direction labels do not overlap)."""
    n = int(n)
    horizon = int(horizon)
    if n <= 0:
        return np.array([], dtype=float)
    if horizon <= 1:
        return np.ones(n, dtype=float)
    length = n + horizon - 1
    diff = np.zeros(length + 1, dtype=float)
    idx = np.arange(n)
    np.add.at(diff, idx, 1.0)
    np.add.at(diff, idx + horizon, -1.0)
    concurrency = np.cumsum(diff)[:length]
    inv = 1.0 / np.maximum(concurrency, 1.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    u = (prefix[idx + horizon] - prefix[idx]) / horizon
    m = float(u.mean())
    return u / m if m > 0 else np.ones(n, dtype=float)


def calibrate_and_abstain(val_prob, val_target, test_prob, eps):
    """Fit an isotonic calibrator on (val_prob, val_target), apply it to BOTH val and
    test (so the meta-learner trains and scores on consistent calibrated inputs), then
    abstain: set entries within `eps` of 0.5 to exactly 0.5. Degrades to identity when
    calibration is undefined; never raises."""
    from core.calibration import apply_calibrator, fit_calibrator
    cal = fit_calibrator(val_prob, val_target)
    cv = np.asarray(apply_calibrator(cal, val_prob), dtype=float)
    ct = np.asarray(apply_calibrator(cal, test_prob), dtype=float)
    if eps and eps > 0:
        cv = np.where(np.abs(cv - 0.5) < eps, 0.5, cv)
        ct = np.where(np.abs(ct - 0.5) < eps, 0.5, ct)
    return cv, ct
