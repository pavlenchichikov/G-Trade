"""Tests for core.calibration - isotonic probability calibration."""

import numpy as np

from core.calibration import (
    apply_calibrator,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
)


def test_identity_when_none():
    probs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(apply_calibrator(None, probs), probs)


def test_returns_none_on_single_class():
    probs = np.linspace(0.3, 0.7, 100)
    targets = np.ones(100, dtype=int)
    assert fit_calibrator(probs, targets) is None


def test_returns_none_on_too_few_samples():
    assert fit_calibrator([0.4, 0.6], [0, 1]) is None


def test_calibration_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    # Overconfident raw scores: true freq grows with score but compressed.
    probs = rng.uniform(0, 1, 500)
    targets = (rng.uniform(0, 1, 500) < probs).astype(int)
    calib = fit_calibrator(probs, targets)
    assert calib is not None
    grid = np.linspace(0, 1, 50)
    out = apply_calibrator(calib, grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Isotonic output must be non-decreasing.
    assert np.all(np.diff(out) >= -1e-9)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 300)
    targets = (rng.uniform(0, 1, 300) < probs).astype(int)
    calib = fit_calibrator(probs, targets)
    save_calibrator(calib, str(tmp_path), "foo")
    loaded = load_calibrator(str(tmp_path), "foo")
    assert loaded is not None
    np.testing.assert_allclose(
        apply_calibrator(calib, probs), apply_calibrator(loaded, probs)
    )


def test_save_none_is_noop(tmp_path):
    assert save_calibrator(None, str(tmp_path), "foo") is None
    assert load_calibrator(str(tmp_path), "foo") is None


# --- global LIVE calibration layer tests ---


def test_apply_live_global_identity_without_file(tmp_path):
    from core.calibration import apply_live_global
    assert apply_live_global(0.7, str(tmp_path)) == 0.7


def test_live_global_monotone_round_trip(tmp_path):
    from core import calibration
    # properly calibrated synthetic stream: higher prob, higher up-rate
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.05, 0.95, 600)
    ups = (rng.uniform(0, 1, 600) < probs).astype(int)
    iso = calibration.fit_calibrator(probs, ups)
    assert iso is not None
    calibration.save_live_global(iso, {"n": 600}, str(tmp_path))
    lo = calibration.apply_live_global(0.2, str(tmp_path))
    hi = calibration.apply_live_global(0.8, str(tmp_path))
    assert lo < hi  # monotone preserved through save/load/apply


def test_live_global_shrinks_anti_calibrated_tails(tmp_path):
    from core import calibration
    # mid-range informative, tails INVERTED (the measured live pathology):
    # the increasing isotonic flattens the tails toward the mid rate, draining
    # false conviction instead of flipping the signal.
    rng = np.random.default_rng(7)
    mid_p = rng.uniform(0.35, 0.65, 400)
    mid_up = (rng.uniform(0, 1, 400) < mid_p).astype(int)
    tail_p = np.concatenate([rng.uniform(0.85, 0.99, 100), rng.uniform(0.01, 0.15, 100)])
    tail_up = np.concatenate([np.zeros(100, int), np.ones(100, int)])  # inverted tails
    probs = np.concatenate([mid_p, tail_p])
    ups = np.concatenate([mid_up, tail_up])
    iso = calibration.fit_calibrator(probs, ups)
    assert iso is not None
    calibration.save_live_global(iso, {"n": 600}, str(tmp_path))
    assert calibration.apply_live_global(0.95, str(tmp_path)) < 0.75
    assert calibration.apply_live_global(0.05, str(tmp_path)) > 0.25


def test_live_global_cache_sees_new_save(tmp_path):
    from core import calibration
    import time
    rng = np.random.default_rng(1)
    p1 = rng.uniform(0.05, 0.95, 300)
    u1 = (rng.uniform(0, 1, 300) < p1).astype(int)
    iso1 = calibration.fit_calibrator(p1, u1)
    calibration.save_live_global(iso1, {"n": 300}, str(tmp_path))
    before = calibration.apply_live_global(0.9, str(tmp_path))
    time.sleep(0.02)
    iso2 = calibration.fit_calibrator(p1, 1 - u1 if hasattr(u1, "__neg__") else u1)
    if iso2 is None:
        iso2 = iso1
    calibration.save_live_global(iso2, {"n": 300}, str(tmp_path))
    after = calibration.apply_live_global(0.9, str(tmp_path))
    # the cache must serve the NEW model after a save (values may coincide only
    # if both fits map 0.9 identically, which the inverted targets prevent)
    assert after != before


def test_apply_live_global_survives_corrupt_file(tmp_path):
    from core.calibration import LIVE_GLOBAL_FILENAME, apply_live_global
    (tmp_path / LIVE_GLOBAL_FILENAME).write_bytes(b"junk")
    assert apply_live_global(0.7, str(tmp_path)) == 0.7
