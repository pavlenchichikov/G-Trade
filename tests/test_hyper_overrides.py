"""Relative hyperparameter overrides in train_hybrid (the auto-research
model-hyperparameter axis backend). Defaults must be byte-identical to the
pre-override behavior; overrides compose with the per-asset optuna baseline."""

import pytest

th = pytest.importorskip("train_hybrid")


OPT = {"cb_iterations": 1000, "cb_depth": 6, "cb_lr": 0.05}
PROFILE = {"lookback": 15}


def _clear(monkeypatch):
    for k in ("GTRADE_CB_ITER_MULT", "GTRADE_CB_DEPTH_DELTA", "GTRADE_CB_LR_MULT",
              "GTRADE_LOOKBACK_DELTA"):
        monkeypatch.delenv(k, raising=False)


def test_defaults_reproduce_baseline(monkeypatch):
    _clear(monkeypatch)
    assert th.cb_params_for(OPT) == (1000, 6, 0.05)
    assert th.cb_params_for({}) == (700, 8, 0.03)
    assert th.lookback_for(OPT | {"lookback": 25}, PROFILE) == 25
    assert th.lookback_for({}, PROFILE) == 15


def test_relative_overrides_apply(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GTRADE_CB_ITER_MULT", "1.5")
    monkeypatch.setenv("GTRADE_CB_DEPTH_DELTA", "-1")
    monkeypatch.setenv("GTRADE_CB_LR_MULT", "2.0")
    monkeypatch.setenv("GTRADE_LOOKBACK_DELTA", "5")
    assert th.cb_params_for(OPT) == (1500, 5, 0.1)
    assert th.lookback_for({}, PROFILE) == 20


def test_overrides_are_clamped(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GTRADE_CB_ITER_MULT", "100")
    monkeypatch.setenv("GTRADE_CB_DEPTH_DELTA", "9")
    monkeypatch.setenv("GTRADE_CB_LR_MULT", "0.0001")
    monkeypatch.setenv("GTRADE_LOOKBACK_DELTA", "-100")
    iters, depth, lr = th.cb_params_for(OPT)
    assert iters == 3000 and depth == 10 and lr == 0.005
    assert th.lookback_for({}, PROFILE) == 5


def test_garbage_env_falls_back_to_default(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GTRADE_CB_ITER_MULT", "abc")
    monkeypatch.setenv("GTRADE_CB_DEPTH_DELTA", "")
    assert th.cb_params_for(OPT) == (1000, 6, 0.05)


def _clear_thr(monkeypatch):
    for k in ("GTRADE_THR_MARGIN", "GTRADE_BAND_DELTA"):
        monkeypatch.delenv(k, raising=False)


def test_thr_margin_default_identity(monkeypatch):
    _clear_thr(monkeypatch)
    assert th.apply_thr_margin(0.55, 0.45) == (0.55, 0.45)


def test_thr_margin_shifts_outward(monkeypatch):
    _clear_thr(monkeypatch)
    monkeypatch.setenv("GTRADE_THR_MARGIN", "0.02")
    b, s = th.apply_thr_margin(0.55, 0.45)
    assert b == pytest.approx(0.57) and s == pytest.approx(0.43)


def test_thr_margin_clamps_and_caps(monkeypatch):
    _clear_thr(monkeypatch)
    monkeypatch.setenv("GTRADE_THR_MARGIN", "0.05")
    b, s = th.apply_thr_margin(0.94, 0.06)
    assert b == 0.95 and s == pytest.approx(0.05)
    # an absurd env margin is capped at 0.10, never disabling the shift
    monkeypatch.setenv("GTRADE_THR_MARGIN", "0.30")
    b, s = th.apply_thr_margin(0.52, 0.48)
    assert b == pytest.approx(0.62) and s == pytest.approx(0.38)
    # palette margins always take effect on realistic tight gaps
    monkeypatch.setenv("GTRADE_THR_MARGIN", "0.05")
    b, s = th.apply_thr_margin(0.54, 0.46)
    assert b == pytest.approx(0.59) and s == pytest.approx(0.41)


def test_band_for_delta_and_floor(monkeypatch):
    _clear_thr(monkeypatch)
    prof = {"no_trade_band": 0.01}
    assert th.band_for(prof) == pytest.approx(0.01)
    monkeypatch.setenv("GTRADE_BAND_DELTA", "0.01")
    assert th.band_for(prof) == pytest.approx(0.02)
    monkeypatch.setenv("GTRADE_BAND_DELTA", "-0.05")
    assert th.band_for(prof) == 0.0
