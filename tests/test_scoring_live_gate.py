"""score_asset carries both the raw and the gated signal (core/scoring.py).
The heavy model-loading path is not exercised here - gate wiring is tested by
monkeypatching the pieces around it."""

import numpy as np
import pandas as pd
import pytest

from core import scoring


@pytest.fixture()
def fake_asset(tmp_path, monkeypatch):
    """Minimal df + a fake CatBoost so score_asset reaches the gate wiring."""
    n = 120
    df = pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "trend_strength": np.full(n, 0.02),
        **{f"f{i}": np.random.rand(n) for i in range(6)},
    })
    reg = {"features": [f"f{i}" for i in range(6)], "score": 5.0}

    class _FakeCB:
        def load_model(self, path):
            return None

        def predict_proba(self, x):
            return np.array([[0.3, 0.7]])

    monkeypatch.setattr(scoring, "CatBoostClassifier", _FakeCB)
    monkeypatch.setattr(scoring, "load_or_fit_scaler",
                        lambda md, t, x: (__import__("sklearn.preprocessing",
                                          fromlist=["StandardScaler"]).StandardScaler().fit(x), "fit"))
    # champion file must "exist"
    cb = tmp_path / "aapl_cb.cbm"
    cb.write_bytes(b"x")
    return df, reg, str(tmp_path)


def test_gate_applied_and_raw_preserved(fake_asset, monkeypatch):
    df, reg, model_dir = fake_asset
    monkeypatch.setattr(scoring.live_gate, "gate",
                        lambda name, prob, sig: ("WAIT", "live-gate: test"))
    res = scoring.score_asset(df, "AAPL", "aapl", reg,
                              {"AAPL": {"buy": 0.55, "sell": 0.45}}, model_dir)
    assert res is not None
    assert res["sig"] == "WAIT"
    assert res["sig_raw"] == "BUY"          # 0.7 > 0.55 pre-gate
    assert res["gate_reason"] == "live-gate: test"
    assert res["prob_raw"] == pytest.approx(res["prob"])  # no live pkl -> identity


def test_no_gate_no_reason(fake_asset, monkeypatch):
    df, reg, model_dir = fake_asset
    monkeypatch.setattr(scoring.live_gate, "gate", lambda name, prob, sig: (sig, None))
    res = scoring.score_asset(df, "AAPL", "aapl", reg,
                              {"AAPL": {"buy": 0.55, "sell": 0.45}}, model_dir)
    assert res["sig"] == res["sig_raw"] == "BUY"
    assert res["gate_reason"] is None
