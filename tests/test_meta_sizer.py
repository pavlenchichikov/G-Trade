import importlib

import pytest


def _ms(monkeypatch, mode=None, thr=None):
    if mode is None:
        monkeypatch.delenv("GTRADE_META_SIZING", raising=False)
    else:
        monkeypatch.setenv("GTRADE_META_SIZING", mode)
    if thr is not None:
        monkeypatch.setenv("GTRADE_META_GATE_THR", str(thr))
    import core.meta_sizer as ms
    return importlib.reload(ms)


def test_meta_enabled_reads_env(monkeypatch):
    assert _ms(monkeypatch).meta_enabled() == "off"                 # unset -> off
    assert _ms(monkeypatch, "shadow").meta_enabled() == "shadow"
    assert _ms(monkeypatch, "active").meta_enabled() == "active"
    assert _ms(monkeypatch, "bogus").meta_enabled() == "off"        # unknown -> off


def test_gate_active_suppresses_weak_signal(monkeypatch):
    ms = _ms(monkeypatch, "active", thr=0.5)
    out, info = ms.gate("BUY", 0.4)
    assert out == "WAIT" and info["meta_gated"] is True and info["meta_prob"] == 0.4
    out2, info2 = ms.gate("BUY", 0.7)
    assert out2 == "BUY" and info2["meta_gated"] is False
    assert ms.gate("WAIT", 0.1)[0] == "WAIT"                        # WAIT stays WAIT
    assert ms.gate("SELL", None)[0] == "SELL"                       # no meta_prob -> no gate


def test_gate_shadow_keeps_signal_but_reports(monkeypatch):
    ms = _ms(monkeypatch, "shadow", thr=0.5)
    out, info = ms.gate("BUY", 0.4)
    assert out == "BUY"                                             # shadow never changes signal
    assert info["meta_gated"] is True and info["meta_prob"] == 0.4  # but reports it WOULD gate


def test_gate_off_is_noop(monkeypatch):
    ms = _ms(monkeypatch)
    out, info = ms.gate("BUY", 0.1)
    assert out == "BUY" and info["meta_gated"] is False


def _cb_ok():
    try:
        import catboost  # noqa: F401
        return True
    except Exception:
        return False


class _MockModel:
    """Module-level stub so joblib/pickle can locate the class by module path."""
    def predict_proba(self, X):
        import numpy as np
        return np.tile([0.3, 0.7], (len(X), 1))


def test_save_load_meta_roundtrip(tmp_path, monkeypatch):
    import core.meta_sizer as ms
    monkeypatch.setattr(ms, "META_DIR", str(tmp_path / "meta"))

    ms.save_meta("BTC", _MockModel())
    m = ms.load_meta("BTC")
    assert m is not None
    assert abs(ms.meta_prob(m, [0.0, 1.0, 2.0]) - 0.7) < 1e-9
    assert ms.load_meta("NOPE") is None                            # absent -> None


@pytest.mark.skipif(not _cb_ok(), reason="catboost not installed")
def test_build_meta_label_and_train(tmp_path):
    import numpy as np
    import pandas as pd
    import core.meta_sizer as ms
    rng = np.random.RandomState(0)
    n = 500
    f0 = rng.randn(n).astype("float32")
    feat = pd.DataFrame({"f0": f0, "f1": rng.randn(n).astype("float32"),
                         "rsi": rng.randn(n).astype("float32"),
                         "target": (f0 > 0).astype("float32")})
    X, y = ms.build_meta_label(feat, ["f0", "f1"], ["f0", "rsi"], "target", n_blocks=5)
    assert X is not None and X.shape[0] == y.shape[0] and X.shape[1] == 2
    assert set(np.unique(y)).issubset({0.0, 1.0})
    model = ms.train_meta_model(X, y)
    assert model is not None
    assert 0.0 <= ms.meta_prob(model, X[0]) <= 1.0


def test_train_meta_model_guards():
    import numpy as np
    import core.meta_sizer as ms
    assert ms.train_meta_model(None, None) is None
    X = np.zeros((10, 2), dtype="float32")
    assert ms.train_meta_model(X, np.zeros(10, dtype="float32")) is None   # single-class/too-few


@pytest.mark.skipif(not _cb_ok(), reason="catboost not installed")
def test_train_and_save_persists(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import core.meta_sizer as ms
    monkeypatch.setattr(ms, "META_DIR", str(tmp_path / "meta"))
    rng = np.random.RandomState(1)
    n = 500
    cols = {c: rng.randn(n).astype("float32") for c in ms.OPINION_COLS + ms.REGIME_COLS}
    # target NOT a clean function of a CB feature, so CB is not perfect and y_meta has
    # both classes (a single-class y_meta would correctly make train_and_save skip).
    cols["target"] = (rng.rand(n) > 0.5).astype("float32")
    feat = pd.DataFrame(cols)
    assert ms.train_and_save("DAX", feat) is True
    assert ms.load_meta("DAX") is not None


def test_train_and_save_never_raises_on_bad_frame(tmp_path, monkeypatch):
    import pandas as pd
    import core.meta_sizer as ms
    monkeypatch.setattr(ms, "META_DIR", str(tmp_path / "meta"))
    assert ms.train_and_save("X", pd.DataFrame({"nope": [1, 2, 3]})) is False   # no cols
