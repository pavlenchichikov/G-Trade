import numpy as np
import pandas as pd
import pytest

from core import chronos_features as cf


def _fake_forecaster(context_arr, horizon):
    # deterministic: forecast quantiles = last value scaled, so features are computable
    last = float(context_arr[-1])
    return {0.1: last * 0.98, 0.5: last * 1.01, 0.9: last * 1.04}


def test_forecast_features_shape_and_warmup():
    close = pd.Series(np.linspace(100, 120, 80),
                      index=pd.date_range("2020-01-01", periods=80))
    out = cf.forecast_features(close, context=64, horizon=5, forecaster=_fake_forecaster)
    assert list(out.columns) == ["chronos_ret", "chronos_spread", "chronos_dir"]
    assert len(out) == len(close)
    assert out.iloc[:64].isna().all().all()          # warm-up NaN
    assert out.iloc[64:].notna().all().all()          # computed after context


def test_forecast_features_values():
    close = pd.Series(np.full(80, 100.0),
                      index=pd.date_range("2020-01-01", periods=80))
    out = cf.forecast_features(close, context=64, horizon=5, forecaster=_fake_forecaster)
    row = out.iloc[70]
    # q50=101 -> ret = 101/100 - 1 = 0.01; spread = (104-98)/100 = 0.06; dir = 1 (q50>close)
    assert abs(row["chronos_ret"] - 0.01) < 1e-9
    assert abs(row["chronos_spread"] - 0.06) < 1e-9
    assert row["chronos_dir"] == 1.0


def test_forecast_features_dir_down():
    def down(context_arr, horizon):
        last = float(context_arr[-1])
        return {0.1: last * 0.9, 0.5: last * 0.97, 0.9: last * 1.01}
    close = pd.Series(np.full(80, 100.0),
                      index=pd.date_range("2020-01-01", periods=80))
    out = cf.forecast_features(close, context=64, forecaster=down)
    assert out.iloc[70]["chronos_dir"] == 0.0        # q50 < close


def test_load_chronos_missing_dep_raises(monkeypatch):
    # force the import to fail and assert the error names the requirements file
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("chronos") or name == "torch":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cf.chronos_available() is False
    with pytest.raises(RuntimeError, match="requirements-chronos.txt"):
        cf._load_chronos("amazon/chronos-t5-tiny")
