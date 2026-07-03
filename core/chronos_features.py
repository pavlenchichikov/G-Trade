"""Zero-shot forecast features from a pretrained time-series model (Chronos).

The torch/chronos deps are OPTIONAL and imported lazily inside _load_chronos, so the
rest of the stack installs and runs without them. forecast_features takes an injected
`forecaster` so it is fully testable with a fake (no torch, no model download). These
features are PRECOMPUTED and cached (see precompute_chronos.py); training reads the
cache, never this model.
"""

import numpy as np
import pandas as pd


def chronos_available():
    """True iff the optional Chronos deps import."""
    try:
        import torch  # noqa: F401
        import chronos  # noqa: F401
        return True
    except Exception:
        return False


def _load_chronos(model):
    """Lazily load a Chronos pipeline and return a forecaster callable
    (context_array, horizon) -> {quantile: forecast_value}. Raises a clear error
    when the optional deps are missing."""
    try:
        import torch
        from chronos import ChronosPipeline
    except Exception as exc:
        raise RuntimeError(
            "Chronos forecast features need optional deps; install "
            "requirements-chronos.txt (torch + chronos).") from exc
    pipe = ChronosPipeline.from_pretrained(model)

    def _forecast(context_arr, horizon):
        ctx = torch.tensor(np.asarray(context_arr, dtype="float32"))
        samples = pipe.predict(ctx, prediction_length=horizon)  # [1, n_samples, horizon]
        end = np.asarray(samples[0, :, -1])                      # horizon-end samples
        return {0.1: float(np.quantile(end, 0.1)),
                0.5: float(np.quantile(end, 0.5)),
                0.9: float(np.quantile(end, 0.9))}
    return _forecast


def forecast_features(close, context=64, horizon=5,
                      model="amazon/chronos-t5-tiny", forecaster=None):
    """Rolling zero-shot forecast features for a close-price series. At each bar t
    (with a full `context` window) forecast `horizon` bars ahead and derive stationary
    features from the forecast quantiles:
      chronos_ret    = q50 / close_t - 1        (median forecast return)
      chronos_spread = (q90 - q10) / close_t    (forecast uncertainty / vol proxy)
      chronos_dir    = 1.0 if q50 > close_t else 0.0
    The first `context` bars are NaN. `forecaster` is injected for testing; when None
    the Chronos pipeline named by `model` is loaded lazily."""
    close = pd.Series(close, dtype="float64")
    n = len(close)
    ret = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if forecaster is None:
        forecaster = _load_chronos(model)
    vals = close.to_numpy()
    for t in range(context, n):
        q = forecaster(vals[t - context:t], horizon)
        c = vals[t - 1]
        if c == 0 or not np.isfinite(c):
            continue
        ret[t] = q[0.5] / c - 1.0
        spread[t] = (q[0.9] - q[0.1]) / c
        direction[t] = 1.0 if q[0.5] > c else 0.0
    return pd.DataFrame(
        {"chronos_ret": ret, "chronos_spread": spread, "chronos_dir": direction},
        index=close.index)
