"""Serve-side model loading (core/model_io.py).

Guards the 2026-07-16 finding: natively saved Keras 3 champions must survive the
save/load/predict round-trip. Before load_keras_native existed, the fixed-size
V50/V49 rebuild paths silently partial-loaded adaptive-sized champions (members
predicted ~0.5 noise) and the cast Lambda in the transformer/TCN builders died
at predict time on reload ("could not infer the shape of the Lambda's output").
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from core.architectures import (  # noqa: E402
    build_lstm_multitask, build_tcn, build_transformer_encoder,
)
from core.model_io import load_keras_native, load_lstm_model  # noqa: E402


def _first_output(pred):
    return pred[0] if isinstance(pred, (list, tuple)) else pred


@pytest.mark.parametrize("builder,shape", [
    (build_tcn, (20, 8)),
    (build_transformer_encoder, (10, 8)),
    (build_lstm_multitask, (10, 8)),
])
def test_native_round_trip_predicts(tmp_path, builder, shape):
    model = builder(shape)
    path = str(tmp_path / "m.keras")
    model.save(path)
    loaded = load_keras_native(path)
    assert loaded is not None
    x = np.random.rand(2, *shape).astype("float32")
    want = _first_output(model.predict(x, verbose=0)).ravel()
    got = _first_output(loaded.predict(x, verbose=0)).ravel()
    # the reload must reproduce the trained model, not a re-initialized skeleton
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_native_load_missing_or_broken_returns_none(tmp_path):
    assert load_keras_native(str(tmp_path / "absent.keras")) is None
    bad = tmp_path / "corrupt.keras"
    bad.write_bytes(b"not a zip at all")
    assert load_keras_native(str(bad)) is None


def test_load_lstm_model_prefers_native_for_adaptive_sizes(tmp_path):
    """An adaptive-sized champion (units far from the fixed 192/96 / 128/64
    skeletons) must load exactly via Method 0 instead of a partial name-match."""
    shape = (10, 8)
    model = build_lstm_multitask(shape, units1=48, units2=24, head_dim=24)
    path = str(tmp_path / "asset_lstm.keras")
    model.save(path)
    loaded, mode, lookback = load_lstm_model(path, lookback=10, n_features=8)
    assert loaded is not None
    assert lookback == 10
    x = np.random.rand(1, *shape).astype("float32")
    want = _first_output(model.predict(x, verbose=0)).ravel()
    got = _first_output(loaded.predict(x, verbose=0)).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_lambda_class_not_left_patched(tmp_path):
    """load_keras_native patches the Lambda class only for the duration of the
    load; the process-wide class must come back untouched."""
    orig_call = tf.keras.layers.Lambda.call
    model = build_tcn((20, 8))
    path = str(tmp_path / "m.keras")
    model.save(path)
    assert load_keras_native(path) is not None
    assert tf.keras.layers.Lambda.call is orig_call
