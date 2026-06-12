"""predict.py должен брать сохранённый train-скейлер, а не фитить новый
на окне инференса (train/serve skew)."""

import os
import re

import numpy as np

from core.scaling import load_or_fit_scaler, save_scaler
from sklearn.preprocessing import StandardScaler

PREDICT_SRC = os.path.join(os.path.dirname(__file__), "..", "predict.py")


def _read_predict():
    with open(PREDICT_SRC, encoding="utf-8") as f:
        return f.read()


def test_predict_uses_shared_scaler_loader():
    src = _read_predict()
    assert "load_or_fit_scaler" in src


def test_predict_does_not_refit_scaler_on_window():
    src = _read_predict()
    # No StandardScaler().fit_transform(...) on the inference window anymore.
    assert not re.search(r"StandardScaler\(\)\s*\n?\s*X_all\s*=\s*scaler\.fit_transform", src)
    assert "scaler.fit_transform(" not in src


def test_predict_applies_calibrator():
    src = _read_predict()
    assert "apply_calibrator" in src


def test_parity_saved_scaler_used_at_serve(tmp_path):
    """End-to-end: a scaler fit on 'train' is reused verbatim at 'serve' time."""
    train = np.random.randn(300, 6) * 4 + 2
    serve_window = np.random.randn(50, 6) * 4 + 2
    save_scaler(StandardScaler().fit(train), str(tmp_path), "xau")

    scaler, source = load_or_fit_scaler(str(tmp_path), "xau", serve_window)
    assert source == "saved"
    # Transform must match the saved scaler exactly (no re-fit on serve window).
    expected = StandardScaler().fit(train).transform(serve_window)
    np.testing.assert_allclose(scaler.transform(serve_window), expected)
