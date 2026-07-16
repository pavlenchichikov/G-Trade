"""Loading of saved champion models.

Moved out of backtest.py: the loader is also needed by predict.py, alert_bot.py
and signal_engine.py, so importing it from the backtest script was wrong.
"""

import json
import os
import threading
import types

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Flatten, Input, Multiply, Permute, RepeatVector,
)
from tensorflow.keras.models import Model

from core.architectures import ReduceSumLayer, build_lstm_attention
from core.logger import get_logger

logger = get_logger("model_io")


_SERVE_CUSTOM_OBJECTS = {"ReduceSumLayer": ReduceSumLayer}
_LAMBDA_PATCH_LOCK = threading.Lock()


def _lambda_cast_call(self, inputs, mask=None, training=None):
    """Replacement body for reloaded Lambda layers. Every Lambda this project
    ever saved is the dtype cast `lambda t: tf.cast(t, compute_dt)`; Keras 3
    reloads a marshalled lambda without its globals (`tf` missing) and without
    an output shape, so the original dies at first call. Replaying the cast
    (shape-preserving) restores the saved model's real behavior."""
    return tf.cast(inputs, tf.keras.mixed_precision.global_policy().compute_dtype)


def _lambda_identity_shape(self, input_shape):
    return input_shape


def load_keras_native(path):
    """The straight Keras 3 load of a natively saved .keras champion - the format
    every model since the 2026-06 full retrain is in. Returns the model or None.

    This must be tried FIRST: the legacy V50/V49 rebuild paths reconstruct
    fixed-size architectures (192/96, 128/64) and cannot fit the adaptive-sized
    champions - their name-matched partial weight loads used to leave layers at
    random init and the member quietly predicted ~0.5 noise.

    Lambda handling: custom_objects cannot override the builtin Lambda class, so
    the class is patched (see _lambda_cast_call) only for the duration of the
    load, then the cast is pinned onto the loaded Lambda INSTANCES and the class
    restored - process-wide Lambda behavior is untouched."""
    lam_cls = tf.keras.layers.Lambda
    with _LAMBDA_PATCH_LOCK:
        orig_call = lam_cls.call
        orig_shape = lam_cls.compute_output_shape
        lam_cls.call = _lambda_cast_call
        lam_cls.compute_output_shape = _lambda_identity_shape
        try:
            model = tf.keras.models.load_model(
                path, safe_mode=False, custom_objects=_SERVE_CUSTOM_OBJECTS)
        except Exception as e:
            logger.debug("Native keras load failed for %s: %s", path, e)
            return None
        finally:
            lam_cls.call = orig_call
            lam_cls.compute_output_shape = orig_shape
    for layer in model.layers:
        if isinstance(layer, lam_cls):
            layer.call = types.MethodType(_lambda_cast_call, layer)
            layer.compute_output_shape = types.MethodType(_lambda_identity_shape, layer)
    return model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_lstm_legacy(input_shape):
    """V49 architecture: LSTM(128)+Dropout+LSTM(64), Bahdanau-style attention, Lambda-ReduceSum."""
    timesteps = input_shape[0]
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    # V49 attention: Dense(1,tanh), Flatten, Dense(ts,softmax), RepeatVector, Permute, Multiply
    a = Dense(1, activation='tanh')(x)
    a = Flatten()(a)
    a = Dense(timesteps, activation='softmax')(a)
    a = RepeatVector(64)(a)
    a = Permute((2, 1))(a)
    x = Multiply()([x, a])
    x = ReduceSumLayer()(x)
    x = Dense(32, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def _detect_format(path):
    """Detect if file is ZIP (.keras native) or HDF5."""
    with open(path, 'rb') as f:
        header = f.read(4)
    return 'zip' if header[:2] == b'PK' else 'hdf5'


def _load_weights_keras3(model, h5_path):
    """Load weights from Keras 3.x h5 format (layers/<name>/vars/N) into a Keras 2.x model."""
    import h5py
    import numpy as np
    with h5py.File(h5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError("Not a Keras 3.x h5 file")
        h5_layers = f['layers']
        for layer in model.layers:
            if layer.name in h5_layers and 'vars' in h5_layers[layer.name]:
                vars_grp = h5_layers[layer.name]['vars']
                weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                layer.set_weights(weights)
            # Handle LSTM cell weights (stored as layers/<name>/cell/vars/N)
            elif layer.name in h5_layers and 'cell' in h5_layers[layer.name]:
                cell_grp = h5_layers[layer.name]['cell']
                if 'vars' in cell_grp:
                    vars_grp = cell_grp['vars']
                    weights = [np.array(vars_grp[str(i)]) for i in range(len(vars_grp))]
                    layer.set_weights(weights)


def get_lookback(reg_entry, asset_name):
    """Resolve actual lookback: registry.lookback - optuna_params - profile fallback."""
    if reg_entry and 'lookback' in reg_entry:
        return int(reg_entry['lookback'])
    optuna = load_json(OPTUNA_PARAMS_PATH)
    if asset_name in optuna and 'lookback' in optuna[asset_name]:
        return int(optuna[asset_name]['lookback'])
    return int(reg_entry.get('profile', {}).get('lookback', 10)) if reg_entry else 10


def detect_lookback_from_h5(h5_path):
    """Detect actual lookback from saved weights by finding the attention Dense(timesteps) square kernel."""
    import h5py

    def _find_square_kernel(group):
        """Recursively search for a square kernel tensor (NxN, 5<=N<=100)."""
        for key in group.keys():
            item = group[key]
            if hasattr(item, 'shape'):
                if 'kernel' in key and len(item.shape) == 2 and item.shape[0] == item.shape[1] and 5 <= item.shape[0] <= 100:
                    return int(item.shape[0])
            elif hasattr(item, 'keys'):
                result = _find_square_kernel(item)
                if result is not None:
                    return result
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'model_weights' in f:
                return _find_square_kernel(f['model_weights'])
            if 'layers' in f:
                return _find_square_kernel(f['layers'])
    except Exception as e:
        logger.debug("Lookback detection failed for %s: %s", h5_path, e)
    return None


def load_lstm_model(lstm_path, lookback, n_features):
    """Load LSTM handling V49, V50, HDF5, ZIP, and Keras 3.x formats."""
    import shutil
    import tempfile
    import zipfile
    fmt = _detect_format(lstm_path)

    # For HDF5 files saved as .keras: need .h5 extension for load_weights
    if fmt == 'hdf5':
        h5_path = lstm_path.replace('.keras', '.tmp.h5')
        shutil.copy2(lstm_path, h5_path)
        weights_path = h5_path
    else:
        weights_path = lstm_path
        h5_path = None

    try:
        # Method 0: native Keras 3 load - the correct path for every champion
        # saved since the 2026-06 retrain (adaptive sizes live in the saved
        # config, so no rebuilt skeleton can drift from them).
        if fmt == 'zip':
            model = load_keras_native(lstm_path)
            if model is not None:
                native_lb = getattr(model, 'input_shape', (None, lookback))[1] or lookback
                return model, "DUAL (AI)", int(native_lb)

        # Auto-detect lookback from saved weights (handles frozen champions with outdated optuna params)
        detected_lb = detect_lookback_from_h5(weights_path if fmt == 'hdf5' else lstm_path)
        if detected_lb is not None and detected_lb != lookback:
            logger.debug("Lookback override for %s: %d - %d (from weights)", lstm_path, lookback, detected_lb)
            lookback = detected_lb
        input_shape = (lookback, n_features)

        # Method 1: V50 architecture (192/96+ReduceSumLayer), Keras 2.x load_weights
        try:
            model = build_lstm_attention(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (AI)", lookback
        except Exception as e:
            logger.debug("V50 load failed: %s", e)

        # Method 2: V49 legacy architecture, Keras 2.x load_weights
        try:
            model = build_lstm_legacy(input_shape)
            model.load_weights(weights_path)
            return model, "DUAL (V49)", lookback
        except Exception as e:
            logger.debug("V49 load failed: %s", e)

        # Method 3: ZIP with Keras 3.x weights - extract h5, load manually
        if fmt == 'zip':
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(lstm_path, 'r') as z:
                    z.extractall(tmpdir)
                inner_h5 = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(inner_h5):
                    # Auto-detect lookback from extracted weights
                    zip_lb = detect_lookback_from_h5(inner_h5)
                    if zip_lb is not None:
                        input_shape = (zip_lb, n_features)
                    zip_lookback = zip_lb if zip_lb is not None else lookback
                    # Try V49 legacy with Keras 3.x weights
                    try:
                        model = build_lstm_legacy(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras3 V49 load failed: %s", e)
                    # Try V50 with Keras 3.x weights
                    try:
                        model = build_lstm_attention(input_shape)
                        _load_weights_keras3(model, inner_h5)
                        return model, "DUAL (AI)", zip_lookback
                    except Exception as e:
                        logger.debug("Keras 3.x load failed for %s: %s", lstm_path, e)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        logger.debug("All LSTM load methods failed for %s", lstm_path)
        return None, "CB ONLY (Err)", lookback
    finally:
        if h5_path and os.path.exists(h5_path):
            os.remove(h5_path)
