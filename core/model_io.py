"""Загрузка сохранённых моделей-чемпионов.

Перенесено из backtest.py: загрузчик нужен также predict.py, alert_bot.py и
signal_dashboard.py, импортировать его из скрипта бэктеста было неправильно.
"""

import json
import os

from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Flatten, Input, Multiply, Permute, RepeatVector,
)
from tensorflow.keras.models import Model

from core.architectures import ReduceSumLayer, build_lstm_attention
from core.logger import get_logger

logger = get_logger("model_io")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OPTUNA_PARAMS_PATH = os.path.join(MODEL_DIR, "optuna_params.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_lstm_legacy(input_shape):
    """V49 architecture: LSTM(128)+Dropout+LSTM(64), Bahdanau-style attention, Lambda->ReduceSum."""
    timesteps = input_shape[0]
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    # V49 attention: Dense(1,tanh)->Flatten->Dense(ts,softmax)->RepeatVector->Permute->Multiply
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
    """Resolve actual lookback: registry.lookback -> optuna_params -> profile fallback."""
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
        # Auto-detect lookback from saved weights (handles frozen champions with outdated optuna params)
        detected_lb = detect_lookback_from_h5(weights_path if fmt == 'hdf5' else lstm_path)
        if detected_lb is not None and detected_lb != lookback:
            logger.debug("Lookback override for %s: %d -> %d (from weights)", lstm_path, lookback, detected_lb)
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
