"""Neural network architectures for the ensemble pipeline.

Provides four model builders:
  • build_lstm_attention    — 2-layer LSTM + scaled attention
  • build_transformer_encoder — 2-block Transformer encoder
  • build_lstm_multitask    — LSTM+Attention with direction + magnitude heads
  • build_tcn              — Temporal Convolutional Network (dilated causal)

All builders return compiled Keras Models ready for .fit().
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Flatten, Input, Multiply, Permute, RepeatVector,
)
from tensorflow.keras.models import Model

# legacy.Adam avoids LossScaleOptimizer bug with CosineDecay + mixed_float16.
# Keras 3 removed legacy entirely — fall back to standard Adam.
try:
    _Adam = tf.keras.optimizers.legacy.Adam
    # Verify it's actually callable (Keras 3 stub raises ImportError on __init__)
    _Adam(0.001)
except Exception:
    _Adam = tf.keras.optimizers.Adam


# ── Helpers ──────────────────────────────────────────────────────────────────

class ReduceSumLayer(tf.keras.layers.Layer):
    """Reduce-sum along the time axis (axis=1) for attention output."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)


def attention_block(lstm_output, time_steps):
    """Scaled dot-product attention over LSTM hidden states."""
    a = Dense(1, activation='tanh')(lstm_output)
    a = Flatten()(a)
    a = Dense(time_steps, activation='softmax')(a)
    a = RepeatVector(lstm_output.shape[-1])(a)
    a = Permute([2, 1])(a)
    res = Multiply()([lstm_output, a])
    return ReduceSumLayer()(res)


# ── Architecture 1: LSTM + Attention ─────────────────────────────────────────

def build_lstm_attention(input_shape):
    """Two-layer LSTM (192→96 units) with attention pooling."""
    inputs = Input(shape=input_shape)
    x = LSTM(192, return_sequences=True)(inputs)
    x = LSTM(96, return_sequences=True)(x)
    x = attention_block(x, input_shape[0])
    x = Dropout(0.2)(x)
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation='swish')(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=_Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Architecture 2: Transformer Encoder ──────────────────────────────────────

def build_transformer_encoder(input_shape, num_heads=4, ff_dim=128, dropout=0.1,
                              n_train_samples=500):
    """2-block Transformer encoder with positional embedding and cosine-decay LR."""
    inputs = Input(shape=input_shape)
    n_feat = input_shape[1]
    compute_dt = tf.keras.mixed_precision.global_policy().compute_dtype
    x = tf.keras.layers.Lambda(lambda t: tf.cast(t, compute_dt))(inputs)
    pos_emb = tf.keras.layers.Embedding(input_shape[0], n_feat)(
        tf.cast(tf.range(input_shape[0]), tf.int32))
    x = x + tf.keras.layers.Lambda(lambda t: tf.cast(t, compute_dt))(pos_emb)
    for _ in range(2):
        key_dim = max(1, n_feat // num_heads)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim)(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = Dense(ff_dim, activation='relu')(x)
        ff = Dense(n_feat)(ff)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='swish')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    steps_per_epoch = max(1, n_train_samples // 128)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4,
        decay_steps=steps_per_epoch * 80,
        alpha=5e-5,
    )
    model.compile(optimizer=_Adam(lr_schedule),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ── Architecture 3: LSTM Multitask (Direction + Magnitude) ───────────────────

def build_lstm_multitask(input_shape, n_train_samples=500):
    """LSTM+Attention with direction (primary) and magnitude (auxiliary) heads.

    Trains with richer signal. Save direction-only model for prediction.
    Uses cosine-decay LR scheduler for better convergence.
    """
    inputs = Input(shape=input_shape)
    x = LSTM(192, return_sequences=True)(inputs)
    x = LSTM(96, return_sequences=True)(x)
    x = attention_block(x, input_shape[0])
    x = Dropout(0.2)(x)
    shared = Dense(64, activation='swish')(x)
    shared = Dropout(0.15)(shared)
    # Direction head (primary task: up/down)
    d = Dense(32, activation='swish')(shared)
    dir_out = Dense(1, activation='sigmoid', dtype='float32', name='direction')(d)
    # Magnitude head (auxiliary task: normalized next_ret)
    m = Dense(16, activation='swish')(shared)
    mag_out = Dense(1, dtype='float32', name='magnitude')(m)
    model = Model(inputs=inputs, outputs=[dir_out, mag_out])
    steps_per_epoch = max(1, n_train_samples // 128)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.002,
        decay_steps=steps_per_epoch * 120,
        alpha=0.0001,
    )
    model.compile(
        optimizer=_Adam(lr_schedule),
        loss={'direction': 'binary_crossentropy', 'magnitude': 'huber'},
        loss_weights={'direction': 1.0, 'magnitude': 0.2},
        metrics={'direction': 'accuracy'},
    )
    return model


# ── Architecture 4: Temporal Convolutional Network ───────────────────────────

def build_tcn(input_shape, n_filters=64, kernel_size=3, n_blocks=3, dropout=0.15,
              n_train_samples=500):
    """Dilated causal convolutions capture multi-scale temporal patterns
    complementary to LSTM (recurrent) and Transformer (global attention).
    """
    inputs = Input(shape=input_shape)
    compute_dt = tf.keras.mixed_precision.global_policy().compute_dtype
    x = tf.keras.layers.Lambda(lambda t: tf.cast(t, compute_dt))(inputs)
    for i in range(n_blocks):
        dilation = 2 ** i
        res = x
        x = tf.keras.layers.Conv1D(n_filters, kernel_size, padding='causal',
                                    dilation_rate=dilation)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(n_filters, kernel_size, padding='causal',
                                    dilation_rate=dilation)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        if res.shape[-1] != n_filters:
            res = tf.keras.layers.Conv1D(n_filters, 1)(res)
        x = x + res
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='swish')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    steps_per_epoch = max(1, n_train_samples // 128)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4,
        decay_steps=steps_per_epoch * 60,
        alpha=5e-5,
    )
    model.compile(optimizer=_Adam(lr_schedule),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
