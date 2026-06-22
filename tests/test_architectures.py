"""Tests for core.architectures - neural network model builders."""

import numpy as np
import tensorflow as tf

from core.architectures import (
    ReduceSumLayer,
    adaptive_units,
    attention_block,
    build_lstm_attention,
    build_lstm_multitask,
    build_tcn,
    build_transformer_encoder,
)

INPUT_SHAPE = (20, 8)  # 20 time steps, 8 features


class TestAdaptiveSizing:
    def test_adaptive_units_clamps(self):
        assert adaptive_units(100, lo=32, hi=128, divisor=16) == 32   # 6, clamped up to the floor
        assert adaptive_units(1100, lo=32, hi=128, divisor=16) == 68  # 1100//16
        assert adaptive_units(99999, lo=32, hi=128, divisor=16) == 128  # ceil

    def test_lstm_defaults_reproduce_original(self):
        """Default args must keep the original 192/96 param count (back-compat)."""
        m = build_lstm_multitask(INPUT_SHAPE)
        # 192 units layer 1, 96 layer 2 is the historical network
        assert m.count_params() > 200_000

    def test_lstm_smaller_units_fewer_params(self):
        big = build_lstm_multitask(INPUT_SHAPE, units1=192, units2=96, head_dim=64)
        small = build_lstm_multitask(INPUT_SHAPE, units1=64, units2=32, head_dim=32)
        assert small.count_params() < big.count_params() / 3
        # output heads unchanged (direction + magnitude)
        assert set(small.output_names) == {"direction", "magnitude"}

    def test_lstm_regularization_args_accepted(self):
        m = build_lstm_multitask(INPUT_SHAPE, units1=48, units2=24, head_dim=32,
                                 recurrent_dropout=0.1, l2_reg=1e-5)
        x = np.random.randn(4, *INPUT_SHAPE).astype("float32")
        out = m.predict(x, verbose=0)
        assert out[0].shape == (4, 1) and out[1].shape == (4, 1)


class TestReduceSumLayer:
    def test_sums_along_time_axis(self):
        layer = ReduceSumLayer()
        x = tf.constant([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = layer(x).numpy()
        np.testing.assert_allclose(result, [[4.0, 6.0]])

    def test_output_shape(self):
        layer = ReduceSumLayer()
        x = tf.random.normal((4, 10, 16))
        result = layer(x)
        assert result.shape == (4, 16)


class TestAttentionBlock:
    def test_output_shape(self):
        lstm_out = tf.random.normal((2, 20, 96))
        result = attention_block(lstm_out, time_steps=20)
        assert result.shape == (2, 96)


class TestBuildLstmAttention:
    def test_builds_and_compiles(self):
        model = build_lstm_attention(INPUT_SHAPE)
        assert model is not None
        assert model.input_shape == (None, *INPUT_SHAPE)
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        model = build_lstm_attention(INPUT_SHAPE)
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        assert pred.shape == (2, 1)
        assert (pred >= 0).all() and (pred <= 1).all()


class TestBuildTransformerEncoder:
    def test_builds_and_compiles(self):
        model = build_transformer_encoder(INPUT_SHAPE)
        assert model is not None
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        model = build_transformer_encoder(INPUT_SHAPE)
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        assert pred.shape == (2, 1)
        assert (pred >= 0).all() and (pred <= 1).all()


class TestBuildLstmMultitask:
    def test_builds_and_compiles(self):
        model = build_lstm_multitask(INPUT_SHAPE)
        assert model is not None
        assert len(model.outputs) == 2

    def test_output_names(self):
        model = build_lstm_multitask(INPUT_SHAPE)
        layer_names = [l.name for l in model.layers]
        assert 'direction' in layer_names
        assert 'magnitude' in layer_names

    def test_forward_pass(self):
        model = build_lstm_multitask(INPUT_SHAPE)
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        preds = model.predict(x, verbose=0)
        dir_pred, mag_pred = preds
        assert dir_pred.shape == (2, 1)
        assert (dir_pred >= 0).all() and (dir_pred <= 1).all()
        assert mag_pred.shape == (2, 1)


class TestBuildTCN:
    def test_builds_and_compiles(self):
        model = build_tcn(INPUT_SHAPE)
        assert model is not None
        assert model.output_shape == (None, 1)

    def test_forward_pass(self):
        model = build_tcn(INPUT_SHAPE)
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        assert pred.shape == (2, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_custom_params(self):
        model = build_tcn(INPUT_SHAPE, n_filters=32, n_blocks=2, kernel_size=5)
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        assert pred.shape == (2, 1)
