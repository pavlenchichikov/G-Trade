"""Tests for core.ensemble — gating, stacking, weight tuning."""

import numpy as np
import pytest

from core.ensemble import (
    build_stacking_features,
    ensemble_with_gating,
    tune_ensemble_weights,
)


class TestEnsembleWithGating:
    def test_equal_inputs_return_midpoint(self):
        cb = np.array([0.6, 0.6, 0.6])
        lstm = np.array([0.6, 0.6, 0.6])
        trend = np.array([0.01, 0.01, 0.01])
        result = ensemble_with_gating(cb, lstm, trend)
        np.testing.assert_allclose(result, 0.6, atol=1e-6)

    def test_output_range_0_to_1(self):
        np.random.seed(42)
        cb = np.random.rand(100)
        lstm = np.random.rand(100)
        trend = np.random.rand(100) * 0.05
        result = ensemble_with_gating(cb, lstm, trend)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_override_weight(self):
        cb = np.array([0.4])
        lstm = np.array([0.8])
        trend = np.array([0.01])
        # w_lstm=1.0 should return pure LSTM (adjusted by disagreement)
        result = ensemble_with_gating(cb, lstm, trend, w_lstm_override=1.0)
        assert result[0] > 0.6  # Should lean toward LSTM's 0.8

    def test_disagreement_reduces_confidence(self):
        cb = np.array([0.3])
        lstm = np.array([0.9])
        trend = np.array([0.01])
        result = ensemble_with_gating(cb, lstm, trend, w_lstm_override=0.5)
        # Average of 0.3 and 0.9 is 0.6, but disagreement pulls toward 0.5
        assert result[0] < 0.6

    def test_agreement_preserves_signal(self):
        cb = np.array([0.8])
        lstm = np.array([0.8])
        trend = np.array([0.01])
        result = ensemble_with_gating(cb, lstm, trend)
        # Both agree on 0.8, disagreement = 0, so confidence = 1.0
        assert abs(result[0] - 0.8) < 0.01

    def test_high_trend_increases_lstm_weight(self):
        cb = np.array([0.4])
        lstm = np.array([0.7])
        trend_low = np.array([0.001])
        trend_high = np.array([0.05])
        res_low = ensemble_with_gating(cb, lstm, trend_low)
        res_high = ensemble_with_gating(cb, lstm, trend_high)
        # Higher trend → more LSTM weight → result closer to 0.7
        assert res_high[0] > res_low[0]


class TestTuneEnsembleWeights:
    def test_returns_valid_weight(self):
        np.random.seed(42)
        n = 200
        y_val = np.random.randint(0, 2, n)
        cb = y_val * 0.3 + 0.35 + np.random.randn(n) * 0.1
        lstm = y_val * 0.3 + 0.35 + np.random.randn(n) * 0.1
        trend = np.random.rand(n) * 0.02
        w, acc = tune_ensemble_weights(cb, lstm, y_val, trend, gate=0.01)
        assert 0.0 <= w <= 1.0
        assert 0.0 <= acc <= 1.0

    def test_low_lstm_acc_limits_weight(self):
        np.random.seed(42)
        n = 200
        y_val = np.random.randint(0, 2, n)
        cb = y_val * 0.4 + 0.3
        lstm = np.random.rand(n)  # Random = bad
        trend = np.random.rand(n) * 0.02
        w, _ = tune_ensemble_weights(cb, lstm, y_val, trend, gate=0.01, lstm_acc=0.48)
        assert w <= 0.40  # Should be capped due to low LSTM accuracy

    def test_optimal_weight_improves_accuracy(self):
        np.random.seed(42)
        n = 200
        y_val = np.random.randint(0, 2, n)
        cb = y_val * 0.3 + 0.35
        lstm = y_val * 0.3 + 0.35
        trend = np.ones(n) * 0.01
        _, best_acc = tune_ensemble_weights(cb, lstm, y_val, trend, gate=0.01)
        # Best accuracy should be at least 50% (better than random)
        assert best_acc >= 0.5


class TestBuildStackingFeatures:
    def test_output_shape(self):
        n = 50
        cb = np.random.rand(n)
        lstm = np.random.rand(n)
        tf = np.random.rand(n)
        tcn = np.random.rand(n)
        trend = np.random.rand(n)
        result = build_stacking_features(cb, lstm, tf, tcn, trend)
        assert result.shape == (n, 7)  # 4 models + disagree + mean + trend

    def test_first_four_columns_are_model_probs(self):
        cb = np.array([0.6])
        lstm = np.array([0.7])
        tf = np.array([0.5])
        tcn = np.array([0.8])
        trend = np.array([0.01])
        result = build_stacking_features(cb, lstm, tf, tcn, trend)
        np.testing.assert_allclose(result[0, :4], [0.6, 0.7, 0.5, 0.8])

    def test_disagreement_column(self):
        # All models agree → disagreement = 0
        cb = np.array([0.6])
        result = build_stacking_features(cb, cb, cb, cb, np.array([0.01]))
        assert result[0, 4] == 0.0  # std of [0.6, 0.6, 0.6, 0.6] = 0

    def test_trend_is_last_column(self):
        trend = np.array([0.042])
        result = build_stacking_features(
            np.array([0.5]), np.array([0.5]),
            np.array([0.5]), np.array([0.5]), trend,
        )
        assert result[0, -1] == pytest.approx(0.042)
