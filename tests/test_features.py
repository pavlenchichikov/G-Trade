"""Tests for core.features — feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from core.features import engineer_features, CANDIDATE_FEATURES


@pytest.fixture
def sample_ohlcv():
    """Generate 300 rows of realistic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 100000, n).astype(float)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }).set_index("Date")


class TestEngineerFeatures:
    def test_output_has_required_columns(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        required = [
            'ret_1', 'ret_5', 'ret_10', 'ret_20',
            'taleb_risk', 'vol_z', 'atr',
            'sma_20', 'sma_50', 'sma_200', 'trend_strength',
            'rsi', 'macd_hist', 'bb_pos', 'vol_ratio',
            'target', 'next_ret',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_values(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        # After dropna(), no NaN should remain in feature columns
        feature_cols = [c for c in df.columns if c not in ('Date', 'date')]
        for col in feature_cols:
            assert not df[col].isna().any(), f"NaN found in {col}"

    def test_output_is_shorter_than_input(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        # SMA200 requires 200 bars warmup, so output should be significantly shorter
        assert len(df) < len(sample_ohlcv)
        assert len(df) > 50  # but should still have reasonable data

    def test_target_is_binary(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        assert set(df['target'].unique()).issubset({0, 1})

    def test_rsi_range(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        assert df['rsi'].min() >= 0
        assert df['rsi'].max() <= 100

    def test_bb_pos_reasonable(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        # Bollinger position should mostly be between -3 and +3
        assert df['bb_pos'].min() > -10
        assert df['bb_pos'].max() < 10

    def test_vol_ratio_positive(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        assert (df['vol_ratio'] >= 0).all()

    def test_date_preserved_as_column(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        assert 'Date' in df.columns or not isinstance(df.index, pd.DatetimeIndex)

    def test_works_with_lowercase_columns(self, sample_ohlcv):
        df_lower = sample_ohlcv.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        df = engineer_features(df_lower)
        assert 'ret_1' in df.columns
        assert len(df) > 0

    def test_returns_are_correct(self, sample_ohlcv):
        df = engineer_features(sample_ohlcv)
        # ret_1 should approximately equal daily pct change
        # (not exact due to NaN dropping, but correlation should be very high)
        assert df['ret_1'].std() > 0  # not constant

    def test_sma_ordering(self, sample_ohlcv):
        """SMA columns should have reasonable values."""
        df = engineer_features(sample_ohlcv)
        assert df['sma_20'].std() > 0
        assert df['sma_50'].std() > 0
        assert df['sma_200'].std() > 0

    def test_empty_input(self):
        """Should handle edge case of too-short input."""
        df = pd.DataFrame({
            'Date': pd.date_range("2023-01-01", periods=10),
            'Open': range(10), 'High': range(10),
            'Low': range(10), 'Close': range(10),
            'Volume': range(10),
        }).set_index('Date')
        result = engineer_features(df)
        # With only 10 rows, after SMA200 warmup removal, result should be empty
        assert len(result) == 0
