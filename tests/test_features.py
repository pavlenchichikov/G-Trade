"""Tests for core.features - feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from core.features import (
    add_macro_features,
    compute_taleb_risk,
    engineer_features,
    latest_taleb_risk,
    _MACRO_FEATURES,
)


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


class TestTalebRisk:
    def test_matches_engineer_features_column(self, sample_ohlcv):
        """The extracted helper must reproduce the in-pipeline taleb_risk exactly."""
        close = sample_ohlcv["Close"]
        helper = compute_taleb_risk(close)
        # Same formula engineer_features uses before the warm-up median fill.
        log_ret = np.log(close / close.shift(1))
        expected = log_ret.rolling(window=60, min_periods=30).kurt()
        pd.testing.assert_series_equal(helper, expected, check_names=False)

    def test_latest_returns_last_value(self, sample_ohlcv):
        closes = list(sample_ohlcv["Close"])
        val = latest_taleb_risk(closes)
        assert val is not None
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_latest_fills_warmup_with_median(self):
        """A short series (only warm-up rows are NaN) still yields a number, not NaN."""
        # 40 closes: rolling(60, min_periods=30) gives a value only from row 30 on.
        closes = list(100 + np.cumsum(np.random.RandomState(0).randn(40) * 0.5))
        val = latest_taleb_risk(closes)
        assert val is not None and np.isfinite(val)

    def test_latest_none_when_too_short(self):
        assert latest_taleb_risk([100, 101, 102]) is None
        assert latest_taleb_risk([]) is None


class TestMacroFeatures:
    def _engine(self, tmp_path):
        from sqlalchemy import create_engine
        return create_engine(f"sqlite:///{tmp_path / 'macro_test.db'}")

    def test_missing_tables_fill_zero(self, tmp_path):
        """No tnx/vix/dxy tables: all macro columns present and 0.0."""
        eng = self._engine(tmp_path)
        df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5),
                           "close": [1, 2, 3, 4, 5]})
        out = add_macro_features(df, eng)
        for c in _MACRO_FEATURES:
            assert c in out.columns
            assert (out[c] == 0.0).all()

    def test_as_of_forward_fill(self, tmp_path):
        """A bar between two source dates gets the last value known on/before it."""
        eng = self._engine(tmp_path)
        pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"],
                      "close": [4.0, 4.5]}).to_sql("tnx", eng, index=False)
        pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"],
                      "close": [13.0, 20.0]}).to_sql("vix", eng, index=False)
        pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"],
                      "close": [100.0, 101.0]}).to_sql("dxy", eng, index=False)
        # asset bar on 2024-01-05 is between the two dates, so it takes the 01-01 value
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05", "2024-01-11"]),
                           "close": [10.0, 11.0]})
        out = add_macro_features(df, eng).set_index("Date")
        assert out.loc["2024-01-05", "macro_tnx"] == 4.0
        assert out.loc["2024-01-05", "macro_vix"] == 13.0
        assert out.loc["2024-01-11", "macro_tnx"] == 4.5  # after the 01-10 update
        assert np.isfinite(out["macro_vix_chg5"]).all()


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

    def test_tail_risk_columns_present(self, sample_ohlcv):
        """Taleb block now exposes asymmetry (skew) and downside tail (VaR)."""
        df = engineer_features(sample_ohlcv)
        for col in ('taleb_risk', 'ret_skew', 'var_5'):
            assert col in df.columns, f"Missing tail-risk column: {col}"
        # var_5 is a 5% left-tail return: should be non-positive on average
        assert df['var_5'].mean() <= 0

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


def test_candidate_lists_base_and_ext():
    import core.features as F
    # base: has close/volume, no macro
    assert "close" in F.CANDIDATE_FEATURES and "volume" in F.CANDIDATE_FEATURES
    assert not any(c.startswith("macro_") for c in F.CANDIDATE_FEATURES)
    assert len(F.CANDIDATE_FEATURES) == 24
    # ext: no close/volume, has macro + new features, keeps sma
    assert "close" not in F.CANDIDATE_FEATURES_EXT
    assert "volume" not in F.CANDIDATE_FEATURES_EXT
    assert "sma_20" in F.CANDIDATE_FEATURES_EXT
    for c in ("macro_tnx", "ret_1_vn", "lead_sp500_ret", "cal_dow"):
        assert c in F.CANDIDATE_FEATURES_EXT
    assert len(F.CANDIDATE_FEATURES_EXT) == 34


def test_active_candidate_features_flag(monkeypatch):
    import core.features as F
    monkeypatch.delenv("GTRADE_FEATURE_SET", raising=False)
    assert F.active_candidate_features() == F.CANDIDATE_FEATURES
    monkeypatch.setenv("GTRADE_FEATURE_SET", "ext")
    assert F.active_candidate_features() == F.CANDIDATE_FEATURES_EXT


def test_feature_version_differs_by_set(monkeypatch):
    import core.features as F
    monkeypatch.delenv("GTRADE_FEATURE_SET", raising=False)
    base_v = F.feature_version()
    monkeypatch.setenv("GTRADE_FEATURE_SET", "ext")
    assert F.feature_version() != base_v


def test_engineer_features_has_vn_and_calendar(sample_ohlcv):
    from core.features import engineer_features
    df = engineer_features(sample_ohlcv)
    for c in ("ret_1_vn", "ret_5_vn", "cal_dow", "cal_mpos"):
        assert c in df.columns
        assert df[c].notna().all()
    assert df["cal_dow"].between(0.0, 1.0).all()
    assert df["cal_mpos"].between(0.0, 1.0).all()


def test_add_cross_lag_features_asof_and_missing(tmp_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from core.features import add_cross_lag_features
    eng = create_engine(f"sqlite:///{tmp_path / 'lead.db'}")
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"], "Close": [100.0, 110.0]}).to_sql("sp500", eng, index=False)
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-10"], "Close": [13.0, 20.0]}).to_sql("vix", eng, index=False)
    # btc table intentionally missing, so lead_btc_ret must be 0.0
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05", "2024-01-11"]), "close": [1.0, 2.0]})
    out = add_cross_lag_features(df, eng)
    for c in ("lead_sp500_ret", "lead_vix_ret", "lead_btc_ret"):
        assert c in out.columns and out[c].notna().all()
    assert (out["lead_btc_ret"] == 0.0).all()
    # the 2024-01-11 bar sees the 2024-01-10 leader values (the prior known move)
    assert out.set_index("Date").loc["2024-01-11", "lead_vix_ret"] != 0.0


def test_ab_features_compare():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("ab_features", os.path.join(os.path.dirname(os.path.dirname(__file__)), "ab_features.py"))
    abf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abf)
    base = [{"Asset": "BTC", "Score": 2.0, "LSTM_Acc": 0.50}]
    ext = [{"Asset": "BTC", "Score": 2.5, "LSTM_Acc": 0.56}]
    rows = abf.compare(base, ext)
    r = rows[0]
    assert r["asset"] == "BTC"
    assert round(r["score_delta"], 2) == 0.5
    assert round(r["lstm_delta"], 2) == 0.06
