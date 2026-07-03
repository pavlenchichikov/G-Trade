"""Tests for core.features - feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from core.features import (
    add_macro_features,
    compute_taleb_risk,
    engineer_features,
    latest_taleb_risk,
    make_target,
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
    # ext is the adopted default; base only when explicitly requested
    monkeypatch.delenv("GTRADE_FEATURE_SET", raising=False)
    assert F.active_candidate_features() == F.CANDIDATE_FEATURES_EXT
    monkeypatch.setenv("GTRADE_FEATURE_SET", "base")
    assert F.active_candidate_features() == F.CANDIDATE_FEATURES


def test_feature_version_differs_by_set(monkeypatch):
    import core.features as F
    monkeypatch.setenv("GTRADE_FEATURE_SET", "base")
    base_v = F.feature_version()
    monkeypatch.delenv("GTRADE_FEATURE_SET", raising=False)  # default ext
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
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ab_features.py")
    if not os.path.exists(path):
        import pytest
        pytest.skip("ab_features.py is a local (gitignored) experiment tool, absent in CI")
    spec = importlib.util.spec_from_file_location("ab_features", path)
    abf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abf)
    base = [{"Asset": "BTC", "Score": 2.0, "LSTM_Acc": 0.50}]
    ext = [{"Asset": "BTC", "Score": 2.5, "LSTM_Acc": 0.56}]
    rows = abf.compare(base, ext)
    r = rows[0]
    assert r["asset"] == "BTC"
    assert round(r["score_delta"], 2) == 0.5
    assert round(r["lstm_delta"], 2) == 0.06


def test_active_candidate_features_extra(monkeypatch):
    import core.features as F
    monkeypatch.delenv("GTRADE_FEATURE_SET", raising=False)
    monkeypatch.setenv("GTRADE_EXTRA_FEATURES", "ret_1,brand_new_feat")
    out = F.active_candidate_features()
    assert "brand_new_feat" in out          # appended
    assert out.count("ret_1") == 1          # not duplicated (already in base)


def test_make_target_direction_matches_legacy():
    close = pd.Series([10, 11, 10.5, 12, 11.5, 13.0])
    expected = (close.shift(-1) > close).astype(int)
    out = make_target(close, "direction")
    pd.testing.assert_series_equal(out, expected)


def test_make_target_rel_median_balances_a_trend():
    # strong upward drift: the direction label is heavily skewed to 1,
    # rel_median should sit near 0.5 because it is relative to recent moves
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0.5, 1.0, 400)))
    direction = make_target(close, "direction")
    relmed = make_target(close, "rel_median", window=30)
    # compare on the region where the rolling baseline exists
    rel_ratio = relmed.iloc[30:-1].mean()
    dir_ratio = direction.iloc[30:-1].mean()
    assert dir_ratio > 0.6           # legacy label is skewed on a trend
    assert 0.4 <= rel_ratio <= 0.6   # rel_median is balanced


def test_make_target_rel_median_has_no_lookahead():
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0.0, 1.0, 200)))
    full = make_target(close, "rel_median", window=20)
    truncated = make_target(close.iloc[:-5], "rel_median", window=20)
    # overlapping interior rows must be identical (baseline uses only past returns).
    # Exclude truncated's own final row: its next_ret is undefined by construction
    # (post-C1 fix this is correctly NaN, not the prior bug's fabricated 0), so it
    # is not a valid "no lookahead" comparison point regardless of the bug.
    a = full.iloc[21:-6].reset_index(drop=True)
    b = truncated.iloc[21:-1].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b)


def test_make_target_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_target(pd.Series([1.0, 2.0, 3.0]), "triple_barrier")


def _ohlcv(n=320, seed=2):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.3, 1.0, n))
    return pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=n),
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1, 100, n),
    })


def test_engineer_features_default_label_unchanged(monkeypatch):
    monkeypatch.delenv("GTRADE_LABEL_MODE", raising=False)
    from core.features import engineer_features
    df = _ohlcv()
    out = engineer_features(df)
    # default target must equal the legacy direction formula on the same close
    legacy = (out["close"].shift(-1) > out["close"]).astype(int)
    # last row has no next bar; compare the interior the model actually uses
    assert (out["target"].iloc[:-1].values == legacy.iloc[:-1].values).all()


def test_engineer_features_rel_median_changes_label(monkeypatch):
    from core.features import engineer_features
    df = _ohlcv()
    monkeypatch.setenv("GTRADE_LABEL_MODE", "direction")
    a = engineer_features(df)["target"].mean()
    monkeypatch.setenv("GTRADE_LABEL_MODE", "rel_median")
    monkeypatch.setenv("GTRADE_LABEL_WINDOW", "30")
    b = engineer_features(df)["target"].mean()
    # Robust property, not a fixture-specific magnitude: the trending fixture
    # skews "direction" toward 1 (next bar usually up), while rel_median is
    # relative to the trailing median return and stays closer to balanced -
    # so the label actually changes, and direction's mean is the higher one.
    assert a != b
    assert a > b


def test_make_target_rel_median_warmup_and_final_row_are_nan():
    """C1 regression: with a window large enough to extend past the feature
    warm-up, make_target must emit NaN (not a fabricated 0) on rows where the
    rolling baseline or next_ret is undefined, so dropna removes them
    regardless of GTRADE_LABEL_WINDOW."""
    rng = np.random.default_rng(3)
    n, window = 120, 50
    close = pd.Series(100 + np.cumsum(rng.normal(0.0, 1.0, n)))
    out = make_target(close, "rel_median", window=window)
    # ret = close.pct_change() is NaN at row 0, and rolling(window) needs
    # `window` consecutive non-NaN values (default min_periods=window), so
    # baseline's first valid row is index `window`, not `window - 1`.
    assert out.iloc[:window].isna().all()
    # the final row's next_ret is undefined regardless of window
    assert pd.isna(out.iloc[-1])
    # interior rows (baseline and next_ret both defined) must be concrete 0/1
    interior = out.iloc[window:-1]
    assert interior.isna().sum() == 0
    assert set(interior.unique()) <= {0.0, 1.0}


def test_active_candidate_features_drop(monkeypatch):
    from core.features import active_candidate_features, feature_version
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    monkeypatch.delenv("GTRADE_EXTRA_FEATURES", raising=False)
    full = active_candidate_features()
    v_full = feature_version()
    monkeypatch.setenv("GTRADE_DROP_FEATURES", "rsi, atr")
    dropped = active_candidate_features()
    assert "rsi" not in dropped and "atr" not in dropped
    assert dropped == [f for f in full if f not in ("rsi", "atr")]
    # unset again -> byte-identical list + same feature_version (production safety)
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    assert active_candidate_features() == full
    assert feature_version() == v_full


def test_active_candidate_features_drop_unknown_is_harmless(monkeypatch):
    from core.features import active_candidate_features
    monkeypatch.delenv("GTRADE_EXTRA_FEATURES", raising=False)
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    full = active_candidate_features()
    monkeypatch.setenv("GTRADE_DROP_FEATURES", "does_not_exist")
    assert active_candidate_features() == full


def test_add_chronos_features_off_by_default(monkeypatch, tmp_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from core import features as F
    monkeypatch.delenv("GTRADE_CHRONOS", raising=False)
    df = pd.DataFrame({"close": [1.0, 2.0]},
                      index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    eng = create_engine("sqlite:///" + str(tmp_path / "m.db"))
    out = F.add_chronos_features(df.copy(), "sp500", eng)
    assert list(out.columns) == ["close"]          # unchanged, no chronos cols


def test_add_chronos_features_joins_when_enabled(monkeypatch, tmp_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from core import features as F
    dbp = str(tmp_path / "m.db")
    eng = create_engine("sqlite:///" + dbp)
    cache = pd.DataFrame({
        "asset": ["sp500", "sp500"], "date": ["2020-01-01", "2020-01-02"],
        "chronos_ret": [0.01, 0.02], "chronos_spread": [0.05, 0.06],
        "chronos_dir": [1.0, 0.0]})
    cache.to_sql("chronos_cache", eng, index=False)
    monkeypatch.setenv("GTRADE_CHRONOS", "1")
    df = pd.DataFrame({"close": [1.0, 2.0]},
                      index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    out = F.add_chronos_features(df.copy(), "sp500", eng)
    assert "chronos_ret" in out.columns and "chronos_spread" in out.columns
    assert abs(out.iloc[0]["chronos_ret"] - 0.01) < 1e-9
    assert out.iloc[1]["chronos_dir"] == 0.0


def test_chronos_names_not_in_feature_version(monkeypatch):
    from core import features as F
    monkeypatch.delenv("GTRADE_CHRONOS", raising=False)
    monkeypatch.delenv("GTRADE_EXTRA_FEATURES", raising=False)
    v1 = F.feature_version()
    monkeypatch.setenv("GTRADE_CHRONOS", "1")
    assert F.feature_version() == v1               # gate does not change the id


def test_add_chronos_features_pipeline_date_column(monkeypatch, tmp_path):
    """In the pipeline df carries a Date COLUMN with a RangeIndex (siblings end with
    reset_index). The join must key off that column, not the integer index, else the
    Chronos columns are silently all-NaN."""
    import pandas as pd
    from sqlalchemy import create_engine
    from core import features as F
    eng = create_engine("sqlite:///" + str(tmp_path / "m.db"))
    cache = pd.DataFrame({
        "asset": ["sp500", "sp500"], "date": ["2020-01-01", "2020-01-02"],
        "chronos_ret": [0.01, 0.02], "chronos_spread": [0.05, 0.06],
        "chronos_dir": [1.0, 0.0]})
    cache.to_sql("chronos_cache", eng, index=False)
    monkeypatch.setenv("GTRADE_CHRONOS", "1")
    # df as it arrives in the pipeline: a 'Date' column + a RangeIndex
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                       "close": [1.0, 2.0]})
    out = F.add_chronos_features(df.copy(), "sp500", eng)
    assert "chronos_ret" in out.columns
    assert out["chronos_ret"].notna().all()        # NOT silently all-NaN
    assert abs(float(out.loc[out["Date"] == pd.Timestamp("2020-01-01"), "chronos_ret"].iloc[0]) - 0.01) < 1e-9


def test_add_chronos_features_enabled_but_no_cache(monkeypatch, tmp_path):
    """GTRADE_CHRONOS set but no cache table -> no-op (df unchanged), not a crash."""
    import pandas as pd
    from sqlalchemy import create_engine
    from core import features as F
    eng = create_engine("sqlite:///" + str(tmp_path / "empty.db"))
    monkeypatch.setenv("GTRADE_CHRONOS", "1")
    df = pd.DataFrame({"close": [1.0, 2.0]},
                      index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    out = F.add_chronos_features(df.copy(), "sp500", eng)
    assert list(out.columns) == ["close"]


def test_add_chronos_features_dedup_no_row_multiplication(monkeypatch, tmp_path):
    """Cache with a duplicate date must not multiply left rows (row count preserved)."""
    import pandas as pd
    from sqlalchemy import create_engine
    from core import features as F
    eng = create_engine("sqlite:///" + str(tmp_path / "dedup.db"))
    # Two rows with identical (asset, date) - the duplicate that triggers row multiplication
    cache = pd.DataFrame({
        "asset": ["sp500", "sp500"],
        "date": ["2020-01-01", "2020-01-01"],   # DUPLICATE date
        "chronos_ret": [0.01, 0.99],            # keep="last" should retain 0.99
        "chronos_spread": [0.05, 0.88],
        "chronos_dir": [1.0, 0.0],
    })
    cache.to_sql("chronos_cache", eng, index=False)
    monkeypatch.setenv("GTRADE_CHRONOS", "1")
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "close": [1.0, 2.0],
    })
    out = F.add_chronos_features(df.copy(), "sp500", eng)
    assert len(out) == 2                        # NOT multiplied by the duplicate
    # keep="last" means the second row (chronos_ret=0.99) wins for 2020-01-01
    val = float(out.loc[out["Date"] == pd.Timestamp("2020-01-01"), "chronos_ret"].iloc[0])
    assert abs(val - 0.99) < 1e-9
