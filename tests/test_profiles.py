"""Tests for core.profiles — asset profile mapping."""

from core.profiles import (
    FOREX,
    MEANREV,
    PROFILE_DEFAULT,
    PROFILE_FOREX,
    PROFILE_MEANREV,
    PROFILE_RUS,
    PROFILE_TRENDY,
    RUS,
    TRENDY,
    get_profile,
)


class TestGetProfile:
    def test_trendy_assets(self):
        for asset in ["BTC", "ETH", "NVDA", "TSLA"]:
            assert get_profile(asset) == PROFILE_TRENDY

    def test_meanrev_assets(self):
        for asset in ["VIX", "DXY", "TNX"]:
            assert get_profile(asset) == PROFILE_MEANREV

    def test_russian_assets(self):
        for asset in ["SBER", "GAZP", "LKOH"]:
            assert get_profile(asset) == PROFILE_RUS

    def test_forex_assets(self):
        for asset in ["EURUSD", "GBPUSD", "USDJPY"]:
            assert get_profile(asset) == PROFILE_FOREX

    def test_default_profile(self):
        assert get_profile("UNKNOWN_ASSET") == PROFILE_DEFAULT
        assert get_profile("AAPL") == PROFILE_DEFAULT

    def test_no_overlap_between_categories(self):
        all_sets = [TRENDY, MEANREV, RUS, FOREX]
        for i, s1 in enumerate(all_sets):
            for j, s2 in enumerate(all_sets):
                if i != j:
                    overlap = s1 & s2
                    assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_profile_has_required_keys(self):
        required = ['lookback', 'thr_buy_grid', 'thr_sell_grid',
                     'no_trade_band', 'regime_risk_cap', 'trend_gate', 'top_k_features']
        for profile in [PROFILE_DEFAULT, PROFILE_TRENDY, PROFILE_MEANREV, PROFILE_RUS, PROFILE_FOREX]:
            for key in required:
                assert key in profile, f"Missing key '{key}' in profile"

    def test_lookback_positive(self):
        for profile in [PROFILE_DEFAULT, PROFILE_TRENDY, PROFILE_MEANREV, PROFILE_RUS, PROFILE_FOREX]:
            assert profile['lookback'] > 0

    def test_buy_threshold_above_sell(self):
        for profile in [PROFILE_DEFAULT, PROFILE_TRENDY, PROFILE_MEANREV, PROFILE_RUS, PROFILE_FOREX]:
            assert min(profile['thr_buy_grid']) > max(profile['thr_sell_grid'])
