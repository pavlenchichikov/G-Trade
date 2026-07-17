"""Tests for core.backtesting - walk-forward splits, PnL, scoring."""

import numpy as np

from core.backtesting import (
    adaptive_split_params,
    apply_regime_filter,
    make_signals,
    make_walk_forward_splits,
    max_drawdown_from_returns,
    pnl_from_signals,
    score_strategy,
    sharpe_from_returns,
)


class TestAdaptiveSplitParams:
    def test_large_dataset(self):
        params = adaptive_split_params(5000)
        assert params is not None
        assert params["min_train"] == 500
        assert params["step"] == 360

    def test_medium_dataset(self):
        params = adaptive_split_params(1000)
        assert params is not None
        assert params["min_train"] == 500

    def test_small_dataset(self):
        params = adaptive_split_params(400)
        assert params is not None
        assert params["min_train"] == 220

    def test_tiny_dataset(self):
        params = adaptive_split_params(100)
        assert params is None

    def test_boundary_values(self):
        assert adaptive_split_params(220) is not None
        assert adaptive_split_params(219) is None


class TestWalkForwardSplits:
    def test_generates_splits(self):
        splits = make_walk_forward_splits(1000, min_train=200, val_size=100, test_size=100, step=100)
        assert len(splits) > 0

    def test_no_data_leak(self):
        """Train must come before val, val before test."""
        splits = make_walk_forward_splits(1000, min_train=200, val_size=100, test_size=100, step=100)
        for train_s, val_s, test_s in splits:
            assert train_s.stop <= val_s.start
            assert val_s.stop <= test_s.start

    def test_expanding_window(self):
        """First split should have less training data than last."""
        splits = make_walk_forward_splits(2000, min_train=200, val_size=100, test_size=100, step=200)
        if len(splits) > 1:
            first_train = splits[0][0].stop
            last_train = splits[-1][0].stop
            assert last_train > first_train

    def test_empty_on_small_data(self):
        splits = make_walk_forward_splits(50, min_train=200, val_size=100, test_size=100)
        assert len(splits) == 0

    def test_test_does_not_exceed_data(self):
        splits = make_walk_forward_splits(800, min_train=500, val_size=120, test_size=120, step=120)
        for _, _, test_s in splits:
            assert test_s.stop <= 800

    def test_embargo_inserts_gap(self):
        """With embargo, a gap separates train/val and val/test (purged CV)."""
        splits = make_walk_forward_splits(
            1000, min_train=200, val_size=100, test_size=100, step=100, embargo=10
        )
        assert len(splits) > 0
        for train_s, val_s, test_s in splits:
            assert val_s.start - train_s.stop == 10
            assert test_s.start - val_s.stop == 10

    def test_embargo_zero_matches_contiguous(self):
        """embargo=0 reproduces the original contiguous behavior."""
        splits = make_walk_forward_splits(
            1000, min_train=200, val_size=100, test_size=100, step=100, embargo=0
        )
        for train_s, val_s, test_s in splits:
            assert val_s.start == train_s.stop
            assert test_s.start == val_s.stop


class TestPnLFromSignals:
    def test_no_trades(self):
        signals = np.zeros(100)
        returns = np.random.randn(100) * 0.01
        profit, trades, winrate = pnl_from_signals(signals, returns)
        assert trades == 0
        assert profit == 0.0
        assert winrate == 0.0

    def test_all_winning_trades(self):
        signals = np.ones(100)
        returns = np.ones(100) * 0.02  # 2% up every day
        profit, trades, winrate = pnl_from_signals(signals, returns)
        assert trades == 100
        assert winrate > 0  # Most should win (2% - costs > 0)
        assert profit > 0

    def test_sell_signals_in_downtrend(self):
        signals = -np.ones(50)
        returns = -np.ones(50) * 0.02  # 2% down every day
        profit, trades, winrate = pnl_from_signals(signals, returns)
        assert trades == 50
        assert profit > 0  # Short in downtrend = profit

    def test_nan_returns_skipped(self):
        signals = np.array([1, 1, 1, 0, 1])
        returns = np.array([0.01, np.nan, 0.01, 0.01, 0.01])
        profit, trades, _ = pnl_from_signals(signals, returns)
        assert trades == 3  # NaN and 0-signal skipped

    def test_commission_reduces_profit(self):
        signals = np.ones(50)
        returns = np.ones(50) * 0.003  # Small positive return
        profit_low, _, _ = pnl_from_signals(signals, returns, commission=0.001, slippage=0.0)
        profit_high, _, _ = pnl_from_signals(signals, returns, commission=0.005, slippage=0.0)
        assert profit_low > profit_high


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = np.ones(100) * 0.01  # Always positive
        dd = max_drawdown_from_returns(returns)
        assert dd == 0.0

    def test_full_drawdown(self):
        returns = np.array([0.5, -0.99])  # Massive loss
        dd = max_drawdown_from_returns(returns)
        assert dd > 50

    def test_empty_returns(self):
        assert max_drawdown_from_returns([]) == 0.0

    def test_realistic_drawdown(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        dd = max_drawdown_from_returns(returns)
        assert 0 <= dd <= 100


class TestScoreStrategy:
    def test_insufficient_trades(self):
        assert score_strategy(10.0, 5.0, 60.0, 5) == -999.0

    def test_positive_score(self):
        score = score_strategy(20.0, 5.0, 65.0, 50)
        assert score > 0

    def test_higher_profit_higher_score(self):
        score1 = score_strategy(10.0, 5.0, 55.0, 50)
        score2 = score_strategy(20.0, 5.0, 55.0, 50)
        assert score2 > score1

    def test_higher_drawdown_lower_score(self):
        score1 = score_strategy(10.0, 5.0, 55.0, 50)
        score2 = score_strategy(10.0, 20.0, 55.0, 50)
        assert score1 > score2

    def test_sharpe_rewards_consistency(self):
        """Same profit/dd/winrate but higher Sharpe must score higher."""
        low = score_strategy(10.0, 5.0, 55.0, 50, sharpe=0.5)
        high = score_strategy(10.0, 5.0, 55.0, 50, sharpe=2.5)
        assert high > low

    def test_sharpe_none_is_legacy_formula(self):
        assert score_strategy(20.0, 5.0, 65.0, 50, sharpe=None) == score_strategy(
            20.0, 5.0, 65.0, 50
        )


class TestSharpeFromReturns:
    def test_too_few_returns(self):
        assert sharpe_from_returns([0.01]) == 0.0

    def test_zero_variance(self):
        assert sharpe_from_returns([0.01, 0.01, 0.01]) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.002, 0.01, 200)
        assert sharpe_from_returns(r) > 0

    def test_nan_skipped(self):
        a = sharpe_from_returns([0.01, 0.02, -0.01, 0.015])
        b = sharpe_from_returns([0.01, 0.02, np.nan, -0.01, 0.015])
        assert abs(a - b) < 1e-9


class TestMakeSignals:
    def test_buy_above_threshold(self):
        prob = np.array([0.6, 0.7, 0.8])
        signals = make_signals(prob, buy_thr=0.55, sell_thr=0.45, no_trade_band=0.02)
        assert (signals == 1).all()

    def test_sell_below_threshold(self):
        prob = np.array([0.3, 0.2, 0.1])
        signals = make_signals(prob, buy_thr=0.55, sell_thr=0.45, no_trade_band=0.02)
        assert (signals == -1).all()

    def test_neutral_in_band(self):
        prob = np.array([0.50, 0.48, 0.52])
        signals = make_signals(prob, buy_thr=0.55, sell_thr=0.45, no_trade_band=0.05)
        assert (signals == 0).all()

    def test_mixed_signals(self):
        prob = np.array([0.7, 0.5, 0.3])
        signals = make_signals(prob, buy_thr=0.55, sell_thr=0.45, no_trade_band=0.02)
        assert signals[0] == 1
        assert signals[1] == 0
        assert signals[2] == -1


class TestRegimeFilter:
    def test_suppresses_buy_in_downtrend(self):
        signals = np.array([1, 1, 1])
        close = np.array([90.0, 85.0, 80.0])
        sma200 = np.array([100.0, 100.0, 100.0])
        taleb = np.array([1.0, 1.0, 1.0])
        filt = apply_regime_filter(signals, close, sma200, taleb, risk_cap=5.0)
        assert (filt == 0).all()  # All buys suppressed (close < sma200)

    def test_allows_buy_in_uptrend(self):
        signals = np.array([1, 1, 1])
        close = np.array([110.0, 115.0, 120.0])
        sma200 = np.array([100.0, 100.0, 100.0])
        taleb = np.array([1.0, 1.0, 1.0])
        filt = apply_regime_filter(signals, close, sma200, taleb, risk_cap=5.0)
        assert (filt == 1).all()

    def test_blocks_on_high_taleb(self):
        signals = np.array([1, -1, 1])
        close = np.array([110.0, 90.0, 110.0])
        sma200 = np.array([100.0, 100.0, 100.0])
        taleb = np.array([6.0, 6.0, 6.0])  # Above risk_cap
        filt = apply_regime_filter(signals, close, sma200, taleb, risk_cap=5.0)
        assert (filt == 0).all()

    def test_preserves_hold_signals(self):
        signals = np.array([0, 0, 0])
        close = np.array([90.0, 110.0, 100.0])
        sma200 = np.array([100.0, 100.0, 100.0])
        taleb = np.array([1.0, 1.0, 1.0])
        filt = apply_regime_filter(signals, close, sma200, taleb, risk_cap=5.0)
        assert (filt == 0).all()

    def test_regime_filter_modes(self):
        sig = np.array([1, -1, 1, -1])
        close = np.array([90.0, 110.0, 110.0, 90.0])
        sma = np.array([100.0, 100.0, 100.0, 100.0])
        taleb = np.array([0.0, 0.0, 9.0, 9.0])
        cap = 5.0
        # default (both): i0 BUY below SMA blocked, i1 SELL above SMA blocked,
        # i2/i3 taleb-capped
        assert apply_regime_filter(sig, close, sma, taleb, cap).tolist() == [0, 0, 0, 0]
        # explicit mode="both" identical to the default
        assert apply_regime_filter(sig, close, sma, taleb, cap, mode="both").tolist() == [0, 0, 0, 0]
        # off: untouched copy
        out = apply_regime_filter(sig, close, sma, taleb, cap, mode="off")
        assert out.tolist() == [1, -1, 1, -1]
        out[0] = 0
        assert sig[0] == 1  # copy, not a view
        # sma_only: taleb ignored -> i2 BUY above SMA survives, i3 SELL below SMA survives
        assert apply_regime_filter(sig, close, sma, taleb, cap, mode="sma_only").tolist() == [0, 0, 1, -1]
        # taleb_only: trend ignored -> i0/i1 survive, i2/i3 capped
        assert apply_regime_filter(sig, close, sma, taleb, cap, mode="taleb_only").tolist() == [1, -1, 0, 0]
        # unknown mode falls back to both
        assert apply_regime_filter(sig, close, sma, taleb, cap, mode="bogus").tolist() == [0, 0, 0, 0]
