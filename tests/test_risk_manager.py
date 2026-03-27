"""Tests for risk_manager.py — Kelly sizing, circuit breakers, signal gating."""

import os
import sys
import json
import tempfile

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_manager import RiskManager, RISK_CONFIG


@pytest.fixture
def rm():
    """Fresh RiskManager with no persisted state."""
    # Temporarily override state path to avoid loading real state
    import risk_manager as rm_mod
    original_path = rm_mod.RISK_STATE_PATH
    rm_mod.RISK_STATE_PATH = os.path.join(tempfile.mkdtemp(), "test_risk_state.json")
    manager = RiskManager(initial_capital=10_000)
    yield manager
    rm_mod.RISK_STATE_PATH = original_path


class TestKellyFraction:
    def test_positive_edge(self, rm):
        size = rm.kelly_fraction(win_rate=0.60, avg_win=0.025, avg_loss=0.012)
        assert size > 0
        assert size <= RISK_CONFIG["max_single_position"]

    def test_no_edge_returns_zero(self, rm):
        # win_rate=0.3 with equal win/loss → negative Kelly
        size = rm.kelly_fraction(win_rate=0.30, avg_win=0.01, avg_loss=0.01)
        assert size == 0.0

    def test_high_taleb_reduces_size(self, rm):
        size_low = rm.kelly_fraction(win_rate=0.60, taleb_risk=1.0)
        size_high = rm.kelly_fraction(win_rate=0.60, taleb_risk=4.0)
        assert size_low > size_high

    def test_correlation_reduces_size(self, rm):
        size_0 = rm.kelly_fraction(win_rate=0.60, n_correlated=0)
        size_3 = rm.kelly_fraction(win_rate=0.60, n_correlated=3)
        assert size_0 > size_3

    def test_respects_max_single_position(self, rm):
        size = rm.kelly_fraction(win_rate=0.90, avg_win=0.10, avg_loss=0.001)
        assert size <= RISK_CONFIG["max_single_position"]

    def test_below_min_threshold_returns_zero(self, rm):
        # Very small edge → Kelly below min threshold
        size = rm.kelly_fraction(win_rate=0.505, avg_win=0.005, avg_loss=0.005)
        assert size == 0.0

    def test_invalid_inputs(self, rm):
        assert rm.kelly_fraction(win_rate=0.0) == 0.0
        assert rm.kelly_fraction(win_rate=1.0) == 0.0
        assert rm.kelly_fraction(win_rate=-0.1) == 0.0


class TestCircuitBreakers:
    def test_no_halt_on_fresh_state(self, rm):
        halted, reason = rm.is_trading_halted()
        assert halted is False
        assert reason == ""

    def test_halt_on_max_drawdown(self, rm):
        rm.peak_capital = 10_000
        rm.current_capital = 8_000  # -20% drawdown
        halted, reason = rm.is_trading_halted()
        assert halted is True
        assert "DRAWDOWN" in reason

    def test_halt_on_daily_loss(self, rm):
        rm.daily_start_capital = 10_000
        rm.current_capital = 9_400  # -6% daily loss
        halted, reason = rm.is_trading_halted()
        assert halted is True
        assert "DAILY" in reason

    def test_no_halt_within_limits(self, rm):
        rm.peak_capital = 10_000
        rm.current_capital = 9_600  # -4% drawdown (< 15% halt, < 5% daily)
        rm.daily_start_capital = 10_000
        halted, _ = rm.is_trading_halted()
        assert halted is False


class TestCheckSignal:
    def test_wait_always_approved(self, rm):
        result = rm.check_signal("BTC", "WAIT", confidence=0.5)
        assert result["approved"] is True

    def test_buy_with_good_confidence(self, rm):
        result = rm.check_signal("BTC", "BUY", confidence=0.65)
        assert result["approved"] is True
        assert result["position_size_usd"] > 0

    def test_buy_blocked_by_high_taleb(self, rm):
        result = rm.check_signal("BTC", "BUY", confidence=0.65, taleb_risk=6.0)
        assert result["approved"] is False
        assert "TAIL RISK" in result["reason"]

    def test_buy_blocked_by_drawdown_halt(self, rm):
        rm.peak_capital = 10_000
        rm.current_capital = 8_000
        result = rm.check_signal("BTC", "BUY", confidence=0.65)
        assert result["approved"] is False
        assert "HALTED" in result["reason"]

    def test_sell_signal_processed(self, rm):
        result = rm.check_signal("BTC", "SELL", confidence=0.35)
        # SELL with confidence 0.35 → effective win_rate = 1-0.35 = 0.65
        assert result["approved"] is True


class TestProperties:
    def test_current_drawdown(self, rm):
        rm.peak_capital = 10_000
        rm.current_capital = 9_000
        assert abs(rm.current_drawdown - 0.10) < 0.001

    def test_daily_pnl(self, rm):
        rm.daily_start_capital = 10_000
        rm.current_capital = 10_500
        assert abs(rm.daily_pnl - 0.05) < 0.001

    def test_total_exposure_empty(self, rm):
        assert rm.total_exposure == 0.0

    def test_total_exposure_with_positions(self, rm):
        rm.open_positions = {
            "BTC": {"size_usd": 1000},
            "ETH": {"size_usd": 500},
        }
        assert abs(rm.total_exposure - 0.15) < 0.001

    def test_remaining_capacity(self, rm):
        rm.open_positions = {"BTC": {"size_usd": 2000}}  # 20%
        remaining = rm.remaining_capacity
        assert abs(remaining - 0.10) < 0.001  # 30% - 20% = 10%


class TestPersistence:
    def test_save_and_load(self, rm, tmp_path):
        import risk_manager as rm_mod
        rm_mod.RISK_STATE_PATH = str(tmp_path / "risk_state.json")

        rm.current_capital = 9500
        rm.peak_capital = 10000
        rm.open_positions = {"BTC": {"size_usd": 500}}
        rm.save_state()

        rm2 = RiskManager(initial_capital=10000)
        assert rm2.current_capital == 9500
        assert rm2.peak_capital == 10000
        assert "BTC" in rm2.open_positions
