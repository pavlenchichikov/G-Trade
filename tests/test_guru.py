"""Tests for core.guru — Guru Council fundamental analysis."""

import pytest

from core.guru import (
    calc_peg,
    calc_graham_number,
    technical_context,
    lynch_analysis,
    buffett_analysis,
    graham_analysis,
    munger_analysis,
    get_guru_analysis,
)


class TestCalcPeg:
    def test_normal(self):
        assert calc_peg(20, 0.20) == pytest.approx(1.0)

    def test_high_growth(self):
        peg = calc_peg(30, 0.50)
        assert peg == pytest.approx(0.6)

    def test_zero_pe(self):
        assert calc_peg(0, 0.20) is None

    def test_zero_growth(self):
        assert calc_peg(20, 0) is None

    def test_negative(self):
        assert calc_peg(-5, 0.10) is None


class TestCalcGrahamNumber:
    def test_normal(self):
        gn = calc_graham_number(5.0, 30.0)
        assert gn is not None
        assert gn > 0

    def test_zero_eps(self):
        assert calc_graham_number(0, 30) is None

    def test_negative_bv(self):
        assert calc_graham_number(5, -10) is None


class TestLynchAnalysis:
    def test_undervalued(self):
        fund = {'pe': 10, 'growth': 0.20, 'peg_ratio': 0.5}
        status, desc, score = lynch_analysis(fund, None)
        assert "BUY" in status
        assert score == 2

    def test_overvalued(self):
        fund = {'pe': 50, 'growth': 0.05, 'peg_ratio': 10.0}
        status, desc, score = lynch_analysis(fund, None)
        assert "EXP" in status
        assert score == 0

    def test_technical_fallback(self):
        tech = {'above_50': True, 'above_200': True, 'rsi': 55,
                'pct_52w': 60, 'vol_30d': 0.2, 'macd_bull': True, 'sma_rising': True}
        status, desc, score = lynch_analysis(None, tech)
        assert "MOMENTUM" in status
        assert score == 2

    def test_no_data(self):
        status, desc, score = lynch_analysis(None, None)
        assert "N/A" in status
        assert score == 0


class TestBuffettAnalysis:
    def test_high_quality(self):
        fund = {'roe': 0.25, 'debt_equity': 0.2, 'profit_margin': 0.30,
                'gross_margin': 0.50, 'dividend_yield': 5, 'fcf': 1e9,
                'fcf_positive_streak': 4, 'price_to_fcf': 15, 'retained_earnings': 1e9}
        status, desc, score = buffett_analysis(fund, None)
        assert "GEM" in status or "QUALITY" in status
        assert score >= 1

    def test_weak(self):
        fund = {'roe': 0.02, 'debt_equity': 3.0, 'profit_margin': -0.05,
                'gross_margin': 0.10, 'dividend_yield': 0, 'fcf': -1e6}
        status, desc, score = buffett_analysis(fund, None)
        assert "WEAK" in status
        assert score == 0


class TestGrahamAnalysis:
    def test_cheap_stock(self):
        fund = {'pe': 8, 'eps': 5.0, 'book_value': 50.0, 'price': 30.0,
                'debt_equity': 0.3, 'current_ratio': 2.5, 'ncav_per_share': 35}
        status, desc, score = graham_analysis(fund, None)
        assert score >= 2

    def test_expensive_stock(self):
        fund = {'pe': 80, 'eps': 1.0, 'book_value': 5.0, 'price': 200.0,
                'debt_equity': 2.0, 'current_ratio': 0.8}
        status, desc, score = graham_analysis(fund, None)
        assert score <= 1


class TestMungerAnalysis:
    def test_clean(self):
        fund = {'pe': 15, 'debt_equity': 0.3, 'roe': 0.20, 'profit_margin': 0.15}
        tech = {'rsi': 55, 'vol_30d': 0.20, 'pct_52w': 50}
        status, desc, score = munger_analysis(fund, tech)
        assert "CLEAN" in status or "MINOR" in status

    def test_danger(self):
        fund = {'pe': 100, 'debt_equity': 5.0, 'roe': -0.30, 'profit_margin': -0.20,
                'revenue_qoq': -0.25, 'fcf': -1e9, 'fcf_positive_streak': 0}
        tech = {'rsi': 85, 'vol_30d': 0.80, 'pct_52w': 95}
        status, desc, score = munger_analysis(fund, tech)
        assert "DANGER" in status or "WARNING" in status


class TestGetGuruAnalysis:
    def test_returns_all_gurus(self):
        fund = {'pe': 15, 'growth': 0.15, 'roe': 0.18, 'debt_equity': 0.5,
                'profit_margin': 0.12, 'fcf': 1e8, 'book_value': 30, 'eps': 3,
                'price': 50, 'source': 'test'}
        tech = {'close': 50, 'rsi': 55, 'above_50': True, 'above_200': True,
                'pct_52w': 60, 'vol_30d': 0.2, 'macd_bull': True, 'sma_rising': True}
        result = get_guru_analysis(fund, tech)
        assert 'lynch' in result
        assert 'buffett' in result
        assert 'graham' in result
        assert 'munger' in result
        assert 'council' in result
        assert result['council']['verdict'] in ('BUY', 'HOLD', 'AVOID')

    def test_no_data(self):
        result = get_guru_analysis(None, None)
        assert result['council']['verdict'] == 'AVOID'
