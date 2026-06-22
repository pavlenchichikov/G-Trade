"""Tests for core.guru - Guru Council fundamental analysis."""

import pytest

import core.guru as guru
from core.guru import (
    calc_peg,
    calc_graham_number,
    lynch_analysis,
    buffett_analysis,
    graham_analysis,
    munger_analysis,
    get_guru_analysis,
    council_weights_from_log,
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

    def test_data_source_uses_underscore_source_key(self):
        """Regression: the fund dict key is '_source', not 'source'. A real stock
        must report its true source, not be mislabeled 'technical'."""
        fund = {'pe': 20, 'roe': 0.25, 'eps': 5, 'book_value': 40,
                'sector': 'Technology', 'price': 100, '_source': 'yfinance_live'}
        res = get_guru_analysis(fund, None)
        assert res['data_source'] == 'yfinance_live'

    def test_priced_but_no_fundamentals_is_technical(self):
        """yfinance returns a priced .info for forex/crypto too - with no real
        fundamentals it must count as technical (shows N/A in the UI)."""
        fund = {'price': 1.14, 'pe': 0, 'roe': 0, 'eps': 0, 'book_value': 0,
                'sector': '', '_source': 'yfinance_live'}
        res = get_guru_analysis(fund, None)
        assert res['data_source'] == 'technical'


class TestMungerVeto:
    def test_danger_blocks_buy(self, monkeypatch):
        """3 bullish gurus (6/8 = BUY) must be vetoed to HOLD by a Munger DANGER."""
        monkeypatch.setattr(guru, "lynch_analysis", lambda f, t: ("[OK] BUY", "", 2))
        monkeypatch.setattr(guru, "buffett_analysis", lambda f, t, **k: ("[TOP] GEM", "", 2))
        monkeypatch.setattr(guru, "graham_analysis", lambda f, t, **k: ("[OK] BUY", "", 2))
        monkeypatch.setattr(guru, "munger_analysis", lambda f, t, **k: ("[!!] DANGER", "", 0))
        res = get_guru_analysis({'source': 'x'}, {})
        assert res['council']['verdict'] != 'BUY'
        assert res['council']['vetoed'] is True

    def test_clean_munger_does_not_veto(self, monkeypatch):
        monkeypatch.setattr(guru, "lynch_analysis", lambda f, t: ("[OK] BUY", "", 2))
        monkeypatch.setattr(guru, "buffett_analysis", lambda f, t, **k: ("[TOP] GEM", "", 2))
        monkeypatch.setattr(guru, "graham_analysis", lambda f, t, **k: ("[OK] BUY", "", 2))
        monkeypatch.setattr(guru, "munger_analysis", lambda f, t, **k: ("[OK] CLEAN", "", 2))
        res = get_guru_analysis({'source': 'x'}, {})
        assert res['council']['verdict'] == 'BUY'
        assert res['council']['vetoed'] is False


class TestSectorAware:
    def test_financial_relaxes_buffett_debt(self):
        """High debt should not sink a bank under Buffett once sector is known."""
        fund = {'roe': 0.16, 'debt_equity': 5.0, 'profit_margin': 0.10, 'fcf': 1e8}
        base = buffett_analysis(fund, None)[2]
        fin = buffett_analysis(fund, None, sector="US FINANCE")[2]
        assert fin > base

    def test_financial_relaxes_munger_leverage(self):
        """A bank's high debt + high liabilities must not read as Munger DANGER."""
        fund = {'pe': 12, 'debt_equity': 6.0, 'roe': 0.14, 'profit_margin': 0.20,
                'total_assets': 100, 'total_liabilities': 92}
        generic = munger_analysis(fund, None)[2]
        fin = munger_analysis(fund, None, sector="RUS FINANCE")[2]
        assert fin >= generic


class TestTechnicalDiscount:
    def test_technical_only_flagged_and_shaved(self):
        tech = {'above_50': True, 'above_200': True, 'rsi': 55, 'pct_52w': 60,
                'vol_30d': 0.2, 'macd_bull': True, 'sma_rising': True}
        res = get_guru_analysis(None, tech)
        assert res['data_source'] == 'technical'
        assert 'tech-only' in res['council']['text']


class TestCouncilWeights:
    def test_fallback_equal_when_no_db(self, tmp_path):
        w = council_weights_from_log(str(tmp_path / "missing.db"))
        assert w == {'lynch': 1.0, 'buffett': 1.0, 'graham': 1.0, 'munger': 1.0}

    def test_weights_keep_council_scale(self, tmp_path):
        """Calibrated weights normalise to sum 4 (mean 1), preserving the 0-8 scale."""
        import sqlite3
        db = str(tmp_path / "market.db")
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE guru_log (lynch_score INT, buffett_score INT,
                       graham_score INT, munger_score INT, ret_5d REAL)""")
        # buffett's '2' precedes strong returns; others flat, so buffett gets weight.
        rows = []
        for i in range(60):
            rows.append((2 if i % 2 else 0, 2 if i % 2 else 0,
                         1, 1, 0.05 if i % 2 else -0.01))
        con.executemany("INSERT INTO guru_log VALUES (?,?,?,?,?)", rows)
        con.commit(); con.close()
        w = council_weights_from_log(db)
        assert abs(sum(w.values()) - 4.0) < 1e-6
        assert all(v >= 0 for v in w.values())
