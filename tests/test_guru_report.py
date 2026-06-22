"""Tests for guru_report.py's fundamentals-resolution helper."""

import guru_report as gr


def test_resolve_fundamentals_prefers_smartlab():
    smartlab = {"SBER": {"pe": 4.1, "roe": 0.23, "debt": 0.0, "div": 12.0}}
    fund = gr.resolve_fundamentals("SBER", "SBER", smartlab)
    assert fund["_source"] == "smartlab"
    assert fund["pe"] == 4.1
    assert fund["roe"] == 0.23


def test_resolve_fundamentals_yfinance_for_non_moex(monkeypatch):
    monkeypatch.setattr(gr, "fetch_yf_deep",
                         lambda symbol: {"_source": "yfinance_live", "pe": 20.0})
    fund = gr.resolve_fundamentals("TSLA", "TSLA", {})
    assert fund == {"_source": "yfinance_live", "pe": 20.0}


def test_resolve_fundamentals_backup_fallback_non_moex(monkeypatch):
    monkeypatch.setattr(gr, "fetch_yf_deep", lambda symbol: None)
    fund = gr.resolve_fundamentals("TSLA", "TSLA", {})
    assert fund is None  # TSLA has no GLOBAL_BACKUP entry


def test_resolve_fundamentals_moex_backup():
    fund = gr.resolve_fundamentals("SBER", "SBER", {})  # smartlab empty, MOEX backup branch
    assert fund["_source"] == "backup"
    assert fund["pe"] == 4.2  # GLOBAL_BACKUP['SBER']['pe']


def test_resolve_fundamentals_moex_no_backup_returns_none():
    fund = gr.resolve_fundamentals("IMOEX", "IMOEX", {})  # MOEX asset, no GLOBAL_BACKUP entry
    assert fund is None
