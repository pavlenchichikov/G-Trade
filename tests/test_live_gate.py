# tests/test_live_gate.py
"""Live-accuracy gate rules (core/live_gate.py). Every rule fires at its
boundary and never on insufficient n; WAIT passes through; the master switch
and a missing table disable everything silently."""

import sqlite3

import pytest

from core import live_gate


def _make_db(tmp_path, rows):
    """rows: list of (asset, signal, correct) - dated today so the 45d window
    always includes them."""
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, correct INTEGER)")
    con.executemany(
        "INSERT INTO prediction_log VALUES (date('now'), ?, ?, 0.6, ?)", rows)
    con.commit()
    con.close()
    return db


def _forex_majors_assets():
    import config
    for group, assets in config.ASSET_TYPES.items():
        if "FOREX" in group and "MAJOR" in group.upper():
            return group, assets
    pytest.skip("no forex majors group in config")


def test_class_gate_fires_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("GTRADE_LIVE_GATE_MIN_N", "10")
    group, assets = _forex_majors_assets()
    # 10 verified rows across the class at 30% accuracy
    rows = [(assets[0], "BUY", 1 if i < 3 else 0) for i in range(10)]
    db = _make_db(tmp_path, rows)
    sig, reason = live_gate.gate(assets[0], 0.6, "BUY", db_path=db)
    assert sig == "WAIT"
    assert reason is not None and group in reason and "n=10" in reason


def test_class_gate_needs_min_n(tmp_path, monkeypatch):
    monkeypatch.setenv("GTRADE_LIVE_GATE_MIN_N", "100")
    monkeypatch.setenv("GTRADE_LIVE_GATE_ASSET_N", "100")
    group, assets = _forex_majors_assets()
    rows = [(assets[0], "BUY", 0) for _ in range(10)]  # awful but tiny n
    db = _make_db(tmp_path, rows)
    sig, reason = live_gate.gate(assets[0], 0.6, "BUY", db_path=db)
    assert sig == "BUY" and reason is None


def test_asset_gate_fires(tmp_path, monkeypatch):
    monkeypatch.setenv("GTRADE_LIVE_GATE_MIN_N", "1000")  # class rule off
    group, assets = _forex_majors_assets()
    rows = [(assets[0], "SELL", 1 if i < 7 else 0) for i in range(20)]  # 35%
    db = _make_db(tmp_path, rows)
    sig, reason = live_gate.gate(assets[0], 0.4, "SELL", db_path=db)
    assert sig == "WAIT"
    assert assets[0] in reason


def test_tail_gate_fires_without_any_stats(tmp_path):
    db = _make_db(tmp_path, [])
    assert live_gate.gate("AAPL", 0.9, "BUY", db_path=db)[0] == "WAIT"
    assert live_gate.gate("AAPL", 0.1, "SELL", db_path=db)[0] == "WAIT"
    assert live_gate.gate("AAPL", 0.6, "BUY", db_path=db) == ("BUY", None)


def test_wait_passes_through(tmp_path):
    db = _make_db(tmp_path, [])
    assert live_gate.gate("AAPL", 0.95, "WAIT", db_path=db) == ("WAIT", None)


def test_master_switch_disables(tmp_path, monkeypatch):
    monkeypatch.setenv("GTRADE_LIVE_GATE", "0")
    db = _make_db(tmp_path, [("EURUSD", "BUY", 0)] * 200)
    assert live_gate.gate("EURUSD", 0.99, "BUY", db_path=db) == ("BUY", None)


def test_missing_table_is_noop(tmp_path):
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()
    # only the tail rule (needs no stats) may fire; a mid prob passes
    assert live_gate.gate("AAPL", 0.6, "BUY", db_path=db) == ("BUY", None)


def test_meta_groups_do_not_own_assets():
    """Assets listed in a curated meta-group (TOP SIGNALS) must attribute to
    their REAL sector class, and the meta-group itself never gates anyone."""
    import config
    meta = set()
    for g in live_gate.META_GROUPS:
        meta |= set(config.ASSET_TYPES.get(g, []))
    if not meta:
        import pytest as _pytest
        _pytest.skip("no meta-group assets in config")
    for asset in meta:
        assert live_gate._asset_class(asset) not in live_gate.META_GROUPS
