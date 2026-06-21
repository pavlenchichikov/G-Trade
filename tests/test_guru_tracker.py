"""Tests for guru_tracker.py - the 60-day value-horizon track record."""

import sqlite3

import pandas as pd
import pytest

import guru_tracker as gt


@pytest.fixture
def gt_db(tmp_path, monkeypatch):
    """Point guru_tracker at an isolated DB (never the real market.db)."""
    path = str(tmp_path / "market.db")
    monkeypatch.setattr(gt, "DB_PATH", path)
    monkeypatch.setattr(gt, "_ENGINE", None)  # reset the cached sqlalchemy engine
    return path


def test_60d_return_and_correctness_computed(gt_db):
    # 70 business days of strictly rising closes for AAPL.
    dates = pd.bdate_range("2024-01-01", periods=70).strftime("%Y-%m-%d")
    con = sqlite3.connect(gt_db)
    pd.DataFrame({"Date": dates, "close": [100 + i for i in range(70)]}).to_sql(
        "aapl", con, index=False)
    con.close()

    gt.log_guru_verdict("AAPL", 2, 1, 2, 1, 80.0, "BUY", "yfinance_live", 100.0)
    # pin the verdict's date to day 0 so 60 forward bars exist
    con = sqlite3.connect(gt_db)
    con.execute("UPDATE guru_log SET date=?", (dates[0],))
    con.commit()
    con.close()

    gt.update_actuals()

    con = sqlite3.connect(gt_db)
    ret_60d, correct_60d = con.execute(
        "SELECT ret_60d, correct_60d FROM guru_log").fetchone()
    con.close()
    assert ret_60d is not None and ret_60d > 0   # price rose over 60 days
    assert correct_60d == 1                       # BUY was correct


def test_accuracy_horizon_60d(gt_db):
    dates = pd.bdate_range("2024-01-01", periods=70).strftime("%Y-%m-%d")
    con = sqlite3.connect(gt_db)
    pd.DataFrame({"Date": dates, "close": [100 + i for i in range(70)]}).to_sql(
        "aapl", con, index=False)
    con.close()
    gt.log_guru_verdict("AAPL", 2, 1, 2, 1, 80.0, "BUY", "yfinance_live", 100.0)
    con = sqlite3.connect(gt_db)
    con.execute("UPDATE guru_log SET date=?", (dates[0],))
    con.commit()
    con.close()
    gt.update_actuals()

    acc = gt.get_guru_accuracy(days=100000, horizon="60d")
    assert acc["horizon"] == "60d"
    assert acc["total"] == 1 and acc["correct"] == 1


def test_migration_adds_60d_columns(gt_db):
    # Old-schema guru_log predating the 60-day horizon.
    con = sqlite3.connect(gt_db)
    con.execute("CREATE TABLE guru_log "
                "(date TEXT, asset TEXT, council_verdict TEXT, ret_20d REAL)")
    con.commit()
    cur = con.cursor()
    gt._ensure_table(cur)  # must ALTER ADD the missing 60d columns
    con.commit()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(guru_log)")}
    con.close()
    assert "ret_60d" in cols and "correct_60d" in cols
