"""update_actuals must reconcile logged predictions against the next-day close.

Regression test for the silent KeyError('Close') bug: price tables store columns
lowercase (data_engine lowercases them), so df["Close"] raised and every row was
skipped, leaving accuracy "0 verified" forever.
"""
import sqlite3
from datetime import datetime

import pytest


def _seed(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    # real data_engine schema: capital "Date" index column, lowercase OHLC
    cur.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    cur.executemany("INSERT INTO btc VALUES (?,?,?,?,?,?)", [
        ("2026-06-12", 1, 1, 1, 100.0, 1),
        ("2026-06-13", 1, 1, 1, 110.0, 1),   # +10% next day
    ])
    cur.execute(
        "CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, probability REAL, "
        "actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL)"
    )
    cur.execute("INSERT INTO prediction_log VALUES ('2026-06-12','BTC','BUY',0.7,NULL,NULL,NULL,NULL)")
    con.commit()
    con.close()


def test_update_actuals_reconciles_against_next_close(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "t.db")
    _seed(db)
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)  # rebuild engine against temp DB

    pt.update_actuals()

    con = sqlite3.connect(db)
    ret, correct = con.execute(
        "SELECT actual_next_ret, correct FROM prediction_log"
    ).fetchone()
    con.close()
    assert ret is not None and abs(ret - 0.10) < 1e-9   # 100 to 110
    assert correct == 1                                  # BUY and price rose


def test_log_prediction_tags_current_version(tmp_path, monkeypatch):
    import performance_tracker as pt
    from core.features import feature_version
    db = str(tmp_path / "m.db")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, close REAL)")
    con.execute("INSERT INTO btc VALUES (?, 100.0)", (today,))   # real bar for today
    con.commit()
    con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt.log_prediction("BTC", "BUY", 0.7)
    con = sqlite3.connect(db)
    mv = con.execute("SELECT model_version FROM prediction_log").fetchone()[0]
    con.close()
    assert mv == feature_version()


def test_log_prediction_skips_when_no_bar_for_that_day(tmp_path, monkeypatch):
    """No price bar for the prediction date (weekend/holiday/not-fetched yet) means the
    signal can never be reconciled, so the row must NOT be logged - otherwise it sits
    'awaiting check' forever and inflates the pending count."""
    import performance_tracker as pt
    db = str(tmp_path / "nb.db")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE aapl (Date TEXT, close REAL)")
    con.execute("INSERT INTO aapl VALUES ('2020-01-02', 100.0)")   # an old bar, not today
    con.commit()
    con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)

    pt.log_prediction("AAPL", "BUY", 0.7)                          # no bar for today -> skip
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] == 0
    con.execute("INSERT INTO aapl VALUES (?, 101.0)", (today,))    # today's bar arrives
    con.commit()
    con.close()

    pt.log_prediction("AAPL", "BUY", 0.7)                          # now it logs
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] == 1
    con.close()


def test_migrate_adds_column_and_keeps_old_rows_legacy(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, "
                "probability REAL, actual_next_ret REAL, correct INTEGER, "
                "cb_prob REAL, lstm_prob REAL)")
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-01','BTC','BUY',0.7,0.01,1,0.7,NULL)")
    con.commit(); con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt._prepare()
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(prediction_log)")]
    legacy = con.execute("SELECT model_version FROM prediction_log").fetchone()[0]
    con.close()
    assert "model_version" in cols
    assert legacy is None   # pre-migration row stays a distinct legacy generation


def test_get_accuracy_scopes_by_version(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "scoped.db")
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt._prepare()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    con.executemany(
        "INSERT INTO prediction_log (date,asset,signal,probability,actual_next_ret,"
        "correct,cb_prob,lstm_prob,model_version) VALUES (?,?,?,?,?,?,?,?,?)",
        [(today, "BTC", "BUY", 0.6, 0.01, 1, None, None, "oldv"),
         (today, "ETH", "BUY", 0.6, 0.01, 1, None, None, "oldv"),
         (today, "SOL", "BUY", 0.6, -0.01, 0, None, None, "newv")],
    )
    con.commit(); con.close()
    assert pt.get_accuracy(days=30)["accuracy"] == pytest.approx(2 / 3)   # all
    assert pt.get_accuracy(days=30, model_version="newv")["accuracy"] == 0.0
    assert pt.get_accuracy(days=30, model_version="oldv")["accuracy"] == 1.0


def test_update_actuals_returns_counts(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, Close REAL)")
    con.execute("INSERT INTO btc VALUES ('2026-06-10', 100.0)")
    con.execute("INSERT INTO btc VALUES ('2026-06-11', 110.0)")
    con.execute("""CREATE TABLE prediction_log (
        date TEXT, asset TEXT, signal TEXT, probability REAL,
        actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL,
        model_version TEXT)""")
    # 2026-06-10 has a next bar (reconcilable); 2026-06-11 does not (stays pending)
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-10','BTC','BUY',0.6,NULL,NULL,NULL,NULL,'v1')")
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-11','BTC','BUY',0.6,NULL,NULL,NULL,NULL,'v1')")
    con.commit()
    con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)   # reset the cached engine
    res = pt.update_actuals()
    assert res == {"pending": 2, "reconciled": 1, "excluded": 0}


def test_update_actuals_excludes_prediction_on_closed_market_day(tmp_path, monkeypatch):
    """A prediction dated on a day the asset did not trade (no bar) while a later bar
    already exists is a stale duplicate of the prior close - it cannot be objectively
    verified, so it is dropped, not mapped onto a neighbouring day and counted."""
    import performance_tracker as pt
    db = str(tmp_path / "wk.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    con.executemany("INSERT INTO btc VALUES (?,?,?,?,?,?)", [
        ("2026-06-12", 1, 1, 1, 100.0, 1),   # Friday
        ("2026-06-15", 1, 1, 1, 110.0, 1),   # Monday; 06-13/06-14 are a closed market, no bars
    ])
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, probability REAL, "
                "actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL, model_version TEXT)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-06-13','BTC','BUY',0.7,NULL,NULL,NULL,NULL,'v1')")
    con.commit(); con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    res = pt.update_actuals()
    assert res == {"pending": 1, "reconciled": 0, "excluded": 1}
    con = sqlite3.connect(db)
    remaining = con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
    con.close()
    assert remaining == 0   # the closed-market prediction is gone, not counted


def test_update_actuals_scores_real_weekend_bar_for_crypto(tmp_path, monkeypatch):
    """When the asset genuinely trades that day (crypto has a real Saturday bar) the
    prediction is a real trading day and must be scored, not dropped."""
    import performance_tracker as pt
    db = str(tmp_path / "cr.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    con.executemany("INSERT INTO btc VALUES (?,?,?,?,?,?)", [
        ("2026-06-13", 1, 1, 1, 100.0, 1),   # Saturday - a real crypto bar
        ("2026-06-14", 1, 1, 1, 110.0, 1),   # Sunday (+10%)
    ])
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, probability REAL, "
                "actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL, model_version TEXT)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-06-13','BTC','BUY',0.7,NULL,NULL,NULL,NULL,'v1')")
    con.commit(); con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    res = pt.update_actuals()
    assert res == {"pending": 1, "reconciled": 1, "excluded": 0}
    con = sqlite3.connect(db)
    ret, correct = con.execute("SELECT actual_next_ret, correct FROM prediction_log").fetchone()
    con.close()
    assert abs(ret - 0.10) < 1e-9 and correct == 1


def test_update_actuals_keeps_today_prediction_before_bar_lands(tmp_path, monkeypatch):
    """A prediction dated after the latest bar (today's, before its close is fetched)
    is a real trading day whose outcome has not formed - it must stay pending, never
    be deleted as a non-trading day."""
    import performance_tracker as pt
    db = str(tmp_path / "td.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    con.execute("INSERT INTO btc VALUES ('2026-06-12', 1, 1, 1, 100.0, 1)")   # only bar so far
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, probability REAL, "
                "actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL, model_version TEXT)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-06-15','BTC','BUY',0.7,NULL,NULL,NULL,NULL,'v1')")
    con.commit(); con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    res = pt.update_actuals()
    assert res == {"pending": 1, "reconciled": 0, "excluded": 0}
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM prediction_log WHERE actual_next_ret IS NULL").fetchone()[0]
    con.close()
    assert n == 1   # still pending, not deleted


def test_log_prediction_stamps_bar_date(tmp_path, monkeypatch):
    """log_prediction dates the row by the passed market-bar date, not wall clock,
    so a weekend run for a closed market folds into the real trading-day row."""
    import performance_tracker as pt
    db = str(tmp_path / "bd.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, close REAL)")
    con.execute("INSERT INTO btc VALUES ('2026-06-12', 100.0)")   # real bar for that date
    con.commit()
    con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt.log_prediction("BTC", "BUY", 0.7, date="2026-06-12")
    # a later same-asset call stamped with the same bar date must dedup, not add a row
    pt.log_prediction("BTC", "SELL", 0.3, date="2026-06-12")
    con = sqlite3.connect(db)
    rows = con.execute("SELECT date, signal FROM prediction_log").fetchall()
    con.close()
    assert rows == [("2026-06-12", "BUY")]


def test_meta_shadow_report_scores_the_gate(tmp_path, monkeypatch):
    """meta_shadow_report: over reconciled directional predictions that carry a
    meta_prob, keeping only meta_prob>=thr should surface the accuracy lift."""
    import performance_tracker as pt
    db = str(tmp_path / "shadow.db")
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt._prepare()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    con.executemany(
        "INSERT INTO prediction_log (date,asset,signal,probability,actual_next_ret,"
        "correct,cb_prob,lstm_prob,model_version,meta_prob) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(today, "AAA", "BUY", 0.6, 0.01, 1, None, None, "v1", 0.70),
         (today, "BBB", "BUY", 0.6, 0.01, 1, None, None, "v1", 0.60),
         (today, "CCC", "SELL", 0.4, 0.01, 0, None, None, "v1", 0.30),
         (today, "DDD", "SELL", 0.4, 0.01, 0, None, None, "v1", 0.40),
         # a WAIT and a not-yet-reconciled row must be ignored
         (today, "EEE", "WAIT", 0.5, 0.0, None, None, None, "v1", 0.9),
         (today, "FFF", "BUY", 0.6, None, None, None, None, "v1", 0.9)],
    )
    con.commit(); con.close()
    rep = pt.meta_shadow_report(days=30)
    assert rep["rows"] == 4
    assert rep["baseline_accuracy"] == 0.5
    at_50 = next(s for s in rep["sweep"] if s["thr"] == 0.50)
    assert at_50["kept"] == 2 and at_50["accuracy"] == 1.0 and at_50["coverage"] == 0.5
    assert rep["discrimination"]["high_meta_acc"] == 1.0
    assert rep["discrimination"]["low_meta_acc"] == 0.0


def test_meta_shadow_report_empty_when_no_shadow_data(tmp_path, monkeypatch):
    import performance_tracker as pt
    db = str(tmp_path / "empty.db")
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    assert pt.meta_shadow_report(days=30) == {"rows": 0}


def test_log_prediction_records_meta_prob(tmp_path, monkeypatch):
    import sqlite3
    import performance_tracker as pt
    db = str(tmp_path / "mp.db")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE btc (Date TEXT, close REAL)")
    con.execute("INSERT INTO btc VALUES (?, 100.0)", (today,))   # real bar for today
    con.commit()
    con.close()
    monkeypatch.setattr(pt, "DB_PATH", db)
    monkeypatch.setattr(pt, "_ENGINE", None)
    pt.log_prediction("BTC", "BUY", 0.7, meta_prob=0.42)
    con = sqlite3.connect(db)
    val = con.execute("SELECT meta_prob FROM prediction_log").fetchone()[0]
    con.close()
    assert abs(val - 0.42) < 1e-9
