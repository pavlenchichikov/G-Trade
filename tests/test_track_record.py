"""Tests for core.track_record against a temporary SQLite DB."""

import sqlite3

import pytest

from core import track_record


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE prediction_log (
            date TEXT, asset TEXT, signal TEXT, probability REAL,
            actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL
        )
    """)
    rows = [
        # BTC: three days of history, latest is BUY
        ("2026-06-08", "BTC", "BUY", 0.61, 0.012, 1, 0.61, None),
        ("2026-06-09", "BTC", "SELL", 0.58, 0.004, 0, 0.58, None),
        ("2026-06-10", "BTC", "BUY", 0.62, None, None, 0.62, None),
        # TSLA: one day
        ("2026-06-10", "TSLA", "WAIT", 0.51, None, None, 0.51, None),
    ]
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?)", rows)
    # price table for stale_assets
    con.execute('CREATE TABLE btc (Date TEXT, Close REAL)')
    con.execute("INSERT INTO btc VALUES ('2026-06-10', 100.0)")
    con.execute('CREATE TABLE tsla (Date TEXT, Close REAL)')
    con.execute("INSERT INTO tsla VALUES ('2026-01-01', 200.0)")
    con.commit()
    con.close()
    return path


def test_latest_signals_returns_last_row_per_asset(db):
    sigs = {s["asset"]: s for s in track_record.latest_signals(db_path=db)}
    assert sigs["BTC"]["date"] == "2026-06-10"
    assert sigs["BTC"]["signal"] == "BUY"
    assert sigs["TSLA"]["signal"] == "WAIT"


def test_latest_signals_empty_table(tmp_path):
    path = str(tmp_path / "empty.db")
    sqlite3.connect(path).close()
    assert track_record.latest_signals(db_path=path) == []


def test_accuracy_scoped_to_current_version_when_column_present(tmp_path):
    """With a model_version column, accuracy must ignore legacy-generation rows."""
    from core.features import feature_version
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, "
        "probability REAL, actual_next_ret REAL, correct INTEGER, cb_prob REAL, "
        "lstm_prob REAL, model_version TEXT)"
    )
    cur_v = feature_version()
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?,?)", [
        ("2026-06-01", "BTC", "BUY", 0.6, 0.01, 1, None, None, "legacy"),
        ("2026-06-02", "BTC", "BUY", 0.6, 0.01, 1, None, None, "legacy"),
        ("2026-06-03", "BTC", "BUY", 0.6, -0.01, 0, None, None, cur_v),
    ])
    con.commit(); con.close()
    acc = track_record.asset_accuracy("BTC", db_path=path)
    assert acc["n"] == 1        # only the current-generation row counts
    assert acc["acc"] == 0.0


def test_asset_accuracy_counts_only_verified(db):
    acc = track_record.asset_accuracy("BTC", db_path=db)
    assert acc["n"] == 2          # the row with correct=NULL isn't counted
    assert acc["correct"] == 1
    assert acc["acc"] == 0.5


def test_asset_accuracy_no_data(db):
    acc = track_record.asset_accuracy("NOPE", db_path=db)
    assert acc["n"] == 0
    assert acc["acc"] is None


def test_asset_track_orders_desc_and_limits(db):
    track = track_record.asset_track("BTC", limit=2, db_path=db)
    assert len(track) == 2
    assert track[0]["date"] == "2026-06-10"
    assert track[1]["date"] == "2026-06-09"


def test_stale_assets_flags_old_data(db):
    stale = track_record.stale_assets(
        max_age_days=7, assets=["BTC", "TSLA"], db_path=db,
        today="2026-06-12",
    )
    names = [s["asset"] for s in stale]
    assert "TSLA" in names
    assert "BTC" not in names


def test_price_series_ascending(db):
    series = track_record.price_series("BTC", days=10, db_path=db)
    assert series == [{"date": "2026-06-10", "close": 100.0}]


def test_price_series_missing_table(db):
    assert track_record.price_series("GOLD", db_path=db) == []


def test_ohlc_series_returns_bars(tmp_path):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE btc (Date TEXT, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL)")
    con.executemany("INSERT INTO btc VALUES (?,?,?,?,?,?)", [
        ("2026-06-10", 100, 110, 95, 105, 1),
        ("2026-06-11", 105, 120, 104, 118, 1),
    ])
    con.commit(); con.close()
    bars = track_record.ohlc_series("BTC", db_path=path)
    assert len(bars) == 2
    assert bars[0]["date"] == "2026-06-10" and bars[0]["high"] == 110
    assert bars[1]["open"] == 105 and bars[1]["close"] == 118


def test_ohlc_series_missing_table(db):
    assert track_record.ohlc_series("GOLD", db_path=db) == []


def test_stale_assets_missing_table_reported(db):
    stale = track_record.stale_assets(
        max_age_days=7, assets=["GOLD"], db_path=db, today="2026-06-12",
    )
    assert stale[0]["asset"] == "GOLD"
    assert stale[0]["last_date"] is None


def test_latest_signals_prefers_gated_display(tmp_path):
    import sqlite3
    from core import track_record
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, actual_next_ret REAL, correct INTEGER,"
                " cb_prob REAL, lstm_prob REAL, model_version TEXT, meta_prob REAL,"
                " sig_shown TEXT, gate_reason TEXT)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-16','EURUSD','BUY',0.9,"
                "NULL,NULL,NULL,NULL,NULL,NULL,'WAIT','live-gate: FOREX MAJORS 34% (n=126)')")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-16','AAPL','SELL',0.3,"
                "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)")
    con.commit(); con.close()
    rows = {r["asset"]: r for r in track_record.latest_signals(db_path=db)}
    assert rows["EURUSD"]["signal"] == "WAIT"
    assert rows["EURUSD"]["signal_raw"] == "BUY"
    assert "live-gate" in rows["EURUSD"]["gate_reason"]
    assert rows["AAPL"]["signal"] == "SELL" and rows["AAPL"]["gate_reason"] is None


def test_latest_signals_pre_migration_schema(tmp_path):
    import sqlite3
    from core import track_record
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, correct INTEGER)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-16','BTC','BUY',0.6,NULL)")
    con.commit(); con.close()
    rows = track_record.latest_signals(db_path=db)
    assert rows[0]["signal"] == "BUY" and rows[0]["gate_reason"] is None


def test_latest_gated_picks_newest_row_post_migration(tmp_path):
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, actual_next_ret REAL, correct INTEGER,"
                " cb_prob REAL, lstm_prob REAL, model_version TEXT, meta_prob REAL,"
                " sig_shown TEXT, gate_reason TEXT)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-15','EURUSD','SELL',0.4,"
                "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-16','EURUSD','BUY',0.9,"
                "NULL,NULL,NULL,NULL,NULL,NULL,'WAIT','live-gate: FOREX MAJORS 34% (n=126)')")
    con.commit(); con.close()
    gated = track_record.latest_gated("EURUSD", db_path=db)
    assert gated["signal"] == "WAIT"
    assert gated["signal_raw"] == "BUY"
    assert "live-gate" in gated["gate_reason"]


def test_latest_gated_pre_migration_schema(tmp_path):
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, correct INTEGER)")
    con.execute("INSERT INTO prediction_log VALUES ('2026-07-16','BTC','BUY',0.6,NULL)")
    con.commit(); con.close()
    gated = track_record.latest_gated("BTC", db_path=db)
    assert gated["signal"] == "BUY"
    assert gated["gate_reason"] is None


def test_latest_gated_missing_table(tmp_path):
    path = str(tmp_path / "empty.db")
    sqlite3.connect(path).close()
    assert track_record.latest_gated("BTC", db_path=path) == {}
