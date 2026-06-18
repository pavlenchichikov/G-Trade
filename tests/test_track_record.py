"""Тесты core.track_record на временной SQLite."""

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
        # BTC: три дня истории, последняя - BUY
        ("2026-06-08", "BTC", "BUY", 0.61, 0.012, 1, 0.61, None),
        ("2026-06-09", "BTC", "SELL", 0.58, 0.004, 0, 0.58, None),
        ("2026-06-10", "BTC", "BUY", 0.62, None, None, 0.62, None),
        # TSLA: один день
        ("2026-06-10", "TSLA", "WAIT", 0.51, None, None, 0.51, None),
    ]
    con.executemany("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?)", rows)
    # ценовая таблица для stale_assets
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
    assert acc["n"] == 2          # строка с correct=NULL не считается
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


def test_stale_assets_missing_table_reported(db):
    stale = track_record.stale_assets(
        max_age_days=7, assets=["GOLD"], db_path=db, today="2026-06-12",
    )
    assert stale[0]["asset"] == "GOLD"
    assert stale[0]["last_date"] is None
