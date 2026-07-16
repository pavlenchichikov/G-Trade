"""Fitting the global live calibration layer from prediction_log outcomes."""

import sqlite3

import recalibrate_live as rl
from core import calibration


def _db(tmp_path, rows):
    """rows: (signal, probability, correct)"""
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,"
                " probability REAL, correct INTEGER)")
    con.executemany(
        "INSERT INTO prediction_log VALUES (date('now'), 'X', ?, ?, ?)", rows)
    con.commit()
    con.close()
    return db


def test_collect_pairs_converts_sell_to_went_up(tmp_path):
    db = _db(tmp_path, [("BUY", 0.7, 1), ("BUY", 0.6, 0),
                        ("SELL", 0.3, 1), ("SELL", 0.4, 0)])
    probs, ups = rl.collect_pairs(db_path=db)
    assert probs == [0.7, 0.6, 0.3, 0.4]
    # BUY correct -> up; SELL correct -> DOWN (not up); SELL wrong -> up
    assert ups == [1, 0, 0, 1]


def test_main_refuses_below_min_n(tmp_path, capsys):
    db = _db(tmp_path, [("BUY", 0.7, 1)] * 5)
    out = rl.main(min_n=300, model_dir=str(tmp_path), db_path=db)
    assert out is None
    assert "not fitting" in capsys.readouterr().out


def test_main_fits_and_writes(tmp_path):
    rows = [("BUY", 0.4 + 0.001 * i, 1 if i % 3 else 0) for i in range(400)]
    db = _db(tmp_path, rows)
    path = rl.main(min_n=100, model_dir=str(tmp_path), db_path=db)
    assert path == calibration.live_global_path(str(tmp_path))
    p = calibration.apply_live_global(0.6, str(tmp_path))
    assert 0.0 <= p <= 1.0
