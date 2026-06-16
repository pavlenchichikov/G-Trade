"""Чтение сигналов и их истории из market.db для веба, бота и дайджеста.

Интерфейсы ничего не считают: predict.py пишет в prediction_log
(performance_tracker), здесь - только выборки.
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")

ACC_LAST_N = 20  # сколько последних проверенных сигналов идёт в точность


def _table_name(asset: str) -> str:
    # та же нормализация, что в predict.py
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _connect(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def _accuracy(con, asset: str, last_n: int) -> dict:
    rows = con.execute(
        "SELECT correct FROM prediction_log "
        "WHERE asset=? AND correct IS NOT NULL ORDER BY date DESC LIMIT ?",
        (asset, last_n),
    ).fetchall()
    n = len(rows)
    correct = sum(r[0] for r in rows)
    return {"n": n, "correct": correct, "acc": (correct / n) if n else None}


def asset_accuracy(asset: str, last_n: int = ACC_LAST_N, db_path=None) -> dict:
    with _connect(db_path) as con:
        try:
            return _accuracy(con, asset, last_n)
        except sqlite3.OperationalError:
            return {"n": 0, "correct": 0, "acc": None}


def latest_signals(db_path=None, acc_last_n: int = ACC_LAST_N) -> list:
    """Последний сигнал по каждому активу + точность последних проверенных."""
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                "SELECT p.asset, p.date, p.signal, p.probability "
                "FROM prediction_log p "
                "JOIN (SELECT asset, MAX(date) AS d FROM prediction_log GROUP BY asset) m "
                "ON p.asset = m.asset AND p.date = m.d "
                "ORDER BY p.asset"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for asset, date, signal, prob in rows:
            out.append({
                "asset": asset,
                "date": date,
                "signal": signal,
                "probability": prob,
                "acc": _accuracy(con, asset, acc_last_n),
            })
        return out


def asset_track(asset: str, limit: int = 30, db_path=None) -> list:
    """История сигналов актива, свежие первыми."""
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                "SELECT date, signal, probability, actual_next_ret, correct, "
                "cb_prob, lstm_prob "
                "FROM prediction_log WHERE asset=? ORDER BY date DESC LIMIT ?",
                (asset, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {"date": d, "signal": s, "probability": p,
             "actual_next_ret": r, "correct": c,
             "cb_prob": cb, "lstm_prob": lstm}
            for d, s, p, r, c, cb, lstm in rows
        ]


def price_series(asset: str, days: int = 60, db_path=None) -> list:
    """Последние closes актива по возрастанию даты: [{'date','close'}, ...]."""
    table = _table_name(asset)
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                f'SELECT Date, Close FROM "{table}" ORDER BY Date DESC LIMIT ?',
                (days,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    rows.reverse()
    return [{"date": str(d)[:10], "close": c} for d, c in rows if c is not None]


def stale_assets(max_age_days: int = 7, assets=None, db_path=None, today=None) -> list:
    """Активы, у которых данные в market.db старше порога (или отсутствуют)."""
    if assets is None:
        from config import FULL_ASSET_MAP
        assets = list(FULL_ASSET_MAP.keys())
    today_dt = datetime.strptime(today, "%Y-%m-%d") if today else datetime.now()

    out = []
    with _connect(db_path) as con:
        for asset in assets:
            table = _table_name(asset)
            try:
                row = con.execute(f'SELECT MAX(Date) FROM "{table}"').fetchone()
                last = row[0] if row else None
            except sqlite3.OperationalError:
                last = None
            if last is None:
                out.append({"asset": asset, "last_date": None, "age_days": None})
                continue
            try:
                last_dt = datetime.strptime(str(last)[:10], "%Y-%m-%d")
            except ValueError:
                continue
            age = (today_dt - last_dt).days
            if age > max_age_days:
                out.append({"asset": asset, "last_date": str(last)[:10], "age_days": age})
    return out
