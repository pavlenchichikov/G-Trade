import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(f"sqlite:///{DB_PATH}")
    return _ENGINE


def _conn():
    return sqlite3.connect(DB_PATH)


def _ensure_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            date TEXT,
            asset TEXT,
            signal TEXT,
            probability REAL,
            actual_next_ret REAL,
            correct INTEGER,
            cb_prob REAL,
            lstm_prob REAL
        )
    """)


def log_prediction(asset, signal, probability, cb_prob=None, lstm_prob=None):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        # Skip if already logged today for this asset
        dup = cur.execute(
            "SELECT 1 FROM prediction_log WHERE date=? AND asset=?",
            (today, asset),
        ).fetchone()
        if dup:
            return
        cur.execute(
            """INSERT INTO prediction_log
               (date, asset, signal, probability, actual_next_ret, correct, cb_prob, lstm_prob)
               VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)""",
            (today, asset, signal, probability, cb_prob, lstm_prob),
        )
        con.commit()


def update_actuals():
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        rows = cur.execute(
            "SELECT rowid, date, asset, signal FROM prediction_log WHERE actual_next_ret IS NULL"
        ).fetchall()

        for rowid, date_str, asset, signal in rows:
            table = asset.lower()
            try:
                df = pd.read_sql(
                    f'SELECT Date, Close FROM "{table}" ORDER BY Date',
                    _engine(),
                    index_col="Date",
                )
                df = df[~df.index.duplicated(keep="last")].sort_index()
                if date_str not in df.index:
                    continue
                idx = df.index.get_loc(date_str)
                if idx + 1 >= len(df):
                    continue
                today_close = df["Close"].iloc[idx]
                next_close = df["Close"].iloc[idx + 1]
                if today_close == 0:
                    continue
                ret = (next_close - today_close) / today_close

                if signal == "WAIT":
                    correct = None
                elif signal == "BUY":
                    correct = 1 if ret > 0 else 0
                else:  # SELL
                    correct = 1 if ret < 0 else 0

                cur.execute(
                    "UPDATE prediction_log SET actual_next_ret=?, correct=? WHERE rowid=?",
                    (ret, correct, rowid),
                )
            except Exception:
                continue

        con.commit()


def _date_filter(days):
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return cutoff


def get_accuracy(asset=None, days=30):
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)

    df = pd.read_sql(
        f"SELECT signal, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
    )

    if df.empty:
        return {"accuracy": None, "total_predictions": 0, "correct_count": 0, "by_signal": {}}

    total = len(df)
    correct = int(df["correct"].sum())
    accuracy = correct / total if total else None

    by_signal = {}
    for sig in ["BUY", "SELL"]:
        sub = df[df["signal"] == sig]
        count = len(sub)
        acc = sub["correct"].sum() / count if count else None
        by_signal[sig] = {"acc": acc, "count": count}

    return {
        "accuracy": accuracy,
        "total_predictions": total,
        "correct_count": correct,
        "by_signal": by_signal,
    }


def get_accuracy_history(asset=None, window=7):
    params = []
    where = "WHERE signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)

    df = pd.read_sql(
        f"SELECT date, correct FROM prediction_log {where} ORDER BY date",
        _engine(),
        params=tuple(params) if params else None,
    )

    if df.empty:
        return pd.DataFrame(columns=["date", "rolling_acc", "predictions_count"])

    daily = df.groupby("date").agg(correct_sum=("correct", "sum"), count=("correct", "count")).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_correct"] = daily["correct_sum"].rolling(window, min_periods=1).sum()
    daily["rolling_count"] = daily["count"].rolling(window, min_periods=1).sum()
    daily["rolling_acc"] = daily["rolling_correct"] / daily["rolling_count"]
    daily["predictions_count"] = daily["rolling_count"].astype(int)
    return daily[["date", "rolling_acc", "predictions_count"]]


def get_leaderboard(days=30):
    cutoff = _date_filter(days)
    df = pd.read_sql(
        """SELECT asset, correct FROM prediction_log
           WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL""",
        _engine(),
        params=(cutoff,),
    )

    if df.empty:
        return pd.DataFrame(columns=["Asset", "Accuracy", "Predictions", "Correct"])

    grp = df.groupby("asset").agg(
        Predictions=("correct", "count"),
        Correct=("correct", "sum"),
    ).reset_index()
    grp = grp[grp["Predictions"] >= 5].copy()
    grp["Accuracy"] = grp["Correct"] / grp["Predictions"]
    grp = grp.rename(columns={"asset": "Asset"})
    grp = grp.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return grp[["Asset", "Accuracy", "Predictions", "Correct"]]


def get_daily_stats(days=30):
    cutoff = _date_filter(days)
    df = pd.read_sql(
        """SELECT date, correct FROM prediction_log
           WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL""",
        _engine(),
        params=(cutoff,),
    )

    if df.empty:
        return pd.DataFrame(columns=["Date", "Predictions", "Correct", "Accuracy"])

    daily = df.groupby("date").agg(
        Predictions=("correct", "count"),
        Correct=("correct", "sum"),
    ).reset_index()
    daily["Accuracy"] = daily["Correct"] / daily["Predictions"]
    daily = daily.rename(columns={"date": "Date"})
    return daily.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    print("Updating actuals...")
    update_actuals()

    lb = get_leaderboard(days=30)
    if lb.empty:
        print("No leaderboard data (need >= 5 predictions per asset).")
    else:
        lb_display = lb.copy()
        lb_display["Accuracy"] = lb_display["Accuracy"].map("{:.1%}".format)
        print("\n=== LEADERBOARD (last 30 days) ===")
        print(lb_display.to_string(index=False))

    overall = get_accuracy(days=30)
    print("\n=== OVERALL ACCURACY (last 30 days) ===")
    if overall["accuracy"] is None:
        print("No data yet.")
    else:
        print(f"Accuracy : {overall['accuracy']:.1%}")
        print(f"Total    : {overall['total_predictions']}")
        print(f"Correct  : {overall['correct_count']}")
        for sig, stats in overall["by_signal"].items():
            acc_str = f"{stats['acc']:.1%}" if stats["acc"] is not None else "N/A"
            print(f"  {sig}: {acc_str} ({stats['count']} predictions)")
