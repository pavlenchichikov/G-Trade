"""
guru_tracker.py — Track Guru Council recommendations vs actual outcomes.
Logs each guru's verdict + council vote, then compares with price movement
after 1, 5, and 20 trading days.
"""

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
        CREATE TABLE IF NOT EXISTS guru_log (
            date TEXT,
            asset TEXT,
            lynch_score INTEGER,
            buffett_score INTEGER,
            graham_score INTEGER,
            munger_score INTEGER,
            council_pct REAL,
            council_verdict TEXT,
            data_source TEXT,
            price_at_signal REAL,
            ret_1d REAL,
            ret_5d REAL,
            ret_20d REAL,
            correct_1d INTEGER,
            correct_5d INTEGER,
            correct_20d INTEGER
        )
    """)


def log_guru_verdict(asset, lynch_score, buffett_score, graham_score, munger_score,
                     council_pct, council_verdict, data_source, price):
    """Log a guru council verdict for an asset."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        # Don't duplicate — one entry per asset per day
        existing = cur.execute(
            "SELECT rowid FROM guru_log WHERE date = ? AND asset = ?",
            (today, asset)
        ).fetchone()
        if existing:
            cur.execute(
                """UPDATE guru_log SET lynch_score=?, buffett_score=?, graham_score=?,
                   munger_score=?, council_pct=?, council_verdict=?, data_source=?,
                   price_at_signal=? WHERE rowid=?""",
                (lynch_score, buffett_score, graham_score, munger_score,
                 council_pct, council_verdict, data_source, price, existing[0])
            )
        else:
            cur.execute(
                """INSERT INTO guru_log
                   (date, asset, lynch_score, buffett_score, graham_score, munger_score,
                    council_pct, council_verdict, data_source, price_at_signal,
                    ret_1d, ret_5d, ret_20d, correct_1d, correct_5d, correct_20d)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)""",
                (today, asset, lynch_score, buffett_score, graham_score, munger_score,
                 council_pct, council_verdict, data_source, price)
            )
        con.commit()


def update_actuals():
    """Fill in actual returns for past guru predictions."""
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        rows = cur.execute(
            """SELECT rowid, date, asset, council_verdict
               FROM guru_log WHERE ret_1d IS NULL OR ret_5d IS NULL OR ret_20d IS NULL"""
        ).fetchall()

        for rowid, date_str, asset, verdict in rows:
            table = asset.lower().replace("^", "").replace(".", "").replace("-", "")
            try:
                df = pd.read_sql(
                    f'SELECT Date, close FROM "{table}" ORDER BY Date',
                    _engine(), index_col="Date",
                )
                df = df[~df.index.duplicated(keep="last")].sort_index()
                if date_str not in df.index:
                    continue
                idx = df.index.get_loc(date_str)
                price_0 = df["close"].iloc[idx]
                if price_0 == 0:
                    continue

                updates = {}
                # 1-day return
                if idx + 1 < len(df):
                    ret_1 = (df["close"].iloc[idx + 1] - price_0) / price_0
                    updates["ret_1d"] = ret_1
                    if verdict == "BUY":
                        updates["correct_1d"] = 1 if ret_1 > 0 else 0
                    elif verdict == "AVOID":
                        updates["correct_1d"] = 1 if ret_1 < 0 else 0
                    else:
                        updates["correct_1d"] = None

                # 5-day return
                if idx + 5 < len(df):
                    ret_5 = (df["close"].iloc[idx + 5] - price_0) / price_0
                    updates["ret_5d"] = ret_5
                    if verdict == "BUY":
                        updates["correct_5d"] = 1 if ret_5 > 0 else 0
                    elif verdict == "AVOID":
                        updates["correct_5d"] = 1 if ret_5 < 0 else 0
                    else:
                        updates["correct_5d"] = None

                # 20-day return
                if idx + 20 < len(df):
                    ret_20 = (df["close"].iloc[idx + 20] - price_0) / price_0
                    updates["ret_20d"] = ret_20
                    if verdict == "BUY":
                        updates["correct_20d"] = 1 if ret_20 > 0 else 0
                    elif verdict == "AVOID":
                        updates["correct_20d"] = 1 if ret_20 < 0 else 0
                    else:
                        updates["correct_20d"] = None

                if updates:
                    set_clause = ", ".join(f"{k}=?" for k in updates)
                    cur.execute(
                        f"UPDATE guru_log SET {set_clause} WHERE rowid=?",
                        list(updates.values()) + [rowid]
                    )
            except Exception:
                continue

        con.commit()


def get_guru_accuracy(days=30, horizon="5d"):
    """
    Overall accuracy of the council verdict (BUY/AVOID) over given horizon.
    horizon: '1d', '5d', or '20d'
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    col_correct = f"correct_{horizon}"
    col_ret = f"ret_{horizon}"

    df = pd.read_sql(
        f"""SELECT asset, council_verdict, council_pct, {col_correct}, {col_ret}
            FROM guru_log
            WHERE date >= ? AND council_verdict != 'HOLD' AND {col_correct} IS NOT NULL""",
        _engine(), params=(cutoff,),
    )

    if df.empty:
        return {"accuracy": None, "total": 0, "correct": 0, "avg_return": None,
                "by_verdict": {}, "horizon": horizon}

    total = len(df)
    correct = int(df[col_correct].sum())
    accuracy = correct / total if total else None
    avg_ret = float(df[col_ret].mean())

    by_verdict = {}
    for v in ["BUY", "AVOID"]:
        sub = df[df["council_verdict"] == v]
        cnt = len(sub)
        if cnt > 0:
            by_verdict[v] = {
                "accuracy": float(sub[col_correct].sum()) / cnt,
                "count": cnt,
                "avg_return": float(sub[col_ret].mean()),
            }

    return {
        "accuracy": accuracy, "total": total, "correct": correct,
        "avg_return": avg_ret, "by_verdict": by_verdict, "horizon": horizon,
    }


def get_guru_individual_accuracy(days=30, horizon="5d"):
    """
    Per-guru accuracy: how often each guru's positive score (>=1) predicted positive returns.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    col_ret = f"ret_{horizon}"

    df = pd.read_sql(
        f"""SELECT lynch_score, buffett_score, graham_score, munger_score, {col_ret}
            FROM guru_log WHERE date >= ? AND {col_ret} IS NOT NULL""",
        _engine(), params=(cutoff,),
    )

    if df.empty:
        return {}

    results = {}
    for guru, col in [("Lynch", "lynch_score"), ("Buffett", "buffett_score"),
                      ("Graham", "graham_score"), ("Munger", "munger_score")]:
        positive = df[df[col] >= 1]
        negative = df[df[col] == 0]

        pos_correct = int((positive[col_ret] > 0).sum()) if len(positive) else 0
        neg_correct = int((negative[col_ret] < 0).sum()) if len(negative) else 0

        total_calls = len(positive) + len(negative)
        total_correct = pos_correct + neg_correct

        results[guru] = {
            "accuracy": total_correct / total_calls if total_calls else None,
            "bullish_calls": len(positive),
            "bullish_correct": pos_correct,
            "bearish_calls": len(negative),
            "bearish_correct": neg_correct,
            "avg_ret_when_bullish": float(positive[col_ret].mean()) if len(positive) else None,
        }

    return results


def get_guru_leaderboard(days=30, horizon="5d"):
    """Asset leaderboard by council accuracy."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    col_correct = f"correct_{horizon}"
    col_ret = f"ret_{horizon}"

    df = pd.read_sql(
        f"""SELECT asset, council_verdict, {col_correct}, {col_ret}
            FROM guru_log
            WHERE date >= ? AND council_verdict != 'HOLD' AND {col_correct} IS NOT NULL""",
        _engine(), params=(cutoff,),
    )

    if df.empty:
        return pd.DataFrame(columns=["Asset", "Predictions", "Correct", "Accuracy", "Avg_Return"])

    grp = df.groupby("asset").agg(
        Predictions=(col_correct, "count"),
        Correct=(col_correct, "sum"),
        Avg_Return=(col_ret, "mean"),
    ).reset_index()
    grp = grp[grp["Predictions"] >= 3].copy()
    grp["Accuracy"] = grp["Correct"] / grp["Predictions"]
    grp = grp.rename(columns={"asset": "Asset"})
    return grp.sort_values("Accuracy", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Updating guru actuals...")
    update_actuals()

    for horizon in ["1d", "5d", "20d"]:
        acc = get_guru_accuracy(days=60, horizon=horizon)
        if acc["accuracy"] is not None:
            print(f"\n=== COUNCIL ACCURACY ({horizon}, last 60 days) ===")
            print(f"  Accuracy: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")
            print(f"  Avg Return: {acc['avg_return']:.2%}")
            for v, s in acc["by_verdict"].items():
                print(f"    {v}: {s['accuracy']:.1%} ({s['count']} calls, avg ret {s['avg_return']:.2%})")
        else:
            print(f"\n=== {horizon}: No data yet ===")

    gurus = get_guru_individual_accuracy(days=60, horizon="5d")
    if gurus:
        print("\n=== INDIVIDUAL GURU ACCURACY (5d) ===")
        for name, s in gurus.items():
            acc_str = f"{s['accuracy']:.1%}" if s['accuracy'] is not None else "N/A"
            print(f"  {name:<10} Accuracy: {acc_str}  Bullish: {s['bullish_calls']}  Bearish: {s['bearish_calls']}")
