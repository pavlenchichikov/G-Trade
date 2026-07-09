import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine

from core.features import feature_version

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


def current_model_version() -> str:
    """Feature-space id of the model writing predictions now (see feature_version)."""
    return feature_version()


def _migrate(cur):
    """Add model_version and meta_prob to an old prediction_log in place (rows before the
    migration keep NULL = legacy generation, so they never blend with the new
    model's track record)."""
    cols = [r[1] for r in cur.execute("PRAGMA table_info(prediction_log)").fetchall()]
    if cols and "model_version" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN model_version TEXT")
    if cols and "meta_prob" not in cols:
        cur.execute("ALTER TABLE prediction_log ADD COLUMN meta_prob REAL")


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
            lstm_prob REAL,
            model_version TEXT,
            meta_prob REAL
        )
    """)
    _migrate(cur)


def _prepare():
    """Ensure the table exists and is migrated before an aggregate read."""
    with _conn() as con:
        _ensure_table(con.cursor())
        con.commit()


def log_prediction(asset, signal, probability, cb_prob=None, lstm_prob=None,
                   model_version=None, meta_prob=None, date=None):
    # Stamp the prediction with the market bar it was computed from, not the wall
    # clock. A radar run on a Saturday for a stock sees only Friday's close, so the
    # caller passes date="Friday" and the per-asset dedup collapses it into the real
    # Friday prediction instead of minting a phantom "Saturday" row for a closed
    # market. date=None keeps the old wall-clock default for callers that lack a bar.
    today = date or datetime.utcnow().strftime("%Y-%m-%d")
    if model_version is None:
        model_version = feature_version()
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
               (date, asset, signal, probability, actual_next_ret, correct,
                cb_prob, lstm_prob, model_version, meta_prob)
               VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)""",
            (today, asset, signal, probability, cb_prob, lstm_prob,
             model_version, meta_prob),
        )
        con.commit()


def _load_bars(asset, cache):
    """Return a per-asset (Date-indexed, lowercase-columns) close series, cached
    for the duration of one reconcile pass. None if the price table is missing."""
    if asset in cache:
        return cache[asset]
    table = asset.lower()
    try:
        df = pd.read_sql(
            f'SELECT Date, Close FROM "{table}" ORDER BY Date',
            _engine(),
            index_col="Date",
        )
        # data_engine stores columns lowercase (close); normalize so df["close"]
        # never raises KeyError and silently skips the actual-vs-predicted check.
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        df = None
    cache[asset] = df
    return df


def update_actuals():
    """Reconcile logged predictions against the next trading bar. Returns counters
    so callers (the Web UI reconcile button) can report progress.

    A prediction is objectively verifiable only if its date is a real trading bar
    for that asset. Trading days are per-asset: a Saturday is a live bar for crypto
    but a closed market for a stock. So for each row:

      - date is an exact bar with a following bar  -> score it (BUY correct if the
        next close rose, SELL if it fell, WAIT never scored).
      - date is an exact bar but the latest one    -> stays pending (outcome not
        formed yet: today/yesterday before the next close lands).
      - date has no bar but a later bar exists      -> the market was closed that day
        (weekend/holiday); the prediction just reused the prior close and cannot be
        verified, so it is dropped (excluded) rather than mapped onto a neighbouring
        day and double-counted. This is why non-trading days never reach the stats.
      - date has no bar and none follow yet         -> a real trading day whose bar is
        not fetched yet -> stays pending (never deleted).
    """
    reconciled = 0
    excluded = 0
    cache = {}
    with _conn() as con:
        cur = con.cursor()
        _ensure_table(cur)
        pending = cur.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE actual_next_ret IS NULL"
        ).fetchone()[0]

        # Scan every row (not just pending) so historical phantom rows already
        # scored on a closed-market day get purged too - one reconcile self-heals.
        rows = cur.execute(
            "SELECT rowid, date, asset, signal, actual_next_ret FROM prediction_log"
        ).fetchall()

        for rowid, date_str, asset, signal, anr in rows:
            df = _load_bars(asset, cache)
            if df is None or len(df) == 0:
                continue
            D = pd.Timestamp(date_str).normalize()
            pos = int(df.index.searchsorted(D, side="right")) - 1
            exact = pos >= 0 and df.index[pos] == D
            has_future = pos + 1 < len(df)

            if not exact:
                if has_future:
                    # Non-trading day for this asset (market was closed), yet it has
                    # since traded: unverifiable stale duplicate -> remove it.
                    cur.execute("DELETE FROM prediction_log WHERE rowid=?", (rowid,))
                    excluded += 1
                # else: bar simply not fetched yet -> leave pending, never delete.
                continue

            if anr is not None:
                continue  # already scored on a real bar
            if not has_future:
                continue  # exact bar but latest -> outcome not formed yet

            today_close = df["close"].iloc[pos]
            next_close = df["close"].iloc[pos + 1]
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
            reconciled += 1

        con.commit()
    return {"pending": pending, "reconciled": reconciled, "excluded": excluded}


def _date_filter(days):
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return cutoff


def get_accuracy(asset=None, days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)

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


def get_accuracy_history(asset=None, window=7, model_version=None):
    _prepare()
    params = []
    where = "WHERE signal != 'WAIT' AND correct IS NOT NULL"
    if asset:
        where += " AND asset = ?"
        params.append(asset)
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)

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


def get_leaderboard(days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)
    df = pd.read_sql(
        f"SELECT asset, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
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


def get_daily_stats(days=30, model_version=None):
    _prepare()
    cutoff = _date_filter(days)
    params = [cutoff]
    where = "WHERE date >= ? AND signal != 'WAIT' AND correct IS NOT NULL"
    if model_version:
        where += " AND model_version = ?"
        params.append(model_version)
    df = pd.read_sql(
        f"SELECT date, correct FROM prediction_log {where}",
        _engine(),
        params=tuple(params),
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
    res = update_actuals()
    print("Reconciled %d of %d pending." % (res["reconciled"], res["pending"]))
    if res.get("excluded"):
        print("Excluded %d prediction(s) on non-trading days (market closed)." % res["excluded"])

    lb = get_leaderboard(days=30)
    if lb.empty:
        print("No leaderboard data (need >= 5 predictions per asset).")
    else:
        lb_display = lb.copy()
        lb_display["Accuracy"] = lb_display["Accuracy"].map("{:.1%}".format)
        print("\n=== LEADERBOARD (last 30 days) ===")
        print(lb_display.to_string(index=False))

    ver = current_model_version()
    for label, mv in (("ALL GENERATIONS", None), (f"CURRENT MODEL [{ver}]", ver)):
        overall = get_accuracy(days=30, model_version=mv)
        print(f"\n=== OVERALL ACCURACY (last 30 days) - {label} ===")
        if overall["accuracy"] is None:
            print("No data yet.")
        else:
            print(f"Accuracy : {overall['accuracy']:.1%}")
            print(f"Total    : {overall['total_predictions']}")
            print(f"Correct  : {overall['correct_count']}")
            for sig, stats in overall["by_signal"].items():
                acc_str = f"{stats['acc']:.1%}" if stats["acc"] is not None else "N/A"
                print(f"  {sig}: {acc_str} ({stats['count']} predictions)")
