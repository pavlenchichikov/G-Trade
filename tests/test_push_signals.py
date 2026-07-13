"""Unit tests for push_signals.build_payload (pure, no network / no DB)."""

from push_signals import _rest_base, build_payload


def test_rest_base_tolerates_pasted_path():
    want = "https://x.supabase.co/rest/v1"
    assert _rest_base("https://x.supabase.co") == want
    assert _rest_base("https://x.supabase.co/") == want
    assert _rest_base("https://x.supabase.co/rest/v1") == want
    assert _rest_base("https://x.supabase.co/rest/v1/") == want


def _sig(asset, signal, prob, acc=None, date="2026-07-12"):
    return {
        "asset": asset,
        "date": date,
        "signal": signal,
        "probability": prob,
        "acc": {"n": 10, "correct": 6, "acc": acc},
    }


def test_counts_and_breadth():
    sigs = [
        _sig("BTC", "BUY", 0.62, 0.6),
        _sig("ETH", "WAIT", 0.51, None),
        _sig("SBER", "SELL", 0.38, 0.5),
    ]
    rows, stats = build_payload(sigs)
    assert len(rows) == 3
    assert (stats["n_buy"], stats["n_sell"], stats["n_wait"]) == (1, 1, 1)
    # breadth = actionable / total = 2/3
    assert abs(stats["breadth"] - 2 / 3) < 1e-9
    # accuracy = mean of present accs (0.6, 0.5)
    assert abs(stats["accuracy"] - 0.55) < 1e-9
    assert stats["snapshot_date"] == "2026-07-12"


def test_row_mapping_and_missing_action():
    rows, stats = build_payload([{"asset": "GOLD", "date": "2026-07-11", "probability": 0.58}])
    row = rows[0]
    assert row["asset"] == "GOLD"
    assert row["action"] == "WAIT"        # missing signal - WAIT
    assert row["prob"] == 0.58
    assert row["mode"] is None and row["taleb"] is None
    assert stats["n_wait"] == 1
    assert stats["accuracy"] is None       # no accuracies present


def test_empty_snapshot_uses_today():
    rows, stats = build_payload([])
    assert rows == []
    assert stats["n_buy"] == stats["n_sell"] == stats["n_wait"] == 0
    assert stats["breadth"] == 0.0
    assert stats["snapshot_date"]          # today's date, non-empty


import sqlite3

import push_signals


def _seed_history_db(tmp_path):
    db = str(tmp_path / "m.db")
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE btc (Date TEXT, open REAL, close REAL, high REAL,'
                ' low REAL, volume REAL)')
    for i in range(200):
        d = f"2026-{i:03d}"  # synthetic ascending dates, 8 chars (str[:10]-safe)
        con.execute("INSERT INTO btc VALUES (?, 1, 2, 3, 0.5, 9)", (d,))
    con.execute('CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT,'
                ' probability REAL, actual_next_ret REAL, correct INTEGER)')
    for i in range(120):
        con.execute("INSERT INTO prediction_log VALUES (?, 'BTC', 'BUY', 0.6, 0.01, 1)",
                    (f"2026-{i:03d}",))
    con.commit()
    con.close()
    return db


def test_fetch_history_rows_limits_and_order(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP", {"BTC": "BTC-USD"},
                        raising=False)
    db = _seed_history_db(tmp_path)
    bars, hist = push_signals.fetch_history_rows(db_path=db, bar_limit=180,
                                                 sig_limit=90)
    assert len(bars) == 180 and len(hist) == 90
    # ascending after the DESC-limit read
    assert bars[0]["date"] < bars[-1]["date"]
    assert bars[-1]["date"] == "2026-199"
    assert set(bars[0]) == {"asset", "date", "open", "high", "low", "close"}
    assert set(hist[0]) == {"asset", "date", "signal", "prob", "actual_next_ret",
                            "correct"}


def test_fetch_history_rows_skips_missing_table(tmp_path, monkeypatch):
    monkeypatch.setattr(push_signals, "FULL_ASSET_MAP",
                        {"BTC": "BTC-USD", "NEWCO": "NEW"}, raising=False)
    db = _seed_history_db(tmp_path)
    bars, hist = push_signals.fetch_history_rows(db_path=db)
    assert {b["asset"] for b in bars} == {"BTC"}  # NEWCO has no table - skipped


def test_push_history_full_refresh(monkeypatch):
    calls = []

    class R:
        status_code = 200
        def raise_for_status(self):
            pass

    monkeypatch.setattr(push_signals.requests, "delete",
                        lambda url, **kw: calls.append(("DELETE", url)) or R())
    monkeypatch.setattr(push_signals.requests, "post",
                        lambda url, **kw: calls.append(("POST", url)) or R())
    bars = [{"asset": "BTC", "date": "2026-07-13"}] * 2500  # forces 3 chunks
    push_signals.push_history("https://x.supabase.co", "k", bars, [], [], None)
    deletes = [u for m, u in calls if m == "DELETE"]
    posts = [u for m, u in calls if m == "POST"]
    assert sum("/bars" in u for u in deletes) == 1
    assert sum("/bars" in u for u in posts) == 3          # 2500 rows / 1000 chunk
    assert sum("/signal_history" in u for u in deletes) == 1
    assert sum("/guru?" in u or u.endswith("/guru") for u in deletes) == 1
    assert not any("/guru_stats" in u for u in posts)      # guru_stats None - skipped


def test_fetch_guru_rows_maps_dashboard_payload(monkeypatch):
    from core import dashboard

    monkeypatch.setattr(dashboard, "guru_latest", lambda: [{
        "asset": "AAPL", "date": "2026-07-12", "verdict": "BUY", "pct": 75.0,
        "lynch": 2, "buffett": 2, "graham": 1, "munger": 1,
        "source": "yf", "correct_5d": 1,
    }])
    monkeypatch.setattr(dashboard, "guru_accuracy", lambda: {
        "council": {"accuracy": 0.61, "total": 40, "correct": 24,
                    "avg_return": 0.01, "by_verdict": {}, "horizon": "60d"},
        "individual": {},
    })
    rows, stats = push_signals.fetch_guru_rows()
    assert rows == [{"asset": "AAPL", "verdict": "BUY", "council_pct": 75.0,
                     "lynch": 2, "buffett": 2, "graham": 1, "munger": 1,
                     "source": "yf", "date": "2026-07-12", "correct_5d": 1}]
    assert stats == {"id": 1, "accuracy": 0.61, "n": 40, "horizon": "60d"}


def test_build_push_text_ranks_by_confidence():
    rows = [
        {"asset": "BTC", "action": "BUY", "prob": 0.60},
        {"asset": "SBER", "action": "SELL", "prob": 0.30},   # conf 0.70 - first
        {"asset": "ETH", "action": "WAIT", "prob": 0.50},    # excluded
        {"asset": "GOLD", "action": "BUY", "prob": None},    # excluded
    ]
    stats = {"n_buy": 2, "n_sell": 1}
    title, body = push_signals.build_push_text(rows, stats)
    assert title == "Atratus: 2 BUY / 1 SELL"
    assert body == "SBER SELL 70% | BTC BUY 60%"


def test_build_push_text_empty_body_fallback():
    title, body = push_signals.build_push_text([], {"n_buy": 0, "n_sell": 0})
    assert body == "New signals snapshot"


def test_send_push_noop_without_creds(monkeypatch):
    monkeypatch.delenv("GTRADE_FCM_CREDS", raising=False)
    # must not touch the network or firebase at all
    assert push_signals.send_push("https://x.supabase.co", "k", [], {}) == 0
