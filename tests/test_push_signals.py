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
    assert row["action"] == "WAIT"        # missing signal -> WAIT
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
