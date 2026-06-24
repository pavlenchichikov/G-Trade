# tests/test_loop_state.py
from core.loop_state import load_state, save_state, approve, dismiss


def test_load_missing_returns_skeleton(tmp_path):
    st = load_state(str(tmp_path / "none.json"))
    assert st["proposed"] == [] and st["approved"] == [] and st["last_run"] is None


def test_save_then_load_roundtrip(tmp_path):
    p = str(tmp_path / "s.json")
    save_state(p, {"last_run": "2026-06-24", "steps": {}, "assets": [],
                   "proposed": ["AAPL"], "approved": [], "history": []})
    assert load_state(p)["proposed"] == ["AAPL"]


def test_approve_moves_only_proposed(tmp_path):
    p = str(tmp_path / "s.json")
    save_state(p, {"last_run": None, "steps": {}, "assets": [],
                   "proposed": ["AAPL", "MSFT"], "approved": [], "history": []})
    st = approve(p, ["AAPL", "NOPE"])  # NOPE not in proposed, so it is ignored
    assert st["approved"] == ["AAPL"]


def test_dismiss_removes_from_proposed(tmp_path):
    p = str(tmp_path / "s.json")
    save_state(p, {"last_run": None, "steps": {}, "assets": [],
                   "proposed": ["AAPL", "MSFT"], "approved": [], "history": []})
    st = dismiss(p, "AAPL")
    assert st["proposed"] == ["MSFT"]
