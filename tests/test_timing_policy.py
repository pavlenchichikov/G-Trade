"""Tests for core/timing_policy.py (spec 2026-07-18-rl-timing-policy)."""
import numpy as np

from core import timing_policy as tp


def _arr(*vals):
    return np.asarray(vals, dtype=float)


def _apply(policy, probs, **kw):
    n = len(probs)
    kw.setdefault("buy_thr", 0.55)
    kw.setdefault("sell_thr", 0.45)
    kw.setdefault("atr", np.full(n, 0.02))
    kw.setdefault("taleb_hi", np.zeros(n, dtype=bool))
    kw.setdefault("risky", False)
    return policy.apply(_arr(*probs), **kw)


class TestBaselineIdentity:
    def test_default_params_equal_raw_signals(self):
        pol = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
        probs = [0.60, 0.60, 0.50, 0.40, 0.40, 0.50, 0.70]
        sides, actions, reasons = _apply(pol, probs)
        raw = [1, 1, 0, -1, -1, 0, 1]
        assert list(sides) == raw


class TestEntryRules:
    def test_entry_margin_blocks_weak_signal(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.03})
        sides, actions, reasons = _apply(pol, [0.56, 0.56])
        assert list(sides) == [0, 0]
        assert reasons[0] == "entry_margin"

    def test_confirm_days_delays_entry(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "confirm_days": 2})
        sides, actions, reasons = _apply(pol, [0.60, 0.60, 0.60])
        assert list(sides) == [0, 1, 1]
        assert reasons[0] == "confirm" and actions[1] == "ENTER"

    def test_taleb_margin_added_only_in_high_regime(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS,
                              "entry_margin_hi_taleb": 0.10})
        hi = np.asarray([True, False])
        sides, _, reasons = _apply(pol, [0.60, 0.60], taleb_hi=hi)
        assert list(sides) == [0, 1]
        assert reasons[0] == "entry_margin"

    def test_risky_margin(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS,
                              "entry_margin_risky": 0.10})
        sides, _, _ = _apply(pol, [0.60], risky=True)
        assert list(sides) == [0]

    def test_short_entry_symmetric(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.03})
        sides, _, reasons = _apply(pol, [0.44, 0.40])
        assert list(sides) == [0, -1]
        assert reasons[0] == "entry_margin"


class TestExitRules:
    def test_hysteresis_holds_through_wobble(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "exit_hysteresis": 0.05})
        # enters at 0.60; prob dips to 0.52 (below buy_thr but above 0.55-0.05)
        sides, actions, _ = _apply(pol, [0.60, 0.52, 0.49])
        assert list(sides) == [1, 1, 0]
        assert actions[1] == "HOLD" and actions[2] == "EXIT"

    def test_flip_exits(self):
        pol = tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
        sides, actions, reasons = _apply(pol, [0.60, 0.40])
        assert actions[1] == "EXIT" and reasons[1] == "flip"

    def test_max_hold_exits(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "max_hold_days": 2})
        sides, actions, reasons = _apply(pol, [0.60, 0.60, 0.60, 0.60])
        assert actions[2] == "EXIT" and reasons[2] == "max_hold"
        assert list(sides)[:3] == [1, 1, 0]

    def test_cooldown_blocks_reentry(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "max_hold_days": 2,
                              "cooldown_days": 2})
        sides, actions, reasons = _apply(pol, [0.60] * 5)
        # enter@0, hold@1, max_hold exit@2, cooldown@3, cooldown expired -> enter@4
        assert actions[2] == "EXIT"
        assert actions[3] == "STAY_OUT" and reasons[3] == "cooldown"
        assert actions[4] == "ENTER"

    def test_trail_stop(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "trail_atr": 2.0})
        probs = [0.60, 0.60, 0.60, 0.60]
        next_ret = _arr(0.05, -0.03, -0.03, 0.0)  # peak .05, dd .06 > 2*.02
        sides, actions, reasons = pol.apply(
            _arr(*probs), buy_thr=0.55, sell_thr=0.45,
            atr=np.full(4, 0.02), taleb_hi=np.zeros(4, dtype=bool),
            risky=False, next_ret=next_ret)
        assert "trail_stop" in reasons
        i = reasons.index("trail_stop")
        assert actions[i] == "EXIT"


class TestStepAndLoad:
    def test_policy_step_matches_apply(self):
        pol = tp.RulesPolicy({**tp.DEFAULT_PARAMS, "confirm_days": 1,
                              "exit_hysteresis": 0.05})
        probs = [0.60, 0.60, 0.52, 0.40]
        sides, actions, reasons = _apply(pol, probs)
        state = tp.FRESH_STATE.copy()
        for i, p in enumerate(probs):
            action, reason, state = tp.policy_step(
                pol, p, 0.55, 0.45, 0.02, False, False, state)
            assert action == actions[i], f"bar {i}"

    def test_load_policy_missing(self, tmp_path):
        assert tp.load_policy(str(tmp_path / "nope.json")) is None

    def test_load_policy_roundtrip(self, tmp_path):
        import json
        p = tmp_path / "timing_policy.json"
        p.write_text(json.dumps(
            {"version": 1, "params": dict(tp.DEFAULT_PARAMS)}))
        pol = tp.load_policy(str(p))
        assert isinstance(pol, tp.RulesPolicy)

    def test_timing_on(self, monkeypatch):
        monkeypatch.delenv("GTRADE_TIMING_POLICY", raising=False)
        assert tp.timing_on() is False
        monkeypatch.setenv("GTRADE_TIMING_POLICY", "1")
        assert tp.timing_on() is True


def _seed_bar(db_path, asset, day, close=100.0):
    """Insert one price bar so log_prediction's _has_bar gate lets the row
    through (real schema: capital 'Date', lowercase table name/columns)."""
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute(f"CREATE TABLE IF NOT EXISTS {asset.lower()} (Date TEXT, close REAL)")
    con.execute(f"INSERT INTO {asset.lower()} VALUES (?, ?)", (day, close))
    con.commit()
    con.close()


class TestServeShadow:
    """Serve-side shadow integration (Task 5): log_prediction gains two
    optional timing_* columns and performance_tracker.timing_state rebuilds
    the policy's position state from them. NOTE: the brief's sketch assumed
    log_prediction had no date-override hook and asked for a new
    `_day_override` kwarg; the real signature already has `date=` for this
    exact purpose (see its docstring), so these tests reuse that instead of
    adding a redundant parameter. The real signature also gates every insert
    on `_has_bar` (a real price bar must exist for that asset/date), so each
    test seeds a matching price bar first."""

    def test_log_prediction_accepts_timing_fields(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime

        import performance_tracker as pt

        db = str(tmp_path / "predictions.db")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        _seed_bar(db, "TEST", today)
        monkeypatch.setattr(pt, "DB_PATH", db, raising=False)

        pt.log_prediction("TEST", "BUY", 0.61, cb_prob=0.6, lstm_prob=0.5,
                          model_version="v1", meta_prob=None,
                          sig_shown="BUY", gate_reason=None,
                          timing_action="ENTER:+1", timing_reason="ok")

        con = sqlite3.connect(db)
        row = con.execute("SELECT timing_action, timing_reason FROM "
                          "prediction_log WHERE asset='TEST'").fetchone()
        con.close()
        assert row == ("ENTER:+1", "ok")

    def test_log_prediction_off_path_nulls(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime

        import performance_tracker as pt

        db = str(tmp_path / "predictions.db")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        _seed_bar(db, "TEST2", today)
        monkeypatch.setattr(pt, "DB_PATH", db, raising=False)

        pt.log_prediction("TEST2", "BUY", 0.61, cb_prob=0.6, lstm_prob=0.5,
                          model_version="v1", meta_prob=None,
                          sig_shown="BUY", gate_reason=None)

        con = sqlite3.connect(db)
        row = con.execute("SELECT timing_action, timing_reason FROM "
                          "prediction_log WHERE asset='TEST2'").fetchone()
        con.close()
        assert row == (None, None)

    def test_timing_state_rebuild(self, tmp_path, monkeypatch):
        import performance_tracker as pt

        db = str(tmp_path / "predictions.db")
        rows = [("2026-07-01", "ENTER:+1"), ("2026-07-02", "HOLD"),
                ("2026-07-03", "HOLD")]
        for day, _act in rows:
            _seed_bar(db, "T3", day)
        monkeypatch.setattr(pt, "DB_PATH", db, raising=False)

        for day, act in rows:
            pt.log_prediction("T3", "BUY", 0.6, cb_prob=0.6, lstm_prob=0.5,
                              model_version="v1", meta_prob=None,
                              sig_shown="BUY", gate_reason=None,
                              timing_action=act, timing_reason="ok",
                              date=day)

        st = pt.timing_state("T3")
        assert st["pos"] == 1
        assert st["days_held"] == 2

    def test_timing_state_rebuilds_streak_and_cooldown(self, tmp_path, monkeypatch):
        import performance_tracker as pt

        db = str(tmp_path / "predictions.db")
        rows = [("2026-07-01", "ENTER:+1"), ("2026-07-02", "EXIT"),
                ("2026-07-03", "STAY_OUT")]
        for day, _act in rows:
            _seed_bar(db, "T4", day)
        monkeypatch.setattr(pt, "DB_PATH", db, raising=False)

        for day, act in rows:
            pt.log_prediction("T4", "BUY", 0.6, cb_prob=0.6, lstm_prob=0.5,
                              model_version="v1", meta_prob=None,
                              sig_shown="BUY", gate_reason=None,
                              timing_action=act, timing_reason="ok",
                              date=day)

        st = pt.timing_state("T4", cooldown_days=3)
        assert st == {"pos": 0, "days_held": 0, "seg_peak": 0.0, "seg_ret": 0.0,
                      "cooldown_left": 2, "streak": 3, "last_raw": 1}


class TestScoringOffPathGuard:
    """score_asset's timing-policy shadow block must be unreachable unless
    GTRADE_TIMING_POLICY=1 - checked structurally (rather than through a full
    score_asset() run, which needs a fitted champion + scaler + calibrator
    fixture set) because the guard is what makes the OFF path byte-identical:
    no load_policy() call, no timing_state()/DB touch, no policy file read."""

    def test_timing_block_gated_by_timing_on_before_load_policy(self):
        import inspect

        from core import scoring as scoring_mod

        src = inspect.getsource(scoring_mod.score_asset)
        assert "timing_policy.timing_on()" in src
        assert "timing_policy.load_policy()" in src
        on_idx = src.index("timing_policy.timing_on()")
        load_idx = src.index("timing_policy.load_policy()")
        assert on_idx < load_idx, (
            "load_policy() must appear textually after the timing_on() guard")
