"""Tests for core/ar_rl.py - RL search controller (spec 2026-07-18)."""
import random

import pytest

from core import ar_memory, ar_rl


class TestBlobStore:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ar_memory, "_RL_BLOB_DIR", str(tmp_path), raising=False)
        ar_memory.blob_put("rl_test", {"a": 1})
        assert ar_memory.blob_get("rl_test") == {"a": 1}

    def test_missing_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ar_memory, "_RL_BLOB_DIR", str(tmp_path), raising=False)
        assert ar_memory.blob_get("nope", default={}) == {}


class TestPhase:
    def test_fill_below_half(self):
        assert ar_rl.phase_of(0.49) == "fill"

    def test_refine_at_half(self):
        assert ar_rl.phase_of(0.5) == "refine"


class TestScheduler:
    def test_fresh_state_uniform_prior(self):
        s = ar_rl.Scheduler()
        assert s.posterior_mean("feat", "fill") == pytest.approx(0.5)

    def test_update_moves_posterior(self):
        s = ar_rl.Scheduler()
        for _ in range(10):
            s.update("feat", "fill", True)
        assert s.posterior_mean("feat", "fill") > 0.8

    def test_phases_are_independent(self):
        s = ar_rl.Scheduler()
        for _ in range(10):
            s.update("feat", "fill", True)
        assert s.posterior_mean("feat", "refine") == pytest.approx(0.5)

    def test_discount_forgets(self):
        s = ar_rl.Scheduler()
        for _ in range(50):
            s.update("feat", "fill", True)
        high = s.posterior_mean("feat", "fill")
        for _ in range(50):
            s.update("feat", "fill", False)
        # with gamma=0.99 discounting, old successes decay: posterior falls well below high
        assert s.posterior_mean("feat", "fill") < high - 0.3

    def test_choose_prefers_good_arm(self):
        rng = random.Random(7)
        s = ar_rl.Scheduler(rng=rng)
        for _ in range(30):
            s.update("cma", "fill", True)
            s.update("feat", "fill", False)
        picks = [s.choose(["feat", "cma"], "fill")[0] for _ in range(200)]
        assert picks.count("cma") > 120

    def test_floor_guarantees_exploration(self):
        rng = random.Random(7)
        s = ar_rl.Scheduler(rng=rng)
        for _ in range(60):
            s.update("cma", "fill", True)
            s.update("feat", "fill", False)
        picks = [s.choose(["feat", "cma"], "fill")[0] for _ in range(2000)]
        # floor_total = 0.03 * 2 = 6 percent uniform => feat gets >= ~3 percent
        assert picks.count("feat") >= 20

    def test_choose_only_available(self):
        s = ar_rl.Scheduler(rng=random.Random(1))
        for _ in range(50):
            arm, _ = s.choose(["nets"], "refine")
            assert arm == "nets"

    def test_bonus_applies_n_successes(self):
        s, s2 = ar_rl.Scheduler(), ar_rl.Scheduler()
        s.bonus("llm", "fill")
        for _ in range(ar_rl.ADOPT_BONUS):
            s2.update("llm", "fill", True)
        assert s.posterior_mean("llm", "fill") == pytest.approx(
            s2.posterior_mean("llm", "fill"), abs=0.05)

    def test_state_roundtrip(self):
        s = ar_rl.Scheduler(rng=random.Random(3))
        s.update("surr", "refine", True)
        s2 = ar_rl.Scheduler(state=s.to_state())
        assert s2.posterior_mean("surr", "refine") == pytest.approx(
            s.posterior_mean("surr", "refine"))

    def test_corrupt_state_starts_fresh(self):
        s = ar_rl.Scheduler(state={"posteriors": "garbage", "version": 99})
        assert s.posterior_mean("feat", "fill") == pytest.approx(0.5)

    def test_halve(self):
        s = ar_rl.Scheduler()
        for _ in range(20):
            s.update("feat", "fill", True)
        before = s.posterior_mean("feat", "fill")
        s.halve()
        # mean roughly preserved, confidence (counts) halved
        assert s.posterior_mean("feat", "fill") == pytest.approx(before, abs=0.1)
        a, b = s.to_state()["posteriors"]["fill"]["feat"]
        assert a + b < 13


class TestCuriosity:
    def test_default_score(self):
        c = ar_rl.CuriosityMap()
        assert c.score("sigA") == pytest.approx(1.0)

    def test_reward_and_penalize(self):
        c = ar_rl.CuriosityMap()
        c.reward("sigA")
        assert c.score("sigA") == pytest.approx(2.0)
        for _ in range(10):
            c.penalize("sigA")
        assert c.score("sigA") == pytest.approx(0.1)  # floored

    def test_pick_prefers_curious(self):
        c = ar_rl.CuriosityMap()
        for _ in range(5):
            c.reward("hot")
            c.penalize("cold")
        rng = random.Random(11)
        picks = [c.pick(["hot", "cold"], rng) for _ in range(300)]
        assert picks.count("hot") > 200

    def test_prune(self):
        c = ar_rl.CuriosityMap()
        c.reward("gone")
        c.prune({"kept"})
        assert c.to_state()["scores"].get("gone") is None

    def test_state_roundtrip(self):
        c = ar_rl.CuriosityMap()
        c.reward("x")
        c2 = ar_rl.CuriosityMap(state=c.to_state())
        assert c2.score("x") == pytest.approx(2.0)


class TestFallback:
    def test_not_tripped_before_min_samples(self):
        m = ar_rl.FallbackMonitor()
        for _ in range(19):
            m.record(True, True)
        for _ in range(49):
            m.record(False, False)
        assert m.tripped() is False

    def test_tripped_when_scheduler_underperforms(self):
        m = ar_rl.FallbackMonitor()
        for _ in range(30):
            m.record(True, True)      # floor draws hit 100 percent
        for _ in range(50):
            m.record(False, False)    # scheduler draws hit 0 percent
        assert m.tripped() is True

    def test_not_tripped_when_scheduler_wins(self):
        m = ar_rl.FallbackMonitor()
        for _ in range(30):
            m.record(True, False)
        for _ in range(50):
            m.record(False, True)
        assert m.tripped() is False


class _FakeGenome:
    """Minimal stand-in with the six numeric genes."""
    def __init__(self):
        self.thr_margin = 0.0
        self.band_delta = 0.0
        self.cb_lr_mult = 1.0
        self.cb_iter_mult = 1.0
        self.cb_depth_delta = 0
        self.lookback_delta = 0


class TestCmaEmitter:
    def test_ask_stays_in_hull(self):
        e = ar_rl.CmaEmitter(rng=random.Random(5))
        e.seed_from(_FakeGenome())
        for _ in range(100):
            child = e.ask(_FakeGenome())
            assert 0.0 <= child.thr_margin <= 0.05
            assert -0.01 <= child.band_delta <= 0.02
            assert 0.5 <= child.cb_lr_mult <= 2.0
            assert 0.7 <= child.cb_iter_mult <= 1.5
            assert child.cb_depth_delta in (-2, -1, 0, 1, 2)
            assert -10 <= child.lookback_delta <= 10
            assert isinstance(child.cb_depth_delta, int)
            assert isinstance(child.lookback_delta, int)

    def test_ask_rounds_4_decimals(self):
        e = ar_rl.CmaEmitter(rng=random.Random(5))
        e.seed_from(_FakeGenome())
        child = e.ask(_FakeGenome())
        assert child.thr_margin == round(child.thr_margin, 4)

    def test_tell_moves_mean_toward_winners(self):
        e = ar_rl.CmaEmitter(rng=random.Random(5))
        e.seed_from(_FakeGenome())
        # feed lambda=6 evals: high thr_margin wins
        for i in range(6):
            x = [0.05 if i < 3 else 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            e.tell(x, fit=10.0 - i)
        assert e.to_state()["mean"][0] > 0.02

    def test_state_roundtrip(self):
        e = ar_rl.CmaEmitter(rng=random.Random(5))
        e.seed_from(_FakeGenome())
        e.tell([0.01, 0.0, 1.0, 1.0, 0.0, 0.0], 1.0)
        e2 = ar_rl.CmaEmitter(state=e.to_state(), rng=random.Random(5))
        assert e2.to_state()["mean"] == e.to_state()["mean"]
        assert len(e2.to_state()["evals"]) == 1


class TestNoveltyEmitter:
    def _setup(self):
        # 3 count bins x 2 groups; archive occupies (0,0) and (1,0)
        keys = ["0_0_0", "2_1_0"]  # floor_count_group keys: count bins 0 and 1, group 0
        elites = [
            {"genome": "gA", "fitness": 1.0},
            {"genome": "gB", "fitness": 2.0},
        ]
        counts = {"gA": 5, "gB": 15}
        return keys, elites, counts

    def test_targets_missing_projection(self):
        keys, elites, counts = self._setup()
        captured = {}

        def mutate_toward(parent, tbin, tgroup):
            captured["target"] = (tbin, tgroup)
            captured["parent"] = parent
            return "child"

        e = ar_rl.NoveltyEmitter(count_bins=3, groups=2, rng=random.Random(2))
        child = e.emit(keys, elites,
                       count_of=lambda g: counts[g],
                       group_of=lambda g: 0,
                       count_bin_of=lambda c: 0 if c < 10 else (1 if c < 20 else 2),
                       mutate_toward=mutate_toward)
        assert child == "child"
        # occupied projections: (0,0) and (1,0); target must be one of the rest
        assert captured["target"] in {(2, 0), (0, 1), (1, 1), (2, 1)}

    def test_none_when_grid_full(self):
        e = ar_rl.NoveltyEmitter(count_bins=1, groups=1, rng=random.Random(2))
        child = e.emit(["0_0_0"], [{"genome": "gA", "fitness": 1.0}],
                       count_of=lambda g: 5, group_of=lambda g: 0,
                       count_bin_of=lambda c: 0,
                       mutate_toward=lambda p, tb, tg: "child")
        assert child is None

    def test_gives_up_after_attempts(self):
        keys, elites, counts = self._setup()
        calls = []

        def mutate_toward(parent, tbin, tgroup):
            calls.append(1)
            return None

        e = ar_rl.NoveltyEmitter(count_bins=3, groups=2, rng=random.Random(2))
        child = e.emit(keys, elites,
                       count_of=lambda g: counts[g], group_of=lambda g: 0,
                       count_bin_of=lambda c: 0 if c < 10 else (1 if c < 20 else 2),
                       mutate_toward=mutate_toward)
        assert child is None
        assert len(calls) == 5
