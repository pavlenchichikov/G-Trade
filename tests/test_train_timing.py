"""Tests for train_timing.py (dataset-free parts use synthetic series)."""
import os

import numpy as np
import pytest

import train_timing as tt
from core import timing_policy as tp


def _series(n=120, seed=3):
    rng = np.random.default_rng(seed)
    probs = np.clip(0.5 + rng.normal(0, 0.08, n), 0.05, 0.95)
    next_ret = rng.normal(0.0005, 0.01, n)
    next_ret[-1] = np.nan
    return {
        "probs": probs, "next_ret": next_ret,
        "atr": np.full(n, 0.015), "taleb_hi": np.zeros(n, dtype=bool),
        "buy_thr": 0.55, "sell_thr": 0.45, "risky": False,
        "dates": np.arange(n),
    }


class TestEvalPolicy:
    def test_baseline_equals_default_policy(self):
        s = _series()
        base = tt.eval_baseline(s)
        pol = tt.eval_policy(s, tp.RulesPolicy(dict(tp.DEFAULT_PARAMS)))
        assert pol["score"] == pytest.approx(base["score"])
        assert pol["n_trades"] == base["n_trades"]

    def test_stricter_entry_trades_less(self):
        s = _series()
        strict = tt.eval_policy(
            s, tp.RulesPolicy({**tp.DEFAULT_PARAMS, "entry_margin": 0.08}))
        base = tt.eval_baseline(s)
        assert strict["n_trades"] <= base["n_trades"]

    def test_forex_costs_used(self):
        s = _series()
        s["risky"] = True
        s["is_forex"] = True
        r = tt.eval_baseline(s)
        s2 = _series()
        s2["is_forex"] = False
        r2 = tt.eval_baseline(s2)
        # same series, cheaper forex legs -> profit no worse
        assert r["profit"] >= r2["profit"]


class TestFitness:
    def test_median_minus_iqr(self):
        scores = [1.0, 2.0, 3.0, 4.0, 100.0]
        med = np.median(scores)
        iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
        assert tt.fitness(scores) == pytest.approx(med - 0.25 * iqr)

    def test_empty_is_minus_inf(self):
        assert tt.fitness([]) == float("-inf")


class TestSplitFitGate:
    def _series_by_asset(self, k=8, n=240):
        out = {}
        for i in range(k):
            s = _series(n=n, seed=i)
            out[f"A{i}"] = s
        return out

    def test_split_is_time_ordered(self):
        s = _series(n=100)
        tr, va, te = tt.split_series(s)
        assert len(tr["probs"]) == 60 and len(va["probs"]) == 20
        assert len(te["probs"]) == 20
        assert tr["buy_thr"] == s["buy_thr"]

    def test_fit_returns_valid_params(self):
        data = self._series_by_asset()
        params = tt.fit_policy(
            {a: tt.split_series(s)[0] for a, s in data.items()},
            budget=30, seed=7)
        for name, lo, hi, is_int in tp.PARAM_SPECS:
            assert lo <= params[name] <= hi
            if is_int:
                assert float(params[name]).is_integer()

    def test_gate_hold_on_noise(self):
        data = self._series_by_asset()
        test_slices = {a: tt.split_series(s)[2] for a, s in data.items()}
        verdict = tt.gate_policy(test_slices, dict(tp.DEFAULT_PARAMS))
        # default params ARE the baseline -> all deltas 0 -> never ADOPT
        assert verdict["verdict"] == "HOLD"

    def test_save_policy_only_on_adopt(self, tmp_path):
        p = str(tmp_path / "timing_policy.json")
        tt.save_policy(dict(tp.DEFAULT_PARAMS),
                       {"verdict": "HOLD", "per_asset": {}}, path=p)
        assert not os.path.exists(p)
        tt.save_policy(dict(tp.DEFAULT_PARAMS),
                       {"verdict": "ADOPT", "per_asset": {}}, path=p)
        assert os.path.exists(p)
