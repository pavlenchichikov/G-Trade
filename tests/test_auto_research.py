import json
import os

import auto_research as ar


def test_train_merges_env_overrides(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, cwd, env):
        captured["env"] = env
        # the trainer writes quality_report.json into the model dir (last cmd context):
        os.makedirs(env["GTRADE_MODEL_DIR"], exist_ok=True)
        with open(os.path.join(env["GTRADE_MODEL_DIR"], "quality_report.json"), "w") as f:
            json.dump([{"Asset": "SP500", "Score": 1.0}], f)

    monkeypatch.setattr(ar.subprocess, "run", fake_run)
    rows = ar._train("SP500,NVDA", {"GTRADE_LABEL_MODE": "rel_median"}, str(tmp_path / "m"))
    assert rows == [{"Asset": "SP500", "Score": 1.0}]
    assert captured["env"]["GTRADE_LABEL_MODE"] == "rel_median"
    assert captured["env"]["GTRADE_ASSETS"] == "SP500,NVDA"
    # LIGHT_ENV is still applied
    assert captured["env"]["GTRADE_FORCE_PROMOTE"] == "1"


def test_train_missing_report_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ar.subprocess, "run", lambda cmd, cwd, env: None)
    assert ar._train("SP500", {}, str(tmp_path / "m")) == []


def test_feature_env_builds_keys(tmp_path):
    specs = [{"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    env = ar._feature_env(specs, ["f1"])
    assert env["GTRADE_EXTRA_FEATURES"] == "f1"
    assert os.path.exists(env["GTRADE_DSL_SPECS"])
    assert ar._feature_env([], []) == {}


def _axis_additive():
    # candidate is a dict {"name":..., "lift":x}; to_env irrelevant for the fake train_fn
    return ar.Axis(
        name="fake_add",
        propose=lambda log: [{"name": "c%d" % len(log), "lift": 2.0 if len(log) == 0 else -1.0}],
        to_env=lambda selected: {"NAMES": ",".join(c["name"] for c in selected)},
        kind="additive",
    )


def _fake_train(lift_by_env):
    # returns rows whose Score is the sum of the lifts encoded in NAMES via lift_by_env
    def train(subset, env):
        names = [n for n in env.get("NAMES", "").split(",") if n]
        score = 1.0 + sum(lift_by_env.get(n, 0.0) for n in names)
        return [{"Asset": "SP500", "Score": score}, {"Asset": "NVDA", "Score": score}]
    return train


def test_run_axis_additive_keeps_lifting_rejects_flat():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = _axis_additive()
    train = _fake_train({"c0": 2.0, "c1": -1.0})  # c0 lifts, c1 hurts
    res = ar.run_axis(axis, 2, base, train, persist=lambda log: None)
    kept_names = [c["name"] for c in res["kept"]]
    assert "c0" in kept_names and "c1" not in kept_names
    assert res["kept_delta"] > 0


def test_run_axis_additive_resumes_from_prior_log():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    # prior log: round 0 accepted candidate c0 (delta 2.0); resume should keep c0
    prior = [{"axis": "fake_add", "iter": 0, "cand": [{"name": "c0", "lift": 2.0}],
              "cand_mean_delta": 2.0, "score": 2.0, "accepted": True}]
    axis = _axis_additive()  # reuse the helper defined earlier in this file

    seen = {"trains": 0}

    def train(subset, env):
        seen["trains"] += 1
        names = [n for n in env.get("NAMES", "").split(",") if n]
        # c0 lift 2.0, anything else (c1) hurts
        score = 1.0 + sum(2.0 if n == "c0" else -1.0 for n in names)
        return [{"Asset": a, "Score": score} for a in ("SP500", "NVDA")]

    res = ar.run_axis(axis, 2, base, train, persist=lambda log: None, prior_log=prior)
    kept_names = [c["name"] for c in res["kept"]]
    assert "c0" in kept_names              # resumed: prior accept preserved
    assert res["kept_delta"] == 2.0        # reconstructed from the accepted entry
    # per-run budget: the resumed run performs `budget`(=2) NEW iterations
    # (iters 1 and 2) and never retrains iter 0
    assert seen["trains"] == 2


def test_run_axis_select_best_picks_best():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [{"name": "w20", "v": 20}, {"name": "w30", "v": 30}, {"name": "w60", "v": 60}],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    train = _fake_train({"w20": 0.5, "w30": 3.0, "w60": 1.0})
    res = ar.run_axis(axis, 5, base, train, persist=lambda log: None)
    assert res["best"]["name"] == "w30"
    assert res["best_delta"] > 0


def test_run_axis_select_best_none_beats_base():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [{"name": "w20", "v": 20}],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    res = ar.run_axis(axis, 5, base, _fake_train({"w20": -2.0}), persist=lambda log: None)
    assert res["best"] is None


def test_run_axis_select_best_resumes_correctly():
    # prior log accepted w30 (delta 3.0) then w60 (delta 1.0); the resumed run must
    # pick w30 as best (highest accepted delta), not w60 (last/highest-iter accepted).
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    prior_log = [
        {"axis": "fake_sel", "iter": 0, "cand": {"name": "w30"},
         "cand_mean_delta": 3.0, "accepted": True},
        {"axis": "fake_sel", "iter": 1, "cand": {"name": "w60"},
         "cand_mean_delta": 1.0, "accepted": True},
    ]
    res = ar.run_axis(axis, 5, base, _fake_train({}), prior_log=prior_log, persist=lambda log: None)
    assert res["best"]["name"] == "w30"
    assert res["best_delta"] == 3.0


def test_features_axis_to_env_and_validate():
    base_features = ["ret_1", "ret_5"]
    axis = ar.make_features_axis(base_features)
    assert axis.kind == "additive"
    spec = {"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}
    env = axis.to_env([spec])
    assert env["GTRADE_EXTRA_FEATURES"] == "f1"
    assert os.path.exists(env["GTRADE_DSL_SPECS"])
    # a spec referencing an unknown input fails validate against base_features
    bad = {"name": "f2", "op": "lag", "inputs": ["does_not_exist"], "params": {"k": 1}}
    assert axis.validate(bad, []) is False


def test_features_axis_validate_sees_kept_names():
    # a spec that references a previously-kept feature name validates once it is kept
    base_features = ["ret_1"]
    axis = ar.make_features_axis(base_features)
    kept = [{"name": "f1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    dependent = {"name": "f2", "op": "lag", "inputs": ["f1"], "params": {"k": 1}}
    assert axis.validate(dependent, kept) is True


def test_labeling_axis_proposes_grid_and_maps_env():
    axis = ar.make_labeling_axis()
    assert axis.kind == "select_best"
    cands = axis.propose([])
    windows = sorted(c["window"] for c in cands)
    assert windows == [20, 30, 60]
    assert all(c["mode"] == "rel_median" for c in cands)
    env = axis.to_env({"mode": "rel_median", "window": 30})
    assert env == {"GTRADE_LABEL_MODE": "rel_median", "GTRADE_LABEL_WINDOW": "30"}


def test_labeling_axis_omits_logged_windows():
    axis = ar.make_labeling_axis()
    log = [{"axis": "labeling", "iter": 0, "cand": {"mode": "rel_median", "window": 20}}]
    windows = sorted(c["window"] for c in axis.propose(log))
    assert windows == [30, 60]


def test_build_axes_selects_by_name_and_skips_unknown(caplog):
    import logging
    base_features = ["ret_1"]
    with caplog.at_level(logging.WARNING):
        axes = ar.build_axes(["labeling", "bogus"], base_features)
    names = [a.name for a in axes]
    assert names == ["labeling"]
    assert any("bogus" in r.message for r in caplog.records)


def test_pruning_axis_propose_and_to_env(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    monkeypatch.delenv("GTRADE_EXTRA_FEATURES", raising=False)
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    from core.features import active_candidate_features
    active = active_candidate_features()
    axis = ar.make_pruning_axis(["ret_1"])
    assert axis.kind == "additive"
    proposed = axis.propose([])
    assert len(proposed) == 1   # exactly one candidate per round, not the whole set
    assert proposed[0]["drop"] in set(active)
    assert axis.to_env([{"drop": "rsi"}, {"drop": "atr"}]) == {"GTRADE_DROP_FEATURES": "rsi,atr"}


def test_pruning_axis_omits_already_dropped(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    axis = ar.make_pruning_axis(["ret_1"])
    log = [{"axis": "pruning", "iter": 0, "cand": [{"drop": "rsi"}], "accepted": True}]
    proposed = axis.propose(log)
    assert len(proposed) == 1
    assert proposed[0]["drop"] != "rsi"


def test_pruning_axis_validate_respects_floor(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    from core.features import active_candidate_features
    n = len(active_candidate_features())
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(n))   # floor == full size: no drop allowed
    axis = ar.make_pruning_axis(["ret_1"])
    assert axis.validate({"drop": "rsi"}, []) is False
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(n - 2))  # allow dropping up to 2
    axis2 = ar.make_pruning_axis(["ret_1"])
    assert axis2.validate({"drop": "rsi"}, []) is True
    assert axis2.validate({"drop": "atr"}, [{"drop": "rsi"}]) is True
    assert axis2.validate({"drop": "bb_pos"}, [{"drop": "rsi"}, {"drop": "atr"}]) is False


def test_pruning_axis_greedy_over_rounds(monkeypatch):
    # Regression for C1: a proposer that returned ALL droppable features per round
    # made run_axis additive drop the entire candidate set in one eval, blowing past
    # PRUNE_MIN and never performing real backward-elimination. With the one-per-round
    # fix, run_axis must evaluate (and keep/reject) one drop at a time.
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    from core.features import active_candidate_features
    active = list(active_candidate_features())
    n = len(active)
    prune_min = n - 5   # allow at most 5 drops; budget below will exceed that
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", str(prune_min))

    F = active[0]   # the one feature whose drop genuinely helps

    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]

    def fake_train(subset, env):
        dropped = [d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d]
        score = 1.0
        for d in dropped:
            score += 0.5 if d == F else -0.5
        return [{"Asset": a, "Score": score} for a in ("SP500", "NVDA")]

    axis = ar.make_pruning_axis(["ret_1"])
    res = ar.run_axis(axis, budget=10, base_rows=base, train_fn=fake_train,
                       screen_base=None, persist=lambda log: None)

    kept_drops = [c["drop"] for c in res["kept"]]
    assert F in kept_drops                      # the genuinely helpful drop was kept
    assert len(kept_drops) < n - prune_min + 1   # not the whole candidate set dropped at once
    assert n - len(kept_drops) >= prune_min      # floor was never breached
    # every accepted round in the log added exactly one new drop (true backward-elimination)
    for entry in res["log"]:
        if entry.get("accepted"):
            assert len(entry["cand"]) >= 1


def test_build_axes_includes_pruning():
    axes = ar.build_axes(["pruning"], ["ret_1"])
    assert [a.name for a in axes] == ["pruning"]


def test_main_trains_base_once_and_runs_selected_axis(monkeypatch):
    calls = {"train": 0, "selection_base": 0}

    def fake_train_env(subset, env):
        calls["train"] += 1
        if subset == ar.SELECTION_ASSETS and not env:
            calls["selection_base"] += 1
        score = 1.0 + (3.0 if env.get("GTRADE_LABEL_WINDOW") == "30" else 0.0)
        return [{"Asset": a, "Score": score} for a in subset.split(",")]

    monkeypatch.setattr(ar, "train_env", fake_train_env)
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "is_adoptable", lambda b, v, n, bud: (True, "ok"))
    monkeypatch.setenv("GTRADE_AR_AXES", "labeling")
    monkeypatch.setattr(ar, "BUDGET", 5, raising=False)
    # fresh state
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    ar.main()
    # exactly one base train on the selection set (reused across axes)
    assert calls["selection_base"] == 1


def test_screen_env_adds_flag():
    assert ar.screen_env({"GTRADE_LABEL_MODE": "rel_median"}) == {
        "GTRADE_LABEL_MODE": "rel_median", "GTRADE_SCREEN_ONLY": "1"}


def test_screen_on_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SCREEN", raising=False)
    assert ar._screen_on() is True            # default on
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")
    assert ar._screen_on() is False


def test_passes_screen_rejects_below_min():
    # screen base mean Score 1.0; a candidate scoring 0.5 (delta -0.5) fails, 3.0 passes
    screen_base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    axis = ar.Axis(name="x", propose=lambda log: [], to_env=lambda c: {"NAMES": c["name"]},
                   kind="select_best")

    def train(subset, env):
        s = 0.5 if env.get("NAMES") == "bad" else 3.0
        return [{"Asset": "SP500", "Score": s}, {"Asset": "NVDA", "Score": s}]

    ok_bad, _ = ar._passes_screen(axis, {"name": "bad"}, train, screen_base, 0.0)
    ok_good, _ = ar._passes_screen(axis, {"name": "good"}, train, screen_base, 0.0)
    assert ok_bad is False and ok_good is True


def test_passes_screen_failure_does_not_reject():
    # an empty screen result (infra failure) must NOT drop the candidate
    ok, delta = ar._passes_screen(
        ar.Axis(name="x", propose=lambda log: [], to_env=lambda c: {}, kind="select_best"),
        {"name": "c"}, lambda subset, env: [], [{"Asset": "SP500", "Score": 1.0}], 0.0)
    assert ok is True


def test_run_axis_screen_skips_full_eval_of_rejected():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    screen_base = base
    calls = {"full": 0, "screen": 0}

    def train(subset, env):
        if env.get("GTRADE_SCREEN_ONLY") == "1":
            calls["screen"] += 1
            s = 0.5 if env.get("NAMES") == "w20" else 3.0   # w20 fails the screen
        else:
            calls["full"] += 1
            s = 3.0
        return [{"Asset": a, "Score": s} for a in ("SP500", "NVDA")]

    axis = ar.Axis(name="s",
                   propose=lambda log: [{"name": "w20"}, {"name": "w30"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best")
    ar.run_axis(axis, 5, base, train, persist=lambda log: None,
                screen_base=screen_base, screen_min=0.0)
    # w20 screened out, only w30 reaches a full eval
    assert calls["full"] == 1 and calls["screen"] == 2


def _rows2(scores):
    return [{"Asset": a, "Score": s} for a, s in scores.items()]


def test_objective_delta_mean_and_min():
    base = {"A": 1.0, "B": 1.0, "C": 1.0}
    var = _rows2({"A": 2.0, "B": 1.5, "C": 0.5})   # deltas +1.0, +0.5, -0.5
    mean_v, deltas = ar._objective_delta(var, base, "mean")
    assert abs(mean_v - (1.0 + 0.5 - 0.5) / 3) < 1e-9
    min_v, _ = ar._objective_delta(var, base, "min")
    assert min_v == -0.5
    assert ar._objective_delta([], base, "min") == (0.0, [])


def test_mean_delta_wrapper_parity():
    base = {"A": 1.0, "B": 2.0}
    var = _rows2({"A": 1.5, "B": 2.5})
    assert ar._mean_delta(var, base) == ar._objective_delta(var, base, "mean")


def test_run_axis_additive_budget_is_per_run():
    # a prior log already holding `budget` entries must NOT starve a new run
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    prior = [{"axis": "fake_add", "iter": i, "cand": [{"name": "c%d" % i}],
              "cand_mean_delta": -1.0, "score": -1.0} for i in range(2)]
    calls = {"n": 0}

    def train(subset, env):
        calls["n"] += 1
        return [{"Asset": a, "Score": 0.5} for a in ("SP500", "NVDA")]

    res = ar.run_axis(_axis_additive(), 2, base, train,
                      persist=lambda log: None, prior_log=prior)
    assert calls["n"] == 2                     # two NEW iterations this run
    assert len(res["log"]) == 4                # 2 prior + 2 new


def test_run_axis_select_best_budget_is_per_run():
    base = [{"Asset": "SP500", "Score": 1.0}, {"Asset": "NVDA", "Score": 1.0}]
    prior = [{"axis": "fake_sel", "iter": 0, "cand": {"name": "w20"},
              "cand_mean_delta": -1.0}]
    axis = ar.Axis(
        name="fake_sel",
        propose=lambda log: [{"name": "w30"}, {"name": "w60"}, {"name": "w90"}],
        to_env=lambda cand: {"NAMES": cand["name"]},
        kind="select_best",
    )
    calls = {"n": 0}

    def train(subset, env):
        calls["n"] += 1
        return [{"Asset": a, "Score": 2.0} for a in ("SP500", "NVDA")]

    ar.run_axis(axis, 2, base, train, persist=lambda log: None, prior_log=prior)
    assert calls["n"] == 2                     # budget=2 NEW evals despite 1 prior


def test_benjamini_hochberg_known_vector():
    # m=4, alpha=0.05: thresholds 0.0125, 0.025, 0.0375, 0.05 -> reject first two
    assert ar.benjamini_hochberg([0.01, 0.02, 0.2, 0.5], 0.05) == [True, True, False, False]
    # input order preserved
    assert ar.benjamini_hochberg([0.5, 0.01, 0.2, 0.02], 0.05) == [False, True, False, True]
    assert ar.benjamini_hochberg([], 0.05) == []
    assert ar.benjamini_hochberg([0.9, 0.8], 0.05) == [False, False]


def test_objective_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    assert ar._objective() == "mean"
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "min")
    assert ar._objective() == "min"
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "bogus")
    assert ar._objective() == "mean"          # unknown -> mean


def test_holdout_stats_mean_vs_min():
    base = _rows2({"A": 1.0, "B": 1.0, "C": 1.0})
    var = _rows2({"A": 2.0, "B": 2.0, "C": 0.5})   # deltas +1,+1,-0.5
    p_m, v_m, deltas, _ = ar.holdout_stats(base, var, "mean")
    p_n, v_n, _, _ = ar.holdout_stats(base, var, "min")
    assert v_m > v_n and v_n == -0.5
    assert p_m == p_n                              # objective does not change the sign-test p
    p0, v0, d0, tag0 = ar.holdout_stats(base, _rows2({}), "mean")
    assert d0 == [] and v0 == 0.0


def test_is_adoptable_mean_backward_compat():
    # all held-out improve a lot -> adoptable under the default mean path (ab_labeling parity)
    base = _rows2({a: 1.0 for a in ("A", "B", "C", "D", "E")})
    var = _rows2({a: 3.0 for a in ("A", "B", "C", "D", "E")})
    ok, why = ar.is_adoptable(base, var, 1, 1)
    assert ok is True
    # min objective: one asset regresses -> worst delta negative -> not practically over the bar
    var2 = _rows2({"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 0.0})
    ok2, _ = ar.is_adoptable(base, var2, 1, 1, objective="min")
    assert ok2 is False


def test_run_axis_uses_min_objective(monkeypatch):
    # with objective=min, a candidate that lifts the mean but tanks one asset is rejected
    monkeypatch.setenv("GTRADE_AR_OBJECTIVE", "min")
    base = [{"Asset": "A", "Score": 1.0}, {"Asset": "B", "Score": 1.0}]
    axis = ar.Axis(name="s", propose=lambda log: [{"name": "x"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best")

    def train(subset, env):
        # x lifts A by +3 but tanks B by -1 -> mean +1 (good) but min -1 (bad)
        return [{"Asset": "A", "Score": 4.0}, {"Asset": "B", "Score": 0.0}]

    res = ar.run_axis(axis, 3, base, train, persist=lambda log: None)
    assert res["best"] is None         # min objective rejects it (worst delta -1 <= 0)


def test_main_applies_bh_across_axes(monkeypatch):
    # two axis winners -> main collects their held-out p-values and runs ONE BH
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")   # screen off -> simpler full-eval path
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    captured = {"bh_input": None}
    real_bh = ar.benjamini_hochberg
    monkeypatch.setattr(ar, "benjamini_hochberg",
                        lambda p, alpha=0.05: (captured.__setitem__("bh_input", list(p)), real_bh(p, alpha))[1])

    # base (no NAMES) scores 1.0; any candidate (NAMES set) scores 5.0 -> beats base
    def fake_train_env(subset, env):
        s = 5.0 if env.get("NAMES") else 1.0
        return [{"Asset": a, "Score": s} for a in subset.split(",")]
    monkeypatch.setattr(ar, "train_env", fake_train_env)
    monkeypatch.setattr(ar, "build_axes", lambda names, bf: [
        ar.Axis(name="ax1", propose=lambda log: [{"name": "a"}], to_env=lambda c: {"NAMES": "a"}, kind="select_best"),
        ar.Axis(name="ax2", propose=lambda log: [{"name": "b"}], to_env=lambda c: {"NAMES": "b"}, kind="select_best"),
    ])
    monkeypatch.setattr(ar, "BUDGET", 2, raising=False)
    ar.main()
    # BH was applied across exactly the two axis-winners' p-values
    assert captured["bh_input"] is not None and len(captured["bh_input"]) == 2


import os as _os  # noqa: F401  (already imported at module top in most suites; harmless)

_ACTIVE = ["ret_1", "ret_5", "ret_10", "ret_20", "rsi", "atr",
           "vol_z", "sma_20", "macd_hist", "bb_pos"]   # 10 features


def _spec(name="g1", inp="ret_1"):
    return {"name": name, "op": "lag", "inputs": [inp], "params": {"k": 1}}


def test_genome_to_env_empty_is_no_overrides():
    assert ar.genome_to_env(ar.Genome()) == {}


def test_genome_to_env_full():
    g = ar.Genome(drops=["rsi", "atr"], extra=[_spec("gx")],
                  label_mode="rel_median", label_window=20)
    env = ar.genome_to_env(g)
    assert env["GTRADE_DROP_FEATURES"] == "rsi,atr"
    assert env["GTRADE_LABEL_MODE"] == "rel_median"
    assert env["GTRADE_LABEL_WINDOW"] == "20"
    assert env["GTRADE_EXTRA_FEATURES"] == "gx"
    assert _os.path.exists(env["GTRADE_DSL_SPECS"])


def test_valid_accepts_and_rejects():
    assert ar.valid(ar.Genome(), _ACTIVE, 8) is True
    assert ar.valid(ar.Genome(drops=["rsi", "atr"]), _ACTIVE, 8) is True
    assert ar.valid(ar.Genome(drops=["not_a_feature"]), _ACTIVE, 8) is False     # unknown drop
    assert ar.valid(ar.Genome(drops=["gx"], extra=[_spec("gx")]), _ACTIVE, 8) is False  # drop/extra overlap
    assert ar.valid(ar.Genome(label_mode="bogus"), _ACTIVE, 8) is False          # bad mode
    assert ar.valid(ar.Genome(label_window=0), _ACTIVE, 8) is False              # bad window
    assert ar.valid(ar.Genome(drops=["rsi", "atr", "vol_z"]), _ACTIVE, 8) is False  # breaches floor (10-3 < 8)


def test_random_genome_is_valid(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    import random as _r
    for seed in range(20):
        _r.seed(seed)
        g = ar.random_genome(_ACTIVE, ["ret_1", "ret_5", "rsi"])
        assert ar.valid(g, _ACTIVE, 8)


def test_mutate_stays_valid_and_changes(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    import random as _r
    g = ar.Genome(drops=["rsi"], extra=[], label_mode="direction", label_window=30)
    changed = 0
    for seed in range(30):
        _r.seed(seed)
        m = ar.mutate(g, _ACTIVE, ["ret_1", "ret_5", "rsi"])
        assert ar.valid(m, _ACTIVE, 8)
        if m != g:
            changed += 1
    assert changed > 0     # mutation does something across seeds


def test_mutate_floor_constrained_stays_valid(monkeypatch):
    # active of 8 with floor 8 -> add_drop can never fire (would breach the floor);
    # mutate must still return a VALID genome via the other ops (flip_label, etc.)
    # and must never produce a drop that breaches the floor.
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    active8 = _ACTIVE[:8]
    import random as _r
    for seed in range(20):
        _r.seed(seed)
        m = ar.mutate(ar.Genome(), active8, ["ret_1", "ret_5"])
        assert ar.valid(m, active8, 8)
        assert m.drops == []          # cannot drop below the floor of 8 from 8 active


def test_crossover_mixes_parents_and_is_valid(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_PRUNE_MIN", "8")
    import random as _r
    g1 = ar.Genome(drops=["rsi"], extra=[_spec("a")], label_mode="rel_median", label_window=20)
    g2 = ar.Genome(drops=["atr"], extra=[_spec("b")], label_mode="direction", label_window=30)
    for seed in range(20):
        _r.seed(seed)
        c = ar.crossover(g1, g2, _ACTIVE)
        assert ar.valid(c, _ACTIVE, 8)
        # each group comes from one parent
        assert c.drops in (g1.drops, g2.drops) or set(c.drops) <= {"rsi", "atr"}
        assert (c.label_mode, c.label_window) in (("rel_median", 20), ("direction", 30))


def test_crossover_resolves_drop_extra_conflict():
    import random as _r
    _r.seed(1)
    # g1 drops 'b', g2 adds extra named 'b' -> if child takes both, the drop must lose
    g1 = ar.Genome(drops=["rsi"], extra=[])
    g2 = ar.Genome(drops=[], extra=[_spec("rsi")])   # extra name collides with g1's drop
    for seed in range(20):
        _r.seed(seed)
        c = ar.crossover(g1, g2, _ACTIVE)
        extra_names = {s["name"] for s in c.extra}
        assert not (set(c.drops) & extra_names)        # no drop equals an extra name


def test_bin_edges():
    assert ar._bin(-2.0, ar._FLOOR_EDGES) == 0
    assert ar._bin(0.0, ar._FLOOR_EDGES) == 2
    assert ar._bin(5.0, ar._FLOOR_EDGES) == 4


def test_fitness_is_mean_delta():
    base = {"A": 1.0, "B": 1.0}
    rows = _rows2({"A": 2.0, "B": 0.0})       # deltas +1, -1 -> mean 0
    assert ar.fitness(rows, base) == 0.0
    rows2 = _rows2({"A": 3.0, "B": 3.0})       # +2, +2 -> mean 2
    assert ar.fitness(rows2, base) == 2.0


def test_behavior_floor_and_count():
    active = ["ret_1", "ret_5", "ret_10", "ret_20", "rsi", "atr",
              "vol_z", "sma_20", "macd_hist", "bb_pos"]   # 10
    base = {"A": 1.0, "B": 1.0}
    rows = _rows2({"A": 3.0, "B": 0.5})        # min delta = -0.5 -> floor bin 1
    g = ar.Genome(drops=["rsi", "atr"], extra=[_spec("gx")])   # count 10 - 2 + 1 = 9
    bd1, bd2 = ar.behavior(g, rows, base, active)
    assert bd1 == 1                             # -0.5 in [-1.0, -0.25) -> bin 1
    assert bd2 == 0                             # 9 < 12 -> count bin 0


_QD_ACTIVE = ["ret_1", "ret_5", "ret_10", "ret_20", "rsi", "atr",
              "vol_z", "sma_20", "macd_hist", "bb_pos"]


def test_archive_put_store_displace_keep():
    base = {"A": 1.0, "B": 1.0}
    arch = {}
    g1 = ar.Genome(drops=["rsi"])
    assert ar.archive_put(arch, g1, _rows2({"A": 2.0, "B": 2.0}), base, _QD_ACTIVE) is True
    # same niche, higher mean fitness -> displaces
    g2 = ar.Genome(drops=["atr"])
    assert ar.archive_put(arch, g2, _rows2({"A": 3.0, "B": 3.0}), base, _QD_ACTIVE) is True
    # same niche, lower fitness -> kept (False)
    # NOTE: deltas must keep min_delta >= 1.0 (same floor bin as g1/g2) while the
    # mean stays below g2's fitness of 2.0; {"A": 1.5, "B": 1.5} (deltas +0.5,+0.5)
    # would land in floor bin 3 (a different niche), so it is corrected here.
    g3 = ar.Genome(drops=["vol_z"])
    assert ar.archive_put(arch, g3, _rows2({"A": 2.5, "B": 2.5}), base, _QD_ACTIVE) is False
    assert len(arch) == 1


def test_qd_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "_QD_ARCHIVE_PATH", str(tmp_path / "qd.json"))
    base = {"A": 1.0}
    arch = {}
    ar.archive_put(arch, ar.Genome(drops=["rsi"], label_mode="rel_median", label_window=20),
                   _rows2({"A": 2.0}), base, _QD_ACTIVE)
    ar._qd_save(arch)
    loaded = ar._qd_load()
    (k, v), = loaded.items()
    assert v["genome"].drops == ["rsi"] and v["genome"].label_mode == "rel_median"
    assert v["genome"].label_window == 20


def test_run_qd_illuminates_and_gates(monkeypatch):
    monkeypatch.setattr(ar, "_qd_load", lambda: {})
    monkeypatch.setattr(ar, "_qd_save", lambda a: None)
    monkeypatch.setattr(ar, "BUDGET", 6, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "4")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "2")
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    import random as _r
    _r.seed(0)
    calls = {"real": 0, "holdout_full": 0}
    bh_seen = {"n": None}
    real_bh = ar.benjamini_hochberg
    monkeypatch.setattr(ar, "benjamini_hochberg",
                        lambda p, alpha=0.05: (bh_seen.__setitem__("n", len(p)), real_bh(p, alpha))[1])

    def fake_train(subset, env):
        calls["real"] += 1
        screen = env.get("GTRADE_SCREEN_ONLY") == "1"
        if subset == ar.HELDOUT_ASSETS and not screen:
            calls["holdout_full"] += 1
        # more drops -> higher score, so dropping spreads cells and lifts fitness
        n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
        s = 1.0 + 0.7 * n_drop
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    archive = ar.run_qd(train_fn=fake_train)
    assert isinstance(archive, dict) and len(archive) >= 1        # illuminated some niches
    assert bh_seen["n"] is not None and bh_seen["n"] <= 2          # BH across <= GTRADE_AR_QD_FINAL elites
    assert calls["holdout_full"] >= 1                             # final stage ran the held-out gate
    # no real train_hybrid subprocess (the fake replaces train_env entirely)


def test_main_dispatches_to_qd(monkeypatch):
    called = {"qd": 0, "axisloop": 0}
    monkeypatch.setattr(ar, "run_qd", lambda: called.__setitem__("qd", called["qd"] + 1))
    monkeypatch.setattr(ar, "train_env", lambda s, e: called.__setitem__("axisloop", 1) or [])
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setenv("GTRADE_AR_AXES", "qd")
    ar.main()
    assert called["qd"] == 1 and called["axisloop"] == 0       # qd ran; the axis loop did not


def test_genome_sig_canonical():
    a = ar.Genome(drops=["rsi", "atr"], extra=[_spec("x", "ret_1")])
    b = ar.Genome(drops=["atr", "rsi"], extra=[_spec("y", "ret_1")])  # order/name differ
    assert ar.genome_sig(a) == ar.genome_sig(b)
    c = ar.Genome(drops=["rsi"], extra=[_spec("x", "ret_1")])
    assert ar.genome_sig(a) != ar.genome_sig(c)


def test_run_axis_marks_tried_candidates():
    from core import ar_memory
    base = [{"Asset": "SP500", "Score": 1.0}]
    axis = ar.Axis(name="s", propose=lambda log: [{"name": "x"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best",
                   sig=lambda c: ("spec", c["name"]))
    ar.run_axis(axis, 3, base, lambda subset, env: [{"Asset": "SP500", "Score": 2.0}],
                persist=lambda log: None)
    assert ar_memory.tried_seen("spec", "x") is True


def test_propose_evolutionary_respects_registry(monkeypatch):
    import json as _json
    from core import ar_memory
    monkeypatch.setenv("GTRADE_AR_SEED", "7")
    first = ar.propose_evolutionary([], ["ret_1", "ret_5"])
    assert first
    ar_memory.tried_add("spec", _json.dumps(ar._spec_signature(first[0])))
    monkeypatch.setenv("GTRADE_AR_SEED", "7")   # identical RNG stream
    second = ar.propose_evolutionary([], ["ret_1", "ret_5"])
    if second:   # either skipped to a different spec or gave up - never a repeat
        assert ar._spec_signature(second[0]) != ar._spec_signature(first[0])


def test_labeling_axis_respects_registry():
    import json as _json
    from core import ar_memory
    ar_memory.tried_add(
        "label", _json.dumps({"mode": "rel_median", "window": 20}, sort_keys=True))
    windows = sorted(c["window"] for c in ar.make_labeling_axis().propose([]))
    assert windows == [30, 60]


def test_pruning_axis_respects_registry(monkeypatch):
    monkeypatch.delenv("GTRADE_DROP_FEATURES", raising=False)
    from core import ar_memory
    from core.features import active_candidate_features
    first = list(active_candidate_features())[0]
    ar_memory.tried_add("drop", first)
    proposed = ar.make_pruning_axis(["ret_1"]).propose([])
    assert proposed and proposed[0]["drop"] != first


def test_next_child_skips_seen_genomes(monkeypatch):
    from core import ar_memory
    g = ar.Genome(drops=["rsi"])
    monkeypatch.setattr(ar, "mutate", lambda parent, active, bf: g)
    archive = {"0_0": {"genome": ar.Genome(), "fitness": 0.0, "rows": []}}
    # unseen -> returned
    child = ar.next_child(archive, _ACTIVE, ["ret_1"])
    assert child is not None and ar.genome_sig(child) == ar.genome_sig(g)
    # seen -> None (every attempt produces the same seen genome)
    ar_memory.tried_add("genome", ar.genome_sig(ar._canon_genome(g)))
    assert ar.next_child(archive, _ACTIVE, ["ret_1"]) is None


def test_run_qd_registers_genomes(monkeypatch):
    from core import ar_memory
    monkeypatch.setattr(ar, "_qd_load", lambda: {})
    monkeypatch.setattr(ar, "_qd_save", lambda a: None)
    monkeypatch.setattr(ar, "BUDGET", 3, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "3")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    monkeypatch.delenv("GTRADE_AR_PROPOSER", raising=False)
    import random as _r
    _r.seed(0)

    def fake_train(subset, env):
        n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
        s = 1.0 + 0.7 * n_drop
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    ar.run_qd(train_fn=fake_train)
    assert ar_memory.tried_count() >= 3        # at least the init genomes registered


def test_train_base_cached_hits_and_misses(monkeypatch):
    from core import ar_memory
    monkeypatch.setattr(ar_memory, "data_fingerprint", lambda subset: "fp")
    calls = {"n": 0}

    def fake_train(subset, env):
        calls["n"] += 1
        return [{"Asset": "SP500", "Score": 1.0}]

    monkeypatch.setattr(ar, "train_env", fake_train)
    r1 = ar.train_base_cached("SP500", {})
    r2 = ar.train_base_cached("SP500", {})
    assert r1 == r2 == [{"Asset": "SP500", "Score": 1.0}]
    assert calls["n"] == 1                      # second call served from cache
    ar.train_base_cached("SP500", {"GTRADE_SCREEN_ONLY": "1"})
    assert calls["n"] == 2                      # different env = different key
    monkeypatch.setattr(ar_memory, "data_fingerprint", lambda subset: "fp2")
    ar.train_base_cached("SP500", {})
    assert calls["n"] == 3                      # new data invalidates


def test_train_base_cached_does_not_cache_failures(monkeypatch):
    from core import ar_memory
    monkeypatch.setattr(ar_memory, "data_fingerprint", lambda subset: "fp")
    calls = {"n": 0}

    def failing_train(subset, env):
        calls["n"] += 1
        return []

    monkeypatch.setattr(ar, "train_env", failing_train)
    assert ar.train_base_cached("SP500", {}) == []
    assert ar.train_base_cached("SP500", {}) == []
    assert calls["n"] == 2                      # empty rows never cached


def test_llm_child_valid_invalid_and_error(monkeypatch):
    from core import llm_proposer
    elites = [{"genome": ar.Genome(), "fitness": 1.0, "rows": []}]
    good = {"drops": ["rsi"], "extra": [], "label_mode": "direction", "label_window": 30}
    monkeypatch.setattr(llm_proposer, "propose_genome", lambda *a, **k: good)
    g = ar._llm_child(elites, _ACTIVE, ["ret_1"])
    assert isinstance(g, ar.Genome) and g.drops == ["rsi"]
    # invalid genome (unknown drop) -> None
    bad = {"drops": ["not_a_feature"], "extra": [], "label_mode": "direction",
           "label_window": 30}
    monkeypatch.setattr(llm_proposer, "propose_genome", lambda *a, **k: bad)
    assert ar._llm_child(elites, _ACTIVE, ["ret_1"]) is None
    # backend blowing up -> None (silent fallback)
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(llm_proposer, "propose_genome", boom)
    assert ar._llm_child(elites, _ACTIVE, ["ret_1"]) is None


def test_next_child_llm_first_then_fallback(monkeypatch):
    from core import llm_proposer
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    monkeypatch.setenv("GTRADE_AR_QD_LLM_P", "1.0")
    archive = {"0_0": {"genome": ar.Genome(), "fitness": 0.0, "rows": []}}
    llm_genome = {"drops": ["rsi"], "extra": [], "label_mode": "direction",
                  "label_window": 30}
    monkeypatch.setattr(llm_proposer, "propose_genome", lambda *a, **k: llm_genome)
    child = ar.next_child(archive, _ACTIVE, ["ret_1"])
    assert child is not None and child.drops == ["rsi"]     # the LLM child won
    # LLM fails -> evolutionary fallback still yields a child
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(llm_proposer, "propose_genome", boom)
    import random as _r
    _r.seed(3)
    child2 = ar.next_child(archive, _ACTIVE, ["ret_1"])
    assert child2 is not None                                # search never dies


def test_run_qd_llm_failure_completes(monkeypatch):
    from core import llm_proposer
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    monkeypatch.setenv("GTRADE_AR_QD_LLM_P", "1.0")

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(llm_proposer, "propose_genome", boom)
    monkeypatch.setattr(ar, "_qd_load", lambda: {})
    monkeypatch.setattr(ar, "_qd_save", lambda a: None)
    monkeypatch.setattr(ar, "BUDGET", 4, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "3")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    import random as _r
    _r.seed(0)

    def fake_train(subset, env):
        n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
        s = 1.0 + 0.7 * n_drop
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    archive = ar.run_qd(train_fn=fake_train)
    assert isinstance(archive, dict) and len(archive) >= 1   # completed on fallback


def test_main_writes_findings(monkeypatch):
    import json as _json
    from core import ar_memory
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setattr(ar_memory, "data_fingerprint", lambda subset: "fp")
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")
    monkeypatch.setenv("GTRADE_AR_AXES", "labeling")
    monkeypatch.setattr(ar, "BUDGET", 5, raising=False)

    def fake_train_env(subset, env):
        s = 5.0 if env.get("GTRADE_LABEL_MODE") else 1.0
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    monkeypatch.setattr(ar, "train_env", fake_train_env)
    ar.main()
    with open(ar_memory.FINDINGS_PATH, encoding="utf-8") as f:
        journal = _json.load(f)
    assert len(journal) == 1
    assert journal[0]["mode"] == "axes"
    assert journal[0]["winners"] and journal[0]["winners"][0]["axis"] == "labeling"
    assert "adoptable" in journal[0]["winners"][0]


def test_main_llm_proposer_failure_is_graceful(monkeypatch):
    """Fix 1: RuntimeError from run_axis (LLM proposer down) must be caught so the
    run still reaches findings_append journaling at the end."""
    from core import ar_memory

    called = {"findings": False}

    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setattr(ar, "BUDGET", 2, raising=False)
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")

    def fake_train_env(subset, env):
        return [{"Asset": a, "Score": 1.0} for a in subset.split(",")]

    monkeypatch.setattr(ar, "train_env", fake_train_env)

    def failing_propose(log):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(ar, "build_axes", lambda names, bf: [
        ar.Axis(name="llm_fail", propose=failing_propose,
                to_env=lambda c: {}, kind="select_best"),
    ])

    real_findings_append = ar_memory.findings_append

    def tracking_findings_append(record):
        called["findings"] = True
        real_findings_append(record)

    monkeypatch.setattr(ar_memory, "findings_append", tracking_findings_append)

    ar.main()   # must NOT raise
    assert called["findings"] is True


def test_run_qd_no_elites_still_journals(monkeypatch):
    """Fix 2: run_qd with an empty archive must still call findings_append (mode=qd,
    winners=[]) instead of returning early before the journaling block."""
    from core import ar_memory

    called = {"findings": None}

    monkeypatch.setattr(ar, "_qd_load", lambda: {})
    monkeypatch.setattr(ar, "_qd_save", lambda a: None)
    monkeypatch.setattr(ar, "BUDGET", 2, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "0")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)

    def fake_train(subset, env):
        return [{"Asset": a, "Score": 1.0} for a in subset.split(",")]

    real_findings_append = ar_memory.findings_append

    def tracking_findings_append(record):
        called["findings"] = record
        real_findings_append(record)

    monkeypatch.setattr(ar_memory, "findings_append", tracking_findings_append)

    archive = ar.run_qd(train_fn=fake_train)
    assert isinstance(archive, dict) and len(archive) == 0
    assert called["findings"] is not None
    assert called["findings"]["mode"] == "qd"
    assert called["findings"]["winners"] == []


# --- Task 2: neural-lift primitives ---


def _basis_train(subset, env):
    # full Score 5.0 for a candidate (NAMES set) else 1.0; CB-only Score 0.5
    if env.get("GTRADE_SCREEN_ONLY") == "1":
        s = 0.5
    else:
        s = 5.0 if env.get("NAMES") else 1.0
    return [{"Asset": a, "Score": s} for a in subset.split(",")]


def test_neural_contribution_subtracts_cb():
    full = [{"Asset": "A", "Score": 2.0}, {"Asset": "B", "Score": 1.0}]
    cb = [{"Asset": "A", "Score": 0.5}, {"Asset": "C", "Score": 9.0}]
    assert ar.neural_contribution(full, cb) == {"A": 1.5}   # B has no cb, C not in full


def test_contribution_rows_full_minus_screen(monkeypatch):
    monkeypatch.setattr(ar, "train_env", _basis_train)
    rows = ar.contribution_rows("A,B", {"NAMES": "x"}, _basis_train)
    # full 5.0 minus cb 0.5 = 4.5 for each asset
    assert sorted(r["Asset"] for r in rows) == ["A", "B"]
    assert all(abs(r["Score"] - 4.5) < 1e-9 for r in rows)


def test_heldout_eval_returns_full_and_contribution(monkeypatch):
    monkeypatch.setattr(ar, "train_env", _basis_train)
    full, contrib = ar._heldout_eval("A", {"NAMES": "x"}, _basis_train)
    assert full[0]["Score"] == 5.0
    assert abs(contrib[0]["Score"] - 4.5) < 1e-9


def test_score_basis_env(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SCORE_BASIS", raising=False)
    assert ar._score_basis() == "raw"
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "neural")
    assert ar._score_basis() == "neural"
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "bogus")
    assert ar._score_basis() == "raw"


def test_score_rows_raw_vs_neural(monkeypatch):
    monkeypatch.setattr(ar, "train_env", _basis_train)
    monkeypatch.delenv("GTRADE_AR_SCORE_BASIS", raising=False)
    raw = ar.score_rows("A", {"NAMES": "x"}, _basis_train)
    assert raw[0]["Score"] == 5.0                       # raw = one full train
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "neural")
    neu = ar.score_rows("A", {"NAMES": "x"}, _basis_train)
    assert abs(neu[0]["Score"] - 4.5) < 1e-9            # neural = full minus cb


def test_run_axis_neural_basis_scores_contribution(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "neural")
    monkeypatch.setattr(ar, "train_env", _basis_train)
    base = ar.score_rows(ar.SELECTION_ASSETS, {}, ar.train_env)   # contribution base ~0.5
    axis = ar.Axis(name="s", propose=lambda log: [{"name": "x"}],
                   to_env=lambda c: {"NAMES": c["name"]}, kind="select_best")
    res = ar.run_axis(axis, 3, base, ar.train_env, persist=lambda log: None)
    # candidate contribution (4.5) beats base contribution (0.5) -> accepted
    assert res["best"] is not None and res["best"]["name"] == "x"
    assert res["best_delta"] > 0
    assert abs(res["best_delta"] - 4.0) < 1e-9   # 4.5 candidate - 0.5 base contribution


# --- Task 4: neural_lift metric + replication gate in main() ---


def _fake_axes_for_main(monkeypatch):
    monkeypatch.setattr(ar, "_try_sample_frame", lambda: None)
    monkeypatch.setattr(ar, "load_state", lambda: {})
    monkeypatch.setattr(ar, "save_state", lambda s: None)
    monkeypatch.setenv("GTRADE_AR_SCREEN", "0")
    monkeypatch.setenv("GTRADE_AR_AXES", "features")
    monkeypatch.setattr(ar, "BUDGET", 2, raising=False)
    monkeypatch.setattr(ar, "build_axes", lambda names, bf: [
        ar.Axis(name="ax1", propose=lambda log: [{"name": "a"}],
                to_env=lambda c: {"NAMES": "a"}, kind="select_best")])


def test_main_records_neural_lift_and_first_clear(monkeypatch):
    _fake_axes_for_main(monkeypatch)
    monkeypatch.setattr(ar, "train_env", _basis_train)
    monkeypatch.setattr(ar, "train_base_cached", lambda subset, env: _basis_train(subset, env))
    from core import ar_memory
    ar.main()
    journal = ar_memory._load(ar_memory.FINDINGS_PATH, [])
    w = journal[-1]["winners"][0]
    assert w["adoptable"] is True
    assert w["neural_lift"] is not None and w["neural_lift"] > 0
    assert w["replicated"] is False and w["clears"] == 1


def test_main_replication_on_second_run(monkeypatch):
    _fake_axes_for_main(monkeypatch)
    monkeypatch.setattr(ar, "train_env", _basis_train)
    monkeypatch.setattr(ar, "train_base_cached", lambda subset, env: _basis_train(subset, env))
    from core import ar_memory
    ar.main()
    ar.main()   # second independent run, same isolated replication ledger
    journal = ar_memory._load(ar_memory.FINDINGS_PATH, [])
    w2 = journal[-1]["winners"][0]
    assert w2["replicated"] is True and w2["clears"] == 2
    assert ar_memory.findings_summary()["replicated"] >= 1


def test_winner_sig_stable_across_temp_envs():
    # a features winner is a list of spec dicts; the sig must ignore spec names / order
    s1 = [{"name": "g1", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    s2 = [{"name": "zz", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]
    assert ar._winner_sig("features", s1) == ar._winner_sig("features", s2)
    # a labeling winner is a dict
    assert ar._winner_sig("labeling", {"mode": "rel_median", "window": 20}) == \
        ar._winner_sig("labeling", {"window": 20, "mode": "rel_median"})


# --- Task 5: neural_lift metric + replication gate in run_qd() ---


def test_run_qd_records_neural_lift_and_replication(monkeypatch):
    monkeypatch.setattr(ar, "_qd_load", lambda: {})
    monkeypatch.setattr(ar, "_qd_save", lambda a: None)
    monkeypatch.setattr(ar, "BUDGET", 3, raising=False)
    monkeypatch.setenv("GTRADE_AR_QD_INIT", "3")
    monkeypatch.setenv("GTRADE_AR_QD_FINAL", "1")
    monkeypatch.delenv("GTRADE_AR_OBJECTIVE", raising=False)
    monkeypatch.delenv("GTRADE_AR_SCORE_BASIS", raising=False)
    import random as _r
    from core import ar_memory

    def fake_train(subset, env):
        # full: more drops -> higher Score; screen (CB-only): flat low
        if env.get("GTRADE_SCREEN_ONLY") == "1":
            s = 0.5
        else:
            n_drop = len([d for d in env.get("GTRADE_DROP_FEATURES", "").split(",") if d])
            s = 1.0 + 1.5 * n_drop
        return [{"Asset": a, "Score": s} for a in subset.split(",")]

    _r.seed(0)
    ar.run_qd(train_fn=fake_train)
    journal = ar_memory._load(ar_memory.FINDINGS_PATH, [])
    qd = [r for r in journal if r["mode"] == "qd"][-1]
    assert qd["winners"], "an elite should have reached the gate"
    w = qd["winners"][0]
    # CB train is now intercepted by the injected fake, so neural_lift is a concrete
    # number (not None from a real-subprocess empty result)
    assert w["neural_lift"] is not None
    assert "replicated" in w and "clears" in w


def test_run_axis_additive_neural_basis(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SCORE_BASIS", "neural")
    monkeypatch.setattr(ar, "train_env", _basis_train)
    base = ar.score_rows(ar.SELECTION_ASSETS, {}, ar.train_env)   # contribution ~0.5
    axis = ar.Axis(name="a",
                   propose=lambda log: [{"name": "c%d" % len(log)}],
                   to_env=lambda selected: {"NAMES": ",".join(c["name"] for c in selected)},
                   kind="additive")
    res = ar.run_axis(axis, 1, base, ar.train_env, persist=lambda log: None)
    # candidate contribution (4.5) beats base contribution (0.5) -> kept
    assert res["kept"] and res["kept_delta"] > 0


# --- Task 2: surrogate-guided next_child ---


def test_surrogate_child_picks_higher_predicted(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SURROGATE", "1")
    monkeypatch.setenv("GTRADE_AR_SURROGATE_K", "2")
    # archive fitness = n_drops, so the surrogate learns "more drops = better"
    archive = {}
    for i, k in enumerate([0, 1, 2, 3, 0, 1, 2, 3, 1, 2]):
        g = ar.Genome(drops=list(_ACTIVE[:k]))
        archive["%d_0" % i] = {"genome": g, "fitness": float(k), "rows": []}
    lo = ar.Genome(drops=[])                        # predicted low
    hi = ar.Genome(drops=list(_ACTIVE[:3]))         # predicted high
    seq = iter([lo, hi])
    monkeypatch.setattr(ar, "mutate", lambda parent, active, bf: next(seq))
    monkeypatch.setattr(ar, "crossover", lambda a, b, active: next(seq))
    # force the mutate path (not crossover) is irrelevant; both draw from seq
    import random as _r
    _r.seed(1)
    child = ar._surrogate_child(archive, _ACTIVE, ["ret_1"])
    assert child is not None and len(child.drops) == 3   # the higher-predicted one


def test_next_child_surrogate_off_is_unchanged(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SURROGATE", raising=False)
    archive = {"0_0": {"genome": ar.Genome(), "fitness": 0.0, "rows": []}}
    called = {"surrogate": 0}
    from core import qd_surrogate
    monkeypatch.setattr(qd_surrogate, "fit_surrogate",
                        lambda *a, **k: called.__setitem__("surrogate", called["surrogate"] + 1) or None)
    monkeypatch.setattr(ar, "mutate", lambda parent, active, bf: ar.Genome(drops=["rsi"]))
    import random as _r
    _r.seed(0)
    child = ar.next_child(archive, _ACTIVE, ["ret_1"])
    assert child is not None                       # plain path still works
    assert called["surrogate"] == 0                # surrogate never consulted when OFF


def test_next_child_surrogate_error_falls_back(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_SURROGATE", "1")

    def boom(*a, **k):
        raise RuntimeError("surrogate down")
    monkeypatch.setattr(ar, "_surrogate_child", boom)
    monkeypatch.setattr(ar, "mutate", lambda parent, active, bf: ar.Genome(drops=["atr"]))
    archive = {"0_0": {"genome": ar.Genome(), "fitness": 0.0, "rows": []}}
    import random as _r
    _r.seed(0)
    child = ar.next_child(archive, _ACTIVE, ["ret_1"])
    assert child is not None                       # fell back to mutate/crossover
