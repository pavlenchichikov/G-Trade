import numpy as np
import pandas as pd
import pytest

from core.feature_dsl import ALLOWED_OPS, validate_spec, materialize, add_dsl_features

COLS = {"ret_1", "ret_5", "vol_z", "rsi"}


def test_validate_accepts_valid_ops():
    assert validate_spec({"name": "z1", "op": "zscore", "inputs": ["ret_1"], "params": {"window": 20}}, COLS)
    assert validate_spec({"name": "r1", "op": "ratio", "inputs": ["ret_1", "vol_z"]}, COLS)
    assert validate_spec({"name": "l1", "op": "lag", "inputs": ["rsi"], "params": {"k": 3}}, COLS)
    assert validate_spec({"name": "x1", "op": "interaction", "inputs": ["rsi", "vol_z"]}, COLS)
    assert validate_spec({"name": "ll", "op": "lead_lag", "inputs": ["sp500"], "params": {"horizon": 1}}, COLS)


def test_validate_rejects_bad():
    assert not validate_spec({"name": "z", "op": "evil", "inputs": ["ret_1"]}, COLS)        # unknown op
    assert not validate_spec({"name": "z", "op": "zscore", "inputs": ["nope"]}, COLS)        # unknown input
    assert not validate_spec({"name": "z", "op": "zscore", "inputs": ["ret_1"], "params": {"window": 999}}, COLS)  # window oob
    assert not validate_spec({"name": "Z BAD", "op": "ratio", "inputs": ["ret_1", "vol_z"]}, COLS)  # bad name
    assert not validate_spec({"name": "r", "op": "ratio", "inputs": ["ret_1"]}, COLS)        # wrong arity
    assert not validate_spec({"name": "ll", "op": "lead_lag", "inputs": ["fakecoin"]}, COLS)  # unknown leader


def test_materialize_ops_finite():
    df = pd.DataFrame({"ret_1": np.random.randn(50), "vol_z": np.random.randn(50), "rsi": np.random.rand(50) * 100})
    for spec in [
        {"name": "z", "op": "zscore", "inputs": ["ret_1"], "params": {"window": 10}},
        {"name": "r", "op": "ratio", "inputs": ["rsi", "vol_z"]},
        {"name": "l", "op": "lag", "inputs": ["ret_1"], "params": {"k": 2}},
        {"name": "d", "op": "diff", "inputs": ["rsi"], "params": {"k": 1}},
        {"name": "ro", "op": "rolling", "inputs": ["vol_z"], "params": {"window": 5, "agg": "mean"}},
        {"name": "x", "op": "interaction", "inputs": ["rsi", "vol_z"]},
    ]:
        s = materialize(df, spec)
        assert len(s) == 50 and np.isfinite(s.values).all()


def test_materialize_rejects_lead_lag():
    df = pd.DataFrame({"ret_1": [1.0, 2.0]})
    with pytest.raises(ValueError):
        materialize(df, {"name": "ll", "op": "lead_lag", "inputs": ["sp500"]})


def test_allowed_ops_closed():
    assert ALLOWED_OPS == {"zscore", "ratio", "lag", "diff", "rolling", "interaction", "lead_lag"}


def test_add_dsl_features_applies_valid_skips_invalid():
    df = pd.DataFrame({"ret_1": np.random.randn(30), "vol_z": np.random.randn(30)})
    specs = [
        {"name": "good", "op": "zscore", "inputs": ["ret_1"], "params": {"window": 5}},
        {"name": "bad", "op": "zscore", "inputs": ["missing"]},
    ]
    out, skipped = add_dsl_features(df, None, specs)
    assert "good" in out.columns and "bad" not in out.columns
    assert skipped == ["bad"]


def test_add_dsl_features_none_is_noop():
    df = pd.DataFrame({"ret_1": [1.0, 2.0]})
    out, skipped = add_dsl_features(df, None, None)
    assert list(out.columns) == ["ret_1"] and skipped == []


def test_load_dsl_specs_env(tmp_path, monkeypatch):
    import json
    from core.feature_dsl import load_dsl_specs
    monkeypatch.delenv("GTRADE_DSL_SPECS", raising=False)
    assert load_dsl_specs() == []          # unset means no specs
    p = tmp_path / "specs.json"
    p.write_text(json.dumps([{"name": "z", "op": "zscore", "inputs": ["ret_1"]}]))
    monkeypatch.setenv("GTRADE_DSL_SPECS", str(p))
    specs = load_dsl_specs()
    assert specs and specs[0]["name"] == "z"


def test_run_loop_stubbed(monkeypatch, tmp_path):
    ar = _load_auto_research()
    monkeypatch.setattr(ar, "_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(ar, "_LOG_PATH", str(tmp_path / "log.json"))
    calls = {"n": 0, "trains": 0}

    def fake_propose(log, base_features):
        calls["n"] += 1
        return [{"name": "z%d" % calls["n"], "op": "zscore", "inputs": ["ret_1"], "params": {"window": 10}}]

    def fake_train(subset, specs, extra):
        calls["trains"] += 1
        return [{"Asset": "A", "Score": 1.0 + 0.1 * len(specs)}]

    log = ar.run_loop(fake_propose, fake_train, budget=3, base_features=["ret_1"], resume=False)
    assert len(log) == 3 and calls["n"] == 3       # respected the budget
    assert calls["trains"] == 4                     # base cached (1) + one variant per round (3)
    assert ar.load_state().get("kept")              # improving features accumulated


def _load_auto_research():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "auto_research.py")
    if not os.path.exists(path):
        pytest.skip("auto_research.py is a local (gitignored) tool, absent in CI")
    spec = importlib.util.spec_from_file_location("auto_research", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_propose_evolutionary_no_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("GTRADE_AR_SEED", "42")
    ar = _load_auto_research()
    monkeypatch.setattr(ar, "_STATE_PATH", str(tmp_path / "s.json"))
    monkeypatch.setattr(ar, "_LOG_PATH", str(tmp_path / "l.json"))
    bf = ["ret_1", "ret_5", "vol_z", "rsi"]
    # empty log: exploration produces one validated spec, no API/LLM involved
    specs = ar.propose_evolutionary([], bf)
    assert specs and validate_spec(specs[0], set(bf))
    # full loop with the evolutionary proposer and a stubbed single training: no API
    def fake_train(subset, sp, ex):
        return [{"Asset": "A", "Score": 1.0 + 0.01 * len(sp)}]
    log = ar.run_loop(ar.propose_evolutionary, fake_train, budget=4, base_features=bf, resume=False)
    assert len(log) == 4
    ran = [e for e in log if e.get("spec")]
    assert ran  # at least one real experiment with a validated spec


def test_select_proposer_default_is_evolutionary(monkeypatch):
    ar = _load_auto_research()
    monkeypatch.delenv("GTRADE_AR_PROPOSER", raising=False)
    assert ar._select_proposer() is ar.propose_evolutionary
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    assert ar._select_proposer() is ar.propose_next


def test_parse_specs_tolerant():
    ar = _load_auto_research()
    assert ar._parse_specs('[{"name": "z", "op": "zscore", "inputs": ["ret_1"]}]')[0]["name"] == "z"
    assert ar._parse_specs("here you go: [\n{\"name\":\"a\",\"op\":\"lag\",\"inputs\":[\"x\"]}\n] thanks")
    assert ar._parse_specs("sorry, no json") == []
    assert ar._parse_specs("") == []


def test_proposer_prompt_and_provider_dispatch(monkeypatch):
    ar = _load_auto_research()
    p = ar._proposer_prompt([], ["ret_1", "vol_z"])
    assert "zscore" in p and "ret_1" in p          # DSL menu + base columns present
    # unknown provider raises before any network call
    monkeypatch.setenv("GTRADE_AR_LLM", "nope")
    with pytest.raises(RuntimeError):
        ar.propose_next([], ["ret_1"])
    # both real backends are registered
    assert set(ar._LLM_BACKENDS) == {"anthropic", "openai"}


def test_sign_test_adoption():
    ar = _load_auto_research()
    base = [{"Asset": a, "Score": 1.0} for a in "ABCDEF"]
    win = [{"Asset": a, "Score": 2.0} for a in "ABCDEF"]      # 6/6 up, mean +1.0
    ok, why = ar.is_adoptable(base, win, 3, 15)
    assert ok and "p=" in why
    mixed = [{"Asset": "A", "Score": 2.0}, {"Asset": "B", "Score": 0.5},
             {"Asset": "C", "Score": 1.1}, {"Asset": "D", "Score": 0.9},
             {"Asset": "E", "Score": 1.0}, {"Asset": "F", "Score": 0.8}]
    nok, _ = ar.is_adoptable(base, mixed, 3, 15)
    assert not nok                                            # not significant
    over, _ = ar.is_adoptable(base, win, 20, 15)
    assert not over                                           # over budget
    assert ar._sign_test_p([1, 1, 1, 1, 1, 1]) < 0.05
    assert ar._sign_test_p([1, -1, 1, -1]) > 0.05


def test_prescreen_ok():
    import numpy as np
    import pandas as pd
    ar = _load_auto_research()
    rng = np.random.RandomState(0)
    sig = pd.Series(rng.randn(300))
    df = pd.DataFrame({"sig": sig, "noise": pd.Series(rng.randn(300)),
                       "target": sig.shift(1).fillna(0.0)})
    # lag(sig, 1) reproduces the target (high correlation), so it is kept
    assert ar._prescreen_ok({"name": "l", "op": "lag", "inputs": ["sig"], "params": {"k": 1}}, df, threshold=0.1)
    # lag(noise, 1) is unrelated, so it is pruned
    assert not ar._prescreen_ok({"name": "l2", "op": "lag", "inputs": ["noise"], "params": {"k": 1}}, df, threshold=0.2)
    # lead_lag (engine op) and a missing target pass through
    assert ar._prescreen_ok({"name": "ll", "op": "lead_lag", "inputs": ["sp500"]}, df)
    assert ar._prescreen_ok({"name": "z", "op": "zscore", "inputs": ["sig"]}, pd.DataFrame({"sig": [1.0, 2.0]}))


def test_run_loop_resume(monkeypatch, tmp_path):
    import json as _j
    ar = _load_auto_research()
    sp = str(tmp_path / "state.json")
    monkeypatch.setattr(ar, "_STATE_PATH", sp)
    monkeypatch.setattr(ar, "_LOG_PATH", str(tmp_path / "log.json"))
    seed = {"log": [{"iter": 0, "spec": [], "note": "x"}, {"iter": 1, "spec": [], "note": "y"}],
            "kept": [], "kept_delta": 0.0, "base_rows": [{"Asset": "A", "Score": 1.0}]}
    with open(sp, "w") as f:
        _j.dump(seed, f)
    trains = {"n": 0}

    def fake_propose(log, bf):
        return [{"name": "zz", "op": "zscore", "inputs": ["ret_1"]}]

    def fake_train(subset, specs, extra):
        trains["n"] += 1
        return [{"Asset": "A", "Score": 1.5}]

    log = ar.run_loop(fake_propose, fake_train, budget=4, base_features=["ret_1"], resume=True)
    assert len(log) == 4            # continued from the seeded 2 rounds
    assert trains["n"] == 2         # base NOT retrained (cached), only the 2 new rounds
