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


def test_select_proposer_default_is_evolutionary(monkeypatch):
    ar = _load_auto_research()
    from core import llm_proposer
    monkeypatch.delenv("GTRADE_AR_PROPOSER", raising=False)
    assert ar._select_proposer() is ar.propose_evolutionary
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    assert ar._select_proposer() is llm_proposer.propose_specs


def test_parse_specs_tolerant():
    from core import llm_proposer
    assert llm_proposer._parse_specs('[{"name": "z", "op": "zscore", "inputs": ["ret_1"]}]')[0]["name"] == "z"
    assert llm_proposer._parse_specs("here you go: [\n{\"name\":\"a\",\"op\":\"lag\",\"inputs\":[\"x\"]}\n] thanks")
    assert llm_proposer._parse_specs("sorry, no json") == []
    assert llm_proposer._parse_specs("") == []


def test_proposer_prompt_and_provider_dispatch(monkeypatch):
    from core import llm_proposer
    p = llm_proposer._proposer_prompt([], ["ret_1", "vol_z"])
    assert "zscore" in p and "ret_1" in p          # DSL menu + base columns present
    # unknown provider raises before any network call
    monkeypatch.setenv("GTRADE_AR_LLM", "nope")
    with pytest.raises(RuntimeError):
        llm_proposer.propose_specs([], ["ret_1"])
    # known providers are recognized (anthropic and openai)
    monkeypatch.setenv("GTRADE_AR_LLM", "anthropic")
    assert callable(llm_proposer._backend())
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    assert callable(llm_proposer._backend())


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
