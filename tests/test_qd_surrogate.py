from core import qd_surrogate as qs
from auto_research import Genome

_ACTIVE = ["ret_1", "ret_5", "ret_10", "rsi", "atr", "vol_z", "sma_20", "bb_pos"]
_BASE = ["ret_1", "ret_5", "rsi"]


def _spec(op="lag"):
    return {"name": "g", "op": op, "inputs": ["ret_1"], "params": {"k": 1}}


def test_genome_vector_deterministic_and_fixed_length():
    g = Genome(drops=["rsi"], extra=[_spec("lag")], label_mode="rel_median", label_window=20)
    v1 = qs.genome_vector(g, _ACTIVE, _BASE)
    v2 = qs.genome_vector(g, _ACTIVE, _BASE)
    assert v1 == v2
    g2 = Genome()
    assert len(qs.genome_vector(g2, _ACTIVE, _BASE)) == len(v1)   # fixed schema
    # a dropped feature flips its flag
    assert v1[1 + _ACTIVE.index("rsi")] == 1.0
    assert qs.genome_vector(g2, _ACTIVE, _BASE)[1 + _ACTIVE.index("rsi")] == 0.0


def test_fit_surrogate_min_samples_and_degenerate():
    v = qs.genome_vector(Genome(), _ACTIVE, _BASE)
    assert qs.fit_surrogate([(v, 1.0)] * 4, min_samples=8) is None      # too few
    assert qs.fit_surrogate([(v, 1.0)] * 10) is None                    # all equal fitness


def test_fit_surrogate_learns_ranking():
    # fitness = number of drops; the model must rank more-drops higher
    samples = []
    import random as _r
    _r.seed(0)
    for _ in range(40):
        k = _r.randint(0, 3)
        g = Genome(drops=list(_ACTIVE[:k]))
        samples.append((qs.genome_vector(g, _ACTIVE, _BASE), float(k)))
    model = qs.fit_surrogate(samples)
    assert model is not None
    hi = qs.predict(model, qs.genome_vector(Genome(drops=_ACTIVE[:3]), _ACTIVE, _BASE))
    lo = qs.predict(model, qs.genome_vector(Genome(), _ACTIVE, _BASE))
    assert hi > lo


def test_predict_none_model_is_zero():
    assert qs.predict(None, [1.0, 2.0]) == 0.0


def test_env_readers(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_SURROGATE", raising=False)
    monkeypatch.delenv("GTRADE_AR_SURROGATE_K", raising=False)
    assert qs.surrogate_on() is False
    assert qs.n_candidates() == 6
    monkeypatch.setenv("GTRADE_AR_SURROGATE", "1")
    monkeypatch.setenv("GTRADE_AR_SURROGATE_K", "10")
    assert qs.surrogate_on() is True
    assert qs.n_candidates() == 10
    monkeypatch.setenv("GTRADE_AR_SURROGATE_K", "bad")
    assert qs.n_candidates() == 6                                       # bad -> default
