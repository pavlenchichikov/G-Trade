import ab_labeling

evaluate = ab_labeling.evaluate


def _rows(scores):
    return [{"Asset": a, "Score": s} for a, s in scores.items()]


def test_evaluate_reports_deltas_and_verdict():
    selection = ["SP500", "NVDA"]
    # is_adoptable's sign-test needs at least 5 held-out assets all improving
    # to reach p < 0.05 (with n=2 the minimum possible p is 0.25); use 5 here
    # so the ADOPT branch is statistically reachable.
    heldout = ["MSFT", "GOLD", "USDJPY", "ADA", "CAC40"]
    base = _rows({"SP500": 1.0, "NVDA": 2.0, "MSFT": 0.0, "GOLD": -1.0,
                  "USDJPY": 0.0, "ADA": -1.0, "CAC40": 0.5})
    # variant improves every asset by a wide margin -> ADOPT
    var = _rows({"SP500": 3.0, "NVDA": 4.0, "MSFT": 3.0, "GOLD": 2.0,
                 "USDJPY": 3.0, "ADA": 2.0, "CAC40": 3.5})
    res = evaluate(base, var, selection, heldout)
    assert res["selection_mean_delta"] > 0
    assert res["heldout_mean_delta"] > 0
    assert res["verdict"] == "ADOPT"


def test_evaluate_holds_when_no_improvement():
    selection = ["SP500", "NVDA"]
    heldout = ["MSFT", "GOLD"]
    base = _rows({"SP500": 1.0, "NVDA": 2.0, "MSFT": 0.0, "GOLD": -1.0})
    var = _rows({"SP500": 0.9, "NVDA": 1.8, "MSFT": -0.2, "GOLD": -1.5})
    res = evaluate(base, var, selection, heldout)
    assert res["verdict"] == "HOLD"


def test_evaluate_inconclusive_when_asset_missing():
    # GOLD is in the expected held-out set but the trainer dropped it from the
    # variant run (e.g. insufficient history / per-asset crash). The base/var
    # intersection still computes a (false) HOLD-looking delta; evaluate()
    # must surface this as INCONCLUSIVE with the missing asset named, not as
    # an ordinary HOLD indistinguishable from "variant did not help".
    selection = ["SP500", "NVDA"]
    heldout = ["MSFT", "GOLD"]
    base = _rows({"SP500": 1.0, "NVDA": 2.0, "MSFT": 0.0, "GOLD": -1.0})
    var = _rows({"SP500": 3.0, "NVDA": 4.0, "MSFT": 3.0})  # GOLD missing
    res = evaluate(base, var, selection, heldout)
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["missing"] == ["GOLD"]
