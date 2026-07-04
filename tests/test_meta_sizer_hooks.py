def test_gate_contract_for_predict(monkeypatch):
    import importlib
    monkeypatch.setenv("GTRADE_META_SIZING", "active")
    monkeypatch.setenv("GTRADE_META_GATE_THR", "0.5")
    import core.meta_sizer as ms
    ms = importlib.reload(ms)
    # a weak-meta BUY is gated to WAIT; a strong one survives; meta_prob surfaced
    assert ms.gate("BUY", 0.3)[0] == "WAIT"
    assert ms.gate("BUY", 0.9)[0] == "BUY"
    assert ms.gate("BUY", 0.3)[1]["meta_prob"] == 0.3


def test_train_hybrid_hook_gated(monkeypatch, tmp_path):
    """The train_hybrid meta hook calls train_and_save only when enabled."""
    import core.meta_sizer as ms
    calls = []
    monkeypatch.setattr(ms, "train_and_save", lambda asset, df: calls.append(asset) or True)

    # off -> not called
    monkeypatch.delenv("GTRADE_META_SIZING", raising=False)
    if ms.meta_enabled() != "off":
        ms.train_and_save("A", None)
    assert calls == []

    # shadow -> called
    monkeypatch.setenv("GTRADE_META_SIZING", "shadow")
    if ms.meta_enabled() != "off":
        ms.train_and_save("A", None)
    assert calls == ["A"]
