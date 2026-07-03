import numpy as np
import pytest

from core import net_hygiene as nh


def test_env_readers(monkeypatch):
    for k in ("GTRADE_NET_SEEDS", "GTRADE_NET_UNIQUENESS",
              "GTRADE_NET_CALIBRATE", "GTRADE_NET_ABSTAIN_EPS"):
        monkeypatch.delenv(k, raising=False)
    assert nh.net_seeds() == 1
    assert nh.uniqueness_on() is False
    assert nh.calibrate_nets_on() is False
    assert nh.abstain_eps() == 0.0
    monkeypatch.setenv("GTRADE_NET_SEEDS", "5")
    monkeypatch.setenv("GTRADE_NET_UNIQUENESS", "1")
    monkeypatch.setenv("GTRADE_NET_CALIBRATE", "true")
    monkeypatch.setenv("GTRADE_NET_ABSTAIN_EPS", "0.05")
    assert nh.net_seeds() == 5
    assert nh.uniqueness_on() is True
    assert nh.calibrate_nets_on() is True
    assert nh.abstain_eps() == 0.05
    monkeypatch.setenv("GTRADE_NET_SEEDS", "bad")
    assert nh.net_seeds() == 1                 # bad value -> floor 1
    monkeypatch.setenv("GTRADE_NET_SEEDS", "0")
    assert nh.net_seeds() == 1                 # floored at 1


def test_average_probs():
    a = np.array([0.2, 0.8])
    assert np.allclose(nh.average_probs([a]), a)          # single passthrough
    b = np.array([0.4, 0.6])
    assert np.allclose(nh.average_probs([a, b]), [0.3, 0.7])
    with pytest.raises(ValueError):
        nh.average_probs([])


def test_uniqueness_weights_direction_is_uniform():
    w = nh.uniqueness_weights(10, 1)
    assert np.allclose(w, np.ones(10))
    assert np.allclose(nh.uniqueness_weights(10, 0), np.ones(10))


def test_uniqueness_weights_window_downweights_interior():
    w = nh.uniqueness_weights(9, 3)
    assert len(w) == 9
    assert abs(w.mean() - 1.0) < 1e-9                     # normalized to mean 1
    # interior samples overlap fully (concurrency ~horizon) -> lower weight than edges
    assert w[0] > w[4] and w[-1] > w[4]


def test_calibrate_and_abstain_maps_and_abstains():
    # Balanced dataset: 0 to 0.5 all-zero labels, 0.5 to 1 all-one labels.
    # 0.5 appears at the boundary of both halves so the isotonic calibrator
    # must assign it exactly 0.5 -> reliably abstains with eps=0.05.
    val_prob = np.r_[np.linspace(0, 0.5, 200), np.linspace(0.5, 1.0, 200)]
    val_target = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
    test_prob = np.array([0.5, 0.9, 0.1, 0.51])
    cv, ct = nh.calibrate_and_abstain(val_prob, val_target, test_prob, 0.05)
    assert len(ct) == 4 and len(cv) == 400
    # exactly 0.5 calibrates to 0.5 -> |0.5-0.5|=0 < eps -> abstains
    assert ct[0] == 0.5
    # confident 0.9 / 0.1 calibrate to ~1 / ~0 -> not abstained
    assert ct[1] != 0.5 and ct[2] != 0.5


def test_calibrate_and_abstain_one_class_is_identity():
    # one-class targets -> calibrator None -> identity (plus abstention)
    val_prob = np.array([0.3, 0.4, 0.6, 0.7])
    val_target = np.array([1, 1, 1, 1])
    test_prob = np.array([0.2, 0.8])
    cv, ct = nh.calibrate_and_abstain(val_prob, val_target, test_prob, 0.0)
    assert np.allclose(ct, [0.2, 0.8])                    # identity, no abstention (eps 0)
