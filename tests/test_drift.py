from core.drift import DRIFT_CONFIG, miss_streak, acc_trend, classify_asset


def test_miss_streak_counts_trailing_misses():
    assert miss_streak([1, 1, 0, 0, 0]) == 3
    assert miss_streak([0, 0, 1]) == 0
    assert miss_streak([]) == 0


def test_acc_trend_negative_when_degrading():
    # prior 4 all correct, recent 4 all miss, so trend -1.0
    assert acc_trend([1, 1, 1, 1, 0, 0, 0, 0], window=4) == -1.0
    # not enough data
    assert acc_trend([1, 0, 1], window=4) is None


def test_classify_low_accuracy_proposes_retrain():
    r = classify_asset("AAPL", acc=0.40, n=20, baseline_acc=0.62,
                       age_days=5, is_stale=False, recent_outcomes=[1, 0] * 10)
    assert r["status"] == "propose"
    assert any("below floor" in x for x in r["reasons"])


def test_classify_baseline_drop_proposes_retrain():
    r = classify_asset("MSFT", acc=0.52, n=20, baseline_acc=0.65,
                       age_days=5, is_stale=False, recent_outcomes=[1, 0] * 10)
    assert r["status"] == "propose"
    assert any("dropped from baseline" in x for x in r["reasons"])


def test_classify_age_proposes_retrain():
    r = classify_asset("VIX", acc=0.60, n=20, baseline_acc=0.60,
                       age_days=90, is_stale=False, recent_outcomes=[1] * 20)
    assert r["status"] == "propose"
    assert any("age" in x for x in r["reasons"])


def test_classify_insufficient_samples_skips_accuracy():
    # acc looks bad but only 5 reconciled, so accuracy signals skipped
    r = classify_asset("NEW", acc=0.20, n=5, baseline_acc=0.60,
                       age_days=5, is_stale=False, recent_outcomes=[0] * 5)
    assert r["status"] == "ok"
    assert r["reasons"] == []


def test_classify_stale_is_data_flag_not_retrain():
    r = classify_asset("SBER", acc=0.60, n=20, baseline_acc=0.60,
                       age_days=5, is_stale=True, recent_outcomes=[1] * 20)
    assert r["status"] == "data"
    assert "stale data" in r["reasons"]


def test_config_has_expected_keys():
    for k in ("acc_floor", "min_reconciled", "baseline_drop", "max_age_days",
              "stale_days", "trend_window"):
        assert k in DRIFT_CONFIG
