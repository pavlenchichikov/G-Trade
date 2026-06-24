"""Pure outcome-based drift logic for the self-maintaining loop. No I/O here so
it can be unit-tested with plain inputs; loop_cycle.py fetches the real data
(accuracy, registry, stale set) and calls classify_asset per asset."""

DRIFT_CONFIG = {
    "acc_floor": 0.50,        # rolling hit-rate below this proposes a retrain
    "min_reconciled": 15,     # need at least this many reconciled before judging
    "baseline_drop": 0.08,    # live acc this far under trained CB_Acc proposes
    "max_age_days": 45,       # model older than this proposes a challenger
    "stale_days": 7,          # data older than this is a feed flag (not retrain)
    "trend_window": 10,       # window for the accuracy trend
}


def miss_streak(outcomes):
    """Count trailing misses (0s) in a 1=correct/0=miss list, most recent last."""
    streak = 0
    for o in reversed(outcomes):
        if o == 0:
            streak += 1
        else:
            break
    return streak


def acc_trend(outcomes, window):
    """Mean of the last `window` minus the mean of the prior `window`. Negative
    means the model is degrading. None if there is not enough history."""
    if len(outcomes) < 2 * window:
        return None
    recent = outcomes[-window:]
    prior = outcomes[-2 * window:-window]
    return round(sum(recent) / window - sum(prior) / window, 3)


def classify_asset(asset, acc, n, baseline_acc, age_days, is_stale,
                   recent_outcomes, cfg=DRIFT_CONFIG):
    """Classify one asset. status: 'propose' (retrain challenger), 'data' (stale
    feed only), or 'ok'. Accuracy signals are skipped below min_reconciled."""
    reasons = []
    retrain = False
    enough = n >= cfg["min_reconciled"]

    if enough and acc is not None and acc < cfg["acc_floor"]:
        reasons.append("acc %.2f below floor %.2f" % (acc, cfg["acc_floor"]))
        retrain = True
    if (enough and acc is not None and baseline_acc is not None
            and acc < baseline_acc - cfg["baseline_drop"]):
        reasons.append("acc %.2f dropped from baseline %.2f" % (acc, baseline_acc))
        retrain = True
    if age_days is not None and age_days > cfg["max_age_days"]:
        reasons.append("model age %dd over %dd" % (age_days, cfg["max_age_days"]))
        retrain = True
    if is_stale:
        reasons.append("stale data")

    status = "propose" if retrain else ("data" if is_stale else "ok")
    return {
        "asset": asset,
        "status": status,
        "reasons": reasons,
        "acc": acc,
        "baseline_acc": baseline_acc,
        "age_days": age_days,
        "stale": is_stale,
        "miss_streak": miss_streak(recent_outcomes),
        "acc_trend": acc_trend(recent_outcomes, cfg["trend_window"]),
    }
