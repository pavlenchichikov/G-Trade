"""Entry/exit timing policy (Stage A rules), spec 2026-07-18-rl-timing-policy.

The policy decides WHEN to act on the ensemble's signal - never direction
(the ensemble's job) and never size (Kelly + Taleb's job). Actions are
relative to the raw signal; with DEFAULT_PARAMS the policy reproduces the
baseline (signal == action every bar) exactly.

Pure logic: no DB, no model loading. train_timing.py drives .apply over
history; serve drives policy_step one bar at a time (shadow mode).
"""
import json
import math
import os

import numpy as np

PARAM_SPECS = (
    ("entry_margin", 0.0, 0.10, False),
    ("entry_margin_hi_taleb", 0.0, 0.10, False),
    ("entry_margin_risky", 0.0, 0.10, False),
    ("confirm_days", 0, 3, True),
    ("exit_hysteresis", 0.0, 0.10, False),
    ("max_hold_days", 2, 60, True),
    ("trail_atr", 1.0, 8.0, False),
    ("cooldown_days", 0, 10, True),
)

DEFAULT_PARAMS = {
    "entry_margin": 0.0, "entry_margin_hi_taleb": 0.0,
    "entry_margin_risky": 0.0, "confirm_days": 0,
    "exit_hysteresis": 0.0, "max_hold_days": 60,
    "trail_atr": 8.0, "cooldown_days": 0,
}

FRESH_STATE = {"pos": 0, "days_held": 0, "seg_peak": 0.0, "seg_ret": 0.0,
               "cooldown_left": 0, "streak": 0, "last_raw": 0}

POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "timing_policy.json")


def timing_on():
    return os.getenv("GTRADE_TIMING_POLICY") == "1"


def load_policy(path=None):
    """RulesPolicy from timing_policy.json, or None when absent/corrupt."""
    try:
        with open(path or POLICY_PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        params = dict(DEFAULT_PARAMS)
        params.update({k: blob["params"][k] for k in DEFAULT_PARAMS
                       if k in blob.get("params", {})})
        return RulesPolicy(params)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _raw_side(prob, buy_thr, sell_thr):
    return 1 if prob > buy_thr else (-1 if prob < sell_thr else 0)


_TIMING_LABELS = {
    ("STAY_OUT", "entry_margin"): ("policy: too weak to enter", True),
    ("STAY_OUT", "confirm"): ("policy: waiting for confirmation", True),
    ("STAY_OUT", "cooldown"): ("policy: cooling down after exit", True),
    ("HOLD", "ok"): ("policy: holding", False),
    ("EXIT", "flip"): ("policy: exit - signal flipped", True),
    ("EXIT", "hysteresis"): ("policy: exit - momentum faded", True),
    ("EXIT", "max_hold"): ("policy: exit - max hold reached", True),
    ("EXIT", "trail_stop"): ("policy: exit - trailing stop", True),
}


def display_label(action, reason):
    """Map a (timing_action, timing_reason) pair to (text, is_divergence).

    Single source of the display wording (spec 2026-07-23-timing-policy-
    display-layer section 3). `text` is None when nothing should be shown
    (aligned/flat states); `is_divergence` is True when the policy blocks or
    exits against a live signal, which the radar/mobile use to decide whether
    to show a badge at all. `action` may be "ENTER:+1"/"ENTER:-1"."""
    if action and action.startswith("ENTER"):
        return ("policy: entering", False)
    return _TIMING_LABELS.get((action, reason), (None, False))


def policy_step(policy, prob, buy_thr, sell_thr, atr_now, taleb_hi_now,
                risky, state):
    """One-bar decision. Returns (action, reason, new_state).

    `state` keys: see FRESH_STATE. seg_peak/seg_ret are updated by the
    CALLER between bars (they need the realized return); policy_step only
    reads them for the trailing stop and resets them on ENTER/EXIT.
    """
    p = policy.params
    st = dict(state)
    raw = _raw_side(prob, buy_thr, sell_thr)

    last_raw = st.get("last_raw", 0)
    if raw != 0 and raw == last_raw:
        st["streak"] = st.get("streak", 0) + 1
    elif raw != 0:
        st["streak"] = 1
    else:
        st["streak"] = 0
    st["last_raw"] = raw

    if st.get("cooldown_left", 0) > 0:
        st["cooldown_left"] -= 1

    if st["pos"] == 0:
        if raw == 0:
            return "STAY_OUT", "ok", st
        if st["cooldown_left"] > 0:
            return "STAY_OUT", "cooldown", st
        m = p["entry_margin"]
        if taleb_hi_now:
            m += p["entry_margin_hi_taleb"]
        if risky:
            m += p["entry_margin_risky"]
        strong = prob > buy_thr + m if raw == 1 else prob < sell_thr - m
        if not strong:
            return "STAY_OUT", "entry_margin", st
        if st["streak"] < int(p["confirm_days"]):
            return "STAY_OUT", "confirm", st
        st.update(pos=raw, days_held=0, seg_peak=0.0, seg_ret=0.0)
        return "ENTER", "ok", st

    # in position
    st["days_held"] += 1
    if raw == -st["pos"]:
        exit_reason = "flip"
    elif st["pos"] == 1 and prob < buy_thr - p["exit_hysteresis"]:
        exit_reason = "hysteresis"
    elif st["pos"] == -1 and prob > sell_thr + p["exit_hysteresis"]:
        exit_reason = "hysteresis"
    elif st["days_held"] >= int(p["max_hold_days"]):
        exit_reason = "max_hold"
    elif st["seg_peak"] - st["seg_ret"] > p["trail_atr"] * float(atr_now):
        exit_reason = "trail_stop"
    else:
        exit_reason = None

    if exit_reason:
        st.update(pos=0, days_held=0, seg_peak=0.0, seg_ret=0.0,
                  cooldown_left=int(p["cooldown_days"]))
        return "EXIT", exit_reason, st
    return "HOLD", "ok", st


class RulesPolicy:
    def __init__(self, params):
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(params or {})

    def apply(self, probs, buy_thr, sell_thr, atr, taleb_hi, risky,
              next_ret=None):
        """Vector run over history. Returns (sides, actions, reasons).

        `next_ret` (optional): per-bar next returns used ONLY to maintain
        the open segment's running return/peak for the trailing stop;
        when None the trail rule never fires.
        """
        n = len(probs)
        sides = np.zeros(n, dtype=int)
        actions, reasons = [], []
        st = dict(FRESH_STATE)
        for i in range(n):
            action, reason, st = policy_step(
                self, float(probs[i]), buy_thr, sell_thr,
                float(atr[i]), bool(taleb_hi[i]), risky, st)
            sides[i] = st["pos"]
            actions.append(action)
            reasons.append(reason)
            if st["pos"] != 0 and next_ret is not None:
                r = float(next_ret[i])
                if not math.isnan(r):
                    st["seg_ret"] += st["pos"] * r
                    st["seg_peak"] = max(st["seg_peak"], st["seg_ret"])
        return sides, actions, reasons
