"""Daily self-maintaining loop orchestrator (Windows Task Scheduler target).

Runs the safe pipeline (data_engine, predict, reconcile), scans drift, and
writes loop_state.json. Each step is isolated: a failure is recorded and the
cycle still produces a report. NEVER retrains - retraining is approved on the
/loop page and run by loop_retrain.py.

Deploy only AFTER the baseline finishes. Register with run_loop.bat / schtasks.
"""
import datetime as _dt
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import FULL_ASSET_MAP  # noqa: E402
from core import drift, loop_state  # noqa: E402

STATE_PATH = os.path.join(BASE, "loop_state.json")
LOCK_PATH = os.path.join(BASE, "_loop.lock")
REGISTRY_PATH = os.path.join(BASE, "models", "champion_registry.json")
QUALITY_PATH = os.path.join(BASE, "models", "quality_report.json")


def run_step(name, fn):
    """Run one pipeline step, capturing any failure so the cycle continues."""
    try:
        fn()
        return {"step": name, "status": "ok", "msg": ""}
    except Exception as exc:  # noqa: BLE001 - we intentionally catch everything
        return {"step": name, "status": "failed", "msg": str(exc)[:200]}


def scan_assets(rows):
    """Map pre-fetched per-asset rows through the pure drift classifier."""
    return [
        drift.classify_asset(
            r["asset"], r.get("acc"), r.get("n", 0), r.get("baseline_acc"),
            r.get("age_days"), r.get("is_stale", False),
            r.get("recent_outcomes", []))
        for r in rows
    ]


def _run_script(script):
    subprocess.run([sys.executable, script], cwd=BASE, check=True)


def _build_rows():
    """Fetch the inputs drift needs for every asset from the live data."""
    import json
    from core import track_record

    reg = {}
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            reg = json.load(f)

    qmap = {}
    if os.path.exists(QUALITY_PATH):
        try:
            with open(QUALITY_PATH, encoding="utf-8") as f:
                qlist = json.load(f)
            qmap = {rec["Asset"]: rec.get("CB_Acc") for rec in qlist}
        except Exception:
            qmap = {}

    stale = {r["asset"] for r in
              track_record.stale_assets(max_age_days=drift.DRIFT_CONFIG["stale_days"])}

    rows = []
    for asset in FULL_ASSET_MAP:
        acc_info = track_record.asset_accuracy(asset)
        track = track_record.asset_track(asset, limit=40)
        outcomes = [int(t["correct"]) for t in reversed(track)
                    if t.get("correct") is not None]
        entry = reg.get(asset) or {}
        baseline = qmap.get(asset)
        age_days = None
        trained = entry.get("updated_at")
        if trained:
            try:
                d = _dt.datetime.strptime(str(trained)[:10], "%Y-%m-%d")
                age_days = (_dt.datetime.utcnow() - d).days
            except Exception:
                age_days = None
        rows.append({
            "asset": asset, "acc": acc_info.get("acc"), "n": acc_info.get("n", 0),
            "baseline_acc": baseline, "age_days": age_days,
            "is_stale": asset in stale, "recent_outcomes": outcomes,
        })
    return rows


def main():
    if os.path.exists(LOCK_PATH):
        print("[loop] lock present (retrain running?); skipping this cycle.")
        return
    open(LOCK_PATH, "w").close()
    try:
        steps = []
        steps.append(run_step("data_engine", lambda: _run_script("data_engine.py")))
        steps.append(run_step("predict", lambda: _run_script("predict.py")))
        steps.append(run_step("reconcile", lambda: _run_script("performance_tracker.py")))

        assets = []
        drift_step = run_step("drift", lambda: assets.extend(scan_assets(_build_rows())))
        steps.append(drift_step)

        proposed = sorted(a["asset"] for a in assets if a["status"] == "propose")
        state = loop_state.load_state(STATE_PATH)
        state["last_run"] = _dt.datetime.utcnow().isoformat(timespec="seconds")
        state["steps"] = {s["step"]: {"status": s["status"], "msg": s["msg"]} for s in steps}
        state["assets"] = assets
        state["proposed"] = proposed
        # keep approvals that are still proposed; drop the rest
        state["approved"] = [a for a in state.get("approved", []) if a in proposed]
        hist = state.get("history", [])
        hist.insert(0, {"ts": state["last_run"], "proposed": len(proposed),
                        "failed_steps": [s["step"] for s in steps if s["status"] == "failed"]})
        state["history"] = hist[:30]
        loop_state.save_state(STATE_PATH, state)
        print("[loop] cycle done. proposed retrains: %d" % len(proposed))
    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)


if __name__ == "__main__":
    main()
