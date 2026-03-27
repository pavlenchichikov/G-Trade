"""
G-TRADE SCHEDULER -- Auto Task Runner
Runs project tasks on configurable intervals using subprocess.
Uses only stdlib: threading, time, subprocess, json, sys, os, argparse.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import signal
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "scheduler_config.json")
STATE_PATH = os.path.join(BASE_DIR, "scheduler_state.json")

DEFAULT_CONFIG = {
    "tasks": {
        "data_update": {"script": "data_engine.py", "interval_hours": 6, "enabled": True},
        "train": {"script": "train_hybrid.py", "interval_hours": 24, "enabled": False},
        "predict": {"script": "predict.py", "interval_hours": 4, "enabled": True},
        "db_check": {"script": "db_check.py", "args": ["--fix"], "interval_hours": 24, "enabled": True},
        "news_scan": {"script": "news_analyzer.py", "interval_hours": 3, "enabled": True},
        "regime_check": {"script": "regime_detector.py", "interval_hours": 6, "enabled": True},
    }
}

_stop_flag = False


def _now_str():
    return datetime.now().strftime("%H:%M:%S")


def _now_iso():
    return datetime.now().isoformat()


def _load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default if default is not None else {}


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_config():
    """Load scheduler config, creating default if missing."""
    if not os.path.exists(CONFIG_PATH):
        _save_json(CONFIG_PATH, DEFAULT_CONFIG)
        print(f"[{_now_str()}] [INFO] Created default config: {CONFIG_PATH}")
    return _load_json(CONFIG_PATH, DEFAULT_CONFIG)


def load_state():
    """Load last-run state."""
    return _load_json(STATE_PATH, {"last_run": {}})


def save_state(state):
    _save_json(STATE_PATH, state)


def format_duration(seconds):
    """Format seconds into human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if mins:
        return f"{hours}h {mins}m"
    return f"{hours}h"


def format_remaining(seconds):
    """Format remaining time."""
    if seconds <= 0:
        return "due now"
    return format_duration(seconds)


def run_task(task_name, task_cfg):
    """Run a single task via subprocess. Returns (success, duration_seconds)."""
    script = task_cfg["script"]
    script_path = os.path.join(BASE_DIR, script)
    args = task_cfg.get("args", [])

    if not os.path.exists(script_path):
        print(f"[{_now_str()}] [ERR]  {task_name} -- script not found: {script}")
        return False, 0

    cmd = [sys.executable, script_path] + args
    print(f"\n  [RUN]  {task_name}")

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout, _ = proc.communicate(timeout=3600)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f"  [OK]   {task_name}  ({format_duration(elapsed)})")
            return True, elapsed
        else:
            print(f"  [ERR]  {task_name}  exit={proc.returncode}  ({format_duration(elapsed)})")
            if stdout and stdout.strip():
                for line in stdout.strip().split("\n")[-5:]:
                    print(f"           {line}")
            return False, elapsed

    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {task_name}  ({format_duration(elapsed)})")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERR]  {task_name}  {e}")
        return False, elapsed


def seconds_until_due(task_name, task_cfg, state):
    """Return seconds until a task is due. Negative means overdue."""
    last_run_str = state.get("last_run", {}).get(task_name)
    if not last_run_str:
        return -1  # never run, due now
    try:
        last_run = datetime.fromisoformat(last_run_str)
    except (ValueError, TypeError):
        return -1
    interval = timedelta(hours=task_cfg.get("interval_hours", 24))
    next_run = last_run + interval
    return (next_run - datetime.now()).total_seconds()


def cmd_status():
    """Show schedule status and exit."""
    config = load_config()
    state = load_state()
    tasks = config.get("tasks", {})

    enabled_count = sum(1 for t in tasks.values() if t.get("enabled", False))
    total_count = len(tasks)

    print()
    print("=" * 70)
    print("  G-TRADE SCHEDULER  |  Status")
    print(f"  Tasks: {enabled_count} enabled / {total_count} total")
    print("=" * 70)
    print()
    print(f"  {'Task':<18} {'Status':<10} {'Interval':<10} {'Last Run':<20} {'Next In'}")
    print("  " + "-" * 65)

    for name, cfg in tasks.items():
        enabled = cfg.get("enabled", False)
        status = "ON" if enabled else "OFF"
        interval_h = cfg.get("interval_hours", 24)
        interval_str = f"{interval_h}h"

        last_run_str = state.get("last_run", {}).get(name)
        if last_run_str:
            try:
                last_dt = datetime.fromisoformat(last_run_str)
                last_fmt = last_dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                last_fmt = "invalid"
        else:
            last_fmt = "never"

        if enabled:
            remaining = seconds_until_due(name, cfg, state)
            next_str = format_remaining(remaining)
        else:
            next_str = "--"

        print(f"  {name:<16} {status:<10} {interval_str:<10} {last_fmt:<20} {next_str:<12}")

    print()


def cmd_once(task_names):
    """Run specific tasks once and exit."""
    config = load_config()
    state = load_state()
    tasks = config.get("tasks", {})

    for name in task_names:
        if name not in tasks:
            print(f"  [ERR]  Unknown task: {name}")
            print(f"           Available: {', '.join(tasks.keys())}")
            continue
        cfg = tasks[name]
        success, _ = run_task(name, cfg)
        if success:
            if "last_run" not in state:
                state["last_run"] = {}
            state["last_run"][name] = _now_iso()
            save_state(state)


def cmd_run():
    """Main scheduler loop."""
    global _stop_flag

    def _handle_signal(signum, frame):
        global _stop_flag
        _stop_flag = True
        print("\n  [STOP] Shutting down scheduler...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    config = load_config()
    state = load_state()
    tasks = config.get("tasks", {})

    enabled_count = sum(1 for t in tasks.values() if t.get("enabled", False))
    total_count = len(tasks)

    print()
    print("=" * 60)
    print("  G-TRADE SCHEDULER  |  Auto Task Runner")
    print(f"  Tasks : {enabled_count} enabled / {total_count} total")
    print("  Stop  : Ctrl+C")
    print("=" * 60)
    print()

    while not _stop_flag:
        # Reload config each cycle to pick up changes
        config = load_config()
        state = load_state()
        tasks = config.get("tasks", {})

        for name, cfg in tasks.items():
            if _stop_flag:
                break

            if not cfg.get("enabled", False):
                print(f"[{_now_str()}] [SKIP] {name} -- disabled")
                continue

            remaining = seconds_until_due(name, cfg, state)
            if remaining > 0:
                print(f"[{_now_str()}] [WAIT] {name} -- next run in {format_remaining(remaining)}")
                continue

            success, _ = run_task(name, cfg)
            if "last_run" not in state:
                state["last_run"] = {}
            state["last_run"][name] = _now_iso()
            save_state(state)

        if _stop_flag:
            break

        # Sleep in 1-second increments so we can respond to Ctrl+C
        for _ in range(60):
            if _stop_flag:
                break
            time.sleep(1)

    print("  [DONE] Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(description="G-Trade Scheduler -- Auto Task Runner")
    parser.add_argument("--status", action="store_true", help="Show schedule and last run times")
    parser.add_argument("--once", nargs="+", metavar="TASK", help="Run specific tasks once and exit")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.once:
        cmd_once(args.once)
    else:
        cmd_run()


if __name__ == "__main__":
    main()
