"""Slow test: seed-averaging wiring in train_hybrid.py (SP-2 Task 2).

Excluded from the fast suite (marker 'slow').  Run manually when a market.db
is available and GTRADE_NET_SEEDS >= 2 should be exercised end-to-end:

    pytest tests/test_net_seed_avg.py -m slow -s

The test trains one tiny asset with GTRADE_NET_SEEDS=2 (two-seed average),
asserts the run completes, and checks that a quality_report.json is written
with at least one row containing a 'Score' key.
"""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_seed_averaging_trains_and_writes_report(tmp_path):
    """Two-seed neural run completes and writes quality_report.json.

    Requires market.db in the project root.  Skipped automatically if absent.
    """
    if not os.getenv("RUN_SLOW_TRAIN"):
        pytest.skip("set RUN_SLOW_TRAIN=1 to run this real-training slow test")
    if not os.path.exists("market.db"):
        pytest.skip("no market DB - skipping slow seed-averaging test")

    env = dict(os.environ)
    env.update({
        "GTRADE_NET_SEEDS": "2",
        "GTRADE_ASSETS": "SP500",
        "GTRADE_MODEL_DIR": str(tmp_path),
        "GTRADE_MAX_FOLDS": "3",
        "GTRADE_FORCE_PROMOTE": "1",
    })
    result = subprocess.run(
        [sys.executable, "train_hybrid.py"],
        env=env,
        check=False,
        timeout=1800,
    )
    assert result.returncode == 0, f"train_hybrid.py exited with {result.returncode}"

    report = tmp_path / "quality_report.json"
    assert report.exists(), "quality_report.json not written"
    rows = json.load(report.open())
    assert rows and "Score" in rows[0], "quality_report.json missing Score key"
