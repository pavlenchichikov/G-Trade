"""Fit the GLOBAL live calibration layer from verified prediction outcomes.

The 2026-06-12..07-16 live stream was anti-calibrated at the extremes (the
0.9-1.0 probability bucket scored 32% accuracy), so serve-time probabilities
get a second isotonic layer fitted on what actually happened. prediction_log
stores the PRE-layer (raw) probability, so every refit trains raw -> P(up)
on a homogeneous history and REPLACES models/live_calib_global.pkl.

Run weekly:  python recalibrate_live.py [--days 90] [--min-n 300]
Rollback: delete models/live_calib_global.pkl (scoring degrades to identity).
"""

import argparse
import os
import sqlite3
from datetime import datetime

from core.calibration import fit_calibrator, save_live_global

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
MODEL_DIR = os.path.join(BASE, "models")


def collect_pairs(days=90, db_path=None):
    """(probs, went_up) from verified BUY/SELL rows. The stored probability is
    the ensemble P(up), so went_up = correct for BUY and NOT correct for SELL."""
    con = sqlite3.connect(db_path or DB_PATH)
    try:
        rows = con.execute(
            "SELECT signal, probability, correct FROM prediction_log "
            "WHERE correct IS NOT NULL AND signal IN ('BUY','SELL') "
            "AND probability IS NOT NULL AND date >= date('now', ?)",
            ("-%d days" % days,),
        ).fetchall()
    finally:
        con.close()
    probs, ups = [], []
    for sig, p, c in rows:
        probs.append(p)
        ups.append(int(c) if sig == "BUY" else 1 - int(c))
    return probs, ups


def main(days=90, min_n=None, model_dir=None, db_path=None):
    if min_n is None:
        try:
            min_n = int(os.getenv("GTRADE_LIVE_RECAL_MIN_N") or "300")
        except ValueError:
            min_n = 300
    probs, ups = collect_pairs(days, db_path)
    if len(probs) < min_n:
        print("[recalibrate-live] only %d verified rows (< %d) - not fitting"
              % (len(probs), min_n))
        return None
    iso = fit_calibrator(probs, ups)
    if iso is None:
        print("[recalibrate-live] degenerate data - nothing written")
        return None
    path = save_live_global(
        iso, {"n": len(probs), "fitted_at": datetime.utcnow().isoformat(),
              "days": days}, model_dir or MODEL_DIR)
    print("[recalibrate-live] fitted on %d outcomes -> %s" % (len(probs), path))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit the global live calibration layer")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--min-n", type=int, default=None)
    args = ap.parse_args()
    main(days=args.days, min_n=args.min_n)
