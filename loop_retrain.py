"""Run the approved retrain as a champion-challenger, RAM-safe (a fresh process
per chunk), WITHOUT force-promote so the trainer keeps the old champion unless
the challenger beats it. Reads the approved set from loop_state.json and clears
it when done."""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core import loop_state  # noqa: E402

STATE_PATH = os.path.join(BASE, "loop_state.json")
LOCK_PATH = os.path.join(BASE, "_loop.lock")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))

# Same RAM-safe light profile as the baseline, but FORCE_PROMOTE is OFF so the
# trainer's gate decides promotion (challenger must beat the champion).
LIGHT_ENV = {
    "GTRADE_WORKERS": "4", "GTRADE_NEURAL_SLOTS": "4", "GTRADE_TF_THREADS": "2",
    "GTRADE_CB_THREADS": "1", "GTRADE_MAX_FOLDS": "5", "GTRADE_ADAPTIVE_NETS": "1",
    "GTRADE_NET_WARMSTART": "1", "GTRADE_NET_CAP": "80", "GTRADE_EPOCHS_LSTM": "90",
    "GTRADE_EPOCHS_TF": "60", "GTRADE_EPOCHS_TCN": "50", "TF_CPP_MIN_LOG_LEVEL": "2",
}


def chunk(assets, size):
    return [assets[i:i + size] for i in range(0, len(assets), size)]


def main():
    if os.path.exists(LOCK_PATH):
        print("[retrain] loop cycle running; skipping.")
        return
    approved = loop_state.load_state(STATE_PATH).get("approved", [])
    if not approved:
        print("[retrain] nothing approved.")
        return
    open(LOCK_PATH, "w").close()
    try:
        for i, batch in enumerate(chunk(approved, CHUNK_SIZE), 1):
            env = dict(os.environ)
            env.update(LIGHT_ENV)
            env["GTRADE_ASSETS"] = ",".join(batch)
            print("[retrain] batch %d: %s" % (i, ", ".join(batch)))
            rc = subprocess.run([sys.executable, "train_hybrid.py"],
                                cwd=BASE, env=env).returncode
            if rc != 0:
                print("[retrain] batch %d failed (rc=%d); stopping." % (i, rc))
                return
        st = loop_state.load_state(STATE_PATH)
        st["approved"] = []
        loop_state.save_state(STATE_PATH, st)
        print("[retrain] done. Run predict.py to refresh signals.")
    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)


if __name__ == "__main__":
    main()
