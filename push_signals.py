"""Push the latest signals snapshot to Supabase for the Atratus landing.

Reads the per-asset latest signal + accuracy from the local prediction journal
(no models loaded) and upserts:
  - `signals`      : per-asset rows (gated behind the allow-list by RLS)
  - `public_stats` : one anonymized aggregate row (anon-readable teaser)

Run it locally after predict.py / reconcile. It only needs the DB and two env
vars, so it can be scheduled (Task Scheduler) once you are happy with it.

    set SUPABASE_URL=...            (Project URL)
    set SUPABASE_SERVICE_KEY=...    (service_role key - keep secret, local only)
    python push_signals.py
"""

import datetime
import os
import sys

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from core import track_record


def build_payload(signals: list):
    """Turn track_record.latest_signals() into (signal_rows, stats_row)."""
    rows = []
    n_buy = n_sell = n_wait = 0
    accs = []
    max_date = None

    for s in signals:
        action = (s.get("signal") or "WAIT").upper()
        if action == "BUY":
            n_buy += 1
        elif action == "SELL":
            n_sell += 1
        else:
            n_wait += 1

        acc = (s.get("acc") or {}).get("acc")
        if acc is not None:
            accs.append(acc)

        date = s.get("date")
        if date and (max_date is None or date > max_date):
            max_date = date

        rows.append({
            "asset": s["asset"],
            "action": action,
            "prob": s.get("probability"),
            "mode": None,
            "taleb": None,
            "accuracy": acc,
            "snapshot_date": date,
        })

    total = len(signals) or 1
    stats = {
        "id": 1,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_wait": n_wait,
        "accuracy": (sum(accs) / len(accs)) if accs else None,
        "breadth": (n_buy + n_sell) / total,
        "regime": None,
        "sentiment": None,
        "snapshot_date": max_date or datetime.date.today().isoformat(),
    }
    return rows, stats


def _rest_base(url: str) -> str:
    """Normalize a Supabase URL to its REST base, tolerating a pasted
    '/rest/v1' suffix so we never end up with '.../rest/v1/rest/v1'."""
    root = url.strip().rstrip("/")
    if root.endswith("/rest/v1"):
        root = root[: -len("/rest/v1")]
    return root + "/rest/v1"


def push(rows, stats, url: str, key: str):
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Full snapshot: clear the table, then insert fresh (drops delisted assets).
    d = requests.delete(f"{base}/signals?asset=not.is.null", headers=headers, timeout=30)
    d.raise_for_status()
    if rows:
        r = requests.post(
            f"{base}/signals",
            headers={**headers, "Prefer": "resolution=merge-duplicates"},
            json=rows,
            timeout=60,
        )
        r.raise_for_status()
    r = requests.post(
        f"{base}/public_stats?on_conflict=id",
        headers={**headers, "Prefer": "resolution=merge-duplicates"},
        json=stats,
        timeout=30,
    )
    r.raise_for_status()


def main():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (see .env.example).")

    signals = track_record.latest_signals()
    rows, stats = build_payload(signals)
    push(rows, stats, url, key)
    print(f"pushed {len(rows)} signals | "
          f"buy/sell/wait={stats['n_buy']}/{stats['n_sell']}/{stats['n_wait']} | "
          f"as_of={stats['snapshot_date']}")


if __name__ == "__main__":
    main()
