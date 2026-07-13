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

from config import FULL_ASSET_MAP
from core import track_record

BAR_LIMIT = 180   # bars per asset exported for the mobile price chart
SIG_LIMIT = 90    # prediction_log rows per asset for the mobile track record


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


def _chunked(seq, n=1000):
    """Yield n-sized slices so PostgREST request bodies stay small."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_history_rows(db_path=None, bar_limit=BAR_LIMIT, sig_limit=SIG_LIMIT):
    """Per-asset OHLC bars and prediction_log excerpts for the mobile app.

    Reads market.db directly (price tables are named asset.lower() and store
    columns Date/open/high/low/close). Assets without a price table yet (new,
    untrained) are skipped silently.
    """
    import sqlite3
    db = db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "market.db")
    bars, hist = [], []
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        for asset in FULL_ASSET_MAP:
            try:
                rows = cur.execute(
                    f'SELECT Date, open, high, low, close FROM "{asset.lower()}" '
                    "ORDER BY Date DESC LIMIT ?", (bar_limit,)).fetchall()
            except sqlite3.OperationalError:
                continue
            for d, o, h, lo, c in reversed(rows):
                bars.append({"asset": asset, "date": str(d)[:10], "open": o,
                             "high": h, "low": lo, "close": c})
            try:
                sig = cur.execute(
                    "SELECT date, signal, probability, actual_next_ret, correct "
                    "FROM prediction_log WHERE asset = ? ORDER BY date DESC "
                    "LIMIT ?", (asset, sig_limit)).fetchall()
            except sqlite3.OperationalError:
                sig = []
            for d, s, p, r, cr in reversed(sig):
                hist.append({"asset": asset, "date": str(d)[:10], "signal": s,
                             "prob": p, "actual_next_ret": r, "correct": cr})
    finally:
        con.close()
    return bars, hist


def _acc_number(val):
    """Best-effort (accuracy, n) out of guru_accuracy()'s council payload."""
    if isinstance(val, dict):
        acc = val.get("acc")
        if not isinstance(acc, (int, float)):
            acc = val.get("accuracy")
        n = None
        for key in ("n", "count", "total"):
            candidate = val.get(key)
            if isinstance(candidate, int):
                n = candidate
                break
        return (acc if isinstance(acc, (int, float)) else None,
                n if isinstance(n, int) else None)
    if isinstance(val, (int, float)):
        return val, None
    return None, None


def fetch_guru_rows():
    """Latest Guru Council verdicts + one accuracy summary row (or None)."""
    from core import dashboard
    rows = [{"asset": g["asset"], "verdict": g["verdict"],
             "council_pct": g["pct"], "lynch": g["lynch"],
             "buffett": g["buffett"], "graham": g["graham"],
             "munger": g["munger"], "source": g["source"],
             "date": str(g["date"])[:10] if g.get("date") else None,
             "correct_5d": g["correct_5d"]}
            for g in dashboard.guru_latest()]
    acc, n = _acc_number(dashboard.guru_accuracy().get("council"))
    stats = ({"id": 1, "accuracy": acc, "n": n, "horizon": "60d"}
             if acc is not None else None)
    return rows, stats


def push_history(url, key, bars, hist, guru_rows, guru_stats):
    """Full-refresh upsert of the four mobile snapshot tables."""
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    for table, rows in (("bars", bars), ("signal_history", hist),
                        ("guru", guru_rows)):
        d = requests.delete(f"{base}/{table}?asset=not.is.null",
                            headers=headers, timeout=60)
        d.raise_for_status()
        for chunk in _chunked(rows):
            r = requests.post(
                f"{base}/{table}",
                headers={**headers, "Prefer": "resolution=merge-duplicates"},
                json=chunk, timeout=120)
            r.raise_for_status()
    if guru_stats:
        r = requests.post(
            f"{base}/guru_stats?on_conflict=id",
            headers={**headers, "Prefer": "resolution=merge-duplicates"},
            json=guru_stats, timeout=30)
        r.raise_for_status()


def build_push_text(rows, stats):
    """Notification title/body: top-5 non-WAIT signals ranked by confidence.

    prob is the calibrated up-move probability, so confidence is prob for BUY
    and 1-prob for SELL (ASCII separators only - repo convention).
    """
    ranked = []
    for r in rows:
        action, prob = r.get("action"), r.get("prob")
        if action not in ("BUY", "SELL") or prob is None:
            continue
        conf = prob if action == "BUY" else 1.0 - prob
        ranked.append((conf, r["asset"], action))
    ranked.sort(reverse=True)
    title = f"Atratus: {stats.get('n_buy', 0)} BUY / {stats.get('n_sell', 0)} SELL"
    body = " | ".join(f"{a} {act} {conf * 100:.0f}%" for conf, a, act in ranked[:5])
    return title, body or "New signals snapshot"


def send_push(url, key, rows, stats):
    """Personal FCM push to allow-listed devices. Silent no-op without creds.

    Requires GTRADE_FCM_CREDS (path to the Firebase service-account JSON,
    secret, never committed) and the allowed_device_tokens() RPC. Tokens that
    FCM reports as unregistered are deleted (self-cleanup after reinstalls).
    """
    creds_path = os.getenv("GTRADE_FCM_CREDS")
    if not creds_path:
        return 0
    import firebase_admin
    from firebase_admin import credentials, messaging
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(creds_path))

    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    r = requests.post(f"{base}/rpc/allowed_device_tokens", headers=headers,
                      json={}, timeout=30)
    r.raise_for_status()
    tokens = [row["token"] for row in r.json()]
    if not tokens:
        return 0

    title, body = build_push_text(rows, stats)
    msg = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        tokens=tokens,
    )
    resp = messaging.send_each_for_multicast(msg)
    for i, res in enumerate(resp.responses):
        if res.success:
            continue
        name = res.exception.__class__.__name__ if res.exception else ""
        if name in ("UnregisteredError", "SenderIdMismatchError"):
            requests.delete(f"{base}/device_tokens?token=eq.{tokens[i]}",
                            headers=headers, timeout=30)
    return resp.success_count


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

    # Optional extras for the mobile app - each fail-safe: a failure here must
    # never undo or block the signals upsert above.
    try:
        bars, hist = fetch_history_rows()
        guru_rows, guru_stats = fetch_guru_rows()
        push_history(url, key, bars, hist, guru_rows, guru_stats)
        print(f"pushed history: {len(bars)} bars | {len(hist)} signal rows | "
              f"{len(guru_rows)} guru verdicts")
    except Exception as e:
        print(f"WARNING: history push failed: {e}")
    try:
        sent = send_push(url, key, rows, stats)
        if sent:
            print(f"FCM push sent to {sent} device(s)")
    except Exception as e:
        print(f"WARNING: FCM push failed: {e}")


if __name__ == "__main__":
    main()
