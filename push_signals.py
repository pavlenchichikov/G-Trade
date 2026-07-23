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

If SOCKS5_PROXY is set in .env, all Supabase traffic is routed through it and
every request is retried with backoff - Supabase is EU-hosted and a direct
connection is reset on some networks, which used to abort the bulk history push.
"""

import datetime
import os
import socket
import sys
import time
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from config import FULL_ASSET_MAP
from core import timing_policy, track_record

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

        t_act = s.get("timing_action")
        t_rsn = s.get("timing_reason")
        _text, _div = timing_policy.display_label(t_act, t_rsn)
        t_label = _text if _div else None

        rows.append({
            "asset": s["asset"],
            "action": action,
            "prob": s.get("probability"),
            "mode": None,
            "taleb": None,
            "accuracy": acc,
            "snapshot_date": date,
            "timing_action": t_act,
            "timing_reason": t_rsn,
            "timing_label": t_label,
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


CHUNK = 300  # rows per PostgREST upsert; smaller bodies survive a shaky tunnel

# Errors worth retrying: a VPN tunnel drops the connection mid-transfer
# (ConnectionReset / SSL EOF / timeout). A non-2xx HTTP status is NOT retried.
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


_proxy_resolved = False
_proxy_value = None


def _alive(host, port, timeout=1.5):
    """True if something is listening on host:port (a quick TCP probe)."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _proxies():
    """Route Supabase traffic through SOCKS5_PROXY when it is actually up
    (Supabase is EU and a direct connection is reset on some networks). If the
    proxy is unset or not listening, connect directly - so a system-wide VPN
    keeps working without a local proxy. Resolved once per run (like net.py)."""
    global _proxy_resolved, _proxy_value
    if _proxy_resolved:
        return _proxy_value
    _proxy_resolved = True
    p = os.getenv("SOCKS5_PROXY")
    if not p:
        return None
    parsed = urlparse(p)
    if parsed.hostname and parsed.port and _alive(parsed.hostname, parsed.port):
        _proxy_value = {"http": p, "https": p}
    else:
        print(f"note: SOCKS5_PROXY {parsed.hostname}:{parsed.port} not reachable "
              "- connecting directly")
    return _proxy_value


def _send(method, url, *, retries=5, **kw):
    """One Supabase request via the optional SOCKS5 proxy, retried with backoff
    on transient network errors so a single reset does not abort a bulk push.
    Raises on a non-2xx status or once retries are exhausted."""
    kw.setdefault("proxies", _proxies())
    delay = 1.5
    for attempt in range(retries):
        try:
            r = requests.request(method, url, **kw)
            r.raise_for_status()
            return r
        except _TRANSIENT:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise RuntimeError("unreachable")


def _chunked(seq, n=CHUNK):
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
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    for table, rows in (("bars", bars), ("signal_history", hist),
                        ("guru", guru_rows)):
        _send("DELETE", f"{base}/{table}?asset=not.is.null",
              headers=headers, timeout=60)
        for chunk in _chunked(rows):
            _send("POST", f"{base}/{table}", headers=merge, json=chunk,
                  timeout=120)
    if guru_stats:
        _send("POST", f"{base}/guru_stats?on_conflict=id", headers=merge,
              json=guru_stats, timeout=30)


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
    if not os.path.exists(creds_path):
        # Configured but the service-account JSON is not there yet: treat it as
        # "push not set up" and skip quietly, rather than raising every run.
        print(f"note: GTRADE_FCM_CREDS points to {creds_path}, which does not "
              "exist - skipping FCM push (add the Firebase service-account "
              "JSON there to enable notifications)")
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
    r = _send("POST", f"{base}/rpc/allowed_device_tokens", headers=headers,
              json={}, timeout=30)
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
            _send("DELETE", f"{base}/device_tokens?token=eq.{tokens[i]}",
                  headers=headers, timeout=30)
    return resp.success_count


def push(rows, stats, url: str, key: str):
    base = _rest_base(url)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    merge = {**headers, "Prefer": "resolution=merge-duplicates"}
    # Full snapshot: clear the table, then insert fresh (drops delisted assets).
    _send("DELETE", f"{base}/signals?asset=not.is.null", headers=headers, timeout=30)
    for chunk in _chunked(rows):
        _send("POST", f"{base}/signals", headers=merge, json=chunk, timeout=120)
    _send("POST", f"{base}/public_stats?on_conflict=id", headers=merge,
          json=stats, timeout=30)


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
