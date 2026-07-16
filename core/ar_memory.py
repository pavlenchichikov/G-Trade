"""Persistent cross-run memory for the auto-research agent.

Three small JSON files next to the other _auto_research state:
- _ar_tried.json: the permanent registry of every evaluated candidate
  signature (nothing is ever re-tested across runs);
- _ar_eval_cache.json: a cache for expensive trainings, keyed by data +
  feature-space version. Base runs key off the env (base_key); re-gate
  candidate runs key off the genome signature (genome_key), because their
  envs embed temp spec-file paths and would never repeat verbatim;
- _ar_findings.json: the cumulative findings journal (one record per run).

An unreadable file is treated as empty (same tolerance as load_state)."""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIED_PATH = os.path.join(BASE, "_ar_tried.json")
CACHE_PATH = os.path.join(BASE, "_ar_eval_cache.json")
FINDINGS_PATH = os.path.join(BASE, "_ar_findings.json")
DB_PATH = os.path.join(BASE, "market.db")
CACHE_CAP = 120
REPLICATION_PATH = os.path.join(BASE, "_ar_replication.json")


def _load(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def tried_seen(kind, sig):
    """Whether this candidate signature was ever evaluated (any past run)."""
    return sig in _load(TRIED_PATH, {}).get(kind, [])


def tried_add(kind, sig):
    reg = _load(TRIED_PATH, {})
    bucket = reg.setdefault(kind, [])
    if sig not in bucket:
        bucket.append(sig)
        _save(TRIED_PATH, reg)


def tried_count():
    return sum(len(v) for v in _load(TRIED_PATH, {}).values())


def tried_recent(kind, n=20):
    """The last n evaluated signatures for a kind (as stored, oldest-first). Fed to
    the LLM proposer as an 'avoid these' list so it stops re-proposing tried candidates."""
    return _load(TRIED_PATH, {}).get(kind, [])[-n:]


def replication_seen(sig):
    """Whether this candidate signature cleared the held-out gate in any PRIOR run."""
    return bool(_load(REPLICATION_PATH, {}).get(sig))


def replication_add(sig, ts):
    """Record a held-out-gate clear for sig at ISO time ts; return the number of
    distinct runs (timestamps) that have now cleared it."""
    reg = _load(REPLICATION_PATH, {})
    stamps = reg.setdefault(sig, [])
    if ts not in stamps:
        stamps.append(ts)
        _save(REPLICATION_PATH, reg)
    return len(stamps)


def findings_append(record):
    journal = _load(FINDINGS_PATH, [])
    journal.append(record)
    _save(FINDINGS_PATH, journal)


def findings_summary():
    """Cumulative counters for the end-of-run print."""
    journal = _load(FINDINGS_PATH, [])
    adoptable = sum(1 for rec in journal
                    for w in rec.get("winners", []) if w.get("adoptable"))
    replicated = sum(1 for rec in journal
                     for w in rec.get("winners", []) if w.get("replicated"))
    return {"experiments": tried_count(), "adoptable": adoptable,
            "replicated": replicated}


def findings_recent(n=20):
    """The last n findings-journal records, newest first (empty on unreadable file)."""
    return list(reversed(_load(FINDINGS_PATH, [])))[:n]


def findings_all():
    """The full findings journal (oldest first)."""
    return _load(FINDINGS_PATH, [])


def data_fingerprint(subset):
    """Newest bar date per asset table of the subset; changes when new data
    arrives. A missing table is a deterministic marker; a whole-DB failure
    returns a unique value (cache MISS, never a wrong hit)."""
    from core.track_record import _table_name
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            parts = []
            for a in subset.split(","):
                t = _table_name(a.strip())
                try:
                    row = con.execute('SELECT MAX(Date) FROM "%s"' % t).fetchone()
                    parts.append("%s=%s" % (t, row[0]))
                except sqlite3.Error:
                    parts.append("%s=?" % t)
            return "|".join(parts)
        finally:
            con.close()
    except Exception:
        return "err-" + uuid.uuid4().hex


def base_key(subset, env):
    """Cache key for a BASE training: same subset + env + feature space +
    data snapshot means the same quality rows."""
    from core.features import feature_version
    payload = json.dumps(
        [subset, env, feature_version(), data_fingerprint(subset)],
        sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def genome_key(subset, gsig, kind=""):
    """Cache key for a CANDIDATE training: the genome signature stands in for the
    env dict (candidate envs embed temp spec-file paths, so the raw env cannot key
    the cache). kind separates the full train from the CB-only screen train. Same
    invalidation rules as base_key: feature space or new data means a MISS."""
    from core.features import feature_version
    payload = json.dumps(
        [subset, gsig, kind, feature_version(), data_fingerprint(subset)],
        sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def cache_get(key):
    entry = _load(CACHE_PATH, {}).get(key)
    return entry["rows"] if entry else None


def cache_put(key, rows):
    cache = _load(CACHE_PATH, {})
    cache[key] = {"rows": rows, "ts": datetime.utcnow().isoformat()}
    if len(cache) > CACHE_CAP:
        oldest = sorted(cache, key=lambda k: cache[k].get("ts", ""))
        for k in oldest[:len(cache) - CACHE_CAP]:
            del cache[k]
    _save(CACHE_PATH, cache)
