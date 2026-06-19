"""
DB Audit & Fix - audit and repair market.db

  python db_check.py          - full read-only audit (changes nothing)
  python db_check.py --fix    - audit + auto-repair of fixable issues
  python db_check.py --stats  - table statistics only

The audit is split into two blocks:
  FIXABLE (--fix repairs these): duplicates by date, date format, hidden
  duplicates, NULLs, critical OHLC corruption (High<Low, price<=0, negative
  volume). Duplicates are only searched for and fixed in price tables - log
  tables (prediction_log, guru_log) intentionally hold many rows per date
  and are not subject to dedup.
  DATA QUALITY (requires a re-fetch via data_engine, not a DB fix): freshness
  (stale tables), date gaps, coverage vs config, too little data,
  PRAGMA integrity.
"""

import os
import sqlite3
import sys
from datetime import datetime

# Stable UTF-8 to console/launcher pipe (otherwise Cyrillic breaks on cp1251).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")

# Audit thresholds (can be overridden via environment if desired).
STALE_DAILY_DAYS = 7      # daily table with no new bars for longer than this - stale
STALE_WEEKLY_DAYS = 21    # weekly
MAX_GAP_DAYS = 10         # gap between adjacent daily bars larger than this - a hole
GAP_RECENT_DAYS = 730     # don't report gaps older than this (Yahoo gives sparse early history)
MIN_ROWS_TRAIN = 100      # fewer rows than this - too little for training


# -- Utilities ------------------------------------------------------------------

def _connect():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def get_tables(cur):
    return sorted(
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )


# -- Checks -----------------------------------------------------------------

def check_duplicates(cur, tables):
    """Tables with duplicates by Date."""
    problems = {}
    for t in tables:
        rows = cur.execute(
            f"SELECT Date, COUNT(*) AS c FROM {t} GROUP BY Date HAVING c > 1"
        ).fetchall()
        if rows:
            extra = sum(r[1] - 1 for r in rows)
            problems[t] = {"dates": len(rows), "extra_rows": extra}
    return problems


def check_date_formats(cur, tables):
    """Dates not in YYYY-MM-DD format (10 characters)."""
    bad = {}
    for t in tables:
        cnt = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE length(Date) != 10"
        ).fetchone()[0]
        if cnt:
            examples = [
                r[0]
                for r in cur.execute(
                    f"SELECT DISTINCT Date FROM {t} WHERE length(Date) != 10 LIMIT 5"
                ).fetchall()
            ]
            bad[t] = {"count": cnt, "examples": examples}
    return bad


def check_hidden_duplicates(cur, tables):
    """Same date in different formats (hidden duplicates)."""
    hidden = {}
    for t in tables:
        rows = cur.execute(
            f"""SELECT substr(Date,1,10) AS d, COUNT(DISTINCT Date) AS fmts
                FROM {t} GROUP BY d HAVING fmts > 1"""
        ).fetchall()
        if rows:
            hidden[t] = len(rows)
    return hidden


def check_nulls(cur, tables):
    """Rows with NULL in key columns (Date, Close)."""
    bad = {}
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        conditions = []
        if "Date" in cols:
            conditions.append("Date IS NULL")
        for c in cols:
            if c.lower() == "close":
                conditions.append(f"{c} IS NULL")
        if not conditions:
            continue
        cnt = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(conditions)}"
        ).fetchone()[0]
        if cnt:
            bad[t] = cnt
    return bad


def check_empty_tables(cur, tables):
    """Empty tables (0 rows)."""
    empty = []
    for t in tables:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if cnt == 0:
            empty.append(t)
    return empty


# -- Data quality audit (read-only) ----------------------------------------

def _parse_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _price_tables(cur, tables):
    """Only OHLCV asset tables (excludes log tables like guru_log etc.)."""
    out = []
    for t in tables:
        cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
        if {"open", "high", "low", "close"} <= cols:
            out.append(t)
    return out


def check_freshness(cur, tables):
    """How many days ago the last bar was; flags stale tables.

    A stale table is usually a delisted/renamed ticker or a broken fetch
    (see the FIVE-X5, FIXP-FIXR history), not an empty database."""
    today = datetime.now().date()
    stale = {}
    for t in tables:
        d_max = cur.execute(f"SELECT MAX(Date) FROM {t}").fetchone()[0]
        last = _parse_date(d_max)
        if last is None:
            continue
        age = (today - last).days
        limit = STALE_WEEKLY_DAYS if t.endswith("_weekly") else STALE_DAILY_DAYS
        if age > limit:
            stale[t] = age
    return dict(sorted(stale.items(), key=lambda kv: -kv[1]))


def check_ohlc(cur, tables):
    """OHLCV integrity, by severity.

    critical: High<Low, non-positive price, negative volume (data corruption).
    minor:    Open/Close outside the [Low,High] range - a common Yahoo FX feed
              artifact, not corruption, but worth noting."""
    out = {}
    for t in tables:
        cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
        if not {"open", "high", "low", "close"} <= cols:
            continue
        crit = ["High < Low", "Open <= 0", "High <= 0", "Low <= 0", "Close <= 0"]
        if "volume" in cols:
            crit.append("Volume < 0")
        minor = ["Close > High", "Close < Low", "Open > High", "Open < Low"]
        c = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(crit)}").fetchone()[0]
        m = cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(minor)}").fetchone()[0]
        if c or m:
            out[t] = {"critical": c, "minor": m}
    return out


def check_gaps(cur, tables):
    """Biggest gap between adjacent daily bars over the last
    GAP_RECENT_DAYS (old sparse Yahoo history is not counted)."""
    from datetime import timedelta
    cutoff = datetime.now().date() - timedelta(days=GAP_RECENT_DAYS)
    gaps = {}
    for t in tables:
        if t.endswith("_weekly"):
            continue
        dates = [r[0] for r in cur.execute(
            f"SELECT DISTINCT substr(Date,1,10) d FROM {t} ORDER BY d"
        ).fetchall()]
        prev = None
        biggest = 0
        span = None
        for ds in dates:
            d = _parse_date(ds)
            if d is None:
                continue
            # only count a gap if it ends within the recent window
            if prev is not None and d >= cutoff and (d - prev).days > biggest:
                biggest = (d - prev).days
                span = (prev.isoformat(), d.isoformat())
            prev = d
        if biggest > MAX_GAP_DAYS:
            gaps[t] = (biggest, span)
    return dict(sorted(gaps.items(), key=lambda kv: -kv[1][0]))


def check_low_data(cur, tables):
    """Daily tables with too few rows for training."""
    low = {}
    for t in tables:
        if t.endswith("_weekly"):
            continue
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if 0 < cnt < MIN_ROWS_TRAIN:
            low[t] = cnt
    return dict(sorted(low.items(), key=lambda kv: kv[1]))


def _norm_key(key):
    """Table name derived from the asset key (same as in data_engine)."""
    return key.lower().replace("^", "").replace(".", "").replace("-", "")


def check_coverage(tables):
    """Cross-check tables against the asset registry from config: what's missing,
    what's extra, which ones lack a weekly counterpart. None if config is unavailable."""
    try:
        from config import FULL_ASSET_MAP
    except Exception:
        return None
    expected = {_norm_key(k) for k in FULL_ASSET_MAP}
    present = set(tables)
    daily = {t for t in present if not t.endswith("_weekly")}
    return {
        "missing": sorted(expected - daily),
        "orphan": sorted(daily - expected),
        "no_weekly": sorted(a for a in expected
                            if a in daily and f"{a}_weekly" not in present),
    }


def check_integrity(conn):
    """PRAGMA quick_check + file size."""
    res = conn.execute("PRAGMA quick_check").fetchone()[0]
    return {"quick_check": res, "size_mb": os.path.getsize(DB_PATH) / 1024 / 1024}


# -- Fixes --------------------------------------------------------------

def fix_date_formats(cur, tables):
    """Normalizes dates longer than 10 characters to YYYY-MM-DD."""
    total = 0
    for t in tables:
        before = cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE length(Date) > 10"
        ).fetchone()[0]
        if before:
            cur.execute(
                f"UPDATE {t} SET Date = substr(Date,1,10) WHERE length(Date) > 10"
            )
            total += before
            print(f"    {t}: normalized {before} dates")
    return total


def fix_duplicates(cur, tables):
    """Removes duplicates - keeps MAX(rowid) for each date."""
    total = 0
    for t in tables:
        before = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cur.execute(
            f"DELETE FROM {t} WHERE rowid NOT IN "
            f"(SELECT MAX(rowid) FROM {t} GROUP BY Date)"
        )
        after = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        removed = before - after
        if removed:
            total += removed
            print(f"    {t}: removed {removed} duplicates")
    return total


def fix_nulls(cur, tables):
    """Removes rows with NULL Date or NULL Close."""
    total = 0
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        conditions = []
        if "Date" in cols:
            conditions.append("Date IS NULL")
        for c in cols:
            if c.lower() == "close":
                conditions.append(f"{c} IS NULL")
        if not conditions:
            continue
        before = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cur.execute(f"DELETE FROM {t} WHERE {' OR '.join(conditions)}")
        after = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        removed = before - after
        if removed:
            total += removed
            print(f"    {t}: removed {removed} rows with NULL")
    return total


def fix_ohlc(cur, tables):
    """Repairs critical OHLC corruption in price tables (row by row):

      - no positive price at all (O/H/L/C all <=0) - the row is dead, delete it
        (early SHIB: price below storage precision rounded down to zero);
      - some prices are valid - fill zero/negative ones with the median of the
        positive ones, then set High=max, Low=min over the set. Clears High<Low,
        zero O/H/L on tnx/vix (where only Close is valid), and negative Low
        (oil 2020-04 -40.32);
      - Volume<0 - take the absolute value (feed sign error).

    Minor Open/Close deviations outside [Low,High] (Yahoo FX feed artifact) are
    NOT touched. Returns (rows_repaired, rows_deleted)."""
    repaired = deleted = 0
    for t in tables:
        cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
        if not {"open", "high", "low", "close"} <= cols:
            continue
        has_vol = "volume" in cols
        sel_vol = ", Volume" if has_vol else ""
        crit = ["High < Low", "Open <= 0", "High <= 0", "Low <= 0", "Close <= 0"]
        if has_vol:
            crit.append("Volume < 0")
        rows = cur.execute(
            f"SELECT rowid, Open, High, Low, Close{sel_vol} FROM {t} "
            f"WHERE {' OR '.join(crit)}"
        ).fetchall()
        t_rep = t_del = 0
        for row in rows:
            rid, o, h, l, c = row[0], row[1], row[2], row[3], row[4]
            v = row[5] if has_vol else None
            pos = sorted(x for x in (o, h, l, c) if x is not None and x > 0)
            if not pos:
                cur.execute(f"DELETE FROM {t} WHERE rowid = ?", (rid,))
                t_del += 1
                continue
            ref = pos[len(pos) // 2]  # median of the positive prices
            o = o if (o is not None and o > 0) else ref
            h = h if (h is not None and h > 0) else ref
            l = l if (l is not None and l > 0) else ref
            c = c if (c is not None and c > 0) else ref
            hi, lo = max(o, h, l, c), min(o, h, l, c)
            if has_vol:
                nv = abs(v) if v is not None else v
                cur.execute(
                    f"UPDATE {t} SET Open=?, High=?, Low=?, Close=?, Volume=? "
                    f"WHERE rowid=?", (o, hi, lo, c, nv, rid))
            else:
                cur.execute(
                    f"UPDATE {t} SET Open=?, High=?, Low=?, Close=? WHERE rowid=?",
                    (o, hi, lo, c, rid))
            t_rep += 1
        if t_rep or t_del:
            parts = []
            if t_rep:
                parts.append(f"repaired {t_rep}")
            if t_del:
                parts.append(f"deleted {t_del} dead")
            print(f"    {t}: " + ", ".join(parts))
        repaired += t_rep
        deleted += t_del
    return repaired, deleted


def fix_vacuum(conn):
    """VACUUM - compacts the DB file after deletions."""
    before = os.path.getsize(DB_PATH)
    conn.execute("VACUUM")
    after = os.path.getsize(DB_PATH)
    saved = before - after
    if saved > 0:
        print(f"    VACUUM: {before/1024/1024:.1f} MB - {after/1024/1024:.1f} MB (-{saved/1024:.0f} KB)")
    else:
        print(f"    VACUUM: {after/1024/1024:.1f} MB (no change)")
    return saved


# -- Output --------------------------------------------------------------------

def print_stats(cur, tables):
    """Per-table statistics."""
    print(f"\n  {'Table':<25} {'Rows':>8}  {'Min Date':<12} {'Max Date':<12}")
    print(f"  {'-'*58}")
    for t in tables:
        row = cur.execute(
            f"SELECT COUNT(*), MIN(Date), MAX(Date) FROM {t}"
        ).fetchone()
        cnt, d_min, d_max = row
        print(f"  {t:<25} {cnt:>8}  {d_min or '-':<12} {d_max or '-':<12}")


def run_diagnostics(cur, tables):
    """Runs all checks, returns a dict of results."""
    W = 60
    print()
    print("=" * W)
    print(f"  DB CHECK  |  market.db  |  {len(tables)} tables")
    print("=" * W)

    results = {}
    # Duplicates/hidden duplicates are only searched for in price (OHLCV) tables.
    # In log tables (prediction_log, guru_log) multiple rows per date are normal
    # (one row per asset), and auto-dedup by date would wipe out prediction
    # history. _price_tables filters them out (no open/high/low/close).
    price = _price_tables(cur, tables)
    results["price"] = price

    # 1
    print("\n  [1/5] Duplicates by Date")
    dups = check_duplicates(cur, price)
    results["dups"] = dups
    if dups:
        total_extra = sum(v["extra_rows"] for v in dups.values())
        print(f"  [!] Found in {len(dups)} tables  (+{total_extra} extra rows)")
        for t, v in sorted(dups.items()):
            print(f"       {t:<28} {v['dates']} dates  +{v['extra_rows']} rows")
    else:
        print("  [OK] No duplicates")

    # 2
    print("\n  [2/5] Date format (YYYY-MM-DD)")
    bad_fmt = check_date_formats(cur, tables)
    results["bad_fmt"] = bad_fmt
    if bad_fmt:
        print(f"  [!] Found in {len(bad_fmt)} tables")
        for t, v in sorted(bad_fmt.items()):
            print(f"       {t:<28} {v['count']} rows  e.g.: {v['examples'][:2]}")
    else:
        print("  [OK] All dates in correct format")

    # 3
    print("\n  [3/5] Hidden duplicates (mixed date formats)")
    hidden = check_hidden_duplicates(cur, price)
    results["hidden"] = hidden
    if hidden:
        print(f"  [!] Found in {len(hidden)} tables")
        for t, cnt in sorted(hidden.items()):
            print(f"       {t:<28} {cnt} dates")
    else:
        print("  [OK] No hidden duplicates")

    # 4
    print("\n  [4/5] NULL values (Date, Close columns)")
    nulls = check_nulls(cur, tables)
    results["nulls"] = nulls
    if nulls:
        print(f"  [!] Found in {len(nulls)} tables")
        for t, cnt in sorted(nulls.items()):
            print(f"       {t:<28} {cnt} rows")
    else:
        print("  [OK] No NULL values")

    # 5
    print("\n  [5/5] Empty tables")
    empty = check_empty_tables(cur, tables)
    results["empty"] = empty
    if empty:
        print(f"  [!] {len(empty)} empty: {', '.join(empty)}")
    else:
        print("  [OK] No empty tables")

    # Critical OHLC corruption is now also fixable (--fix), so it counts toward
    # the fixable-problems flag. The DATA QUALITY block still prints its report.
    ohlc = check_ohlc(cur, price)
    results["ohlc_crit"] = {t: v["critical"] for t, v in ohlc.items() if v["critical"]}

    results["has_problems"] = bool(
        dups or bad_fmt or hidden or nulls or results["ohlc_crit"]
    )
    return results


def run_quality_audit(conn, cur, tables):
    """Read-only data-quality audit block. Returns the number of warnings."""
    W = 60
    print(f"\n{'='*W}")
    print("  DATA QUALITY  |  OHLC is fixed by --fix; gaps/freshness/low data require a data_engine re-fetch")
    print("=" * W)
    warn = 0
    price = _price_tables(cur, tables)  # OHLCV tables, excluding log tables

    print("\n  [+] Integrity (PRAGMA quick_check)")
    integ = check_integrity(conn)
    if integ["quick_check"] == "ok":
        print(f"  [OK] quick_check ok  |  {integ['size_mb']:.1f} MB  |  {len(price)} price tables")
    else:
        print(f"  [!] quick_check: {integ['quick_check']}"); warn += 1

    print("\n  [+] Freshness (stale tables)")
    stale = check_freshness(cur, price)
    if stale:
        warn += len(stale)
        print(f"  [!] {len(stale)} tables stale "
              f"(daily >{STALE_DAILY_DAYS}d / weekly >{STALE_WEEKLY_DAYS}d):")
        for t, age in list(stale.items())[:20]:
            print(f"       {t:<28} {age} days old")
        if len(stale) > 20:
            print(f"       ... +{len(stale) - 20} more")
    else:
        print("  [OK] all tables fresh")

    print("\n  [+] OHLC integrity")
    ohlc = check_ohlc(cur, price)
    crit = {t: v["critical"] for t, v in ohlc.items() if v["critical"]}
    minor = {t: v["minor"] for t, v in ohlc.items() if v["minor"] and not v["critical"]}
    if crit:
        warn += len(crit)
        print(f"  [!] CRITICAL in {len(crit)} tables (High<Low / price<=0 / volume<0) "
              "- fixed by --fix:")
        for t, c in sorted(crit.items()):
            print(f"       {t:<28} {c} rows")
    else:
        print("  [OK] no critical OHLC corruption")
    if minor:
        rows = sum(minor.values())
        print(f"  [..] minor: {len(minor)} tables, {rows} rows with Open/Close just "
              "outside [Low,High] (Yahoo FX feed quirk, not corruption)")

    print("\n  [+] Date gaps (missing daily bars)")
    gaps = check_gaps(cur, price)
    if gaps:
        warn += len(gaps)
        print(f"  [!] {len(gaps)} tables with a gap > {MAX_GAP_DAYS} days:")
        for t, (g, span) in list(gaps.items())[:15]:
            where = f" ({span[0]}..{span[1]})" if span else ""
            print(f"       {t:<28} {g} days{where}")
        if len(gaps) > 15:
            print(f"       ... +{len(gaps) - 15} more")
    else:
        print("  [OK] no large gaps")

    print(f"\n  [+] Low data (< {MIN_ROWS_TRAIN} rows, weak for training)")
    low = check_low_data(cur, price)
    if low:
        warn += len(low)
        print(f"  [!] {len(low)} thin tables: " +
              ", ".join(f"{t}({c})" for t, c in low.items()))
    else:
        print("  [OK] all daily tables have enough history")

    print("\n  [+] Registry coverage (config.FULL_ASSET_MAP)")
    cov = check_coverage(tables)
    if cov is None:
        print("  [..] config not importable - skipped")
    else:
        if cov["missing"]:
            warn += len(cov["missing"])
            print(f"  [!] {len(cov['missing'])} expected tables MISSING: "
                  + ", ".join(cov["missing"]))
        if cov["no_weekly"]:
            print(f"  [..] {len(cov['no_weekly'])} assets without a _weekly table: "
                  + ", ".join(cov["no_weekly"][:20])
                  + (" ..." if len(cov["no_weekly"]) > 20 else ""))
        if cov["orphan"]:
            print(f"  [..] {len(cov['orphan'])} tables not in config (orphans): "
                  + ", ".join(cov["orphan"]))
        if not cov["missing"] and not cov["orphan"] and not cov["no_weekly"]:
            print("  [OK] tables match the asset registry")

    print(f"\n  Data-quality warnings: {warn}")
    return warn


def run_fix(conn, cur, tables, results):
    """Repairs all problems found."""
    print()
    print("=" * 60)
    print("  DB FIX  |  Auto-repair")
    print("=" * 60)

    fixed_total = 0
    # Dedup only on price tables - log tables must not be touched.
    price = results.get("price") or _price_tables(cur, tables)

    if results.get("bad_fmt"):
        print("\n  [FIX] Normalizing dates:")
        fixed_total += fix_date_formats(cur, tables)

    if results.get("dups") or results.get("hidden"):
        print("\n  [FIX] Removing duplicates:")
        fixed_total += fix_duplicates(cur, price)

    if results.get("nulls"):
        print("\n  [FIX] Removing NULL rows:")
        fixed_total += fix_nulls(cur, tables)

    if results.get("ohlc_crit"):
        print("\n  [FIX] Repairing OHLC (critical corruption):")
        rep, dele = fix_ohlc(cur, price)
        fixed_total += rep + dele

    conn.commit()

    print("\n  [FIX] Compacting DB:")
    fix_vacuum(conn)

    # Re-check
    print(f"\n{'='*60}")
    print("  RE-CHECK")
    print(f"{'='*60}")
    dups2 = check_duplicates(cur, price)
    bad2 = check_date_formats(cur, tables)
    hidden2 = check_hidden_duplicates(cur, price)
    nulls2 = check_nulls(cur, tables)
    ohlc2 = {t: v["critical"] for t, v in check_ohlc(cur, price).items() if v["critical"]}

    if not dups2 and not bad2 and not hidden2 and not nulls2 and not ohlc2:
        print(f"\n  DONE: fixed {fixed_total} rows. Database is clean.")
    else:
        remaining = len(dups2) + len(bad2) + len(hidden2) + len(nulls2) + len(ohlc2)
        print(f"\n  WARNING: {remaining} problems remain. Run again.")

    return fixed_total


# -- Entry points --------------------------------------------------------------

def main_audit():
    """Full read-only audit. Changes nothing (safe to run from the launcher)."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)
    warn = run_quality_audit(conn, cur, tables)

    print(f"\n{'='*60}")
    if not results["has_problems"] and not warn:
        print("  RESULT: database is clean, no issues found")
    else:
        fixable = "yes" if results["has_problems"] else "no"
        print(f"  RESULT: fixable problems - {fixable}; "
              f"quality warnings - {warn}")
        if results["has_problems"]:
            print("  Run  python db_check.py --fix  for auto-repair.")
    print(f"{'='*60}")
    print_stats(cur, tables)
    conn.close()


def main_fix():
    """Audit + auto-repair of fixable problems (no prompts)."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)

    if results["has_problems"]:
        run_fix(conn, cur, tables, results)
    else:
        print("\n  No fixable problems.")

    run_quality_audit(conn, cur, tables)
    print_stats(cur, tables)
    conn.close()


def main_stats():
    """Statistics only."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)
    print(f"\n  DB STATS - market.db ({len(tables)} tables)")
    print_stats(cur, tables)
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DB Audit & Fix for market.db")
    parser.add_argument("--fix", action="store_true", help="Auto-repair fixable problems")
    parser.add_argument("--stats", action="store_true", help="Statistics only")
    # Backward compatibility
    parser.add_argument("--audit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--autofix", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.stats:
        main_stats()
    elif args.fix or args.autofix:
        main_fix()
    else:
        main_audit()
