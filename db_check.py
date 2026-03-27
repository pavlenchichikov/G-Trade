"""
DB Check & Fix — проверка и ремонт market.db
  python db_check.py          — только проверка (интерактивное предложение починить)
  python db_check.py --fix    — проверка + автоматическое исправление
  python db_check.py --stats  — только статистика таблиц
"""

import os
import sqlite3
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")


# ── Утилиты ──────────────────────────────────────────────────────────────────

def _connect():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] База данных не найдена: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def get_tables(cur):
    return sorted(
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )


# ── Проверки ─────────────────────────────────────────────────────────────────

def check_duplicates(cur, tables):
    """Таблицы с дубликатами по Date."""
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
    """Даты не в формате YYYY-MM-DD (10 символов)."""
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
    """Одна дата в разных форматах (скрытые дубли)."""
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
    """Строки с NULL в ключевых колонках (Date, Close)."""
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
    """Пустые таблицы (0 строк)."""
    empty = []
    for t in tables:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if cnt == 0:
            empty.append(t)
    return empty


# ── Исправления ──────────────────────────────────────────────────────────────

def fix_date_formats(cur, tables):
    """Нормализует даты > 10 символов → YYYY-MM-DD."""
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
            print(f"    {t}: нормализовано {before} дат")
    return total


def fix_duplicates(cur, tables):
    """Удаляет дубликаты — оставляет MAX(rowid) для каждой даты."""
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
            print(f"    {t}: удалено {removed} дубликатов")
    return total


def fix_nulls(cur, tables):
    """Удаляет строки с NULL Date или NULL Close."""
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
            print(f"    {t}: удалено {removed} строк с NULL")
    return total


def fix_vacuum(conn):
    """VACUUM — сжатие файла БД после удалений."""
    before = os.path.getsize(DB_PATH)
    conn.execute("VACUUM")
    after = os.path.getsize(DB_PATH)
    saved = before - after
    if saved > 0:
        print(f"    VACUUM: {before/1024/1024:.1f} MB → {after/1024/1024:.1f} MB (−{saved/1024:.0f} KB)")
    else:
        print(f"    VACUUM: {after/1024/1024:.1f} MB (без изменений)")
    return saved


# ── Вывод ────────────────────────────────────────────────────────────────────

def print_stats(cur, tables):
    """Статистика по таблицам."""
    print(f"\n  {'Table':<25} {'Rows':>8}  {'Min Date':<12} {'Max Date':<12}")
    print(f"  {'-'*58}")
    for t in tables:
        row = cur.execute(
            f"SELECT COUNT(*), MIN(Date), MAX(Date) FROM {t}"
        ).fetchone()
        cnt, d_min, d_max = row
        print(f"  {t:<25} {cnt:>8}  {d_min or '—':<12} {d_max or '—':<12}")


def run_diagnostics(cur, tables):
    """Запускает все проверки, возвращает dict результатов."""
    W = 60
    print()
    print("=" * W)
    print(f"  DB CHECK  |  market.db  |  {len(tables)} tables")
    print("=" * W)

    results = {}

    # 1
    print("\n  [1/5] Duplicates by Date")
    dups = check_duplicates(cur, tables)
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
    hidden = check_hidden_duplicates(cur, tables)
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

    results["has_problems"] = bool(dups or bad_fmt or hidden or nulls)
    return results


def run_fix(conn, cur, tables, results):
    """Исправляет все найденные проблемы."""
    print()
    print("=" * 60)
    print("  DB FIX  |  Auto-repair")
    print("=" * 60)

    fixed_total = 0

    if results.get("bad_fmt"):
        print("\n  [FIX] Нормализация дат:")
        fixed_total += fix_date_formats(cur, tables)

    if results.get("dups") or results.get("hidden"):
        print("\n  [FIX] Удаление дубликатов:")
        fixed_total += fix_duplicates(cur, tables)

    if results.get("nulls"):
        print("\n  [FIX] Удаление NULL-строк:")
        fixed_total += fix_nulls(cur, tables)

    conn.commit()

    print("\n  [FIX] Сжатие БД:")
    fix_vacuum(conn)

    # Перепроверка
    print(f"\n{'='*60}")
    print("  ПЕРЕПРОВЕРКА")
    print(f"{'='*60}")
    dups2 = check_duplicates(cur, tables)
    bad2 = check_date_formats(cur, tables)
    hidden2 = check_hidden_duplicates(cur, tables)
    nulls2 = check_nulls(cur, tables)

    if not dups2 and not bad2 and not hidden2 and not nulls2:
        print(f"\n  ГОТОВО: исправлено {fixed_total} строк. База чистая.")
    else:
        remaining = len(dups2) + len(bad2) + len(hidden2) + len(nulls2)
        print(f"\n  ВНИМАНИЕ: осталось {remaining} проблем. Запустите ещё раз.")

    return fixed_total


# ── Точки входа ──────────────────────────────────────────────────────────────

def main_check(autofix=False):
    """Проверка с опциональным исправлением."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)

    if not results["has_problems"]:
        print(f"\n{'='*60}")
        print("  РЕЗУЛЬТАТ: База данных чистая")
        print(f"{'='*60}")
        print_stats(cur, tables)
        conn.close()
        return

    print(f"\n{'='*60}")
    print("  РЕЗУЛЬТАТ: Обнаружены проблемы")
    print(f"{'='*60}")

    if autofix:
        do_fix = True
    else:
        answer = input("\nИсправить автоматически? (y/n): ").strip().lower()
        do_fix = answer in ("y", "yes", "д", "да")

    if do_fix:
        run_fix(conn, cur, tables, results)
        print_stats(cur, tables)
    else:
        print("Отменено.")

    conn.close()


def main_fix():
    """Только исправление (без вопросов)."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)

    results = run_diagnostics(cur, tables)

    if not results["has_problems"]:
        print("\n  База чистая — исправлять нечего.")
        print_stats(cur, tables)
        conn.close()
        return

    run_fix(conn, cur, tables, results)
    print_stats(cur, tables)
    conn.close()


def main_stats():
    """Только статистика."""
    conn = _connect()
    cur = conn.cursor()
    tables = get_tables(cur)
    print(f"\n  DB STATS — market.db ({len(tables)} таблиц)")
    print_stats(cur, tables)
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DB Check & Fix для market.db")
    parser.add_argument("--fix", action="store_true", help="Автоматическое исправление")
    parser.add_argument("--stats", action="store_true", help="Только статистика")
    # Обратная совместимость
    parser.add_argument("--autofix", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.stats:
        main_stats()
    elif args.fix or args.autofix:
        main_fix()
    else:
        main_check()
