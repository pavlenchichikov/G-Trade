import os
import sys
import io
import time
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- НАСТРОЙКИ ---
try:
    from config import SOCKS5_PROXY as PROXY_URL
except ImportError:
    PROXY_URL = "socks5h://127.0.0.1:12334"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

from core.logger import get_logger
logger = get_logger("data_engine")

try:
    from config import FULL_ASSET_MAP
except ImportError:
    exit(f"[ERR] config.py не найден в {BASE_DIR}")

engine = create_engine(f'sqlite:///{os.path.join(BASE_DIR, "market.db")}')
_db_lock = threading.Lock()  # serialize SQLite writes
MOEX_TARGETS = [
    "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS",
    "PLZL", "SIBN", "MGNT",
    "TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM",
    "YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM",
    "HHRU", "SOFL", "ASTR", "WUSH",
    "CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP",
    "IRAO", "HYDR", "FLOT", "AFLT", "PIKK",
    "FEES", "UPRO", "MSNG", "NMTP",
    "PHOR", "SGZH",
    "FIVE", "FIXP", "LENT", "MVID",
    "SMLT", "LSRG",
]

def get_last_date(table_name):
    """Узнаем дату последней свечи в базе"""
    try:
        with engine.connect() as conn:
            # Проверяем наличие таблицы
            check = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
            if not check.fetchone(): return None

            # Берем последнюю дату
            res = conn.execute(text(f"SELECT MAX(Date) FROM {table_name}"))
            last_date_str = res.fetchone()[0]
            if last_date_str:
                return pd.to_datetime(last_date_str)
    except Exception as e:
        print(f"SQL Error: {e}")
    return None


def _drop_existing_dates(df, table_name):
    """Убираем из df даты, которые уже есть в БД — защита от дубликатов."""
    try:
        with engine.connect() as conn:
            check = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
            if not check.fetchone():
                return df  # таблицы нет — всё новое
            existing = pd.read_sql(f"SELECT DISTINCT Date FROM {table_name}", conn)
            if existing.empty:
                return df
            existing_set = set(existing['Date'].astype(str).str[:10])
            mask = ~df.index.astype(str).str[:10].isin(existing_set)
            return df[mask]
    except Exception:
        return df  # при ошибке — пишем как есть

def fetch_yahoo_smart(symbol, last_date):
    y_sym = FULL_ASSET_MAP.get(symbol, symbol)
    if symbol in ['SOL', 'BNB', 'DOGE', 'XRP', 'TON', 'BTC', 'ETH'] and '-USD' not in y_sym:
        y_sym = f"{symbol}-USD"

    now_ts = int(datetime.now().timestamp())
    _orig_last_date = last_date

    if last_date is not None:
        start_ts = int(last_date.timestamp()) + 86400
        if start_ts >= now_ts:
            print(f"   -> [YAHOO] {y_sym:<12} [OK] (UP_TO_DATE)")
            return None
    else:
        # Start from ~5 years ago (1825 days) instead of range=max
        # range=max with interval=1d returns quarterly data for long histories
        start_ts = now_ts - 1825 * 86400

    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1d"
    end_ts = max(now_ts - 300, start_ts + 1)
    url = f"{base_url}&period1={start_ts}&period2={end_ts}"

    print(f"   -> [YAHOO] {y_sym:<12}", end=" ", flush=True)

    def _parse_response(r, route_label):
        if r.status_code != 200:
            snippet = (r.text or '')[:180].replace('\n', ' ')
            msg = f"HTTP {r.status_code} | route={route_label} | body={snippet}"
            return None, msg

        payload = r.json()
        chart = payload.get('chart', {})
        err = chart.get('error')
        if err:
            return None, f"Yahoo chart error | route={route_label}: {err}"

        result = chart.get('result') or []
        if not result:
            return None, f"empty result | route={route_label}"

        res = result[0]
        if 'timestamp' not in res:
            return "UP_TO_DATE", None

        q = res['indicators']['quote'][0]
        df = pd.DataFrame({
            'Date': [datetime.fromtimestamp(ts) for ts in res['timestamp']],
            'Open': q['open'], 'Close': q['close'], 'High': q['high'], 'Low': q['low'], 'Volume': q['volume']
        }).dropna()

        df['Date'] = pd.to_datetime(df['Date'])
        if _orig_last_date is not None:
            df = df[df['Date'] > _orig_last_date]

        if df.empty:
            return "NO_NEW", None

        return df.set_index('Date'), None

    def _get_with_retry(req_kwargs, route_label, retries=3):
        """GET с повтором при 5xx (экспоненциальная задержка 2s, 4s, 8s)."""
        for attempt in range(retries):
            try:
                r = requests.get(**req_kwargs)
                if r.status_code >= 500 and attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    logging.warning(f"{y_sym} {route_label} HTTP {r.status_code}, retry {attempt+1}/{retries-1} in {wait}s")
                    time.sleep(wait)
                    continue
                return r, None
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return None, str(e)
        return None, "max retries exceeded"

    # 1) Try via SOCKS proxy
    r, exc = _get_with_retry(
        dict(url=url, headers={'User-Agent': 'Mozilla/5.0'},
             proxies={'https': PROXY_URL}, timeout=10, verify=False),
        "proxy"
    )
    if r is not None:
        parsed, err = _parse_response(r, "proxy")
        if err is None:
            if isinstance(parsed, str) and parsed == "UP_TO_DATE":
                print("[OK] (Up to date via proxy)")
                return None
            if isinstance(parsed, str) and parsed == "NO_NEW":
                print("[OK] (No new data via proxy)")
                return None
            print(f"[OK] (+{len(parsed)} new bars via proxy)")
            return parsed
        logging.warning(f"{y_sym} proxy route issue: {err}")
    else:
        logging.warning(f"{y_sym} proxy route exception: {exc}")

    # 2) Fallback direct (no proxy)
    r, exc = _get_with_retry(
        dict(url=url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10, verify=False),
        "direct"
    )
    if r is not None:
        parsed, err = _parse_response(r, "direct")
        if err is None:
            if isinstance(parsed, str) and parsed == "UP_TO_DATE":
                print("[OK] (Up to date via direct)")
                return None
            if isinstance(parsed, str) and parsed == "NO_NEW":
                print("[OK] (No new data via direct)")
                return None
            print(f"[OK] (+{len(parsed)} new bars via direct)")
            return parsed
        logging.error(f"{y_sym} yahoo fail: {err}")
        print(f"[ERR] ({err})")
        return None
    else:
        msg = f"RequestException direct: {exc}"
        logging.error(f"{y_sym} yahoo fail: {msg}")
        print(f"[ERR] ({msg})")
        return None

def fetch_moex_smart(symbol, last_date):
    clean = symbol.split('.')[0]
    print(f"   -> [🇷🇺 MOEX] {clean:<12}", end=" ", flush=True)

    # MOEX позволяет указать дату старта 'YYYY-MM-DD'
    start_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else "2015-01-01"

    all_data = []
    # Качаем порциями (если история большая)
    for offset in [0, 500, 1000]:
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{clean}/candles.json?interval=24&from={start_str}&start={offset}"
        if clean == "IMOEX":
             url = f"https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX/candles.json?interval=24&from={start_str}&start={offset}"

        try:
            r = requests.get(url, timeout=10)
            data = r.json()['candles']['data']
            if not data: break
            all_data.extend(data)
            cols = r.json()['candles']['columns']
            if len(data) < 500: break # Конец данных
        except Exception: break

    if not all_data:
        print("[OK] (Up to date)"); return None

    df = pd.DataFrame(all_data, columns=cols).rename(columns={'begin': 'Date', 'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'])

    if last_date:
        df = df[df['Date'] > last_date]

    if df.empty:
        print("[OK] (No new data)"); return None

    print(f"[OK] (+{len(df)} new bars)")
    return df.set_index('Date')

def fetch_moex_weekly(symbol, last_date):
    """Fetch weekly (interval=7) OHLCV bars from MOEX ISS API."""
    clean = symbol.split('.')[0]
    print(f"   -> [WEEKLY 🇷🇺] {clean:<12}", end=" ", flush=True)

    start_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else "2015-01-01"

    all_data = []
    cols = None
    for offset in [0, 500, 1000, 1500]:
        if clean == "IMOEX":
            url = (f"https://iss.moex.com/iss/engines/stock/markets/index/"
                   f"boards/SNDX/securities/IMOEX/candles.json"
                   f"?interval=7&from={start_str}&start={offset}")
        else:
            url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
                   f"boards/TQBR/securities/{clean}/candles.json"
                   f"?interval=7&from={start_str}&start={offset}")
        try:
            r = requests.get(url, timeout=10)
            payload = r.json()['candles']
            data = payload['data']
            if not data:
                break
            cols = payload['columns']
            all_data.extend(data)
            if len(data) < 500:
                break
        except Exception:
            break

    if not all_data:
        print("[OK] (Up to date)")
        return None

    df = pd.DataFrame(all_data, columns=cols).rename(columns={
        'begin': 'Date', 'open': 'Open', 'close': 'Close',
        'high': 'High', 'low': 'Low', 'volume': 'Volume',
    })
    df['Date'] = pd.to_datetime(df['Date'])

    if last_date:
        df = df[df['Date'] > last_date]

    if df.empty:
        print("[OK] (No new data)")
        return None

    print(f"[OK] (+{len(df)} weekly bars)")
    return df.set_index('Date')


def fetch_yahoo_weekly(symbol, last_date):
    """Fetch weekly (1wk) OHLCV bars from Yahoo Finance for multi-timeframe analysis."""
    y_sym = FULL_ASSET_MAP.get(symbol, symbol)
    if symbol in ['SOL', 'BNB', 'DOGE', 'XRP', 'TON', 'BTC', 'ETH'] and '-USD' not in y_sym:
        y_sym = f"{symbol}-USD"

    now_ts = int(datetime.now().timestamp())
    _orig_last_date = last_date  # сохраняем для фильтрации

    if last_date is not None:
        # Сдвигаем на 1 день (не неделю) — чтобы не пропустить текущую неделю
        start_ts = int(last_date.timestamp()) + 86400
        if start_ts >= now_ts:
            # Данные уже актуальны
            print(f"   -> [WEEKLY] {y_sym:<12} [OK] (UP_TO_DATE)")
            return None
    else:
        start_ts = now_ts - 1825 * 86400  # 5 years

    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y_sym}?interval=1wk"
    url = f"{base_url}&period1={start_ts}&period2={now_ts}"

    print(f"   -> [WEEKLY] {y_sym:<12}", end=" ", flush=True)

    def _try(proxies):
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'},
                             proxies=proxies, timeout=10, verify=False)
            if r.status_code != 200:
                return None
            chart = r.json().get('chart', {})
            if chart.get('error') or not chart.get('result'):
                return None
            res = chart['result'][0]
            if 'timestamp' not in res:
                return "UP_TO_DATE"
            q = res['indicators']['quote'][0]
            df = pd.DataFrame({
                'Date':   [datetime.fromtimestamp(ts) for ts in res['timestamp']],
                'Open':   q['open'], 'Close': q['close'],
                'High':   q['high'], 'Low':   q['low'],
                'Volume': q['volume'],
            }).dropna()
            df['Date'] = pd.to_datetime(df['Date'])
            if _orig_last_date is not None:
                df = df[df['Date'] > _orig_last_date]
            return df.set_index('Date') if not df.empty else "NO_NEW"
        except Exception as exc:
            logging.warning("Weekly fetch error for %s: %s", y_sym, exc)
            return None

    for proxies in [{'https': PROXY_URL}, {}]:
        result = _try(proxies)
        if result is None:
            continue
        if isinstance(result, str):
            print(f"[OK] ({result})")
            return None
        print(f"[OK] (+{len(result)} weekly bars)")
        return result

    print("[ERR] (failed all routes)")
    return None


def _save_df(df, table_name):
    """Normalize and append DataFrame to SQLite."""
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
    df.index = pd.to_datetime(df.index).normalize()
    df.index = df.index.strftime('%Y-%m-%d')
    df = df[~df.index.duplicated(keep='last')]
    df = _drop_existing_dates(df, table_name)
    if not df.empty:
        df.to_sql(table_name, engine, if_exists='append', index=True)
    return len(df)


def _progress_bar(current, total, width=20):
    filled = int(width * current / total) if total else 0
    bar = '#' * filled + '.' * (width - filled)
    return f"[{bar}] {current}/{total}"


# ---------------------------------------------------------------------------
# Thread-safe stdout proxy: worker threads capture print() to thread-local
# buffers instead of writing to real stdout (contextlib.redirect_stdout is
# NOT thread-safe and corrupts sys.stdout when used from multiple threads).
# ---------------------------------------------------------------------------
_tls = threading.local()
_real_stdout = None  # set in main()


class _StdoutProxy:
    """Proxy that routes write() to a thread-local buffer if one is set."""

    def write(self, text):
        buf = getattr(_tls, 'buf', None)
        if buf is not None:
            buf.write(text)
        elif _real_stdout is not None:
            _real_stdout.write(text)

    def flush(self):
        buf = getattr(_tls, 'buf', None)
        if buf is not None:
            buf.flush()
        elif _real_stdout is not None:
            _real_stdout.flush()

    def isatty(self):
        return _real_stdout.isatty() if _real_stdout else False

    def __getattr__(self, name):
        return getattr(_real_stdout, name) if _real_stdout else None


def _fetch_and_save_daily(n, s):
    """Fetch one asset (daily) and save to DB. Thread-safe."""
    table_name = n.lower().replace("^","").replace(".","").replace("-","")
    last_dt = get_last_date(table_name)
    _tls.buf = io.StringIO()
    try:
        if n in MOEX_TARGETS:
            df = fetch_moex_smart(s, last_dt)
        else:
            df = fetch_yahoo_smart(n, last_dt)
        raw_out = _tls.buf.getvalue()
    except Exception as e:
        logging.error(f"Fetch error {n}: {e}")
        return n, 'ERR', 0
    finally:
        _tls.buf = None
    bars = 0
    if df is not None:
        with _db_lock:
            bars = _save_df(df, table_name)
        status = 'NEW'
    elif '[ERR]' in raw_out or ('ERR' in raw_out.upper() and 'UP_TO_DATE' not in raw_out):
        status = 'ERR'
    else:
        status = 'UP_TO_DATE'
    return n, status, bars


def _fetch_and_save_weekly(n, s):
    """Fetch one asset (weekly) and save to DB. Thread-safe."""
    table_name = n.lower().replace("^","").replace(".","").replace("-","") + "_weekly"
    last_dt = get_last_date(table_name)
    _tls.buf = io.StringIO()
    try:
        if n in MOEX_TARGETS:
            df = fetch_moex_weekly(s, last_dt)
        else:
            df = fetch_yahoo_weekly(n, last_dt)
        raw_out = _tls.buf.getvalue()
    except Exception as e:
        logging.error(f"Weekly fetch error {n}: {e}")
        return n, 'ERR', 0
    finally:
        _tls.buf = None
    bars = 0
    if df is not None:
        with _db_lock:
            bars = _save_df(df, table_name)
        status = 'NEW'
    elif '[ERR]' in raw_out:
        status = 'ERR'
    else:
        status = 'UP_TO_DATE'
    return n, status, bars


def main():
    global _real_stdout
    _real_stdout = sys.stdout
    sys.stdout = _StdoutProxy()

    t_start = time.time()
    W = 60  # line width

    print()
    print('=' * W)
    print('  G-TRADE DATA ENGINE  |  Smart Update V104')
    print(f'  {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}')
    print('=' * W)

    assets = list(FULL_ASSET_MAP.items())
    total = len(assets)

    # ── DAILY ──────────────────────────────────────────────────────────────
    print()
    print('  DAILY BARS')
    print('  ' + '-' * (W - 2))

    # Categorize assets into groups (used for summary display)
    from config import ASSET_TYPES
    cat_map = {}
    for cat, members in ASSET_TYPES.items():
        for m in members:
            cat_map[m] = cat

    group_label = {
        'INDICES & MACRO': 'INDICES', 'COMMODITIES': 'COMMODITIES',
        'US TECH': 'US TECH', 'US HEALTHCARE': 'US OTHER', 'US FINANCE': 'US OTHER',
        'US CONSUMER': 'US OTHER', 'US INDUSTRIAL': 'US OTHER',
        'US SEMI': 'US OTHER', 'US SOFTWARE': 'US OTHER',
        'CRYPTO': 'CRYPTO',
        'FOREX MAJORS': 'FOREX', 'FOREX CROSSES': 'FOREX', 'FOREX EXOTIC': 'FOREX',
        'RUS BLUE CHIPS': 'MOEX', 'RUS FINANCE': 'MOEX',
        'RUS TECH': 'MOEX', 'RUS METALS': 'MOEX', 'RUS INFRA': 'MOEX',
        'RUS CONSUMER': 'MOEX', 'RUS PROPERTY': 'MOEX',
    }

    results = []
    _done = [0]
    _done_lock = threading.Lock()
    _is_tty = sys.stdout.isatty()

    def _track_daily(future):
        res = future.result()
        results.append(res)
        name, status, bars = res
        with _done_lock:
            _done[0] += 1
            if _is_tty:
                sys.stdout.write(f"\r  Fetching {_done[0]}/{total}  {name:<12}")
                sys.stdout.flush()
            else:
                # GUI / pipe: one line per asset
                if status == 'NEW':
                    tag = f'+{bars} bars'
                elif status == 'ERR':
                    tag = 'ERROR'
                else:
                    tag = 'up to date'
                print(f"  [{_done[0]:>2}/{total}] {name:<14} {tag}")

    with ThreadPoolExecutor(max_workers=12) as executor:
        futs = {executor.submit(_fetch_and_save_daily, n, s): (n, s)
                for n, s in assets}
        for fut in as_completed(futs):
            _track_daily(fut)
    if _is_tty:
        print()

    stats = {
        'ok':       sum(1 for _, st, _ in results if st == 'NEW'),
        'uptodate': sum(1 for _, st, _ in results if st == 'UP_TO_DATE'),
        'err':      sum(1 for _, st, _ in results if st == 'ERR'),
        'new_bars': sum(b for _, _, b in results),
    }

    # Quick summary line (visible in GUI)
    if stats['ok'] == 0 and stats['err'] == 0:
        print(f"  >> All {stats['uptodate']} assets up to date")
    else:
        parts = []
        if stats['ok']:
            parts.append(f"{stats['ok']} updated (+{stats['new_bars']} bars)")
        if stats['uptodate']:
            parts.append(f"{stats['uptodate']} up to date")
        if stats['err']:
            parts.append(f"{stats['err']} errors")
        print(f"  >> {' | '.join(parts)}")

    # Print grouped summary (terminal only — GUI already shows per-asset lines)
    if _is_tty:
        result_map = {n: (st, b) for n, st, b in results}
        group_order = ['INDICES', 'COMMODITIES', 'US TECH', 'US OTHER', 'CRYPTO', 'MOEX', 'FOREX', 'OTHER']
        label_map = {
            'INDICES': 'Indices & Macro', 'COMMODITIES': 'Commodities',
            'US TECH': 'US Tech', 'US OTHER': 'US Sectors',
            'CRYPTO': 'Crypto',
            'MOEX': 'MOEX (Russia)', 'FOREX': 'Forex', 'OTHER': 'Other',
        }
        for grp in group_order:
            members = [n for n, s in assets
                       if group_label.get(cat_map.get(n, ''), 'OTHER') == grp and n in result_map]
            if not members:
                continue
            print()
            print(f"  {label_map.get(grp, grp)}")
            for name in members:
                status, bars = result_map[name]
                if status == 'NEW':
                    tag = f'+{bars} bars'
                    mark = '[+]'
                elif status == 'ERR':
                    tag = 'ERROR'
                    mark = '[!]'
                else:
                    tag = 'up to date'
                    mark = '   '
                print(f"    {mark} {name:<14} {tag}")

    # ── WEEKLY ─────────────────────────────────────────────────────────────
    weekly_assets = list(assets)  # all assets including MOEX
    total_w = len(weekly_assets)

    print()
    print()
    print('  WEEKLY BARS')
    print('  ' + '-' * (W - 2))

    w_results = []
    _done_w = [0]

    def _track_weekly(future):
        res = future.result()
        w_results.append(res)
        name, status, bars = res
        with _done_lock:
            _done_w[0] += 1
            if _is_tty:
                sys.stdout.write(f"\r  Fetching {_done_w[0]}/{total_w}  {name:<12}")
                sys.stdout.flush()
            else:
                if status == 'NEW':
                    tag = f'+{bars} bars'
                elif status == 'ERR':
                    tag = 'ERROR'
                else:
                    tag = 'up to date'
                print(f"  [{_done_w[0]:>2}/{total_w}] {name:<14} {tag}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futs_w = {executor.submit(_fetch_and_save_weekly, n, s): (n, s)
                  for n, s in weekly_assets}
        for fut in as_completed(futs_w):
            _track_weekly(fut)
    if _is_tty:
        print()

    w_ok   = sum(1 for _, st, _ in w_results if st == 'NEW')
    w_upd  = sum(1 for _, st, _ in w_results if st == 'UP_TO_DATE')
    w_err  = sum(1 for _, st, _ in w_results if st == 'ERR')
    w_bars = sum(b for _, _, b in w_results)

    # Quick weekly summary line
    if w_ok == 0 and w_err == 0:
        print(f"  >> All {w_upd} weekly assets up to date")
    else:
        parts = []
        if w_ok:
            parts.append(f"{w_ok} updated (+{w_bars} bars)")
        if w_upd:
            parts.append(f"{w_upd} up to date")
        if w_err:
            parts.append(f"{w_err} errors")
        print(f"  >> {' | '.join(parts)}")

    # ── SUMMARY ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print('=' * W)
    print('  SUMMARY')
    print('  ' + '-' * (W - 2))
    print(f'  Daily   : {stats["ok"]} updated  |  {stats["uptodate"]} current  |  {stats["err"]} errors  |  +{stats["new_bars"]} bars')
    print(f'  Weekly  : {w_ok} updated  |  {w_upd} current  |  {w_err} errors  |  +{w_bars} bars')
    print(f'  Time    : {elapsed:.1f}s')
    print('=' * W)
    print()

if __name__ == "__main__": main()
