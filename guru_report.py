"""
guru_report.py — Console Guru Council report with full financial data.
Shows raw fundamentals + guru verdicts for any asset or all assets.

Usage:
    python guru_report.py              # Все активы (краткий)
    python guru_report.py TSLA         # Один актив (полный отчёт)
    python guru_report.py TSLA SBER    # Несколько активов
    python guru_report.py --all        # Все активы (полный отчёт)
    python guru_report.py --sector US  # По группе (US, RUS, CRYPTO, COMMODITY)
"""

import os
import sys
import warnings
import math

import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import FULL_ASSET_MAP
from sqlalchemy import create_engine

DB_PATH = os.path.join(BASE_DIR, "market.db")
_engine = create_engine(f"sqlite:///{DB_PATH}")

# ── MOEX list (no yfinance fundamentals) ──
MOEX_ASSETS = {
    "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS",
    "PLZL", "SIBN", "MGNT", "TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM",
    "YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM",
    "HHRU", "SOFL", "ASTR", "WUSH",
    "CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP",
    "IRAO", "HYDR", "FLOT", "AFLT", "PIKK",
    "FEES", "UPRO", "MSNG", "NMTP",
    "PHOR", "SGZH", "FIVE", "FIXP", "LENT", "MVID",
    "SMLT", "LSRG",
}

CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XRP", "TON", "DOGE", "BNB",
                 "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR"}
COMMODITY_ASSETS = {"GOLD", "SILVER", "OIL", "GAS"}

US_ASSETS = {"NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "COIN", "MSTR",
             "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK",
             "JPM", "BAC", "GS", "V", "MA", "WFC",
             "WMT", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX", "SBUX",
             "BA", "CAT", "XOM", "CVX", "COP",
             "INTC", "QCOM", "AVGO", "MU",
             "CRM", "ORCL", "ADBE", "UBER", "PYPL"}

# Sectors for --sector filter
SECTORS = {
    "US": US_ASSETS,
    "RUS": MOEX_ASSETS,
    "CRYPTO": CRYPTO_ASSETS,
    "COMMODITY": COMMODITY_ASSETS,
    "INDEX": {"SP500", "NASDAQ", "DOW", "IMOEX", "VIX", "DXY", "TNX"},
}

# ── BACKUP fundamentals (same as app.py) ──
GLOBAL_BACKUP = {
    'SBER': {'pe': 4.2, 'roe': 0.24, 'debt': 0.0, 'div': 11.5},
    'GAZP': {'pe': 3.5, 'roe': 0.04, 'debt': 1.8, 'div': 0.0},
    'LKOH': {'pe': 5.5, 'roe': 0.18, 'debt': 0.2, 'div': 16.0},
    'ROSN': {'pe': 5.9, 'roe': 0.15, 'debt': 0.9, 'div': 9.0},
    'NVTK': {'pe': 7.1, 'roe': 0.19, 'debt': 0.4, 'div': 8.5},
    'YNDX': {'pe': 14.5, 'roe': 0.25, 'debt': 0.1, 'div': 3.5},
    'TCSG': {'pe': 7.8, 'roe': 0.32, 'debt': 0.0, 'div': 5.0},
    'OZON': {'pe': 99.0, 'roe': -0.10, 'debt': 2.5, 'div': 0.0},
    'VKCO': {'pe': 99.0, 'roe': -0.05, 'debt': 1.2, 'div': 0.0},
    'POSI': {'pe': 25.0, 'roe': 0.35, 'debt': 0.1, 'div': 3.5},
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yf_deep(symbol):
    """Full yfinance fetch: .info + financials + balance sheet + cashflow."""
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        if info.get('currentPrice', 0) == 0 and info.get('regularMarketPrice', 0) == 0:
            return None

        result = {'_info': info, '_source': 'yfinance_live'}

        # Core metrics
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        result['price'] = price
        result['pe'] = info.get('trailingPE') or info.get('forwardPE') or 0
        result['fwd_pe'] = info.get('forwardPE') or 0
        result['peg_ratio'] = info.get('pegRatio') or 0
        result['roe'] = info.get('returnOnEquity') or 0
        result['debt_equity'] = (info.get('debtToEquity') or 0) / 100
        result['growth'] = info.get('earningsGrowth') or info.get('revenueGrowth') or 0
        result['revenue_growth'] = info.get('revenueGrowth') or 0
        result['profit_margin'] = info.get('profitMargins') or 0
        result['gross_margin'] = info.get('grossMargins') or 0
        result['operating_margin'] = info.get('operatingMargins') or 0
        result['fcf'] = info.get('freeCashflow') or 0
        result['market_cap'] = info.get('marketCap') or 0
        result['book_value'] = info.get('bookValue') or 0
        result['eps'] = info.get('trailingEps') or 0
        result['dividend_yield'] = (info.get('dividendYield') or 0) * 100
        result['payout_ratio'] = info.get('payoutRatio') or 0
        result['beta'] = info.get('beta') or 0
        result['current_ratio'] = info.get('currentRatio') or 0
        result['quick_ratio'] = info.get('quickRatio') or 0
        result['sector'] = info.get('sector', '')
        result['industry'] = info.get('industry', '')
        result['shares'] = info.get('sharesOutstanding') or 0

        # Quarterly financials
        try:
            fin = t.quarterly_financials
            if fin is not None and not fin.empty:
                result['_financials'] = fin
                if 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].dropna().sort_index()
                    result['revenue_quarters'] = rev
                for label in ['Net Income', 'Net Income From Continuing Operations']:
                    if label in fin.index:
                        ni = fin.loc[label].dropna().sort_index()
                        result['net_income_quarters'] = ni
                        break
                if 'Gross Profit' in fin.index:
                    result['gross_profit_quarters'] = fin.loc['Gross Profit'].dropna().sort_index()
                for label in ['Operating Income', 'EBIT']:
                    if label in fin.index:
                        result['operating_income_quarters'] = fin.loc[label].dropna().sort_index()
                        break
        except Exception:
            pass

        # Balance sheet
        try:
            bs = t.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                result['_balance_sheet'] = bs
                latest = bs.iloc[:, 0]
                for label in ['Total Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['total_assets'] = float(latest[label])
                for label in ['Total Liabilities Net Minority Interest', 'Total Liab']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['total_liabilities'] = float(latest[label])
                        break
                for label in ['Current Assets', 'Total Current Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['current_assets'] = float(latest[label])
                        break
                for label in ['Current Liabilities', 'Total Current Liabilities']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['current_liabilities'] = float(latest[label])
                        break
                for label in ['Tangible Book Value', 'Net Tangible Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['tangible_bv'] = float(latest[label])
                        break
                if 'Retained Earnings' in latest.index and pd.notna(latest['Retained Earnings']):
                    result['retained_earnings'] = float(latest['Retained Earnings'])
                for label in ['Cash And Cash Equivalents', 'Cash Financial']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['cash'] = float(latest[label])
                        break
                for label in ['Total Debt', 'Long Term Debt']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['total_debt'] = float(latest[label])
                        break
        except Exception:
            pass

        # Cash flow
        try:
            cf = t.quarterly_cashflow
            if cf is not None and not cf.empty:
                result['_cashflow'] = cf
                latest_cf = cf.iloc[:, 0]
                for label in ['Operating Cash Flow', 'Total Cash From Operating Activities']:
                    if label in latest_cf.index and pd.notna(latest_cf[label]):
                        result['operating_cf'] = float(latest_cf[label])
                        break
                for label in ['Capital Expenditure', 'Capital Expenditures']:
                    if label in latest_cf.index and pd.notna(latest_cf[label]):
                        result['capex'] = float(latest_cf[label])
                        break
                if 'Free Cash Flow' in cf.index:
                    fcf_series = cf.loc['Free Cash Flow'].dropna().sort_index()
                    result['fcf_quarters'] = fcf_series
        except Exception:
            pass

        return result
    except Exception as e:
        return None


def fetch_smartlab_data():
    """Fetch Smart-Lab fundamentals for Russian stocks."""
    url = "https://smart-lab.ru/q/shares_fundamental/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    }
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=False)
        if r.status_code != 200:
            return {}
        import io
        dfs = pd.read_html(io.StringIO(r.text))
        if not dfs:
            return {}
        f_map = {}
        df0 = dfs[0]
        if 'Тикер' in df0.columns and 'P/E' in df0.columns:
            for _, row in df0.iterrows():
                try:
                    t = str(row['Тикер'])
                    pe_s = str(row['P/E']).replace('%', '').replace('\xa0', '').strip()
                    pe = float(pe_s) if pe_s not in ('-', 'nan', 'None', '') else 99.0
                    debt_col = row.get('долг/EBITDA', row.get('Долг/EBITDA'))
                    debt_s = str(debt_col).replace('%', '').replace('\xa0', '').strip()
                    debt = float(debt_s) if debt_s not in ('-', 'nan', 'None', '') else 0.0
                    div_col = row.get('ДД ао, %', row.get('Див. доход, %'))
                    div_s = str(div_col).replace('%', '').replace('\xa0', '').strip()
                    div = float(div_s) if div_s not in ('-', 'nan', 'None', '') else 0.0
                    f_map[t] = {'pe': pe, 'roe': 0.0, 'debt': debt, 'div': div, '_source': 'smartlab'}
                except Exception:
                    continue
        if len(dfs) > 1:
            df1 = dfs[1]
            roe_col = [c for c in df1.columns if c.lower() == 'roe']
            if 'Тикер' in df1.columns and 'P/E' in df1.columns:
                for _, row in df1.iterrows():
                    try:
                        t = str(row['Тикер'])
                        pe_s = str(row['P/E']).replace('%', '').replace('\xa0', '').strip()
                        pe = float(pe_s) if pe_s not in ('-', 'nan', 'None', '') else 99.0
                        roe = 0.0
                        if roe_col:
                            roe_s = str(row[roe_col[0]]).replace('%', '').replace('\xa0', '').strip()
                            roe = float(roe_s) / 100 if roe_s not in ('-', 'nan', 'None', '') else 0.0
                        div_col = row.get('ДД ао, %', row.get('Див. доход, %'))
                        div_s = str(div_col).replace('%', '').replace('\xa0', '').strip()
                        div = float(div_s) if div_s not in ('-', 'nan', 'None', '') else 0.0
                        f_map[t] = {'pe': pe, 'roe': roe, 'debt': 0.0, 'div': div, '_source': 'smartlab'}
                    except Exception:
                        continue
        return f_map
    except Exception:
        return {}


def get_technical(name):
    """Load technical data from SQLite."""
    table = name.lower().replace("^", "").replace(".", "").replace("-", "")
    try:
        df = pd.read_sql(table, _engine, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        df = df[~df.index.duplicated(keep='last')].sort_index()

        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_num(val, fmt=",.0f"):
    """Format number with fallback."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "—"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_money(val):
    if val is None:
        return "—"
    if abs(val) >= 1e12:
        return f"${val/1e12:,.1f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.1f}M"
    return f"${val:,.0f}"


def _fmt_pct(val, mult=1):
    if val is None:
        return "—"
    return f"{val * mult:.1f}%"


def _bar(val, max_val=100, width=20):
    """Simple ASCII bar chart."""
    if val is None or max_val == 0:
        return ""
    filled = int(min(val / max_val, 1.0) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ══════════════════════════════════════════════════════════════════════════════
# GURU ALGORITHMS (mirror of app.py)
# ══════════════════════════════════════════════════════════════════════════════

def _calc_peg(pe, growth):
    if pe <= 0 or growth <= 0:
        return None
    return pe / (growth * 100)


def _calc_graham_number(eps, bv):
    if eps <= 0 or bv <= 0:
        return None
    return math.sqrt(22.5 * eps * bv)


def _tech_context(df):
    if df is None or len(df) < 50:
        return None
    last = df.iloc[-1]
    close = last['close']
    window = min(252, len(df))
    series = df['close'].iloc[-window:]
    p_min, p_max = series.min(), series.max()
    pct_52w = (close - p_min) / (p_max - p_min) * 100 if p_max > p_min else 50
    rets = df['close'].pct_change().dropna()
    vol_30d = float(rets.iloc[-30:].std() * (252 ** 0.5)) if len(rets) >= 30 else 0
    return {
        'close': close,
        'rsi': last.get('RSI', 50),
        'above_50': close > last.get('SMA_50', close),
        'above_200': close > last.get('SMA_200', close),
        'pct_52w': pct_52w,
        'vol_30d': vol_30d,
        'macd_bull': last.get('MACD_hist', 0) > 0,
    }


def run_guru(name, fund, tech):
    """Run all 4 gurus, return results dict."""
    results = {}

    # --- Lynch ---
    if fund:
        pe = fund.get('pe', 0)
        growth = fund.get('growth', 0)
        peg = fund.get('peg_ratio') or _calc_peg(pe, growth)
        if peg and peg > 0:
            if peg < 1.0:
                results['lynch'] = ("[OK] BUY", f"PEG={peg:.2f} P/E={pe:.1f} Growth={growth:.0%}", 2)
            elif peg < 2.0:
                results['lynch'] = ("[--] FAIR", f"PEG={peg:.2f} P/E={pe:.1f} Growth={growth:.0%}", 1)
            else:
                results['lynch'] = ("[OFF] EXP", f"PEG={peg:.2f} P/E={pe:.1f} Growth={growth:.0%}", 0)
        elif pe > 0:
            if pe < 12:
                results['lynch'] = ("[OK] CHEAP", f"P/E={pe:.1f} (нет growth)", 2)
            elif pe < 25:
                results['lynch'] = ("[--] FAIR", f"P/E={pe:.1f} (нет growth)", 1)
            else:
                results['lynch'] = ("[OFF] EXP", f"P/E={pe:.1f}", 0)
    if 'lynch' not in results and tech:
        s = int(tech['above_50']) + int(tech['above_200'])
        results['lynch'] = (["[OFF]", "[--]", "[OK]"][s], f"SMA50/200 trend score={s}", s)
    if 'lynch' not in results:
        results['lynch'] = ("[--] N/A", "Нет данных", 0)

    # --- Buffett ---
    if fund:
        roe = fund.get('roe', 0)
        debt = fund.get('debt_equity', 0)
        gross_m = fund.get('gross_margin', 0)
        margin = fund.get('profit_margin', 0)
        div_y = fund.get('dividend_yield', 0)
        fcf = fund.get('fcf', 0)
        sc = 0
        if roe > 0.20: sc += 2
        elif roe > 0.15: sc += 1
        if debt < 0.5: sc += 2
        elif debt < 1.5: sc += 1
        if gross_m > 0.40: sc += 1
        elif margin > 0.20: sc += 1
        if div_y > 3: sc += 1
        if fcf and fcf > 0: sc += 1
        retained = fund.get('retained_earnings', 0)
        if retained and retained > 0: sc += 1
        if sc >= 6:
            results['buffett'] = ("[TOP] GEM", f"Score={sc} ROE={roe:.0%} D/E={debt:.1f}", 2)
        elif sc >= 3:
            results['buffett'] = ("[OK] QUALITY", f"Score={sc} ROE={roe:.0%} D/E={debt:.1f}", 1)
        else:
            results['buffett'] = ("[!] WEAK", f"Score={sc} ROE={roe:.0%} D/E={debt:.1f}", 0)
    elif tech:
        results['buffett'] = ("[--] TECH", f"RSI={tech['rsi']:.0f} SMA200={'+' if tech['above_200'] else '-'}", int(tech['above_200']))
    else:
        results['buffett'] = ("[--] N/A", "Нет данных", 0)

    # --- Graham ---
    if fund:
        pe = fund.get('pe', 0)
        eps = fund.get('eps', 0)
        bv = fund.get('book_value', 0)
        price = fund.get('price', 0)
        debt = fund.get('debt_equity', 0)
        cr = fund.get('current_ratio', 0)
        sc = 0
        gn = _calc_graham_number(eps, bv)
        ncav = fund.get('ncav_per_share') if 'ncav_per_share' in fund else None
        if ncav and price > 0 and ncav > price: sc += 3
        if gn and price > 0 and gn > price * 1.2: sc += 2
        elif gn and price > 0 and gn > price: sc += 1
        if 0 < pe < 10: sc += 2
        elif 0 < pe < 15: sc += 1
        if cr > 2: sc += 1
        if debt < 0.5: sc += 1
        gn_str = f"GN=${gn:.0f}" if gn else "GN=N/A"
        results['graham'] = (
            "[OK] BUY" if sc >= 5 else "[--] FAIR" if sc >= 2 else "[OFF] EXP",
            f"Score={sc} {gn_str} P/E={pe:.1f} CR={cr:.1f}",
            2 if sc >= 5 else 1 if sc >= 2 else 0
        )
    elif tech:
        pct = tech['pct_52w']
        results['graham'] = (
            "[OK]" if pct < 25 else "[--]" if pct < 60 else "[OFF]",
            f"52W={pct:.0f}%", 2 if pct < 25 else 1 if pct < 60 else 0
        )
    else:
        results['graham'] = ("[--] N/A", "Нет данных", 0)

    # --- Munger ---
    risk_score = 0
    risk_notes = []
    if fund:
        if fund.get('debt_equity', 0) > 3: risk_score += 2; risk_notes.append("Долг>3x")
        elif fund.get('debt_equity', 0) > 1.5: risk_score += 1; risk_notes.append(f"Долг={fund['debt_equity']:.1f}x")
        if fund.get('roe', 0) < 0: risk_score += 2; risk_notes.append("ROE<0")
        if fund.get('pe', 0) > 50: risk_score += 1; risk_notes.append(f"P/E={fund['pe']:.0f}")
        if fund.get('profit_margin', 0) < 0: risk_score += 2; risk_notes.append("Margin<0")
        rev_qoq = fund.get('revenue_qoq', 0)
        if rev_qoq and rev_qoq < -0.10: risk_score += 2; risk_notes.append(f"RevQoQ={rev_qoq:.0%}")
        fcf = fund.get('fcf', 0)
        if fcf and fcf < 0: risk_score += 1; risk_notes.append("FCF<0")
        payout = fund.get('payout_ratio', 0)
        if payout > 1.0: risk_score += 1; risk_notes.append(f"Payout={payout:.0%}")
    if tech:
        if tech['rsi'] > 80: risk_score += 2; risk_notes.append(f"RSI={tech['rsi']:.0f}")
        elif tech['rsi'] > 70: risk_score += 1; risk_notes.append(f"RSI={tech['rsi']:.0f}")
        if tech['vol_30d'] > 0.60: risk_score += 1; risk_notes.append(f"Vol={tech['vol_30d']:.0%}")
    if risk_score == 0:
        results['munger'] = ("[OK] CLEAN", "Рисков нет", 2)
    elif risk_score >= 5:
        results['munger'] = ("[!!] DANGER", " | ".join(risk_notes[:4]), 0)
    elif risk_score >= 2:
        results['munger'] = ("[!] WARNING", " | ".join(risk_notes[:4]), 1)
    else:
        results['munger'] = ("[OK] MINOR", " | ".join(risk_notes[:3]), 2)

    # --- Council Vote ---
    total = results['lynch'][2] + results['buffett'][2] + results['graham'][2] + results['munger'][2]
    pct = total / 8 * 100
    if pct >= 75:
        results['council'] = ("BUY", pct, total)
    elif pct >= 50:
        results['council'] = ("HOLD", pct, total)
    else:
        results['council'] = ("AVOID", pct, total)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_short(name, guru, source):
    """One-line summary."""
    verdict, pct, score = guru['council']
    tag = {"BUY": "+", "HOLD": "~", "AVOID": "-"}[verdict]
    l = guru['lynch'][2]
    b = guru['buffett'][2]
    g = guru['graham'][2]
    m = guru['munger'][2]
    print(f"  [{tag}] {name:<10} {verdict:<5} {pct:5.0f}% ({score}/8)  "
          f"L:{l} B:{b} G:{g} M:{m}  [{source}]")


def print_full(name, symbol, fund, tech, guru):
    """Full detailed report for one asset."""
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {name} ({symbol})")
    print(f"{'=' * w}")

    # --- Source ---
    source = "N/A"
    if fund:
        source = fund.get('_source', 'backup')
    print(f"  Data source: {source}")
    if fund and fund.get('sector'):
        print(f"  Sector: {fund['sector']} / {fund.get('industry', '')}")

    # --- Price & Valuation ---
    print(f"\n  {'─' * (w - 4)}")
    print(f"  VALUATION")
    print(f"  {'─' * (w - 4)}")
    if fund:
        price = fund.get('price', 0)
        print(f"  Price:        {_fmt_num(price, ',.2f')}")
        print(f"  Market Cap:   {_fmt_money(fund.get('market_cap'))}")
        print(f"  P/E (TTM):    {_fmt_num(fund.get('pe'), '.1f')}")
        print(f"  P/E (Fwd):    {_fmt_num(fund.get('fwd_pe'), '.1f')}")
        peg = fund.get('peg_ratio', 0)
        if peg:
            print(f"  PEG Ratio:    {_fmt_num(peg, '.2f')}")
        print(f"  EPS:          {_fmt_num(fund.get('eps'), '.2f')}")
        bv = fund.get('book_value', 0)
        print(f"  Book Value:   {_fmt_num(bv, '.2f')}")
        if bv and price and bv > 0:
            print(f"  P/B:          {_fmt_num(price / bv, '.2f')}")
        gn = _calc_graham_number(fund.get('eps', 0), bv)
        if gn:
            margin = (gn - price) / price * 100 if price > 0 else 0
            print(f"  Graham #:     {_fmt_num(gn, '.2f')}  (margin: {margin:+.0f}%)")
        tbv = fund.get('tangible_bv')
        if tbv and fund.get('shares', 0) > 0:
            tbv_ps = tbv / fund['shares']
            print(f"  Tangible BV/sh: {_fmt_num(tbv_ps, '.2f')}")
    elif tech:
        print(f"  Price:        {_fmt_num(tech['close'], ',.2f')}")
    else:
        print(f"  No price data")

    # --- Profitability ---
    if fund and fund.get('roe', 0) != 0:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  PROFITABILITY")
        print(f"  {'─' * (w - 4)}")
        print(f"  ROE:            {_fmt_pct(fund.get('roe'), 100)}")
        print(f"  Gross Margin:   {_fmt_pct(fund.get('gross_margin'), 100)}")
        print(f"  Oper. Margin:   {_fmt_pct(fund.get('operating_margin'), 100)}")
        print(f"  Net Margin:     {_fmt_pct(fund.get('profit_margin'), 100)}")

    # --- Balance Sheet ---
    if fund and fund.get('total_assets'):
        print(f"\n  {'─' * (w - 4)}")
        print(f"  BALANCE SHEET")
        print(f"  {'─' * (w - 4)}")
        print(f"  Total Assets:      {_fmt_money(fund.get('total_assets'))}")
        print(f"  Total Liabilities: {_fmt_money(fund.get('total_liabilities'))}")
        ta = fund.get('total_assets', 1)
        tl = fund.get('total_liabilities', 0)
        if ta > 0:
            print(f"  Liab/Assets:       {_fmt_pct(tl / ta, 100)}")
        print(f"  Debt/Equity:       {_fmt_num(fund.get('debt_equity'), '.2f')}x")
        print(f"  Total Debt:        {_fmt_money(fund.get('total_debt'))}")
        print(f"  Cash:              {_fmt_money(fund.get('cash'))}")
        cash = fund.get('cash', 0) or 0
        debt = fund.get('total_debt', 0) or 0
        if debt > 0:
            print(f"  Cash/Debt:         {_fmt_pct(cash / debt, 100)}")
        print(f"  Current Ratio:     {_fmt_num(fund.get('current_ratio'), '.2f')}")
        print(f"  Quick Ratio:       {_fmt_num(fund.get('quick_ratio'), '.2f')}")
        ca = fund.get('current_assets', 0) or 0
        if ca > 0 and tl:
            ncav = ca - tl
            shares = fund.get('shares', 0)
            if shares > 0:
                print(f"  NCAV/share:        {_fmt_num(ncav / shares, '.2f')}")
        retained = fund.get('retained_earnings')
        if retained is not None:
            print(f"  Retained Earnings: {_fmt_money(retained)}")

    # --- Cash Flow ---
    if fund and (fund.get('operating_cf') or fund.get('fcf')):
        print(f"\n  {'─' * (w - 4)}")
        print(f"  CASH FLOW")
        print(f"  {'─' * (w - 4)}")
        print(f"  Operating CF:  {_fmt_money(fund.get('operating_cf'))}")
        print(f"  CapEx:         {_fmt_money(fund.get('capex'))}")
        print(f"  Free CF:       {_fmt_money(fund.get('fcf'))}")
        shares = fund.get('shares', 0)
        fcf = fund.get('fcf', 0) or 0
        if shares > 0 and fcf > 0:
            print(f"  FCF/share:     {_fmt_num(fcf / shares, '.2f')}")
            price = fund.get('price', 0) or 0
            if price > 0:
                print(f"  P/FCF:         {_fmt_num(price / (fcf / shares), '.1f')}")

    # --- Quarterly Trends ---
    has_quarters = fund and any(k in fund for k in ['revenue_quarters', 'net_income_quarters', 'fcf_quarters'])
    if has_quarters:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  QUARTERLY TRENDS")
        print(f"  {'─' * (w - 4)}")

        for label, key in [("Revenue", "revenue_quarters"),
                           ("Net Income", "net_income_quarters"),
                           ("Gross Profit", "gross_profit_quarters"),
                           ("Oper. Income", "operating_income_quarters"),
                           ("Free CF", "fcf_quarters")]:
            series = fund.get(key)
            if series is not None and len(series) > 0:
                vals = series.values[-4:]
                dates = series.index[-4:]
                parts = []
                for d, v in zip(dates, vals):
                    q_str = pd.Timestamp(d).strftime("%Y-Q%q") if hasattr(d, 'strftime') else str(d)[:7]
                    parts.append(f"{q_str}: {_fmt_money(v)}")
                print(f"  {label}:")
                for p in parts:
                    print(f"    {p}")
                # QoQ
                if len(vals) >= 2 and vals[-2] != 0:
                    qoq = (vals[-1] - vals[-2]) / abs(vals[-2]) * 100
                    arrow = "+" if qoq > 0 else ""
                    print(f"    QoQ: {arrow}{qoq:.1f}%")

    # --- Dividends ---
    if fund and (fund.get('dividend_yield', 0) > 0 or fund.get('payout_ratio', 0) > 0):
        print(f"\n  {'─' * (w - 4)}")
        print(f"  DIVIDENDS")
        print(f"  {'─' * (w - 4)}")
        print(f"  Div Yield:     {_fmt_num(fund.get('dividend_yield'), '.1f')}%")
        print(f"  Payout Ratio:  {_fmt_pct(fund.get('payout_ratio'), 100)}")

    # --- Risk ---
    if fund:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  RISK")
        print(f"  {'─' * (w - 4)}")
        print(f"  Beta:          {_fmt_num(fund.get('beta'), '.2f')}")

    # --- Technical ---
    if tech:
        print(f"\n  {'─' * (w - 4)}")
        print(f"  TECHNICAL")
        print(f"  {'─' * (w - 4)}")
        print(f"  RSI(14):       {_fmt_num(tech['rsi'], '.1f')}  {_bar(tech['rsi'])}")
        print(f"  52W Range:     {_fmt_num(tech['pct_52w'], '.0f')}%  {_bar(tech['pct_52w'])}")
        print(f"  Vol (30d ann): {_fmt_pct(tech['vol_30d'], 100)}")
        print(f"  > SMA 50:      {'Yes' if tech['above_50'] else 'No'}")
        print(f"  > SMA 200:     {'Yes' if tech['above_200'] else 'No'}")
        print(f"  MACD:          {'Bullish' if tech['macd_bull'] else 'Bearish'}")

    # --- Guru Verdicts ---
    print(f"\n  {'─' * (w - 4)}")
    print(f"  GURU COUNCIL")
    print(f"  {'─' * (w - 4)}")
    for g_name, g_key in [("Lynch (GARP)", "lynch"), ("Buffett (Quality)", "buffett"),
                           ("Graham (Value)", "graham"), ("Munger (Risk)", "munger")]:
        status, desc, score = guru[g_key]
        stars = "*" * score + "." * (2 - score)
        print(f"  {g_name:<20} [{stars}] {status:<14} {desc}")

    verdict, pct, total = guru['council']
    box = {"BUY": ">>> BUY <<<", "HOLD": "--- HOLD ---", "AVOID": "!!! AVOID !!!"}[verdict]
    print(f"\n  {'*' * w}")
    print(f"  COUNCIL VERDICT:  {box}  ({pct:.0f}% consensus, {total}/8)")
    print(f"  {'*' * w}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    # Parse args
    full_mode = False
    sector_filter = None
    assets_requested = []

    i = 0
    while i < len(args):
        if args[i] == '--all':
            full_mode = True
            assets_requested = list(FULL_ASSET_MAP.keys())
        elif args[i] == '--sector' and i + 1 < len(args):
            i += 1
            key = args[i].upper()
            if key in SECTORS:
                assets_requested = [a for a in FULL_ASSET_MAP if a in SECTORS[key]]
                full_mode = True
            else:
                print(f"Unknown sector: {key}. Available: {', '.join(SECTORS.keys())}")
                return
        elif not args[i].startswith('-'):
            assets_requested.append(args[i].upper())
            full_mode = True
        i += 1

    # Default: all assets, short mode
    if not assets_requested:
        assets_requested = list(FULL_ASSET_MAP.keys())
        full_mode = False

    # If 1-3 specific assets, always full mode
    if len(assets_requested) <= 3 and args:
        full_mode = True

    total = len(assets_requested)
    print(f"\n{'=' * 60}")
    print(f"  GURU COUNCIL REPORT  |  {total} asset(s)")
    print(f"  Mode: {'FULL' if full_mode else 'SUMMARY'}")
    print(f"{'=' * 60}")

    # Fetch Smart-Lab for Russian stocks
    print("\n  Loading Smart-Lab data...", end=" ", flush=True)
    smartlab = fetch_smartlab_data()
    print(f"OK ({len(smartlab)} tickers)" if smartlab else "offline")

    buy_list = []
    hold_list = []
    avoid_list = []

    for idx, name in enumerate(assets_requested, 1):
        symbol = FULL_ASSET_MAP.get(name, name)
        if not full_mode:
            print(f"\r  [{idx}/{total}] {name}...", end="", flush=True)

        # Get fundamentals
        fund = None
        clean_name = name.split('.')[0]
        ticker_map = {'YNDX': 'YDEX', 'TCSG': 'T'}
        search = ticker_map.get(clean_name, clean_name)

        if search in smartlab:
            d = smartlab[search]
            fund = {
                'pe': d.get('pe', 99), 'roe': d.get('roe', 0),
                'debt_equity': d.get('debt', 0), 'dividend_yield': d.get('div', 0),
                'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
                'eps': 0, 'beta': 1.0, 'current_ratio': 0, 'price': 0,
                'fwd_pe': 0, 'gross_margin': 0, 'operating_margin': 0,
                'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
                'shares': 0, 'market_cap': 0,
                '_source': 'smartlab',
            }
        elif name not in MOEX_ASSETS:
            # Try yfinance for non-MOEX
            yf_data = fetch_yf_deep(symbol)
            if yf_data:
                fund = yf_data
            else:
                bk = GLOBAL_BACKUP.get(search) or GLOBAL_BACKUP.get(clean_name)
                if bk:
                    fund = {
                        'pe': bk.get('pe', 99), 'roe': bk.get('roe', 0),
                        'debt_equity': bk.get('debt', 0), 'dividend_yield': bk.get('div', 0),
                        'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
                        'eps': 0, 'price': 0, 'gross_margin': 0, 'operating_margin': 0,
                        'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
                        'shares': 0, 'market_cap': 0, 'beta': 1.0, 'current_ratio': 0,
                        'fwd_pe': 0, '_source': 'backup',
                    }
        else:
            bk = GLOBAL_BACKUP.get(search) or GLOBAL_BACKUP.get(clean_name)
            if bk:
                fund = {
                    'pe': bk.get('pe', 99), 'roe': bk.get('roe', 0),
                    'debt_equity': bk.get('debt', 0), 'dividend_yield': bk.get('div', 0),
                    'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
                    'eps': 0, 'price': 0, 'gross_margin': 0, 'operating_margin': 0,
                    'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
                    'shares': 0, 'market_cap': 0, 'beta': 1.0, 'current_ratio': 0,
                    'fwd_pe': 0, '_source': 'backup',
                }

        # Technical
        tech = _tech_context(get_technical(name))

        # Guru
        guru = run_guru(name, fund, tech)
        source = fund.get('_source', 'tech_only') if fund else 'tech_only'

        if full_mode:
            print_full(name, symbol, fund, tech, guru)
        else:
            print(f"\r", end="")
            print_short(name, guru, source)

        # Collect
        verdict = guru['council'][0]
        if verdict == 'BUY':
            buy_list.append((name, guru['council'][1]))
        elif verdict == 'HOLD':
            hold_list.append((name, guru['council'][1]))
        else:
            avoid_list.append((name, guru['council'][1]))

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  BUY:   {len(buy_list)}", end="")
    if buy_list:
        top = sorted(buy_list, key=lambda x: -x[1])[:5]
        print(f"  ({', '.join(f'{n}({p:.0f}%)' for n, p in top)})")
    else:
        print()
    print(f"  HOLD:  {len(hold_list)}", end="")
    if hold_list:
        print(f"  ({', '.join(n for n, _ in hold_list[:8])})")
    else:
        print()
    print(f"  AVOID: {len(avoid_list)}", end="")
    if avoid_list:
        worst = sorted(avoid_list, key=lambda x: x[1])[:5]
        print(f"  ({', '.join(f'{n}({p:.0f}%)' for n, p in worst)})")
    else:
        print()
    print()


if __name__ == "__main__":
    main()
