"""
guru_report.py - Console Guru Council report with full financial data.
Shows raw fundamentals + guru verdicts for any asset or all assets.

Usage:
    python guru_report.py              # All assets (summary)
    python guru_report.py TSLA         # One asset (full report)
    python guru_report.py TSLA SBER    # Several assets
    python guru_report.py --all        # All assets (full report)
    python guru_report.py --sector US  # By group (US, RUS, CRYPTO, COMMODITY)
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
from core.features import compute_rsi
from core.guru import calc_graham_number, get_guru_analysis, technical_context
from net import ssl_verify
from sqlalchemy import create_engine

DB_PATH = os.path.join(BASE_DIR, "market.db")
_engine = create_engine(f"sqlite:///{DB_PATH}")

# -- MOEX list (no yfinance fundamentals) --
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

# -- BACKUP fundamentals (same as app.py) --
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


# ==============================================================================
# DATA FETCHING
# ==============================================================================

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
    except Exception:
        return None


def fetch_smartlab_data():
    """Fetch Smart-Lab fundamentals for Russian stocks."""
    url = "https://smart-lab.ru/q/shares_fundamental/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    }
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=ssl_verify())
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


def resolve_fundamentals(name, symbol, smartlab):
    """Resolve one asset's fundamentals dict: smartlab -> yfinance -> backup -> None.

    `name` is the G-Trade asset code (used for the MOEX_ASSETS/GLOBAL_BACKUP
    lookups, via the same ticker_map remap used elsewhere in this module);
    `symbol` is the actual fetch ticker (FULL_ASSET_MAP value, e.g. BTC ->
    BTC-USD). Shared by main() and webapp.py's live recalculate endpoint so
    both use the exact same fundamentals resolution.
    """
    clean_name = name.split('.')[0]
    ticker_map = {'YNDX': 'YDEX', 'TCSG': 'T'}
    search = ticker_map.get(clean_name, clean_name)

    if search in smartlab:
        d = smartlab[search]
        return {
            'pe': d.get('pe', 99), 'roe': d.get('roe', 0),
            'debt_equity': d.get('debt', 0), 'dividend_yield': d.get('div', 0),
            'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
            'eps': 0, 'beta': 1.0, 'current_ratio': 0, 'price': 0,
            'fwd_pe': 0, 'gross_margin': 0, 'operating_margin': 0,
            'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
            'shares': 0, 'market_cap': 0,
            '_source': 'smartlab',
        }

    if name not in MOEX_ASSETS:
        yf_data = fetch_yf_deep(symbol)
        if yf_data:
            return yf_data
        bk = GLOBAL_BACKUP.get(search) or GLOBAL_BACKUP.get(clean_name)
        if bk:
            return {
                'pe': bk.get('pe', 99), 'roe': bk.get('roe', 0),
                'debt_equity': bk.get('debt', 0), 'dividend_yield': bk.get('div', 0),
                'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
                'eps': 0, 'price': 0, 'gross_margin': 0, 'operating_margin': 0,
                'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
                'shares': 0, 'market_cap': 0, 'beta': 1.0, 'current_ratio': 0,
                'fwd_pe': 0, '_source': 'backup',
            }
        return None

    bk = GLOBAL_BACKUP.get(search) or GLOBAL_BACKUP.get(clean_name)
    if bk:
        return {
            'pe': bk.get('pe', 99), 'roe': bk.get('roe', 0),
            'debt_equity': bk.get('debt', 0), 'dividend_yield': bk.get('div', 0),
            'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
            'eps': 0, 'price': 0, 'gross_margin': 0, 'operating_margin': 0,
            'peg_ratio': 0, 'quick_ratio': 0, 'payout_ratio': 0,
            'shares': 0, 'market_cap': 0, 'beta': 1.0, 'current_ratio': 0,
            'fwd_pe': 0, '_source': 'backup',
        }
    return None


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

        df['RSI'] = compute_rsi(df['close'])

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        return df
    except Exception:
        return None


# ==============================================================================
# FORMATTING HELPERS
# ==============================================================================

def _fmt_num(val, fmt=",.0f"):
    """Format number with fallback."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "-"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_money(val):
    if val is None:
        return "-"
    if abs(val) >= 1e12:
        return f"${val/1e12:,.1f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.1f}M"
    return f"${val:,.0f}"


def _fmt_pct(val, mult=1):
    if val is None:
        return "-"
    return f"{val * mult:.1f}%"


def _bar(val, max_val=100, width=20):
    """Simple ASCII bar chart."""
    if val is None or max_val == 0:
        return ""
    filled = int(min(val / max_val, 1.0) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ==============================================================================
# GURU ALGORITHMS - delegated to core.guru (single source of truth, shared
# with app.py) so this report's verdicts never drift from the live terminal's.
# ==============================================================================

def run_guru(name, fund, tech):
    """Run all 4 gurus via core.guru, reshaped into this script's (status, desc, score) tuples."""
    analysis = get_guru_analysis(fund, tech)
    results = {
        key: (analysis[key]['status'], analysis[key]['desc'], analysis[key]['_score'])
        for key in ('lynch', 'buffett', 'graham', 'munger')
    }
    total = sum(results[key][2] for key in ('lynch', 'buffett', 'graham', 'munger'))
    council = analysis['council']
    results['council'] = (council['verdict'], council['pct'], total)
    return results


# ==============================================================================
# PRINTING
# ==============================================================================

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
    print(f"\n  {'-' * (w - 4)}")
    print("  VALUATION")
    print(f"  {'-' * (w - 4)}")
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
        gn = calc_graham_number(fund.get('eps', 0), bv)
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
        print("  No price data")

    # --- Profitability ---
    if fund and fund.get('roe', 0) != 0:
        print(f"\n  {'-' * (w - 4)}")
        print("  PROFITABILITY")
        print(f"  {'-' * (w - 4)}")
        print(f"  ROE:            {_fmt_pct(fund.get('roe'), 100)}")
        print(f"  Gross Margin:   {_fmt_pct(fund.get('gross_margin'), 100)}")
        print(f"  Oper. Margin:   {_fmt_pct(fund.get('operating_margin'), 100)}")
        print(f"  Net Margin:     {_fmt_pct(fund.get('profit_margin'), 100)}")

    # --- Balance Sheet ---
    if fund and fund.get('total_assets'):
        print(f"\n  {'-' * (w - 4)}")
        print("  BALANCE SHEET")
        print(f"  {'-' * (w - 4)}")
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
        print(f"\n  {'-' * (w - 4)}")
        print("  CASH FLOW")
        print(f"  {'-' * (w - 4)}")
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
        print(f"\n  {'-' * (w - 4)}")
        print("  QUARTERLY TRENDS")
        print(f"  {'-' * (w - 4)}")

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
        print(f"\n  {'-' * (w - 4)}")
        print("  DIVIDENDS")
        print(f"  {'-' * (w - 4)}")
        print(f"  Div Yield:     {_fmt_num(fund.get('dividend_yield'), '.1f')}%")
        print(f"  Payout Ratio:  {_fmt_pct(fund.get('payout_ratio'), 100)}")

    # --- Risk ---
    if fund:
        print(f"\n  {'-' * (w - 4)}")
        print("  RISK")
        print(f"  {'-' * (w - 4)}")
        print(f"  Beta:          {_fmt_num(fund.get('beta'), '.2f')}")

    # --- Technical ---
    if tech:
        print(f"\n  {'-' * (w - 4)}")
        print("  TECHNICAL")
        print(f"  {'-' * (w - 4)}")
        print(f"  RSI(14):       {_fmt_num(tech['rsi'], '.1f')}  {_bar(tech['rsi'])}")
        print(f"  52W Range:     {_fmt_num(tech['pct_52w'], '.0f')}%  {_bar(tech['pct_52w'])}")
        print(f"  Vol (30d ann): {_fmt_pct(tech['vol_30d'], 100)}")
        print(f"  > SMA 50:      {'Yes' if tech['above_50'] else 'No'}")
        print(f"  > SMA 200:     {'Yes' if tech['above_200'] else 'No'}")
        print(f"  MACD:          {'Bullish' if tech['macd_bull'] else 'Bearish'}")

    # --- Guru Verdicts ---
    print(f"\n  {'-' * (w - 4)}")
    print("  GURU COUNCIL")
    print(f"  {'-' * (w - 4)}")
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


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = sys.argv[1:]

    # Parse args
    full_mode = False
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
        fund = resolve_fundamentals(name, symbol, smartlab)

        # Technical
        tech = technical_context(get_technical(name))

        # Guru
        guru = run_guru(name, fund, tech)
        source = fund.get('_source', 'tech_only') if fund else 'tech_only'

        if full_mode:
            print_full(name, symbol, fund, tech, guru)
        else:
            print("\r", end="")
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
    print("  SUMMARY")
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
