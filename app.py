import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
import requests
from sqlalchemy import create_engine
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# --- 1. СИСТЕМНЫЕ НАСТРОЙКИ ---
st.set_page_config(page_title="G-TRADE TERMINAL V97", layout="wide", page_icon="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)

try:
    from config import FULL_ASSET_MAP, ASSET_TYPES
except ImportError:
    st.error("[!] КРИТИЧЕСКАЯ ОШИБКА: Файл config.py не найден!"); st.stop()

try:
    from risk_manager import RiskManager, RISK_CONFIG
    _rm = RiskManager()
    RISK_AVAILABLE = True
except Exception:
    RISK_AVAILABLE = False
    _rm = None

try:
    from portfolio import PortfolioManager, SECTOR_LIMITS
    _pm = PortfolioManager(FULL_ASSET_MAP)
    PORTFOLIO_AVAILABLE = True
except Exception:
    PORTFOLIO_AVAILABLE = False
    _pm = None
    SECTOR_LIMITS = {}

try:
    import news_analyzer as _news
    NEWS_AVAILABLE = True
except Exception:
    NEWS_AVAILABLE = False
    _news = None

try:
    import regime_detector as _regime
    REGIME_AVAILABLE = True
except Exception:
    REGIME_AVAILABLE = False
    _regime = None

try:
    import paper_trading as _paper
    PAPER_AVAILABLE = True
except Exception:
    PAPER_AVAILABLE = False
    _paper = None

try:
    import signal_dashboard as _radar
    RADAR_AVAILABLE = True
except Exception:
    RADAR_AVAILABLE = False
    _radar = None

try:
    import performance_tracker as _perf
    PERF_AVAILABLE = True
except Exception:
    PERF_AVAILABLE = False
    _perf = None

try:
    import sector_rotation as _sector
    SECTOR_AVAILABLE = True
except Exception:
    SECTOR_AVAILABLE = False
    _sector = None

try:
    import whatif_simulator as _whatif
    WHATIF_AVAILABLE = True
except Exception:
    WHATIF_AVAILABLE = False
    _whatif = None

try:
    import model_comparison as _mc
    MC_AVAILABLE = True
except Exception:
    MC_AVAILABLE = False
    _mc = None

try:
    import alert_rules as _alerts
    ALERTS_AVAILABLE = True
except Exception:
    ALERTS_AVAILABLE = False
    _alerts = None

try:
    import guru_tracker as _gt
    GURU_TRACKER = True
except Exception:
    GURU_TRACKER = False
    _gt = None

engine = create_engine(f'sqlite:///{os.path.join(BASE_DIR, "market.db")}')
MODEL_DIR = os.path.join(BASE_DIR, "models")
MOEX_LIST = ["IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "YNDX", "TCSG", "OZON", "VKCO", "POSI", "AFLT"]

# --- 2. ГЛОБАЛЬНАЯ БАЗА ЗНАНИЙ (РЕЗЕРВ) ---
GLOBAL_BACKUP = {
    # РФ (MOEX)
    'SBER': {'pe': 4.2, 'roe': 0.24, 'debt': 0.0, 'div': 11.5, 'desc': 'Лидер сектора'},
    'GAZP': {'pe': 3.5, 'roe': 0.04, 'debt': 1.8, 'div': 0.0, 'desc': 'Инфраструктурные риски'},
    'LKOH': {'pe': 5.5, 'roe': 0.18, 'debt': 0.2, 'div': 16.0, 'desc': 'Дивидендный аристократ'},
    'ROSN': {'pe': 5.9, 'roe': 0.15, 'debt': 0.9, 'div': 9.0, 'desc': 'Стабильный поток'},
    'NVTK': {'pe': 7.1, 'roe': 0.19, 'debt': 0.4, 'div': 8.5, 'desc': 'СПГ проекты'},
    'YNDX': {'pe': 14.5, 'roe': 0.25, 'debt': 0.1, 'div': 3.5, 'desc': 'МКПАО Яндекс (Рост)'}, # ОБНОВЛЕНО
    'YDEX': {'pe': 14.5, 'roe': 0.25, 'debt': 0.1, 'div': 3.5, 'desc': 'МКПАО Яндекс (Рост)'}, # ДУБЛЬ
    'TCSG': {'pe': 7.8, 'roe': 0.32, 'debt': 0.0, 'div': 5.0, 'desc': 'Т-Банк (Рост)'},
    'OZON': {'pe': 99.0, 'roe': -0.10, 'debt': 2.5, 'div': 0.0, 'desc': 'Рост оборота, убыток'},
    'VKCO': {'pe': 99.0, 'roe': -0.05, 'debt': 1.2, 'div': 0.0, 'desc': 'Долговая нагрузка'},
    'POSI': {'pe': 25.0, 'roe': 0.35, 'debt': 0.1, 'div': 3.5, 'desc': 'Кибербезопасность'},
    'AFLT': {'pe': 99.0, 'roe': -0.50, 'debt': 8.0, 'div': 0.0, 'desc': 'Гос. поддержка'},
    'IMOEX': {'pe': 5.5, 'roe': 0.15, 'debt': 0.8, 'div': 9.0, 'desc': 'Индекс Мосбиржи'},
    'TATN': {'pe': 5.0, 'roe': 0.20, 'debt': 0.0, 'div': 12.0, 'desc': 'Татнефть'},
    'SNGS': {'pe': 2.5, 'roe': 0.08, 'debt': 0.0, 'div': 3.0, 'desc': 'Сургутнефтегаз (кубышка)'},
    'PLZL': {'pe': 10.0, 'roe': 0.50, 'debt': 1.5, 'div': 5.0, 'desc': 'Полюс Золото'},
    'SIBN': {'pe': 4.0, 'roe': 0.25, 'debt': 0.3, 'div': 14.0, 'desc': 'Газпром нефть'},
    'MGNT': {'pe': 8.0, 'roe': 0.40, 'debt': 1.5, 'div': 8.0, 'desc': 'Магнит (ритейл)'},
    'VTBR': {'pe': 3.0, 'roe': 0.18, 'debt': 0.0, 'div': 0.0, 'desc': 'ВТБ'},
    'BSPB': {'pe': 3.5, 'roe': 0.19, 'debt': 0.0, 'div': 10.0, 'desc': 'Банк СПб'},
    'MOEX_EX': {'pe': 12.0, 'roe': 0.30, 'debt': 0.0, 'div': 7.0, 'desc': 'Мосбиржа'},
    'MTSS': {'pe': 8.0, 'roe': 0.40, 'debt': 2.0, 'div': 12.0, 'desc': 'МТС'},
    'RTKM': {'pe': 7.0, 'roe': 0.15, 'debt': 2.5, 'div': 8.0, 'desc': 'Ростелеком'},
    'CHMF': {'pe': 6.0, 'roe': 0.30, 'debt': 0.5, 'div': 10.0, 'desc': 'Северсталь'},
    'NLMK': {'pe': 5.5, 'roe': 0.25, 'debt': 0.3, 'div': 10.0, 'desc': 'НЛМК'},
    'MAGN': {'pe': 5.0, 'roe': 0.20, 'debt': 0.0, 'div': 8.0, 'desc': 'ММК'},
    'RUAL': {'pe': 8.0, 'roe': 0.10, 'debt': 2.0, 'div': 0.0, 'desc': 'РУСАЛ'},
    'ALRS': {'pe': 6.0, 'roe': 0.15, 'debt': 0.5, 'div': 6.0, 'desc': 'АЛРОСА'},
    'IRAO': {'pe': 4.0, 'roe': 0.12, 'debt': 0.0, 'div': 10.0, 'desc': 'Интер РАО'},
    'HYDR': {'pe': 5.0, 'roe': 0.10, 'debt': 0.5, 'div': 8.0, 'desc': 'РусГидро'},
    'FLOT': {'pe': 3.5, 'roe': 0.30, 'debt': 1.0, 'div': 12.0, 'desc': 'Совкомфлот'},
    'PIKK': {'pe': 6.0, 'roe': 0.20, 'debt': 2.5, 'div': 5.0, 'desc': 'ПИК'},

    # США (US GIANTS) — updated 2026-03
    'NVDA': {'pe': 38.0, 'growth': 0.96, 'roe': 1.01, 'debt': 0.07, 'marg': 0.56},
    'TSLA': {'pe': 95.0, 'growth': 0.17, 'roe': 0.27, 'debt': 0.13, 'marg': 0.13},
    'AAPL': {'pe': 33.0, 'growth': 0.18, 'roe': 1.52, 'debt': 1.03, 'marg': 0.27},
    'MSFT': {'pe': 31.0, 'growth': 0.11, 'roe': 0.37, 'debt': 0.21, 'marg': 0.36},
    'GOOGL': {'pe': 29.0, 'growth': 0.31, 'roe': 0.36, 'debt': 0.16, 'marg': 0.33},
    'AMZN': {'pe': 31.0, 'growth': 0.52, 'roe': 0.24, 'debt': 0.43, 'marg': 0.10},
    'META': {'pe': 22.0, 'growth': 0.36, 'roe': 0.35, 'debt': 0.18, 'marg': 0.38},
    'AMD':  {'pe': 95.0, 'growth': 0.24, 'roe': 0.05, 'debt': 0.04, 'marg': 0.10},
    'PLTR': {'pe': 150.0, 'growth': 0.36, 'roe': 0.12, 'debt': 0.0, 'marg': 0.18},
    'COIN': {'pe': 18.0, 'growth': 0.40, 'roe': 0.20, 'debt': 0.50, 'marg': 0.30},
    'MSTR': {'pe': 99.0, 'growth': 0.0, 'roe': 0.05, 'debt': 2.0, 'marg': -0.10},
}

# --- 3. ПАРСЕР SMART-LAB (RF LIVE) ---
@st.cache_data(ttl=3600)
def get_smartlab_data():
    """Парсит Smart-Lab, притворяясь браузером"""
    url = "https://smart-lab.ru/q/shares_fundamental/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Cache-Control': 'no-cache'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=False)
        if r.status_code == 200:
            import io
            dfs = pd.read_html(io.StringIO(r.text))
            if not dfs:
                return {}, "[OFF] OFFLINE (Using Backup)"
            f_map = {}

            def _safe_float(val, strip='%'):
                s = str(val).replace(strip, '').replace('\xa0', '').strip()
                if s in ('-', 'nan', 'None', ''):
                    return None
                return float(s)

            # Table 0: general companies (P/E, долг/EBITDA, ДД ао)
            df0 = dfs[0]
            if 'Тикер' in df0.columns and 'P/E' in df0.columns:
                for _, row in df0.iterrows():
                    try:
                        t = str(row['Тикер'])
                        pe = _safe_float(row['P/E']) or 99.0
                        debt_raw = row.get('долг/EBITDA', row.get('Долг/EBITDA'))
                        debt = _safe_float(debt_raw) or 0.0
                        div_raw = row.get('ДД ао, %', row.get('Див. доход, %'))
                        div = _safe_float(div_raw) or 0.0
                        f_map[t] = {'pe': pe, 'roe': 0.0, 'debt': debt, 'div': div}
                    except Exception: continue

            # Table 1: banks (P/E, RoE, ДД ао) — merge/override
            if len(dfs) > 1:
                df1 = dfs[1]
                roe_col = [c for c in df1.columns if c.lower() == 'roe']
                if 'Тикер' in df1.columns and 'P/E' in df1.columns:
                    for _, row in df1.iterrows():
                        try:
                            t = str(row['Тикер'])
                            pe = _safe_float(row['P/E']) or 99.0
                            roe = (_safe_float(row[roe_col[0]]) or 0.0) / 100 if roe_col else 0.0
                            div_raw = row.get('ДД ао, %', row.get('Див. доход, %'))
                            div = _safe_float(div_raw) or 0.0
                            f_map[t] = {'pe': pe, 'roe': roe, 'debt': 0.0, 'div': div}
                        except Exception: continue

            if f_map:
                return f_map, "[ON] LIVE (Smart-Lab)"
    except Exception as e:
        pass
    
    return {}, "[OFF] OFFLINE (Using Backup)"

# --- 4. ТЕХНИЧЕСКИЙ АНАЛИЗ (SQL) ---
def get_technical_data(asset, weekly=False):
    table = asset.lower().replace("^", "").replace(".", "").replace("-", "")
    if weekly:
        table += "_weekly"
    try:
        df = pd.read_sql(table, engine, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]

        # Moving averages
        df['SMA_20']  = df['close'].rolling(20).mean()
        df['SMA_50']  = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()

        # Bollinger Bands
        std20 = df['close'].rolling(20).std()
        df['BB_upper'] = df['SMA_20'] + 2 * std20
        df['BB_lower'] = df['SMA_20'] - 2 * std20

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD']        = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist']   = df['MACD'] - df['MACD_signal']

        # ATR (Average True Range)
        high_low   = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close  = (df['low']  - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Taleb Risk (kurtosis of returns)
        rets = df['close'].pct_change()
        df['taleb_index'] = rets.rolling(30).kurt().fillna(0)

        # Volume SMA for volume analysis
        df['vol_sma20'] = df['volume'].rolling(20).mean()

        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_weekly_data(asset):
    """Load weekly bars for multi-timeframe overlay."""
    table = asset.lower().replace("^", "").replace(".", "").replace("-", "") + "_weekly"
    try:
        df = pd.read_sql(table, engine, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]
        df['SMA_20w'] = df['close'].rolling(20).mean()  # ~5-month MA
        return df
    except Exception:
        return None

def load_ai_report():
    try:
        p = os.path.join(MODEL_DIR, "quality_report.json")
        if os.path.exists(p): return pd.DataFrame(json.load(open(p)))
    except Exception: pass
    return None

def calculate_kelly(acc, taleb):
    if acc < 0.51: return 0.0
    base_kelly = (acc - 0.50) * 200 # Упрощенный Келли
    risk_factor = max(0.2, 1.0 - (taleb / 5.0)) # Штраф за риск Талеба
    return base_kelly * risk_factor

# --- 5. ФУНДАМЕНТАЛЬНЫЙ АНАЛИЗАТОР (GURU COUNCIL V2) ---

# Auto-fetch fundamentals from yfinance (cached 1 hour)
@st.cache_data(ttl=3600)
def _fetch_yf_fundamentals(symbol):
    """Fetch live fundamentals from yfinance: .info + financials + balance sheet."""
    try:
        t = yf.Ticker(symbol)
        i = t.info or {}
        if i.get('currentPrice', 0) == 0 and i.get('regularMarketPrice', 0) == 0:
            return None

        price = i.get('currentPrice') or i.get('regularMarketPrice') or 0

        # --- Base metrics from .info ---
        result = {
            'pe': i.get('trailingPE') or i.get('forwardPE') or 0,
            'fwd_pe': i.get('forwardPE') or 0,
            'roe': i.get('returnOnEquity') or 0,
            'debt_equity': (i.get('debtToEquity') or 0) / 100,
            'growth': i.get('earningsGrowth') or i.get('revenueGrowth') or 0,
            'profit_margin': i.get('profitMargins') or 0,
            'fcf': i.get('freeCashflow') or 0,
            'market_cap': i.get('marketCap') or 0,
            'book_value': i.get('bookValue') or 0,
            'eps': i.get('trailingEps') or 0,
            'dividend_yield': (i.get('dividendYield') or 0) * 100,
            'beta': i.get('beta') or 1.0,
            'revenue_growth': i.get('revenueGrowth') or 0,
            'current_ratio': i.get('currentRatio') or 0,
            'price': price,
            'source': 'yfinance_live',
            # Extra .info fields
            'gross_margin': i.get('grossMargins') or 0,
            'operating_margin': i.get('operatingMargins') or 0,
            'payout_ratio': i.get('payoutRatio') or 0,
            'peg_ratio': i.get('pegRatio') or 0,
            'quick_ratio': i.get('quickRatio') or 0,
            'sector': i.get('sector', ''),
            'industry': i.get('industry', ''),
        }

        # --- Deep dive: quarterly financials for trends ---
        try:
            fin = t.quarterly_financials
            if fin is not None and not fin.empty:
                # Revenue trend (last 4 quarters)
                if 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].dropna().sort_index()
                    if len(rev) >= 2:
                        result['revenue_trend'] = list(rev.values[-4:])
                        # QoQ growth
                        result['revenue_qoq'] = float((rev.iloc[-1] - rev.iloc[-2]) / abs(rev.iloc[-2])) if rev.iloc[-2] != 0 else 0
                # Net Income trend
                for label in ['Net Income', 'Net Income From Continuing Operations']:
                    if label in fin.index:
                        ni = fin.loc[label].dropna().sort_index()
                        if len(ni) >= 2:
                            result['net_income_trend'] = list(ni.values[-4:])
                            result['net_income_qoq'] = float((ni.iloc[-1] - ni.iloc[-2]) / abs(ni.iloc[-2])) if ni.iloc[-2] != 0 else 0
                            # Earnings stability (all positive?)
                            result['earnings_stable'] = bool((ni.values[-4:] > 0).all()) if len(ni) >= 4 else None
                        break
                # Operating Income
                for label in ['Operating Income', 'EBIT']:
                    if label in fin.index:
                        oi = fin.loc[label].dropna().sort_index()
                        if len(oi) >= 1:
                            result['operating_income'] = float(oi.iloc[-1])
                        break
                # Gross Profit
                if 'Gross Profit' in fin.index:
                    gp = fin.loc['Gross Profit'].dropna()
                    if len(gp) >= 1:
                        result['gross_profit'] = float(gp.iloc[-1])
        except Exception:
            pass

        # --- Balance sheet for Graham deep value ---
        try:
            bs = t.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]  # most recent quarter
                # Total Assets / Total Liabilities
                total_assets = 0
                for label in ['Total Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        total_assets = float(latest[label])
                        result['total_assets'] = total_assets
                        break
                total_liab = 0
                for label in ['Total Liabilities Net Minority Interest', 'Total Liab']:
                    if label in latest.index and pd.notna(latest[label]):
                        total_liab = float(latest[label])
                        result['total_liabilities'] = total_liab
                        break
                # Net Current Assets (Graham's liquidation value)
                current_assets = 0
                for label in ['Current Assets', 'Total Current Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        current_assets = float(latest[label])
                        break
                current_liab = 0
                for label in ['Current Liabilities', 'Total Current Liabilities']:
                    if label in latest.index and pd.notna(latest[label]):
                        current_liab = float(latest[label])
                        break
                if current_assets > 0:
                    result['net_current_assets'] = current_assets - total_liab  # NCAV
                    result['working_capital'] = current_assets - current_liab
                # Tangible Book Value
                for label in ['Tangible Book Value', 'Net Tangible Assets']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['tangible_bv'] = float(latest[label])
                        break
                # Retained Earnings (Buffett: consistent compounder)
                if 'Retained Earnings' in latest.index and pd.notna(latest['Retained Earnings']):
                    result['retained_earnings'] = float(latest['Retained Earnings'])
                # Cash & equivalents
                for label in ['Cash And Cash Equivalents', 'Cash Financial']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['cash'] = float(latest[label])
                        break
                # Total Debt
                for label in ['Total Debt', 'Long Term Debt']:
                    if label in latest.index and pd.notna(latest[label]):
                        result['total_debt'] = float(latest[label])
                        break
        except Exception:
            pass

        # --- Cash flow statement ---
        try:
            cf = t.quarterly_cashflow
            if cf is not None and not cf.empty:
                latest_cf = cf.iloc[:, 0]
                # Operating Cash Flow
                for label in ['Operating Cash Flow', 'Total Cash From Operating Activities']:
                    if label in latest_cf.index and pd.notna(latest_cf[label]):
                        result['operating_cf'] = float(latest_cf[label])
                        break
                # Capex
                for label in ['Capital Expenditure', 'Capital Expenditures']:
                    if label in latest_cf.index and pd.notna(latest_cf[label]):
                        result['capex'] = float(latest_cf[label])
                        break
                # FCF from cashflow (more reliable than .info)
                if 'Free Cash Flow' in latest_cf.index and pd.notna(latest_cf['Free Cash Flow']):
                    result['fcf'] = float(latest_cf['Free Cash Flow'])
                elif result.get('operating_cf') and result.get('capex'):
                    result['fcf'] = result['operating_cf'] + result['capex']  # capex is negative
                # FCF trend
                for label in ['Free Cash Flow']:
                    if label in cf.index:
                        fcf_series = cf.loc[label].dropna().sort_index()
                        if len(fcf_series) >= 2:
                            result['fcf_trend'] = list(fcf_series.values[-4:])
                            result['fcf_positive_streak'] = int((fcf_series.values[-4:] > 0).sum()) if len(fcf_series) >= 4 else 0
                        break
        except Exception:
            pass

        # --- Derived metrics ---
        shares = i.get('sharesOutstanding') or 0
        if shares > 0 and result.get('fcf') and result['fcf'] > 0:
            result['fcf_per_share'] = result['fcf'] / shares
            result['price_to_fcf'] = price / (result['fcf'] / shares) if price > 0 else 0
        if result.get('total_assets') and result['total_assets'] > 0:
            result['asset_turnover'] = (i.get('totalRevenue') or 0) / result['total_assets']
        if result.get('tangible_bv') and shares > 0:
            result['tbv_per_share'] = result['tangible_bv'] / shares
        if result.get('net_current_assets') and shares > 0:
            result['ncav_per_share'] = result['net_current_assets'] / shares

        return result
    except Exception:
        return None


def _get_fundamentals(symbol, cl, smart_data):
    """
    Unified fundamental data loader with fallback chain:
    Smart-Lab → yfinance live → GLOBAL_BACKUP → None
    """
    TICKER_MAP = {'YNDX': 'YDEX', 'TCSG': 'T'}
    search_ticker = TICKER_MAP.get(cl, cl)

    # 1. Smart-Lab (Russian stocks)
    if search_ticker in smart_data:
        d = smart_data[search_ticker]
        return {
            'pe': d.get('pe', 99), 'roe': d.get('roe', 0),
            'debt_equity': d.get('debt', 0), 'dividend_yield': d.get('div', 0),
            'growth': 0, 'profit_margin': 0, 'fcf': 0, 'book_value': 0,
            'eps': 0, 'beta': 1.0, 'current_ratio': 0, 'price': 0,
            'fwd_pe': 0, 'revenue_growth': 0, 'market_cap': 0,
            'source': 'smartlab',
        }

    # 2. yfinance live (US + global)
    yf_data = _fetch_yf_fundamentals(symbol)
    if yf_data:
        return yf_data

    # 3. GLOBAL_BACKUP (static fallback)
    bk = GLOBAL_BACKUP.get(search_ticker) or GLOBAL_BACKUP.get(cl)
    if bk:
        return {
            'pe': bk.get('pe', 99), 'roe': bk.get('roe', 0),
            'debt_equity': bk.get('debt', 0), 'dividend_yield': bk.get('div', 0),
            'growth': bk.get('growth', 0), 'profit_margin': bk.get('marg', 0),
            'fcf': 0, 'book_value': 0, 'eps': 0, 'beta': 1.0,
            'current_ratio': 0, 'price': 0, 'fwd_pe': 0,
            'revenue_growth': 0, 'market_cap': 0,
            'source': 'backup',
        }

    return None  # no fundamentals available


from core.guru import technical_context as _technical_context, get_guru_analysis as _guru_analysis


def get_guru_analysis(symbol, df, smart_data_tuple):
    """Thin wrapper: fetches fundamentals then delegates to core.guru."""
    smart_data, _ = smart_data_tuple
    cl = symbol.split('.')[0]
    fund = _get_fundamentals(symbol, cl, smart_data)
    tech = _technical_context(df)
    return _guru_analysis(fund, tech)

# --- 6. ИНТЕРФЕЙС (STREAMLIT UI) ---
st.title("G-TRADE: FINAL TERMINAL V97")

# Загрузка источников
with st.spinner('Подключение к каналам данных...'):
    s_data = get_smartlab_data()

# Индикатор источника
status_icon = "[ON]" if "LIVE" in s_data[1] else "[~]"
st.caption(f"Источник фундаментала: {status_icon} **{s_data[1]}**")

# Сайдбар
category = st.sidebar.selectbox("КАТЕГОРИЯ", list(ASSET_TYPES.keys()))
selected_asset = st.sidebar.selectbox("АКТИВ", ASSET_TYPES[category])
symbol = FULL_ASSET_MAP[selected_asset]
timeframe = st.sidebar.radio("ТАЙМФРЕЙМ", ["Daily", "Weekly"], horizontal=True)

# Загрузка конкретного актива
if timeframe == "Weekly":
    df = get_technical_data(selected_asset, weekly=True)
else:
    df = get_technical_data(selected_asset)
rep = load_ai_report()
stats = rep[rep['Asset']==selected_asset].iloc[0] if rep is not None and not rep[rep['Asset']==selected_asset].empty else None

# Вкладки
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "AI SNIPER",
    "GURU COUNCIL",
    "PORTFOLIO",
    "RISK MANAGER",
    "NEWS & SENTIMENT",
    "MARKET REGIME",
    "PAPER TRADING",
    "SIGNAL RADAR",
    "PERFORMANCE",
    "SECTORS",
    "WHAT-IF",
    "MODEL HEALTH",
    "ALERTS",
])

# === ВКЛАДКА 1: ТРЕЙДИНГ ===
with tab1:
    if df is not None:
        last = df.iloc[-1]
        
        # Метрики
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ЦЕНА", f"{last['close']:,.2f}")
        
        # RSI Logic
        rsi_val = last['RSI']
        if rsi_val > 70: rsi_lbl = "[!] SELL"; rsi_col = "inverse"
        elif rsi_val < 30: rsi_lbl = "[+] BUY"; rsi_col = "normal"
        else: rsi_lbl = "Normal"; rsi_col = "off"
        c2.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_lbl, delta_color=rsi_col)
        
        # Taleb Risk Logic
        risk_val = last['taleb_index']
        risk_lbl = "CRASH" if risk_val > 5 else "SAFE"
        risk_col = "inverse" if risk_val > 5 else "normal"
        c3.metric("RISK", f"{risk_val:.2f}", delta=risk_lbl, delta_color=risk_col)
        
        # Kelly Logic
        lstm_acc = stats['LSTM_Acc'] if stats is not None else 0.50
        kel = calculate_kelly(lstm_acc, risk_val)
        c4.metric("KELLY", f"{kel:.1f}%")
        
        st.divider()
        
        # AI Центр
        st.subheader("Neural Intelligence Decision")
        ac1, ac2, ac3 = st.columns(3)
        cb_acc = stats['CB_Acc'] if stats is not None else 0.50
        ac1.metric("LSTM (Context)", f"{lstm_acc:.1%}")
        ac2.metric("CatBoost (Logic)", f"{cb_acc:.1%}")
        
        # AI VERDICT LOGIC
        if lstm_acc > 0.55: final_dec = "[>>] SNIPER BUY"; final_col = "normal"
        elif lstm_acc < 0.45: final_dec = "[<<] SNIPER SELL"; final_col = "inverse"
        elif rsi_val > 75: final_dec = "[!] TAKE PROFIT"; final_col = "inverse"
        elif lstm_acc > 0.52 and cb_acc > 0.52: final_dec = "[OK] CONSENSUS BUY"; final_col = "normal"
        else: final_dec = "[--] WAIT"; final_col = "off"
        
        ac3.metric("AI VERDICT", final_dec, delta=f"Conf: {lstm_acc:.0%}", delta_color=final_col)

        # Шпаргалка
        with st.expander("ШПАРГАЛКА СНАЙПЕРА"):
            st.markdown("""
            * **[>>] SNIPER BUY:** LSTM > 55%.
            * **[<<] SNIPER SELL:** LSTM < 45%.
            * **[!] TAKE PROFIT:** RSI > 75 (Фиксируй прибыль).
            * **CRASH:** Risk > 5.0 (Не входи).
            """)

        # ── Charts (4 rows: Price, MACD, RSI, Taleb Risk) ───────────────────
        df_w = get_weekly_data(selected_asset)
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.20, 0.18, 0.17],
            subplot_titles=("Price & MA (Daily + Weekly)", "MACD", "RSI (14)", "Tail Risk (Taleb)"),
        )

        # Row 1 — Candlestick + moving averages + Bollinger + weekly overlay
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],  line=dict(color='#ffd700', width=1, dash='dot'), name='SMA 20'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],  line=dict(color='orange',  width=1), name='SMA 50'),  row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#9c27b0', width=1), name='SMA 200'), row=1, col=1)
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(100,150,255,0.4)', width=1), name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(100,150,255,0.4)', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(100,150,255,0.05)', showlegend=False), row=1, col=1)
        # Weekly SMA overlay
        if df_w is not None and 'SMA_20w' in df_w.columns:
            fig.add_trace(go.Scatter(x=df_w.index, y=df_w['SMA_20w'], line=dict(color='#ff6b6b', width=2, dash='longdash'), name='SMA 20W'), row=1, col=1)

        # Row 2 — MACD histogram + lines
        colors_macd = ['#26a69a' if v >= 0 else '#ef5350' for v in df['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], marker_color=colors_macd, name='MACD Hist', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],        line=dict(color='#00F0FF', width=1), name='MACD'),        row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='#FF9800', width=1), name='Signal'),      row=2, col=1)

        # Row 3 — RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#00F0FF', width=2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray",  row=3, col=1)

        # Row 4 — Taleb risk
        fig.add_trace(go.Scatter(x=df.index, y=df['taleb_index'], fill='tozeroy', line=dict(color='#FF2E00', width=1), name='Tail Risk'), row=4, col=1)
        fig.add_hline(y=5, line_dash="dash", line_color="red", row=4, col=1)

        fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Нет данных! Запусти data_engine.py")

# === ВКЛАДКА 2: ИНВЕСТИРОВАНИЕ ===
with tab2:
    st.header(f"Совет Легенд: {selected_asset}")
    
    # Получаем анализ
    current_price = df['close'].iloc[-1] if df is not None else 0
    curs = get_guru_analysis(symbol, df, s_data)
    
    if curs:
        # Council Vote banner
        council = curs.get('council', {})
        src = curs.get('data_source', 'N/A')
        src_labels = {'smartlab': 'Smart-Lab (live)', 'yfinance_live': 'Yahoo Finance (live)',
                      'backup': 'Backup DB (static)', 'technical': 'Technical Only'}
        if council:
            st.markdown(
                f"### :{council['color']}[{council['text']}]"
            )
            st.caption(f"Data: {src_labels.get(src, src)}")
        st.divider()

        col1, col2 = st.columns(2)

        def _guru_color(status):
            if any(k in status for k in ("BUY", "GEM", "QUAL", "CLEAN", "CHEAP", "MOMENTUM", "STRONG", "QUALITY")):
                return "green"
            elif any(k in status for k in ("FAIR", "OK", "SIDEWAYS", "HOLD", "STABLE", "MINOR", "MIXED")):
                return "orange"
            else:
                return "red"

        # Left column
        with col1:
            st.subheader("Питер Линч")
            st.caption("GARP: рост по разумной цене (PEG ratio)")
            res = curs['lynch']
            st.markdown(f":{_guru_color(res['status'])}[**{res['status']}**]")
            st.write(res['desc'])
            st.divider()

            st.subheader("Уоррен Баффет")
            st.caption("Качество: ROE, маржа, низкий долг, дивиденды")
            res = curs['buffett']
            st.markdown(f":{_guru_color(res['status'])}[**{res['status']}**]")
            st.write(res['desc'])

        # Right column
        with col2:
            st.subheader("Бенджамин Грэм")
            st.caption("Value: Graham Number, P/E < 15, запас прочности")
            res = curs['graham']
            st.markdown(f":{_guru_color(res['status'])}[**{res['status']}**]")
            st.write(res['desc'])
            st.divider()

            st.subheader("Чарли Мангер")
            st.caption("Инверсия: долг, убытки, пузырь, волатильность")
            res = curs['munger']
            st.markdown(f":{_guru_color(res['status'])}[**{res['status']}**]")
            st.write(res['desc'])

        # Auto-log guru verdict
        if GURU_TRACKER and council:
            try:
                _gt.log_guru_verdict(
                    asset=selected_asset,
                    lynch_score=curs['lynch'].get('_score', 0) if '_score' in curs['lynch'] else
                        (2 if any(k in curs['lynch']['status'] for k in ('BUY','CHEAP','MOMENTUM')) else
                         1 if any(k in curs['lynch']['status'] for k in ('FAIR','SIDEWAYS')) else 0),
                    buffett_score=2 if any(k in curs['buffett']['status'] for k in ('GEM','QUALITY','STRONG')) else
                        (1 if any(k in curs['buffett']['status'] for k in ('OK','STABLE')) else 0),
                    graham_score=2 if any(k in curs['graham']['status'] for k in ('BUY','CHEAP')) else
                        (1 if 'FAIR' in curs['graham']['status'] else 0),
                    munger_score=2 if any(k in curs['munger']['status'] for k in ('CLEAN','MINOR')) else
                        (1 if 'WARNING' in curs['munger']['status'] else 0),
                    council_pct=council['pct'],
                    council_verdict=council['verdict'],
                    data_source=src,
                    price=current_price,
                )
            except Exception:
                pass

        # --- Guru Performance Report ---
        st.divider()
        st.subheader("Guru Council Track Record")

        if GURU_TRACKER:
            try:
                _gt.update_actuals()
            except Exception:
                pass

            horizon = st.selectbox("Горизонт оценки", ["1d", "5d", "20d"], index=1,
                                   key="guru_horizon")
            days_back = st.slider("Период (дней)", 7, 180, 60, key="guru_days")

            col_a, col_b = st.columns(2)

            with col_a:
                acc = _gt.get_guru_accuracy(days=days_back, horizon=horizon) if _gt else {}
                if acc.get("accuracy") is not None:
                    acc_val = acc["accuracy"]
                    color = "green" if acc_val >= 0.55 else ("orange" if acc_val >= 0.45 else "red")
                    st.metric("Council Accuracy", f"{acc_val:.1%}",
                              delta=f"{acc['correct']}/{acc['total']} correct")
                    if acc.get("avg_return") is not None:
                        st.metric("Avg Return (when acted)", f"{acc['avg_return']:.2%}")
                    for v, s in acc.get("by_verdict", {}).items():
                        st.caption(f"{v}: {s['accuracy']:.1%} ({s['count']} calls, avg {s['avg_return']:.2%})")
                else:
                    st.info("Нет данных. Совет начнёт накапливать историю при просмотре активов.")

            with col_b:
                gurus = _gt.get_guru_individual_accuracy(days=days_back, horizon=horizon) if _gt else {}
                if gurus:
                    st.markdown("**Accuracy per Guru:**")
                    for name, s in gurus.items():
                        acc_str = f"{s['accuracy']:.1%}" if s['accuracy'] is not None else "N/A"
                        bull_str = f"{s['bullish_correct']}/{s['bullish_calls']}" if s['bullish_calls'] else "-"
                        st.caption(f"{name}: **{acc_str}** (Bull: {bull_str})")

            # Leaderboard
            lb = _gt.get_guru_leaderboard(days=days_back, horizon=horizon) if _gt else pd.DataFrame()
            if not lb.empty:
                st.markdown("**Top/Bottom Assets by Council Accuracy:**")
                lb_display = lb.copy()
                lb_display["Accuracy"] = lb_display["Accuracy"].map("{:.1%}".format)
                lb_display["Avg_Return"] = lb_display["Avg_Return"].map("{:.2%}".format)
                st.dataframe(lb_display, use_container_width=True, hide_index=True)
        else:
            st.warning("guru_tracker.py not found.")

    else:
        st.error("Критическая ошибка анализа")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Portfolio Analytics")

    if not PORTFOLIO_AVAILABLE:
        st.warning("portfolio.py not found. Run the setup to enable this tab.")
    else:
        col_l, col_r = st.columns([1, 1])

        # ── Correlation heatmap ────────────────────────────────────────────
        with col_l:
            st.subheader("Asset Correlation Matrix")
            with st.spinner("Computing correlations…"):
                corr = _pm.get_correlation_matrix()

            if not corr.empty:
                # Show a readable subset (top assets by data availability)
                top_assets = [a for a in [
                    "BTC", "ETH", "SOL", "SP500", "NASDAQ", "GOLD",
                    "NVDA", "TSLA", "AAPL", "VIX", "DXY", "OIL",
                ] if a in corr.columns]
                corr_sub = corr.loc[top_assets, top_assets].round(2)

                fig_corr = go.Figure(go.Heatmap(
                    z=corr_sub.values,
                    x=corr_sub.columns.tolist(),
                    y=corr_sub.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0, zmin=-1, zmax=1,
                    text=corr_sub.values.round(2),
                    texttemplate="%{text}",
                    hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                ))
                fig_corr.update_layout(height=420, template="plotly_dark",
                                       margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No correlation data yet — run data_engine.py first.")

        # ── Sector heat bar chart ──────────────────────────────────────────
        with col_r:
            st.subheader("Sector Heat (Example Allocation)")
            sample_positions = {
                "BTC": 0.08, "ETH": 0.06,
                "GOLD": 0.05, "SP500": 0.07,
                "NVDA": 0.06, "SBER": 0.04,
            }
            heat = _pm.get_portfolio_heat(sample_positions)
            active_heat = {k: v for k, v in heat.items() if v > 0}
            limits = {k: v for k, v in SECTOR_LIMITS.items() if k in active_heat}

            if active_heat:
                sectors = list(active_heat.keys())
                values  = [active_heat[s] for s in sectors]
                lims    = [limits.get(s, 0.5) for s in sectors]

                fig_heat = go.Figure()
                fig_heat.add_trace(go.Bar(x=sectors, y=values, name="Current", marker_color="#00ACC1"))
                fig_heat.add_trace(go.Bar(x=sectors, y=lims,   name="Limit",   marker_color="rgba(255,80,80,0.4)"))
                fig_heat.update_layout(
                    barmode="overlay", height=420, template="plotly_dark",
                    yaxis_tickformat=".0%",
                    title="Sector Exposure vs Limits",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            score = _pm.get_diversification_score(sample_positions)
            st.metric("Diversification Score", f"{score:.1f} / 100",
                      delta="Well diversified" if score > 60 else "Concentrated",
                      delta_color="normal" if score > 60 else "inverse")

        # ── Asset statistics table ─────────────────────────────────────────
        st.divider()
        st.subheader("Asset Risk Statistics (Last 120 days)")
        with st.spinner("Computing per-asset statistics…"):
            stats_df = _pm.get_all_stats()

        if not stats_df.empty:
            display = stats_df[["sector", "vol_annual", "sharpe_90d", "max_dd_30d", "skewness", "kurtosis"]].copy()
            display.columns = ["Sector", "Vol (ann.)", "Sharpe 90d", "Max DD 30d", "Skew", "Kurtosis"]
            display["Vol (ann.)"]  = display["Vol (ann.)"].map("{:.1%}".format)
            display["Sharpe 90d"]  = display["Sharpe 90d"].map("{:.2f}".format)
            display["Max DD 30d"]  = display["Max DD 30d"].map("{:.1%}".format)
            display["Skew"]        = display["Skew"].map("{:.2f}".format)
            display["Kurtosis"]    = display["Kurtosis"].map("{:.2f}".format)
            st.dataframe(display, use_container_width=True)
        else:
            st.info("No statistics available yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Risk Manager Dashboard")

    if not RISK_AVAILABLE:
        st.warning("risk_manager.py not found. Run the setup to enable this tab.")
    else:
        summary = _rm.get_summary()
        halted  = summary["trading_halted"]

        # ── Status banner ──────────────────────────────────────────────────
        if halted:
            st.error(f"[OFF] TRADING HALTED — {summary['halt_reason']}")
        else:
            st.success("[ON] Trading Active — all circuit breakers OK")

        st.divider()

        # ── Key metrics row ────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Capital",      f"${summary['current_capital']:,.0f}")
        m2.metric("Total Return", f"{summary['total_return_pct']:+.2f}%",
                  delta_color="normal" if summary['total_return_pct'] >= 0 else "inverse")
        m3.metric("Drawdown",     f"{summary['current_drawdown']:.1%}",
                  delta=f"Limit: {RISK_CONFIG['max_drawdown_halt']:.0%}",
                  delta_color="inverse" if summary['current_drawdown'] > 0.08 else "normal")
        m4.metric("Daily P&L",    f"{summary['daily_pnl']:+.1%}",
                  delta_color="normal" if summary['daily_pnl'] >= 0 else "inverse")
        m5.metric("Exposure",     f"{summary['total_exposure']:.1%}",
                  delta=f"Max: {RISK_CONFIG['max_portfolio_exposure']:.0%}",
                  delta_color="inverse" if summary['total_exposure'] > 0.25 else "normal")

        st.divider()

        # ── Kelly Criterion calculator ─────────────────────────────────────
        st.subheader("Kelly Position Sizer")
        kc1, kc2, kc3, kc4 = st.columns(4)
        k_win_rate  = kc1.slider("Win Rate",   0.40, 0.80, 0.56, 0.01, format="%.2f")
        k_avg_win   = kc2.slider("Avg Win %",  0.005, 0.10, 0.025, 0.005, format="%.3f")
        k_avg_loss  = kc3.slider("Avg Loss %", 0.005, 0.05, 0.012, 0.005, format="%.3f")
        k_taleb     = kc4.slider("Taleb Risk", 0.0,   8.0,  1.0,   0.1)

        kelly_pct = _rm.kelly_fraction(k_win_rate, k_avg_win, k_avg_loss, k_taleb)
        kelly_usd = _rm.current_capital * kelly_pct

        ka, kb, kc = st.columns(3)
        ka.metric("Recommended Size", f"{kelly_pct:.1%}")
        kb.metric("In Dollars",       f"${kelly_usd:,.0f}")
        edge = k_win_rate * (k_avg_win / k_avg_loss) - (1 - k_win_rate)
        kc.metric("Edge (raw Kelly)", f"{edge:.3f}",
                  delta="Positive edge" if edge > 0 else "Negative edge",
                  delta_color="normal" if edge > 0 else "inverse")

        st.divider()

        # ── Risk config display ────────────────────────────────────────────
        st.subheader("Active Risk Rules")
        rules = {
            "Max portfolio exposure":  f"{RISK_CONFIG['max_portfolio_exposure']:.0%}",
            "Max single position":     f"{RISK_CONFIG['max_single_position']:.0%}",
            "Daily loss limit":        f"{RISK_CONFIG['max_daily_loss']:.0%}",
            "Max drawdown halt":       f"{RISK_CONFIG['max_drawdown_halt']:.0%}",
            "Fractional Kelly":        f"{RISK_CONFIG['kelly_fraction']:.0%} of full Kelly",
            "Min trade Kelly":         f"{RISK_CONFIG['min_kelly_threshold']:.1%}",
            "Taleb risk cap (BUY)":    str(RISK_CONFIG['taleb_risk_cap']),
            "Correlation penalty":     f"{RISK_CONFIG['correlation_penalty']:.0%} per correlated pos.",
        }
        rules_df = pd.DataFrame(list(rules.items()), columns=["Rule", "Value"])
        st.table(rules_df.set_index("Rule"))

        # ── Signal check playground ────────────────────────────────────────
        st.subheader("Signal Risk Check Playground")
        p1, p2, p3, p4 = st.columns(4)
        pg_asset   = p1.selectbox("Asset",   list(FULL_ASSET_MAP.keys()), index=0)
        pg_signal  = p2.selectbox("Signal",  ["BUY", "SELL"])
        pg_conf    = p3.slider("Confidence", 0.50, 0.90, 0.60, 0.01)
        pg_taleb   = p4.slider("Taleb",      0.0,  8.0,  1.0,  0.1)

        result = _rm.check_signal(pg_asset, pg_signal, pg_conf, pg_taleb)
        if result["approved"]:
            st.success(f"[OK] {result['reason']}")
            ra, rb, rc = st.columns(3)
            ra.metric("Position Size", f"{result['position_size_pct']:.1%}")
            rb.metric("In Dollars",    f"${result['position_size_usd']:,.0f}")
            rc.metric("Raw Kelly",     f"{result['kelly_raw']:.3f}")
        else:
            st.error(f"[!] REJECTED — {result['reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("News & Sentiment Analysis")

    if not NEWS_AVAILABLE:
        st.warning("news_analyzer.py not found.")
    else:
        # Market Mood
        with st.spinner("Analyzing market mood..."):
            try:
                mood = _news.get_market_mood()
                mood_score = mood["mood_score"]
                mood_label = mood["mood_label"]

                mc1, mc2 = st.columns([1, 3])
                with mc1:
                    mood_color = "normal" if mood_score >= 0 else "inverse"
                    st.metric("Market Mood", mood_label,
                              delta=f"{mood_score:+.3f}",
                              delta_color=mood_color)

                with mc2:
                    if mood["details"]:
                        mood_data = []
                        for d in mood["details"]:
                            mood_data.append({
                                "Asset": d["asset"],
                                "Sentiment": d.get("weighted_sentiment", d.get("avg_sentiment", 0)),
                                "Label": d["sentiment_label"],
                                "Articles": d["articles_count"],
                            })
                        mood_df = pd.DataFrame(mood_data)
                        if not mood_df.empty:
                            colors = ["#22c55e" if s >= 0 else "#ef4444"
                                      for s in mood_df["Sentiment"]]
                            fig_mood = go.Figure(go.Bar(
                                x=mood_df["Asset"], y=mood_df["Sentiment"],
                                marker_color=colors,
                                text=mood_df["Label"],
                                textposition="auto",
                            ))
                            fig_mood.update_layout(
                                height=250, template="plotly_dark",
                                title="Macro Sentiment Heatmap",
                                yaxis_title="Weighted Score",
                                margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig_mood, use_container_width=True)
            except Exception as e:
                st.error(f"Mood analysis error: {e}")

        st.divider()

        # Asset-specific news
        st.subheader(f"News: {selected_asset}")
        with st.spinner(f"Fetching news for {selected_asset}..."):
            try:
                articles = _news.fetch_news(selected_asset, max_articles=12)
                if articles:
                    for art in articles:
                        src = art.get("source", "Unknown")
                        ws = art.get("weighted_score", art.get("sentiment_score", 0))
                        cred = art.get("credibility", 1.0)
                        tier = "[TOP]" if cred >= 1.5 else ("[ + ]" if cred >= 1.3 else "")

                        col_s, col_t = st.columns([1, 5])
                        with col_s:
                            s_color = "green" if ws > 0 else ("red" if ws < 0 else "gray")
                            st.markdown(f":{s_color}[**{ws:+.2f}**]")
                            st.caption(f"{tier} {src}")
                        with col_t:
                            title = art["title"]
                            link = art.get("link", "")
                            if link:
                                st.markdown(f"**[{title}]({link})**")
                            else:
                                st.markdown(f"**{title}**")
                            desc = art.get("description", "")
                            if desc and len(desc) > 15:
                                st.caption(desc[:250])
                        st.markdown("---")
                else:
                    st.info("No news found for this asset.")
            except Exception as e:
                st.error(f"News error: {e}")

        # Top-tier digest
        with st.expander("Full Digest (All Sources)", expanded=False):
            with st.spinner("Loading digest..."):
                try:
                    digest = _news.fetch_authority_digest(
                        max_per_source=3, lang_filter="all",
                        fetch_summaries=False)
                    tier1 = [d for d in digest if d.get("credibility", 1.0) >= 1.3][:20]
                    if tier1:
                        digest_data = []
                        for item in tier1:
                            digest_data.append({
                                "Source": item["source"],
                                "Score": f"{item['weighted_score']:+.2f}",
                                "Headline": item["title"][:80],
                                "Summary": (item.get("description", ""))[:120],
                            })
                        st.dataframe(pd.DataFrame(digest_data),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No digest data available.")
                except Exception as e:
                    st.error(f"Digest error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET REGIME
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Market Regime Detector")

    if not REGIME_AVAILABLE:
        st.warning("regime_detector.py not found.")
    else:
        # Global regime
        try:
            regime = _regime.get_global_regime()
            r1, r2, r3, r4 = st.columns(4)

            regime_name = regime.get("status", "UNKNOWN")
            regime_colors = {
                "RISK-ON": "green", "RISK-OFF": "orange",
                "CRISIS": "red", "SIDEWAYS": "gray"
            }
            rc = regime_colors.get(regime_name, "gray")
            r1.metric("Global Regime", regime_name)
            vix_val = regime.get("vix_value") or 0
            r2.metric("VIX", f"{vix_val:.1f}",
                      delta=regime.get("vix_level", ""),
                      delta_color="inverse" if vix_val > 25 else "normal")
            r3.metric("SP500 Trend", regime.get("sp500_trend", "?"))
            r4.metric("DXY Trend", regime.get("dxy_trend", "?"))

            st.markdown(f"**Regime: :{rc}[{regime_name}]** — {regime.get('description', '')}")

        except Exception as e:
            st.error(f"Global regime error: {e}")

        st.divider()

        # Market breadth
        try:
            breadth = _regime.get_market_breadth()
            if breadth:
                b1, b2, b3, b4 = st.columns(4)
                above_pct = breadth.get("above_sma50_pct", 0)
                pos_pct = breadth.get("positive_20d_pct", 0)
                total = breadth.get("total_assets", 0)
                score = breadth.get("score", "?")
                b1.metric("Above SMA50", f"{above_pct:.1f}%")
                b2.metric("Positive 20D", f"{pos_pct:.1f}%")
                b3.metric("Breadth", score,
                          delta_color="normal" if score == "STRONG" else "inverse")
                b4.metric("Assets Tracked", str(total))
        except Exception:
            pass

        st.divider()

        # Per-asset regime
        st.subheader(f"Asset Regime: {selected_asset}")
        try:
            ar = _regime.get_asset_regime(selected_asset)
            if ar:
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("Trend", ar.get("trend", "?"))
                ac2.metric("Volatility", ar.get("volatility", "?"))
                ac3.metric("Momentum", ar.get("momentum", "?"))
                rsi_r = ar.get("rsi", 0)
                ac4.metric("RSI", f"{rsi_r:.1f}" if isinstance(rsi_r, (int, float)) else str(rsi_r))
        except Exception as e:
            st.error(f"Asset regime error: {e}")

        # All regimes overview
        with st.expander("All Assets Regime Overview", expanded=False):
            with st.spinner("Scanning all assets..."):
                try:
                    all_r = _regime.get_all_regimes()
                    assets_r = all_r.get("assets", {}) if all_r else {}
                    if assets_r:
                        reg_rows = []
                        for asset_name, info in assets_r.items():
                            if isinstance(info, dict):
                                reg_rows.append({
                                    "Asset": asset_name,
                                    "Trend": info.get("trend", "?"),
                                    "Volatility": info.get("volatility", "?"),
                                    "Momentum": info.get("momentum", "?"),
                                    "RSI": f"{info['rsi']:.1f}" if info.get("rsi") else "?",
                                })
                        if reg_rows:
                            st.dataframe(pd.DataFrame(reg_rows),
                                         use_container_width=True, hide_index=True)
                    else:
                        st.info("No regime data available.")
                except Exception as e:
                    st.error(f"All regimes error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PAPER TRADING
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Paper Trading (Virtual Portfolio)")

    if not PAPER_AVAILABLE:
        st.warning("paper_trading.py not found.")
    else:
        try:
            summary = _paper.portfolio_summary(quiet=True)

            # Portfolio overview
            p1, p2, p3, p4 = st.columns(4)
            balance = summary.get("balance", 10000)
            initial = summary.get("initial", 10000)
            equity = summary.get("equity", balance)
            ret_pct = summary.get("return_pct", 0)
            positions = summary.get("positions", [])

            p1.metric("Cash Balance", f"${balance:,.2f}")
            p2.metric("Equity", f"${equity:,.2f}")
            p3.metric("Return", f"{ret_pct:+.1f}%",
                      delta_color="normal" if ret_pct >= 0 else "inverse")
            p4.metric("Open Positions", str(len(positions)))

            st.divider()

            # Open positions table
            if positions:
                st.subheader("Open Positions")
                pos_data = []
                for pos in positions:
                    if isinstance(pos, dict):
                        pos_data.append({
                            "Asset": pos.get("asset", "?"),
                            "Side": pos.get("side", "LONG"),
                            "Entry": f"${pos.get('entry_price', 0):,.2f}",
                            "Current": f"${pos.get('current_price', 0):,.2f}",
                            "Qty": f"{pos.get('quantity', 0):.4f}",
                            "P&L": f"${pos.get('unrealised_pnl', 0):+,.2f}",
                        })
                if pos_data:
                    st.dataframe(pd.DataFrame(pos_data),
                                 use_container_width=True, hide_index=True)

            # Quick trade
            st.divider()
            st.subheader("Quick Trade")
            tc1, tc2, tc3 = st.columns(3)
            trade_asset = tc1.selectbox("Asset", list(FULL_ASSET_MAP.keys()),
                                        key="paper_asset")
            max_amount = max(100, int(balance))
            trade_amount = tc2.number_input("Amount ($)", min_value=100,
                                            max_value=max_amount,
                                            value=min(1000, max_amount),
                                            step=100, key="paper_amount")
            trade_action = tc3.selectbox("Action", ["BUY", "SELL"],
                                         key="paper_action")

            if st.button("Execute Trade", key="paper_execute"):
                try:
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        if trade_action == "BUY":
                            _paper.buy_asset(trade_asset, trade_amount)
                        else:
                            _paper.sell_asset(trade_asset)
                    output = buf.getvalue()
                    if "[OK]" in output:
                        st.success(output.strip())
                        st.rerun()
                    elif "[ERROR]" in output:
                        st.error(output.strip())
                    else:
                        st.info(output.strip() if output.strip() else "Done")
                        st.rerun()
                except Exception as e:
                    st.error(f"Trade error: {e}")

            # Trade history
            with st.expander("Trade History", expanded=False):
                try:
                    closed = _paper._closed_positions()
                    if closed:
                        hist_data = []
                        for t in closed:
                            hist_data.append({
                                "Asset": t.get("asset", "?"),
                                "Side": t.get("side", "?"),
                                "Entry": f"${t.get('entry_price', 0):,.2f}",
                                "Exit": f"${(t.get('exit_price') or 0):,.2f}",
                                "P&L": f"${(t.get('pnl') or 0):+,.2f}",
                                "Date": (t.get("exit_date") or "")[:10],
                            })
                        st.dataframe(pd.DataFrame(hist_data),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No closed trades yet.")
                except Exception as e:
                    st.info(f"No trade history available."  )

        except Exception as e:
            st.error(f"Paper trading error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SIGNAL RADAR
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("Signal Radar — All Assets")
    if not RADAR_AVAILABLE:
        st.warning("signal_dashboard.py not found.")
    else:
        try:
            with st.spinner("Scanning all assets..."):
                sig_df = _radar.get_all_signals(progress=False)
            if sig_df is not None and not sig_df.empty:
                summary = _radar.get_signal_summary(sig_df)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Assets", len(sig_df))
                s2.metric("BUY Signals", summary.get("total_buy", 0))
                s3.metric("SELL Signals", summary.get("total_sell", 0))
                s4.metric("WAIT", summary.get("total_wait", 0))
                st.divider()

                # Color-coded signal filter
                sig_filter = st.selectbox("Filter", ["ALL", "BUY", "SELL", "WAIT"], key="radar_filter")
                display_df = sig_df if sig_filter == "ALL" else sig_df[sig_df["Signal"] == sig_filter]

                # Style the dataframe
                def _color_signal(val):
                    if val == "BUY": return "color: #22c55e; font-weight: bold"
                    if val == "SELL": return "color: #ef4444; font-weight: bold"
                    return "color: #94a3b8"

                styled = display_df.style.applymap(_color_signal, subset=["Signal"])
                styled = styled.format({
                    "Price": "{:,.2f}", "Chg_1d": "{:+.1%}", "Probability": "{:.3f}",
                    "Confidence": "{:.0%}", "RSI": "{:.0f}", "CB_Prob": "{:.3f}",
                    "LSTM_Prob": "{:.3f}", "Kelly_Size": "{:.1f}%"
                })
                st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

                # Strongest signals
                st.divider()
                c1, c2 = st.columns(2)
                buys = sig_df[sig_df["Signal"] == "BUY"].head(5)
                sells = sig_df[sig_df["Signal"] == "SELL"].head(5)
                with c1:
                    st.subheader("Strongest BUY")
                    if not buys.empty:
                        st.dataframe(buys[["Asset", "Probability", "Confidence", "RSI", "Trend"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No BUY signals")
                with c2:
                    st.subheader("Strongest SELL")
                    if not sells.empty:
                        st.dataframe(sells[["Asset", "Probability", "Confidence", "RSI", "Trend"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No SELL signals")
            else:
                st.info("No signal data available.")
        except Exception as e:
            st.error(f"Signal Radar error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("Prediction Performance Tracker")
    if not PERF_AVAILABLE:
        st.warning("performance_tracker.py not found.")
    else:
        try:
            _perf.update_actuals()
            days_sel = st.selectbox("Period", [7, 14, 30, 60, 90], index=2, key="perf_days")

            # Overall stats
            stats_all = _perf.get_accuracy(days=days_sel)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Accuracy", f"{stats_all.get('accuracy', 0):.1%}")
            m2.metric("Total Predictions", stats_all.get("total_predictions", 0))
            m3.metric("Correct", stats_all.get("correct_count", 0))
            by_sig = stats_all.get("by_signal", {})
            buy_acc = by_sig.get("BUY", {}).get("acc", 0)
            sell_acc = by_sig.get("SELL", {}).get("acc", 0)
            m4.metric("BUY / SELL Acc", f"{buy_acc:.0%} / {sell_acc:.0%}")

            st.divider()
            c1, c2 = st.columns(2)

            # Accuracy history chart
            with c1:
                st.subheader("Rolling Accuracy")
                hist = _perf.get_accuracy_history(days=days_sel)
                if hist is not None and not hist.empty and "rolling_acc" in hist.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist["date"], y=hist["rolling_acc"],
                                             mode="lines+markers", name="Accuracy",
                                             line=dict(color="#38bdf8", width=2)))
                    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
                    fig.update_layout(height=300, template="plotly_dark",
                                      yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough prediction data yet.")

            # Leaderboard
            with c2:
                st.subheader("Asset Leaderboard")
                lb = _perf.get_leaderboard(days=days_sel)
                if lb is not None and not lb.empty:
                    st.dataframe(lb, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data for leaderboard.")

            # Daily stats
            st.subheader("Daily Predictions")
            daily = _perf.get_daily_stats(days=days_sel)
            if daily is not None and not daily.empty and "Accuracy" in daily.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=daily["Date"], y=daily["Predictions"],
                                       name="Predictions", marker_color="#334155"))
                fig2.add_trace(go.Scatter(x=daily["Date"], y=daily["Accuracy"],
                                           name="Accuracy", yaxis="y2",
                                           line=dict(color="#22c55e", width=2)))
                fig2.update_layout(height=250, template="plotly_dark",
                                    yaxis2=dict(overlaying="y", side="right", tickformat=".0%"),
                                    margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No daily prediction data yet.")
        except Exception as e:
            st.error(f"Performance Tracker error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — SECTOR ROTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab10:
    st.header("Sector Rotation Analysis")
    if not SECTOR_AVAILABLE:
        st.warning("sector_rotation.py not found.")
    else:
        try:
            weeks_sel = st.slider("Weeks", 4, 12, 8, key="sector_weeks")

            # Momentum table
            st.subheader("Sector Momentum")
            mom = _sector.get_sector_momentum(weeks=min(weeks_sel, 8))
            if mom is not None and not mom.empty:
                def _color_trend(val):
                    if val == "RISING": return "color: #22c55e; font-weight: bold"
                    if val == "FALLING": return "color: #ef4444; font-weight: bold"
                    return "color: #94a3b8"
                st.dataframe(mom.style.applymap(_color_trend, subset=["Trend"]),
                             use_container_width=True, hide_index=True)

            st.divider()

            # Rotation heatmap
            st.subheader("Weekly Returns Heatmap")
            matrix = _sector.get_rotation_matrix(weeks=weeks_sel)
            if matrix is not None and not matrix.empty:
                sectors_list = matrix.index.tolist()
                week_cols = matrix.columns.tolist()
                fig = go.Figure(go.Heatmap(
                    z=matrix.values, x=week_cols, y=sectors_list,
                    colorscale="RdYlGn", zmid=0,
                    text=matrix.values.round(2), texttemplate="%{text:.1f}%",
                    hovertemplate="%{y} / %{x}: %{z:.2f}%<extra></extra>",
                ))
                fig.update_layout(height=350, template="plotly_dark",
                                   margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Drill-down into sector
            st.divider()
            sector_pick = st.selectbox("Drill-down sector", list(_sector.SECTORS.keys()), key="sector_drill")
            detail = _sector.get_asset_returns_by_sector(sector_pick, weeks=min(weeks_sel, 6))
            if detail is not None and not detail.empty:
                st.dataframe(detail.style.format("{:.2f}%", subset=[c for c in detail.columns if c != "Asset"]),
                             use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Sector Rotation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab11:
    st.header("What-If Simulator")
    if not WHATIF_AVAILABLE:
        st.warning("whatif_simulator.py not found.")
    else:
        try:
            wi1, wi2, wi3 = st.columns(3)
            wi_capital = wi1.number_input("Capital ($)", 1000, 100000, 10000, 1000, key="wi_cap")
            wi_days = wi2.selectbox("Days back", [30, 60, 90, 180], index=2, key="wi_days")
            wi_strategy = wi3.selectbox("Strategy", ["equal", "kelly"], key="wi_strat")

            mode = st.radio("Mode", ["Top N by Score", "Custom assets"], horizontal=True, key="wi_mode")
            if mode == "Top N by Score":
                wi_n = st.slider("Top N assets", 3, 15, 5, key="wi_topn")
                sim_assets = None
            else:
                wi_assets_str = st.multiselect("Select assets", list(FULL_ASSET_MAP.keys()),
                                                default=["BTC", "ETH", "NVDA", "GOLD", "SP500"],
                                                key="wi_assets")
                sim_assets = wi_assets_str

            if st.button("Run Simulation", key="wi_run"):
                with st.spinner("Simulating..."):
                    if sim_assets:
                        result = _whatif.simulate(sim_assets, wi_capital, wi_days, wi_strategy)
                    else:
                        result = _whatif.simulate_top_n(wi_n, wi_capital, wi_days)

                if result:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Final Equity", f"${result.get('final', 0):,.2f}")
                    r2.metric("Return", f"{result.get('return_pct', 0):+.1f}%")
                    r3.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.1f}%")
                    r4.metric("Sharpe", f"{result.get('sharpe', 0):.2f}")

                    # Equity curve
                    eq = result.get("equity_curve", [])
                    if eq:
                        eq_df = pd.DataFrame(eq, columns=["Date", "Equity"])
                        fig = go.Figure(go.Scatter(x=eq_df["Date"], y=eq_df["Equity"],
                                                    fill="tozeroy", line=dict(color="#38bdf8", width=2)))
                        fig.update_layout(height=350, template="plotly_dark",
                                           margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    # Per-asset results
                    per = result.get("per_asset", {})
                    if per:
                        st.subheader("Per-Asset Results")
                        pa_data = [{"Asset": a, **v} for a, v in per.items()]
                        st.dataframe(pd.DataFrame(pa_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("Simulation returned no results.")
        except Exception as e:
            st.error(f"What-If error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — MODEL HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab12:
    st.header("Model Health & Comparison")
    if not MC_AVAILABLE:
        st.warning("model_comparison.py not found.")
    else:
        try:
            _mc.save_snapshot()

            c1, c2 = st.columns(2)

            # Model age
            with c1:
                st.subheader("Model Age")
                age_df = _mc.get_model_age()
                if age_df is not None and not age_df.empty:
                    def _color_status(val):
                        if val == "FRESH": return "color: #22c55e"
                        if val == "STALE": return "color: #eab308"
                        if val == "CRITICAL": return "color: #ef4444; font-weight: bold"
                        return ""
                    st.dataframe(age_df.style.applymap(_color_status, subset=["Status"]),
                                 use_container_width=True, hide_index=True, height=400)

            # Comparison vs previous
            with c2:
                st.subheader("Score Change vs Previous")
                comp = _mc.get_comparison_table()
                if comp is not None and not comp.empty:
                    display_cols = [c for c in ["Asset", "Current_Score", "Prev_Score", "Score_Delta",
                                                 "Current_CB", "CB_Delta", "Current_LSTM", "LSTM_Delta"]
                                    if c in comp.columns]
                    st.dataframe(comp[display_cols], use_container_width=True, hide_index=True, height=400)

            # Score history chart
            st.divider()
            st.subheader("Score History (Top Assets)")
            hist = _mc.get_history(metric="score", last_n=20)
            if hist is not None and not hist.empty:
                # Pick top 8 assets by latest score
                latest = hist.iloc[-1].dropna().sort_values(ascending=False)
                top_assets = latest.head(8).index.tolist()
                fig = go.Figure()
                for asset in top_assets:
                    if asset in hist.columns and asset != "Date":
                        fig.add_trace(go.Scatter(x=hist.index if "Date" not in hist.columns else hist["Date"],
                                                  y=hist[asset], name=asset, mode="lines+markers"))
                fig.update_layout(height=350, template="plotly_dark",
                                   margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # Best/Worst
            bw = _mc.get_best_worst(metric="score", n=5)
            if bw:
                b1, b2 = st.columns(2)
                with b1:
                    st.subheader("Top 5 by Score")
                    if bw.get("best"):
                        st.dataframe(pd.DataFrame(bw["best"]), use_container_width=True, hide_index=True)
                with b2:
                    st.subheader("Bottom 5 by Score")
                    if bw.get("worst"):
                        st.dataframe(pd.DataFrame(bw["worst"]), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Model Health error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with tab13:
    st.header("Custom Alert Rules")
    if not ALERTS_AVAILABLE:
        st.warning("alert_rules.py not found.")
    else:
        try:
            # Check current alerts
            triggered = _alerts.check_alerts()
            if triggered:
                st.error(f"{len(triggered)} alerts triggered!")
                for a in triggered:
                    st.warning(a.get("message", str(a)))
            else:
                st.success("No alerts triggered.")

            st.divider()

            # Current rules
            st.subheader("Active Rules")
            rules = _alerts.load_rules()
            if rules:
                rules_data = []
                for r in rules:
                    rules_data.append({
                        "ID": r.get("id"), "Asset": r.get("asset"),
                        "Condition": r.get("condition"), "Value": r.get("value"),
                        "Enabled": r.get("enabled", True),
                        "Last Triggered": (r.get("last_triggered") or "Never")[:10],
                    })
                st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)

                # Toggle/Remove
                tc1, tc2 = st.columns(2)
                toggle_id = tc1.number_input("Toggle rule ID", min_value=1, step=1, key="alert_toggle_id")
                if tc1.button("Toggle", key="alert_toggle_btn"):
                    try:
                        _alerts.toggle_rule(int(toggle_id))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                remove_id = tc2.number_input("Remove rule ID", min_value=1, step=1, key="alert_remove_id")
                if tc2.button("Remove", key="alert_remove_btn"):
                    _alerts.remove_rule(int(remove_id))
                    st.rerun()
            else:
                st.info("No rules defined yet.")

            # Add new rule
            st.divider()
            st.subheader("Add New Rule")
            ac1, ac2, ac3 = st.columns(3)
            al_asset = ac1.selectbox("Asset", list(FULL_ASSET_MAP.keys()), key="alert_asset")
            al_cond = ac2.selectbox("Condition", [
                "rsi_below", "rsi_above", "price_above", "price_below",
                "price_drop_pct", "price_rise_pct", "trend_change"
            ], key="alert_cond")
            al_val = ac3.number_input("Value", value=30.0, step=1.0, key="alert_val")

            if st.button("Add Rule", key="alert_add_btn"):
                try:
                    _alerts.add_rule(al_asset, al_cond, al_val)
                    st.success(f"Rule added: {al_asset} {al_cond} {al_val}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        except Exception as e:
            st.error(f"Alerts error: {e}")