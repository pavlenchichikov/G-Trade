# config.py
import os
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set externally

# --- 1. ПАРАМЕТРЫ МОДЕЛИ ---
SEQ_LEN = 10

# Оптимизированные пороги на основе Backtest V67
THRESHOLDS = {
    "DEFAULT": 0.55,    # Базовый порог

    # ELITE (lower threshold for high-performing assets)
    "TSLA": 0.53,
    "ETH": 0.53,
    "GOLD": 0.54,
    "VIX": 0.53,

    # CAUTIOUS (higher threshold for safety)
    "TON": 0.60,
    "NASDAQ": 0.58,
    "SOL": 0.58,
    "DXY": 0.58,

    # CONSERVATIVE
    "BTC": 0.54,
    "SBER": 0.54
}

# Настройки Telegram — loaded from .env (never hardcode credentials)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "")

# Proxy
SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "socks5h://127.0.0.1:12334")

# --- 2. КАРТА АКТИВОВ ---
FULL_ASSET_MAP = {
    # МИРОВЫЕ ИНДЕКСЫ
    'VIX': '^VIX', 'DXY': 'DX=F', 'TNX': '^TNX',
    'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI',

    # СЫРЬЕ
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'OIL': 'CL=F', 'GAS': 'NG=F',

    # ТЕХНОЛОГИИ США
    'NVDA': 'NVDA', 'TSLA': 'TSLA', 'AAPL': 'AAPL', 'MSFT': 'MSFT',
    'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'META': 'META', 'AMD': 'AMD',
    'PLTR': 'PLTR', 'COIN': 'COIN', 'MSTR': 'MSTR',

    # US HEALTHCARE
    'JNJ': 'JNJ', 'UNH': 'UNH', 'PFE': 'PFE', 'LLY': 'LLY',
    'ABBV': 'ABBV', 'MRK': 'MRK',

    # US FINANCE
    'JPM': 'JPM', 'BAC': 'BAC', 'GS': 'GS', 'V': 'V',
    'MA': 'MA', 'WFC': 'WFC',

    # US CONSUMER
    'WMT': 'WMT', 'KO': 'KO', 'PEP': 'PEP', 'MCD': 'MCD',
    'NKE': 'NKE', 'DIS': 'DIS', 'NFLX': 'NFLX', 'SBUX': 'SBUX',

    # US INDUSTRIAL & ENERGY
    'BA': 'BA', 'CAT': 'CAT', 'XOM': 'XOM', 'CVX': 'CVX', 'COP': 'COP',

    # US SEMICONDUCTORS
    'INTC': 'INTC', 'QCOM': 'QCOM', 'AVGO': 'AVGO', 'MU': 'MU',

    # US SOFTWARE
    'CRM': 'CRM', 'ORCL': 'ORCL', 'ADBE': 'ADBE', 'UBER': 'UBER', 'PYPL': 'PYPL',

    # КРИПТОВАЛЮТА
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'XRP': 'XRP-USD', 'TON': 'TON11419-USD', 'DOGE': 'DOGE-USD', 'BNB': 'BNB-USD',
    'ADA': 'ADA-USD', 'AVAX': 'AVAX-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
    'SHIB': 'SHIB-USD', 'ATOM': 'ATOM-USD', 'UNI': 'UNI-USD', 'NEAR': 'NEAR-USD',

    # РЫНОК РФ — Голубые фишки
    'IMOEX': 'IMOEX', 'SBER': 'SBER', 'GAZP': 'GAZP', 'LKOH': 'LKOH',
    'ROSN': 'ROSN', 'NVTK': 'NVTK', 'TATN': 'TATN', 'SNGS': 'SNGS',
    'PLZL': 'PLZL', 'SIBN': 'SIBN', 'MGNT': 'MGNT',
    # РФ — Банки и финансы
    'TCSG': 'T', 'VTBR': 'VTBR', 'BSPB': 'BSPB', 'MOEX_EX': 'MOEX',
    # РФ — Технологии и рост
    'YNDX': 'YDEX', 'OZON': 'OZON', 'VKCO': 'VKCO', 'POSI': 'POSI',
    'MTSS': 'MTSS', 'RTKM': 'RTKM',
    # РФ — Металлы и промышленность
    'CHMF': 'CHMF', 'NLMK': 'NLMK', 'MAGN': 'MAGN',
    'RUAL': 'RUAL', 'ALRS': 'ALRS',
    # РФ — Энергетика и транспорт
    'IRAO': 'IRAO', 'HYDR': 'HYDR', 'FLOT': 'FLOT',
    'AFLT': 'AFLT', 'PIKK': 'PIKK',
    # РФ — Химия и удобрения
    'PHOR': 'PHOR', 'SGZH': 'SGZH',
    # РФ — Ритейл
    'FIVE': 'FIVE', 'FIXP': 'FIXP', 'LENT': 'LENT', 'MVID': 'MVID',
    # РФ — Строительство и девелопмент
    'SMLT': 'SMLT', 'LSRG': 'LSRG',
    # РФ — Банки и финансы (доп)
    'CBOM': 'CBOM',
    # РФ — Энергетика (доп)
    'FEES': 'FEES', 'UPRO': 'UPRO', 'MSNG': 'MSNG',
    # РФ — Промышленность (доп)
    'TRMK': 'TRMK', 'MTLR': 'MTLR', 'RASP': 'RASP', 'NMTP': 'NMTP',
    # РФ — IT (доп)
    'HHRU': 'HHRU', 'SOFL': 'SOFL', 'ASTR': 'ASTR', 'WUSH': 'WUSH',

    # FOREX — Мажоры
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'CAD=X',
    'NZDUSD': 'NZDUSD=X', 'USDRUB': 'RUB=X',
    # FOREX — Кроссы EUR
    'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X', 'EURCHF': 'EURCHF=X',
    'EURAUD': 'EURAUD=X', 'EURCAD': 'EURCAD=X', 'EURNZD': 'EURNZD=X',
    # FOREX — Кроссы GBP
    'GBPJPY': 'GBPJPY=X', 'GBPAUD': 'GBPAUD=X', 'GBPCAD': 'GBPCAD=X',
    'GBPCHF': 'GBPCHF=X', 'GBPNZD': 'GBPNZD=X',
    # FOREX — Кроссы AUD/NZD/CAD/CHF
    'AUDCAD': 'AUDCAD=X', 'AUDCHF': 'AUDCHF=X', 'AUDJPY': 'AUDJPY=X',
    'AUDNZD': 'AUDNZD=X', 'CADJPY': 'CADJPY=X', 'CHFJPY': 'CHFJPY=X',
    'NZDJPY': 'NZDJPY=X',
    # FOREX — Экзотика
    'USDTRY': 'TRY=X', 'USDMXN': 'MXN=X', 'USDZAR': 'ZAR=X',
    'USDSGD': 'SGD=X', 'USDNOK': 'NOK=X', 'USDSEK': 'SEK=X',
    'USDPLN': 'PLN=X', 'USDCNH': 'CNH=X',
}

# --- 3. ГРУППИРОВКА ---
ASSET_TYPES = {
    "TOP SIGNALS": ["ETH", "TSLA", "GOLD", "VIX", "PLTR", "IMOEX"],
    "CRYPTO": ["BTC", "ETH", "SOL", "XRP", "TON", "DOGE", "BNB",
               "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR"],
    "COMMODITIES": ["GOLD", "SILVER", "OIL", "GAS"],
    "INDICES & MACRO": ["SP500", "NASDAQ", "DOW", "IMOEX", "VIX", "DXY", "TNX"],
    "US TECH": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "COIN", "MSTR"],
    "US HEALTHCARE": ["JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK"],
    "US FINANCE": ["JPM", "BAC", "GS", "V", "MA", "WFC"],
    "US CONSUMER": ["WMT", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX", "SBUX"],
    "US INDUSTRIAL": ["BA", "CAT", "XOM", "CVX", "COP"],
    "US SEMI": ["INTC", "QCOM", "AVGO", "MU"],
    "US SOFTWARE": ["CRM", "ORCL", "ADBE", "UBER", "PYPL"],
    "RUS BLUE CHIPS": ["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "PLZL", "SIBN", "MGNT"],
    "RUS FINANCE": ["TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM"],
    "RUS TECH": ["YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM", "HHRU", "SOFL", "ASTR", "WUSH"],
    "RUS METALS": ["CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP"],
    "RUS INFRA": ["IRAO", "HYDR", "FLOT", "AFLT", "PIKK", "FEES", "UPRO", "MSNG", "NMTP"],
    "RUS CONSUMER": ["FIVE", "FIXP", "LENT", "MVID"],
    "RUS PROPERTY": ["SMLT", "LSRG", "PHOR", "SGZH"],
    "FOREX MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDRUB"],
    "FOREX CROSSES": ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
                      "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
                      "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD",
                      "CADJPY", "CHFJPY", "NZDJPY"],
    "FOREX EXOTIC": ["USDTRY", "USDMXN", "USDZAR", "USDSGD", "USDNOK", "USDSEK", "USDPLN", "USDCNH"],
}
