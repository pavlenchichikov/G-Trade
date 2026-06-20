# config.py
import os
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set externally

# --- 1. MODEL PARAMETERS ---
SEQ_LEN = 10

# Buy/sell thresholds per asset
THRESHOLDS = {
    "DEFAULT": 0.55,    # Base threshold

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

# Telegram settings - loaded from .env (never hardcode credentials)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "")

# Proxy: empty string = no proxy. The address is set only via .env.
SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "")

# --- 2. ASSET MAP ---
FULL_ASSET_MAP = {
    # GLOBAL INDICES
    'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX',
    'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI',

    # COMMODITIES
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'OIL': 'CL=F', 'GAS': 'NG=F',

    # US TECH
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

    # CRYPTO
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'XRP': 'XRP-USD', 'TON': 'TON11419-USD', 'DOGE': 'DOGE-USD', 'BNB': 'BNB-USD',
    'ADA': 'ADA-USD', 'AVAX': 'AVAX-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
    'SHIB': 'SHIB-USD', 'ATOM': 'ATOM-USD', 'UNI': 'UNI7083-USD', 'NEAR': 'NEAR-USD',  # UNI-USD is empty on Yahoo, Uniswap trades under UNI7083-USD

    # RU MARKET - Blue chips
    'IMOEX': 'IMOEX', 'SBER': 'SBER', 'GAZP': 'GAZP', 'LKOH': 'LKOH',
    'ROSN': 'ROSN', 'NVTK': 'NVTK', 'TATN': 'TATN', 'SNGS': 'SNGS',
    'PLZL': 'PLZL', 'SIBN': 'SIBN', 'MGNT': 'MGNT',
    # RU - Banks and finance
    'TCSG': 'T', 'VTBR': 'VTBR', 'BSPB': 'BSPB', 'MOEX_EX': 'MOEX',
    # RU - Tech and growth
    'YNDX': 'YDEX', 'OZON': 'OZON', 'VKCO': 'VKCO', 'POSI': 'POSI',
    'MTSS': 'MTSS', 'RTKM': 'RTKM',
    # RU - Metals and industry
    'CHMF': 'CHMF', 'NLMK': 'NLMK', 'MAGN': 'MAGN',
    'RUAL': 'RUAL', 'ALRS': 'ALRS',
    # RU - Energy and transport
    'IRAO': 'IRAO', 'HYDR': 'HYDR', 'FLOT': 'FLOT',
    'AFLT': 'AFLT', 'PIKK': 'PIKK',
    # RU - Chemicals and fertilizers
    'PHOR': 'PHOR', 'SGZH': 'SGZH',
    # RU - Retail
    'FIVE': 'X5', 'FIXP': 'FIXR', 'LENT': 'LENT', 'MVID': 'MVID',  # FIVE and FIXP renamed on MOEX to X5 and FIXR
    # RU - Construction and development
    'SMLT': 'SMLT', 'LSRG': 'LSRG',
    # RU - Banks and finance (extra)
    'CBOM': 'CBOM',
    # RU - Energy (extra)
    'FEES': 'FEES', 'UPRO': 'UPRO', 'MSNG': 'MSNG',
    # RU - Industry (extra)
    'TRMK': 'TRMK', 'MTLR': 'MTLR', 'RASP': 'RASP', 'NMTP': 'NMTP',
    # RU - IT (extra)
    'HHRU': 'HEAD', 'SOFL': 'SOFL', 'ASTR': 'ASTR', 'WUSH': 'WUSH',  # HHRU renamed on MOEX to HEAD

    # FOREX - Majors
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'JPY=X',
    'USDCHF': 'CHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'CAD=X',
    'NZDUSD': 'NZDUSD=X', 'USDRUB': 'RUB=X',
    # FOREX - EUR crosses
    'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X', 'EURCHF': 'EURCHF=X',
    'EURAUD': 'EURAUD=X', 'EURCAD': 'EURCAD=X', 'EURNZD': 'EURNZD=X',
    # FOREX - GBP crosses
    'GBPJPY': 'GBPJPY=X', 'GBPAUD': 'GBPAUD=X', 'GBPCAD': 'GBPCAD=X',
    'GBPCHF': 'GBPCHF=X', 'GBPNZD': 'GBPNZD=X',
    # FOREX - AUD/NZD/CAD/CHF crosses
    'AUDCAD': 'AUDCAD=X', 'AUDCHF': 'AUDCHF=X', 'AUDJPY': 'AUDJPY=X',
    'AUDNZD': 'AUDNZD=X', 'CADJPY': 'CADJPY=X', 'CHFJPY': 'CHFJPY=X',
    'NZDJPY': 'NZDJPY=X',
    # FOREX - Exotics
    'USDTRY': 'TRY=X', 'USDMXN': 'MXN=X', 'USDZAR': 'ZAR=X',
    'USDSGD': 'SGD=X', 'USDNOK': 'NOK=X', 'USDSEK': 'SEK=X',
    'USDPLN': 'PLN=X', 'USDCNH': 'CNY=X',  # CNH=X is empty on Yahoo (1 bar), CNY=X returns full history; offshore/onshore yuan differ by pips
}

# --- 3. GROUPING ---
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


# --- 3b. CONSOLE RADAR GROUPS (predict.py) ---
# Coarser groups for console output. Assembled from ASSET_TYPES so that adding an
# asset does not require a second list that you have to remember to update.

def _merge_types(*keys: str) -> list:
    out = []
    for k in keys:
        out.extend(ASSET_TYPES[k])
    return out


RADAR_GROUPS = {
    "INDICES & MACRO": [a for a in ASSET_TYPES["INDICES & MACRO"] if a != "IMOEX"],
    "COMMODITIES":     ASSET_TYPES["COMMODITIES"],
    "US TECH":         ASSET_TYPES["US TECH"],
    "US HEALTHCARE":   ASSET_TYPES["US HEALTHCARE"],
    "US FINANCE":      ASSET_TYPES["US FINANCE"],
    "US CONSUMER":     ASSET_TYPES["US CONSUMER"],
    "US INDUSTRIAL":   ASSET_TYPES["US INDUSTRIAL"],
    "US SEMI":         ASSET_TYPES["US SEMI"],
    "US SOFTWARE":     ASSET_TYPES["US SOFTWARE"],
    "CRYPTO":          ASSET_TYPES["CRYPTO"],
    # show IMOEX together with the Russian names
    "MOEX":            ["IMOEX"] + _merge_types(
        "RUS BLUE CHIPS", "RUS FINANCE", "RUS TECH", "RUS METALS",
        "RUS INFRA", "RUS CONSUMER", "RUS PROPERTY"),
    "FOREX":           _merge_types("FOREX MAJORS", "FOREX CROSSES", "FOREX EXOTIC"),
}

# RADAR_GROUPS key -> webapp.py chip-cat-* CSS suffix (style.css). Anything not
# listed here (US sectors, indices, forex) falls back to "us" in radar_category().
_CATEGORY_CSS = {"CRYPTO": "crypto", "MOEX": "ru", "COMMODITIES": "commodity"}


def radar_category(asset: str) -> str:
    """Coarse category for the webapp's chip-cat accent color: crypto/ru/commodity/us."""
    for group, members in RADAR_GROUPS.items():
        if asset in members:
            return _CATEGORY_CSS.get(group, "us")
    return "us"


# --- 4. ENVIRONMENT VALIDATION ---
# Importing config.py must not fail when secrets are unset: most scripts
# (data_engine, train_hybrid, predict, backtest) work without Telegram. Scripts
# that really need credentials call require_env(...) at startup and get a clear
# error instead of failing somewhere deep in network code.

class ConfigError(RuntimeError):
    """A required environment variable is not set."""


def require_env(*names: str) -> None:
    """Fail with a clear message if any of the variables is empty.

    Example: require_env("TELEGRAM_TOKEN", "TELEGRAM_USER_ID") at the start of alert_bot.py.
    """
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ConfigError(
            "Environment variables not set: " + ", ".join(missing) + ". "
            "Copy .env.example to .env and fill in the values."
        )


def validate_telegram_config() -> None:
    """Check that Telegram credentials are set. Call before sending alerts."""
    require_env("TELEGRAM_TOKEN", "TELEGRAM_USER_ID")
