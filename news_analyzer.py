# news_analyzer.py
# News Analyzer -- fetches financial news from authoritative outlets,
# extracts article summaries, and performs weighted sentiment analysis.
# Sources: Bloomberg, Reuters, CNBC, MarketWatch, WSJ, FT, RBC, etc.
# No external NLP libraries required.

import os
import sys
import time
import re
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote_plus
from html import unescape

import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import FULL_ASSET_MAP, SOCKS5_PROXY, ASSET_TYPES

# ---------------------------------------------------------------------------
# Search-friendly names for each asset key
# ---------------------------------------------------------------------------
SEARCH_NAMES = {
    # Indices
    "VIX": "VIX volatility index",
    "DXY": "US Dollar Index DXY",
    "TNX": "US 10-year Treasury yield",
    "SP500": "S&P 500",
    "NASDAQ": "Nasdaq",
    "DOW": "Dow Jones",
    # Commodities
    "GOLD": "Gold",
    "SILVER": "Silver",
    "OIL": "Crude Oil WTI",
    "GAS": "Natural Gas",
    # US Tech
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "AMD": "AMD",
    "PLTR": "Palantir",
    "COIN": "Coinbase",
    "MSTR": "MicroStrategy",
    # Crypto
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "Ripple XRP",
    "TON": "Toncoin TON",
    "DOGE": "Dogecoin",
    "BNB": "BNB Binance",
    # Russia -- blue chips
    "IMOEX": "IMOEX",
    "SBER": "Sberbank",
    "GAZP": "Gazprom",
    "LKOH": "Lukoil",
    "ROSN": "Rosneft",
    "NVTK": "Novatek",
    "TATN": "Tatneft",
    "SNGS": "Surgutneftegaz",
    "PLZL": "Polyus Gold",
    "SIBN": "Gazprom Neft",
    "MGNT": "Magnit",
    # Russia -- finance
    "TCSG": "Tinkoff",
    "VTBR": "VTB Bank",
    "BSPB": "Bank Saint Petersburg",
    "MOEX_EX": "Moscow Exchange MOEX",
    # Russia -- tech
    "YNDX": "Yandex",
    "OZON": "Ozon",
    "VKCO": "VK",
    "POSI": "Positive Technologies",
    "MTSS": "MTS",
    "RTKM": "Rostelecom",
    # Russia -- metals
    "CHMF": "Severstal",
    "NLMK": "NLMK",
    "MAGN": "MMK Magnitogorsk",
    "RUAL": "Rusal",
    "ALRS": "Alrosa",
    # Russia -- infra
    "IRAO": "Inter RAO",
    "HYDR": "RusHydro",
    "FLOT": "Sovcomflot",
    "AFLT": "Aeroflot",
    "PIKK": "PIK Group",
    # Forex
    "EURUSD": "EUR/USD Euro Dollar",
    "GBPUSD": "GBP/USD Pound Dollar",
    "USDJPY": "USD/JPY Dollar Yen",
    "USDCHF": "USD/CHF Dollar Franc",
    "AUDUSD": "AUD/USD Australian Dollar",
    "USDCAD": "USD/CAD Dollar Canadian",
    "NZDUSD": "NZD/USD New Zealand Dollar",
    "EURGBP": "EUR/GBP Euro Pound",
    "EURJPY": "EUR/JPY Euro Yen",
    "GBPJPY": "GBP/JPY Pound Yen",
    "USDRUB": "USD/RUB Dollar Ruble",
}

# Russian search names (for Russian-language RSS)
SEARCH_NAMES_RU = {
    "SBER": "Сбербанк", "GAZP": "Газпром", "LKOH": "Лукойл",
    "ROSN": "Роснефть", "NVTK": "Новатэк", "TATN": "Татнефть",
    "SNGS": "Сургутнефтегаз", "PLZL": "Полюс", "SIBN": "Газпром нефть",
    "MGNT": "Магнит", "TCSG": "Тинькофф", "VTBR": "ВТБ",
    "BSPB": "Банк Санкт-Петербург", "MOEX_EX": "Мосбиржа MOEX",
    "YNDX": "Яндекс", "OZON": "Озон", "VKCO": "ВКонтакте VK",
    "POSI": "Позитив", "MTSS": "МТС", "RTKM": "Ростелеком",
    "CHMF": "Северсталь", "NLMK": "НЛМК", "MAGN": "ММК",
    "RUAL": "Русал", "ALRS": "Алроса", "IRAO": "Интер РАО",
    "HYDR": "РусГидро", "FLOT": "Совкомфлот", "AFLT": "Аэрофлот",
    "PIKK": "ПИК", "IMOEX": "Индекс Мосбиржи",
    "GOLD": "Золото", "SILVER": "Серебро", "OIL": "Нефть",
    "GAS": "Газ природный", "BTC": "Биткоин", "ETH": "Эфириум",
    "USDRUB": "Доллар рубль курс",
}

RU_ASSETS = set(
    ASSET_TYPES.get("RUS BLUE CHIPS", [])
    + ASSET_TYPES.get("RUS FINANCE", [])
    + ASSET_TYPES.get("RUS TECH", [])
    + ASSET_TYPES.get("RUS METALS", [])
    + ASSET_TYPES.get("RUS INFRA", [])
    + ["IMOEX"]
)

# ---------------------------------------------------------------------------
# Sentiment word lists
# ---------------------------------------------------------------------------
POSITIVE_WORDS_EN = {
    "rally", "surge", "gain", "profit", "growth", "bullish", "up", "rise",
    "beat", "strong", "record", "high", "outperform", "buy", "upgrade",
    "breakout", "soar", "boom", "recover", "rebound", "advance", "jumps",
    "positive", "optimism", "confidence", "momentum", "acceleration",
    "dividend", "acquisition", "expansion", "innovation",
}
NEGATIVE_WORDS_EN = {
    "crash", "fall", "drop", "loss", "bear", "decline", "risk", "crisis",
    "selloff", "weak", "recession", "downgrade", "sell", "plunge", "slump",
    "warning", "fear", "dump", "bankruptcy", "fraud", "default", "layoff",
    "inflation", "tariff", "sanctions", "investigation", "lawsuit",
    "tumble", "collapse", "concern", "uncertainty", "volatility",
}
POSITIVE_WORDS_RU = {
    "рост", "прибыль", "рекорд", "повышение", "дивиденды", "покупка",
    "бычий", "подъем", "восстановление", "максимум", "оптимизм",
    "выручка", "спрос", "ралли", "увеличение",
}
NEGATIVE_WORDS_RU = {
    "падение", "обвал", "кризис", "риск", "убыток", "снижение", "продажа",
    "медвежий", "санкции", "дефолт", "банкротство", "минимум", "инфляция",
    "девальвация", "отток", "рецессия", "сокращение", "тариф",
}

ALL_POSITIVE = POSITIVE_WORDS_EN | POSITIVE_WORDS_RU
ALL_NEGATIVE = NEGATIVE_WORDS_EN | NEGATIVE_WORDS_RU

# ---------------------------------------------------------------------------
# RSS URLs (Google News)
# ---------------------------------------------------------------------------
RSS_EN = "https://news.google.com/rss/search?q={query}+stock+market&hl=en-US&gl=US&ceid=US:en"
RSS_RU = "https://news.google.com/rss/search?q={query}+%D0%B0%D0%BA%D1%86%D0%B8%D0%B8&hl=ru&gl=RU&ceid=RU:ru"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 12

# ---------------------------------------------------------------------------
# Authoritative financial news sources -- direct RSS feeds
# ---------------------------------------------------------------------------
# (name, rss_url, language, credibility_weight, category)
# credibility: 1.5 = top-tier, 1.3 = solid, 1.0 = standard, 0.8 = blog
AUTHORITY_FEEDS = [
    # ═══ TIER 1: Premier global financial press ═══
    ("Bloomberg",       "https://feeds.bloomberg.com/markets/news.rss",             "en", 1.5, "markets"),
    ("Bloomberg Biz",   "https://feeds.bloomberg.com/bview/news.rss",               "en", 1.5, "opinion"),
    ("Reuters Biz",     "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", "en", 1.5, "markets"),
    ("Reuters World",   "https://www.reutersagency.com/feed/?taxonomy=best-regions&post_type=best", "en", 1.5, "macro"),
    ("CNBC Top",        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "en", 1.5, "markets"),
    ("CNBC World",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362", "en", 1.5, "macro"),
    ("WSJ Markets",     "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",           "en", 1.5, "markets"),
    ("WSJ World",       "https://feeds.a.dj.com/rss/RSSWorldNews.xml",             "en", 1.5, "macro"),
    ("Financial Times", "https://www.ft.com/rss/home",                              "en", 1.5, "markets"),
    ("Barron's",        "https://feeds.barrons.com/marketcurrents/newshighlights",  "en", 1.5, "analysis"),
    ("AP Business",     "https://rsshub.app/apnews/topics/business",                "en", 1.4, "macro"),

    # ═══ TIER 2: Major financial media ═══
    ("MarketWatch",     "https://feeds.marketwatch.com/marketwatch/topstories/",    "en", 1.4, "markets"),
    ("MarketWatch Bull","https://feeds.marketwatch.com/marketwatch/bulletins/",     "en", 1.4, "breaking"),
    ("Yahoo Finance",   "https://finance.yahoo.com/news/rssindex",                  "en", 1.3, "markets"),
    ("Investing.com",   "https://www.investing.com/rss/news.rss",                   "en", 1.3, "markets"),
    ("Forbes",          "https://www.forbes.com/innovation/feed2",                   "en", 1.2, "tech"),
    ("Business Insider","https://markets.businessinsider.com/rss/news",             "en", 1.2, "markets"),

    # ═══ TIER 2: Crypto & sector specialists ═══
    ("CoinDesk",        "https://www.coindesk.com/arc/outboundfeeds/rss/",          "en", 1.3, "crypto"),
    ("CoinTelegraph",   "https://cointelegraph.com/rss",                            "en", 1.2, "crypto"),
    ("The Block",       "https://www.theblock.co/rss.xml",                          "en", 1.3, "crypto"),
    ("Decrypt",         "https://decrypt.co/feed",                                   "en", 1.1, "crypto"),
    ("Seeking Alpha",   "https://seekingalpha.com/market_currents.xml",             "en", 1.2, "analysis"),
    ("Benzinga",        "https://www.benzinga.com/feed",                             "en", 1.0, "markets"),
    ("Motley Fool",     "https://www.fool.com/feeds/index.aspx",                    "en", 1.0, "analysis"),
    ("Zacks",           "https://www.zacks.com/feeds/",                              "en", 1.0, "analysis"),

    # ═══ Russian authoritative sources ═══
    ("RBC",             "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",        "ru", 1.5, "markets"),
    ("RBC Investments", "https://rssexport.rbc.ru/rbcnews/news/97/full.rss",        "ru", 1.5, "markets"),
    ("Interfax",        "https://www.interfax.ru/rss.asp",                          "ru", 1.5, "macro"),
    ("Kommersant",      "https://www.kommersant.ru/RSS/section-economics.xml",      "ru", 1.4, "macro"),
    ("Kommersant Fin",  "https://www.kommersant.ru/RSS/section-finance.xml",        "ru", 1.4, "markets"),
    ("TASS Economy",    "https://tass.com/rss/v2.xml",                              "ru", 1.4, "macro"),
    ("Vedomosti",       "https://www.vedomosti.ru/rss/articles",                    "ru", 1.4, "macro"),
    ("Vedomosti Fin",   "https://www.vedomosti.ru/rss/rubric/finance",              "ru", 1.4, "markets"),
    ("Finam",           "https://www.finam.ru/analysis/conews/rsspoint/",            "ru", 1.2, "markets"),
    ("BCS Express",     "https://bcs-express.ru/rss",                               "ru", 1.2, "analysis"),
    ("Smart-Lab",       "https://smart-lab.ru/rss/",                                "ru", 1.0, "community"),
    ("Banki.ru",        "https://www.banki.ru/xml/news.rss",                        "ru", 1.1, "finance"),
    ("Lenta Economy",   "https://lenta.ru/rss/articles/economics/",                 "ru", 1.1, "macro"),
    ("Prime TASS",      "https://1prime.ru/rss/main.rss",                           "ru", 1.3, "markets"),
]

# Source name -> credibility weight (quick lookup)
SOURCE_WEIGHTS = {name: weight for name, _, _, weight, _ in AUTHORITY_FEEDS}
SOURCE_WEIGHTS["Google News"] = 1.0

# Known source patterns in Google News titles ("Headline - Source")
_SOURCE_PATTERN = re.compile(r'\s*[-\u2013\u2014]\s*([^-\u2013\u2014]+?)\s*$')

# Meta description pattern for web page summary extraction
_META_DESC = re.compile(
    r'<meta\s+(?:name|property)\s*=\s*["\'](?:description|og:description)["\']'
    r'\s+content\s*=\s*["\']([^"\']{10,500})["\']',
    re.IGNORECASE
)
_META_DESC2 = re.compile(
    r'<meta\s+content\s*=\s*["\']([^"\']{10,500})["\']'
    r'\s+(?:name|property)\s*=\s*["\'](?:description|og:description)["\']',
    re.IGNORECASE
)


def _extract_source(title: str) -> tuple[str, str]:
    """Extract source from Google News title ('Headline - Source')."""
    m = _SOURCE_PATTERN.search(title)
    if m:
        source = m.group(1).strip()
        clean = title[:m.start()].strip()
        return clean, source
    return title, "Unknown"


def _get_source_weight(source_name: str) -> float:
    """Get credibility weight for a source name."""
    source_lower = source_name.lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name.lower() in source_lower or source_lower in name.lower():
            return weight
    _known = {
        "reuters": 1.5, "bloomberg": 1.5, "cnbc": 1.5, "wsj": 1.5,
        "wall street journal": 1.5, "financial times": 1.5, "ft.com": 1.5,
        "barron": 1.5, "economist": 1.4, "forbes": 1.2, "ap news": 1.4,
        "rbc": 1.5, "interfax": 1.5, "kommersant": 1.4, "vedomosti": 1.4,
        "tass": 1.4, "ria": 1.2, "coindesk": 1.3, "marketwatch": 1.4,
        "yahoo finance": 1.3, "business insider": 1.2, "motley fool": 1.0,
        "seeking alpha": 1.2, "the block": 1.3,
    }
    for key, w in _known.items():
        if key in source_lower:
            return w
    return 0.8


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: dict = {}
CACHE_TTL = 1800  # 30 min


def _cache_get(key: str):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _cache[key]
    return None


def _cache_set(key: str, data):
    _cache[key] = (time.time(), data)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _make_session(use_proxy: bool = True) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    if use_proxy and SOCKS5_PROXY:
        s.proxies = {"http": SOCKS5_PROXY, "https": SOCKS5_PROXY}
    return s


def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT) -> str | None:
    """Fetch URL text. Proxy first, then direct."""
    for use_proxy in (True, False):
        try:
            s = _make_session(use_proxy)
            resp = s.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception:
            continue
    return None


def _fetch_article_summary(url: str) -> str:
    """Fetch article page and extract meta description as summary."""
    if not url or "google.com" in url:
        return ""
    cache_key = f"summary_{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        html = _fetch_url(url, timeout=8)
        if not html:
            return ""
        # Only look at first 10KB for speed
        head = html[:10000]
        m = _META_DESC.search(head) or _META_DESC2.search(head)
        if m:
            desc = _clean_html(m.group(1))
            _cache_set(cache_key, desc)
            return desc
    except Exception:
        pass

    _cache_set(cache_key, "")
    return ""


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------
# Namespace map for Atom/RSS extensions
_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "media": "http://search.yahoo.com/mrss/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _parse_rss_items(xml_text: str, source_name: str = "",
                     fetch_summaries: bool = False) -> list[dict]:
    """Parse RSS/Atom XML. Extract title, link, date, description.

    If fetch_summaries=True and description is empty, fetches the article
    page to extract meta description (slower but gives real summaries).
    """
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0: <channel><item>
    channel = root.find("channel")
    entries = channel.findall("item") if channel is not None else []
    # Atom: <entry>
    if not entries:
        ns_atom = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{ns_atom}entry") or root.findall("entry")

    for entry in entries:
        # --- title ---
        title_el = entry.find("title")
        if title_el is None or not title_el.text:
            continue
        raw_title = title_el.text.strip()

        # --- link ---
        link_el = entry.find("link")
        if link_el is not None:
            link = link_el.text.strip() if link_el.text else link_el.get("href", "")
        else:
            link = ""

        # --- published ---
        pub_el = (entry.find("pubDate")
                  or entry.find("published")
                  or entry.find(f"{{{_NS['dc']}}}date"))
        published = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

        # --- description / summary ---
        desc = ""
        # Try <description>, <summary>, <content:encoded>, <media:description>
        for tag in ("description", "summary",
                    f"{{{_NS['content']}}}encoded",
                    f"{{{_NS['media']}}}description"):
            el = entry.find(tag)
            if el is not None and el.text:
                desc = _clean_html(el.text)
                break

        # Limit description length
        if len(desc) > 300:
            # Cut at sentence boundary
            cut = desc[:300].rfind(". ")
            if cut > 100:
                desc = desc[:cut + 1]
            else:
                desc = desc[:297] + "..."

        # --- source ---
        if source_name:
            src = source_name
            clean_title = raw_title
        else:
            clean_title, src = _extract_source(raw_title)

        # Fetch summary from article page if needed
        if fetch_summaries and len(desc) < 20 and link:
            page_desc = _fetch_article_summary(link)
            if page_desc:
                desc = page_desc
                if len(desc) > 300:
                    cut = desc[:300].rfind(". ")
                    desc = desc[:cut + 1] if cut > 100 else desc[:297] + "..."

        items.append({
            "title": clean_title,
            "link": link,
            "published": published,
            "source": src,
            "description": desc,
        })

    return items


# ---------------------------------------------------------------------------
# Authority feed fetcher
# ---------------------------------------------------------------------------
def _fetch_authority_feed(name: str, url: str, lang: str,
                          fetch_summaries: bool = False) -> list[dict]:
    cache_key = f"auth_{name}_{fetch_summaries}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    xml_text = _fetch_url(url)
    if not xml_text:
        return []

    items = _parse_rss_items(xml_text, source_name=name,
                             fetch_summaries=fetch_summaries)
    _cache_set(cache_key, items)
    return items


def fetch_authority_digest(max_per_source: int = 5,
                           lang_filter: str = "all",
                           category_filter: str = "all",
                           fetch_summaries: bool = True) -> list[dict]:
    """Fetch headlines + summaries from all authoritative sources."""
    cache_key = f"digest_{lang_filter}_{category_filter}_{max_per_source}_{fetch_summaries}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    all_items = []
    total = len(AUTHORITY_FEEDS)
    for idx, (name, url, lang, weight, cat) in enumerate(AUTHORITY_FEEDS):
        if lang_filter != "all" and lang != lang_filter:
            continue
        if category_filter != "all" and cat != category_filter:
            continue
        try:
            print(f"\r  [{idx+1}/{total}] {name}...", end="", flush=True)
            items = _fetch_authority_feed(name, url, lang,
                                          fetch_summaries=fetch_summaries)
            for item in items[:max_per_source]:
                score = _score_title(item["title"])
                # Also score description for better accuracy
                desc_score = _score_title(item.get("description", "")) * 0.3
                combined = score + desc_score
                weighted = combined * weight
                all_items.append({
                    "title": item["title"],
                    "link": item["link"],
                    "published": item["published"],
                    "source": name,
                    "description": item.get("description", ""),
                    "category": cat,
                    "credibility": weight,
                    "sentiment_score": round(combined, 3),
                    "weighted_score": round(weighted, 3),
                    "sentiment_label": _sentiment_label(weighted),
                })
        except Exception:
            continue

    print("\r" + " " * 60 + "\r", end="", flush=True)

    all_items.sort(key=lambda x: abs(x["weighted_score"]), reverse=True)
    _cache_set(cache_key, all_items)
    return all_items


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
def _score_title(title: str) -> float:
    """Score text [-1..+1] by keyword matching."""
    if not title:
        return 0.0
    words = title.lower().split()
    pos = neg = 0
    for w in words:
        cleaned = w.strip(".,!?;:\"'()-[]{}#")
        if cleaned in ALL_POSITIVE:
            pos += 1
        if cleaned in ALL_NEGATIVE:
            neg += 1
    score = (pos - neg) / (pos + neg + 1)
    return max(-1.0, min(1.0, score))


def _sentiment_label(score: float) -> str:
    if score > 0.3:
        return "VERY_BULLISH"
    elif score > 0.1:
        return "BULLISH"
    elif score > -0.1:
        return "NEUTRAL"
    elif score > -0.3:
        return "BEARISH"
    else:
        return "VERY_BEARISH"


def _short_label(label: str) -> str:
    return {"VERY_BULLISH": "BULL++", "BULLISH": "BULL",
            "NEUTRAL": "NEUTRAL", "BEARISH": "BEAR",
            "VERY_BEARISH": "BEAR--"}.get(label, label)


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------
def fetch_news(asset: str, max_articles: int = 10,
               fetch_summaries: bool = False) -> list[dict]:
    """Fetch news for asset from Google News + authority feeds."""
    asset = asset.upper()
    cache_key = f"news_{asset}_{max_articles}_{fetch_summaries}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    search_name_en = SEARCH_NAMES.get(asset, asset)
    query_en = quote_plus(search_name_en)
    url_en = RSS_EN.format(query=query_en)

    all_items: list[dict] = []

    # Google News EN
    xml_en = _fetch_url(url_en)
    if xml_en:
        all_items.extend(_parse_rss_items(xml_en,
                                          fetch_summaries=fetch_summaries))

    # Google News RU
    if asset in RU_ASSETS or asset in SEARCH_NAMES_RU:
        search_name_ru = SEARCH_NAMES_RU.get(asset, search_name_en)
        query_ru = quote_plus(search_name_ru)
        url_ru = RSS_RU.format(query=query_ru)
        xml_ru = _fetch_url(url_ru)
        if xml_ru:
            all_items.extend(_parse_rss_items(xml_ru,
                                              fetch_summaries=fetch_summaries))

    # Dedup by title
    seen: set = set()
    unique: list[dict] = []
    for item in all_items:
        key = item["title"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Score with credibility
    results = []
    for item in unique[:max_articles * 2]:
        source = item.get("source", "Unknown")
        weight = _get_source_weight(source)
        score = _score_title(item["title"])
        desc_score = _score_title(item.get("description", "")) * 0.3
        combined = score + desc_score
        weighted = combined * weight
        results.append({
            "title": item["title"],
            "link": item["link"],
            "published": item["published"],
            "source": source,
            "description": item.get("description", ""),
            "credibility": weight,
            "sentiment_score": round(combined, 3),
            "weighted_score": round(weighted, 3),
            "sentiment_label": _sentiment_label(weighted),
        })

    results.sort(key=lambda x: abs(x["weighted_score"]), reverse=True)
    results = results[:max_articles]
    _cache_set(cache_key, results)
    return results


def analyze_asset(asset: str, fetch_summaries: bool = False) -> dict:
    """Analyze sentiment for one asset with source breakdown."""
    asset = asset.upper()
    cache_key = f"analysis_{asset}_{fetch_summaries}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    articles = fetch_news(asset, max_articles=15,
                          fetch_summaries=fetch_summaries)
    if not articles:
        result = {
            "asset": asset, "articles_count": 0,
            "avg_sentiment": 0.0, "weighted_sentiment": 0.0,
            "sentiment_label": "NEUTRAL",
            "top_positive": None, "top_negative": None,
            "top_pos_source": None, "top_neg_source": None,
            "sources_breakdown": {},
        }
        _cache_set(cache_key, result)
        return result

    w_scores = [a["weighted_score"] for a in articles]
    w_avg = sum(w_scores) / len(w_scores)

    positives = [a for a in articles if a["weighted_score"] > 0]
    negatives = [a for a in articles if a["weighted_score"] < 0]
    top_pos = max(positives, key=lambda x: x["weighted_score"]) if positives else None
    top_neg = min(negatives, key=lambda x: x["weighted_score"]) if negatives else None

    src_counts = {}
    for a in articles:
        s = a.get("source", "Unknown")
        src_counts[s] = src_counts.get(s, 0) + 1

    result = {
        "asset": asset,
        "articles_count": len(articles),
        "avg_sentiment": round(sum(a["sentiment_score"] for a in articles) / len(articles), 3),
        "weighted_sentiment": round(w_avg, 3),
        "sentiment_label": _sentiment_label(w_avg),
        "top_positive": top_pos["title"] if top_pos else None,
        "top_negative": top_neg["title"] if top_neg else None,
        "top_pos_source": top_pos["source"] if top_pos else None,
        "top_neg_source": top_neg["source"] if top_neg else None,
        "sources_breakdown": src_counts,
    }
    _cache_set(cache_key, result)
    return result


def scan_all(top_n: int = 10) -> list[dict]:
    """Scan major assets, ranked by sentiment."""
    priority = []
    for group in ["TOP SIGNALS", "CRYPTO", "US TECH", "COMMODITIES",
                   "INDICES & MACRO", "RUS BLUE CHIPS"]:
        for a in ASSET_TYPES.get(group, []):
            if a not in priority:
                priority.append(a)
    for a in FULL_ASSET_MAP:
        if a not in priority:
            priority.append(a)

    targets = priority[:top_n]
    results = []
    for i, asset in enumerate(targets):
        try:
            print(f"\r  Scanning [{i+1}/{len(targets)}] {asset}...", end="", flush=True)
            results.append(analyze_asset(asset))
        except Exception:
            continue
    print("\r" + " " * 50 + "\r", end="", flush=True)

    results.sort(key=lambda x: abs(x.get("weighted_sentiment", 0)), reverse=True)
    return results


def get_market_mood() -> dict:
    """Overall market mood from macro assets."""
    cache_key = "market_mood"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    macro = ["SP500", "NASDAQ", "DOW", "VIX", "DXY", "BTC", "GOLD", "OIL"]
    details, scores = [], []
    for asset in macro:
        try:
            a = analyze_asset(asset)
            details.append(a)
            if a["articles_count"] > 0:
                s = a["weighted_sentiment"]
                if asset == "VIX":
                    s = -s
                scores.append(s)
        except Exception:
            continue

    mood_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    result = {"mood_score": mood_score, "mood_label": _sentiment_label(mood_score),
              "details": details}
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------
_SEP = "=" * 72
_SEP2 = "-" * 72


def _tier_tag(w: float) -> str:
    if w >= 1.5: return "[TOP]"
    if w >= 1.3: return "[ + ]"
    if w >= 1.0: return "[ . ]"
    return "[ - ]"


def _print_header():
    print()
    print(_SEP)
    print("  NEWS ANALYZER -- Authoritative Financial Sources")
    print("  Bloomberg | Reuters | CNBC | WSJ | FT | RBC | Interfax | ...")
    print(_SEP)
    print()


def _print_mood(mood: dict):
    sign = "+" if mood["mood_score"] >= 0 else ""
    print(f"  [MARKET MOOD]  {mood['mood_label']} ({sign}{mood['mood_score']:.2f})")
    print()


def _print_scan(results: list[dict]):
    print("  [TOP MOVERS BY SENTIMENT]")
    print(f"  {'Asset':<10}{'Tone':<12}{'N':<5}{'Source':<16}Headline")
    print(f"  {'-' * 65}")
    for r in results:
        label = _short_label(r["sentiment_label"])
        headline = r.get("top_positive") or r.get("top_negative") or "--"
        if len(headline) > 42:
            headline = headline[:39] + "..."
        src = r.get("top_pos_source") or r.get("top_neg_source") or ""
        print(f"  {r['asset']:<10}{label:<12}{r['articles_count']:<5}{src:<16}\"{headline}\"")
    print()


def _print_asset_detail(asset: str):
    analysis = analyze_asset(asset, fetch_summaries=True)
    articles = fetch_news(asset, max_articles=15, fetch_summaries=True)

    sign = "+" if analysis["weighted_sentiment"] >= 0 else ""
    print(f"  [ASSET: {analysis['asset']}]  "
          f"{analysis['sentiment_label']} ({sign}{analysis['weighted_sentiment']:.2f})  "
          f"{analysis['articles_count']} articles")

    if analysis.get("sources_breakdown"):
        srcs = ", ".join(f"{k}({v})" for k, v in
                         sorted(analysis["sources_breakdown"].items(),
                                key=lambda x: x[1], reverse=True)[:6])
        print(f"  Sources: {srcs}")
    print()

    for a in articles:
        ws = a["weighted_score"]
        src = a.get("source", "?")
        tag = _tier_tag(a.get("credibility", 1.0))
        title = a["title"]
        if len(title) > 60:
            title = title[:57] + "..."
        print(f"  {tag} {ws:+.2f}  [{src}]")
        print(f"         {title}")

        desc = a.get("description", "")
        if desc and len(desc) > 15:
            # Word-wrap description at ~75 chars
            lines = []
            while len(desc) > 75:
                cut = desc[:75].rfind(" ")
                if cut < 30:
                    cut = 75
                lines.append(desc[:cut])
                desc = desc[cut:].lstrip()
            if desc:
                lines.append(desc)
            for line in lines:
                print(f"         > {line}")
        print()
    print()


def _print_digest(digest: list[dict], lang: str, cat: str):
    """Full digest from authoritative sources with descriptions."""
    lang_name = {"en": "International", "ru": "Russian", "all": "All"}
    cat_name = cat.upper() if cat != "all" else "All Categories"
    print(f"  [DIGEST] {lang_name.get(lang, lang)} | {cat_name}")
    print(f"  {len(digest)} articles from authoritative financial press")
    print(_SEP2)

    # Group by source, ordered by credibility
    by_source: dict[str, list] = {}
    for item in digest:
        src = item["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(item)

    # Sort sources by max credibility
    sorted_sources = sorted(by_source.items(),
                            key=lambda x: x[1][0].get("credibility", 0),
                            reverse=True)

    for source, items in sorted_sources:
        weight = items[0].get("credibility", 1.0)
        tag = _tier_tag(weight)
        avg_s = sum(i["weighted_score"] for i in items) / len(items)
        label = _sentiment_label(avg_s)
        cat_s = items[0].get("category", "")

        print(f"\n  {tag} {source}  [{cat_s}]  "
              f"({len(items)} articles, tone: {label} {avg_s:+.2f})")
        print(f"  {'~' * 55}")

        for item in items[:5]:
            title = item["title"]
            if len(title) > 65:
                title = title[:62] + "..."
            ws = item["weighted_score"]
            print(f"    {ws:+.2f}  {title}")

            desc = item.get("description", "")
            if desc and len(desc) > 15:
                if len(desc) > 120:
                    cut = desc[:120].rfind(". ")
                    if cut > 50:
                        desc = desc[:cut + 1]
                    else:
                        desc = desc[:117] + "..."
                print(f"          > {desc}")

    # Summary
    print()
    print(_SEP2)
    if digest:
        avg_all = sum(i["weighted_score"] for i in digest) / len(digest)
        t1 = [i for i in digest if i.get("credibility", 1.0) >= 1.4]
        t1_avg = sum(i["weighted_score"] for i in t1) / len(t1) if t1 else 0

        print(f"  OVERALL TONE:   {_sentiment_label(avg_all)} ({avg_all:+.2f}, "
              f"{len(digest)} articles)")
        if t1:
            print(f"  TOP-TIER TONE:  {_sentiment_label(t1_avg)} ({t1_avg:+.2f}, "
                  f"{len(t1)} articles from Bloomberg/Reuters/WSJ/CNBC/FT/RBC/Interfax/...)")

        # Category breakdown
        cats = {}
        for i in digest:
            c = i.get("category", "other")
            if c not in cats:
                cats[c] = []
            cats[c].append(i["weighted_score"])
        if len(cats) > 1:
            print(f"\n  By category:")
            for c, scores in sorted(cats.items()):
                c_avg = sum(scores) / len(scores)
                print(f"    {c:<12} {_sentiment_label(c_avg)} ({c_avg:+.2f}, {len(scores)} articles)")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="News Analyzer -- Authoritative Financial Sources")
    parser.add_argument("--asset", type=str, default=None,
                        help="Analyze specific asset (BTC, NVDA, SBER...)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all major assets")
    parser.add_argument("--mood", action="store_true",
                        help="Market mood only")
    parser.add_argument("--digest", action="store_true",
                        help="Full digest from authoritative press")
    parser.add_argument("--lang", default="all", choices=["en", "ru", "all"],
                        help="Language filter (default: all)")
    parser.add_argument("--category", default="all",
                        choices=["all", "markets", "macro", "crypto",
                                 "analysis", "opinion", "tech", "breaking",
                                 "finance", "community"],
                        help="Category filter for digest")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of assets to scan (default 10)")
    parser.add_argument("--summaries", action="store_true",
                        help="Fetch full article summaries (slower)")
    args = parser.parse_args()

    _print_header()

    if args.digest:
        print("  Fetching from authoritative financial press...")
        digest = fetch_authority_digest(
            max_per_source=5, lang_filter=args.lang,
            category_filter=args.category,
            fetch_summaries=args.summaries)
        _print_digest(digest, args.lang, args.category)
        return

    if args.asset:
        asset = args.asset.upper()
        if asset not in FULL_ASSET_MAP and asset not in SEARCH_NAMES:
            print(f"  Unknown asset: {asset}")
            print(f"  Available: {', '.join(sorted(FULL_ASSET_MAP.keys()))}")
            return
        _print_asset_detail(asset)
        return

    if args.mood:
        mood = get_market_mood()
        _print_mood(mood)
        return

    if args.scan:
        print("  Scanning all assets...")
        mood = get_market_mood()
        _print_mood(mood)
        results = scan_all(top_n=len(FULL_ASSET_MAP))
        _print_scan(results)
        return

    # Default: mood + top movers + brief digest
    mood = get_market_mood()
    _print_mood(mood)

    results = scan_all(top_n=args.top)
    _print_scan(results)

    # Top-3 detail
    for r in results[:3]:
        _print_asset_detail(r["asset"])

    # Brief top-tier digest
    print(_SEP2)
    print("  [BRIEF DIGEST: Top-Tier Sources]")
    print()
    digest = fetch_authority_digest(max_per_source=3, lang_filter="all",
                                    fetch_summaries=False)
    tier1 = [d for d in digest if d.get("credibility", 1.0) >= 1.4][:12]
    if tier1:
        for item in tier1:
            title = item["title"]
            if len(title) > 58:
                title = title[:55] + "..."
            print(f"  {_tier_tag(item['credibility'])} [{item['source']:<14}] "
                  f"{item['weighted_score']:+.2f}  {title}")
            desc = item.get("description", "")
            if desc and len(desc) > 15:
                if len(desc) > 90:
                    desc = desc[:87] + "..."
                print(f"         > {desc}")
    else:
        print("  (no data from top-tier sources)")
    print()


if __name__ == "__main__":
    main()
