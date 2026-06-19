"""Telegram bot: hourly signal scan, commands, digest, degradation alerts.

The scan computes signals with the same pipeline as predict.py. Commands and
the digest read the log via core.track_record.
"""

import json
import os
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from catboost import CatBoostClassifier
from telebot import TeleBot, apihelper

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

from core.logger import get_logger
from net import ssl_verify
logger = get_logger("alert_bot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# -- Config --
try:
    from config import FULL_ASSET_MAP, TELEGRAM_TOKEN, TELEGRAM_USER_ID
    try:
        from config import SOCKS5_PROXY
    except ImportError:
        SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "")
except ImportError:
    logger.critical("config.py not found")
    sys.exit(1)

# -- Shared ML components (core package) --
from core.features import engineer_features, add_weekly_features, add_crossasset_features
from core.ensemble import ensemble_with_gating
from core.scaling import load_or_fit_scaler
from core.calibration import load_calibrator, apply_calibrator
from core.model_io import get_lookback as _get_lookback, load_lstm_model as _load_lstm_model
from core import reports, track_record
from risk_manager import RISK_CONFIG

from sqlalchemy import create_engine

CHECK_INTERVAL = 3600
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "market.db")
db_engine = create_engine(f"sqlite:///{DB_PATH}")

REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")

MOEX_TARGETS = [
    "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS",
    "PLZL", "SIBN", "MGNT",
    "TCSG", "VTBR", "BSPB", "MOEX_EX",
    "YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM",
    "CHMF", "NLMK", "MAGN", "RUAL", "ALRS",
    "IRAO", "HYDR", "FLOT", "AFLT", "PIKK",
]

# -- Risk & Portfolio --
try:
    from risk_manager import RiskManager
    _rm = RiskManager()
    logger.info("RiskManager loaded")
except Exception as exc:
    logger.warning("RiskManager unavailable: %s", exc)
    _rm = None

try:
    from portfolio import PortfolioManager
    _pm = PortfolioManager(FULL_ASSET_MAP)
    logger.info("PortfolioManager loaded")
except Exception as exc:
    logger.warning("PortfolioManager unavailable: %s", exc)
    _pm = None


def _load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ==============================================================================
# NETWORK HELPERS
# ==============================================================================

def _test_url(url, proxies=None, timeout=5):
    try:
        r = requests.get(url, proxies=proxies or {}, timeout=timeout, verify=ssl_verify())
        return r.status_code == 200
    except Exception:
        return False


def detect_proxy_needed():
    test_url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d"
    if _test_url(test_url):
        logger.info("Direct internet access OK")
        return False
    logger.info("Direct access failed - using proxy")
    return True


def detect_telegram_proxy():
    if _test_url("https://api.telegram.org"):
        logger.info("Direct Telegram access OK")
        return False
    logger.info("Telegram blocked - using proxy")
    return True


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def fetch_moex(symbol, name):
    clean = symbol.split(".")[0]
    start = (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")
    if name == "IMOEX":
        url = (f"https://iss.moex.com/iss/engines/stock/markets/index/"
               f"boards/SNDX/securities/IMOEX/candles.json?interval=24&from={start}")
    else:
        url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
               f"boards/TQBR/securities/{clean}/candles.json?interval=24&from={start}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()["candles"]["data"]
        cols = r.json()["candles"]["columns"]
        if not data:
            return None
        df = pd.DataFrame(data, columns=cols).rename(columns={
            "close": "Close", "high": "High", "low": "Low",
            "volume": "Volume", "open": "Open",
        })
        return df
    except Exception as exc:
        logger.warning("MOEX fetch error %s: %s", name, exc)
        return None


def fetch_world(symbol, use_proxy):
    proxies = {"https": SOCKS5_PROXY} if use_proxy else {}
    try:
        if use_proxy:
            import urllib3
            urllib3.disable_warnings()
            sess = requests.Session()
            sess.proxies = proxies
            sess.verify = False
            df = yf.download(symbol, period="100d", interval="1d",
                             progress=False, session=sess)
        else:
            df = yf.download(symbol, period="100d", interval="1d", progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("Yahoo fetch error %s: %s", symbol, exc)
        return None


# ==============================================================================
# ASSET ANALYSIS
# ==============================================================================

def analyze_asset(df, name, registry, thresholds):
    """Run models on df and return a signal dict or None."""
    try:
        # Engineer features (same as training)
        df = engineer_features(df)
        table = name.lower().replace("^", "").replace(".", "").replace("-", "")
        df = add_weekly_features(df, table, db_engine)
        df = add_crossasset_features(df, table, db_engine)

        if len(df) < 50:
            return None

        current_price = float(df['close'].iloc[-1])
        taleb_val = float(df['taleb_risk'].iloc[-1]) if 'taleb_risk' in df.columns else 0.0

        cb_path = os.path.join(MODEL_DIR, f"{table}_cb.cbm")
        lstm_path = os.path.join(MODEL_DIR, f"{table}_lstm.keras")
        if not os.path.exists(cb_path):
            return None

        # Get features from registry (same as training used)
        reg_entry = registry.get(name)
        if reg_entry and 'features' in reg_entry:
            features = [f for f in reg_entry['features'] if f in df.columns]
        else:
            features = ["close", "volume", "vol_z", "taleb_risk", "ret_1",
                        "ret_5", "trend_strength", "rsi", "sma_20", "sma_50"]
            features = [f for f in features if f in df.columns]

        if len(features) < 3:
            return None

        # Scale using the saved train-fold scaler (train/serve parity); fall back
        # to fitting on the recent window only for legacy models without one.
        n_bars = min(500, len(df))
        x_fit = df[features].iloc[-n_bars:].values
        scaler, _ = load_or_fit_scaler(MODEL_DIR, table, x_fit)
        X_all = scaler.transform(x_fit)

        # CatBoost prediction (last bar)
        cb = CatBoostClassifier()
        cb.load_model(cb_path)
        cb_prob = float(cb.predict_proba(X_all[-1:])[:, 1][0])

        # LSTM prediction
        lstm_prob = None
        mode = "CB"
        if os.path.exists(lstm_path):
            lookback = _get_lookback(reg_entry, name)
            if len(X_all) >= lookback:
                lstm_model, mode, lookback = _load_lstm_model(lstm_path, lookback, len(features))
                if lstm_model is not None:
                    try:
                        X_seq = X_all[-lookback:].reshape(1, lookback, len(features))
                        lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                    except Exception:
                        lstm_prob = None
                        mode = "CB"

        # Ensemble
        if lstm_prob is not None:
            trend_val = float(df['trend_strength'].iloc[-1]) if 'trend_strength' in df.columns else 0.01
            trend_gate = reg_entry.get('profile', {}).get('trend_gate', 0.01) if reg_entry else 0.01
            prob = float(ensemble_with_gating(
                np.array([cb_prob]), np.array([lstm_prob]),
                np.array([trend_val]), trend_gate
            )[0])
        else:
            prob = cb_prob

        # Calibrate to an honest up-move frequency when a calibrator exists.
        prob = float(apply_calibrator(load_calibrator(MODEL_DIR, table), np.array([prob]))[0])

        # Tuned thresholds
        thr = thresholds.get(name, {})
        buy_thr = thr.get('buy', 0.55)
        sell_thr = thr.get('sell', 0.45)

        # Signal decision
        action, signal_type = "WAIT", ""
        if prob > buy_thr:
            action = "BUY"
            signal_type = "SNIPER BUY" if mode != "CB" else "CB BUY"
        elif prob < sell_thr:
            action = "SELL"
            signal_type = "SNIPER SHORT" if mode != "CB" else "CB SHORT"

        confidence = prob

        # Risk gate
        risk_result = None
        if _rm is not None and action != "WAIT":
            n_corr = 0
            if _pm is not None:
                n_corr = len(_pm.get_correlated_assets(name))
            risk_result = _rm.check_signal(
                name, action, confidence, taleb_val, n_correlated=n_corr
            )
            if not risk_result["approved"]:
                logger.info("Signal %s %s REJECTED by risk: %s",
                            action, name, risk_result["reason"])
                action = "WAIT"

        if action == "WAIT":
            return None

        # Portfolio context
        portfolio_line = ""
        if _pm is not None:
            sector = _pm.get_sector(name)
            correlated = _pm.get_correlated_assets(name)
            portfolio_line = f"\n   Sector: `{sector}`"
            if correlated:
                portfolio_line += f"  |  Corr: {', '.join(correlated[:3])}"

        # Position size
        size_line = ""
        if risk_result is not None and risk_result["approved"]:
            size_line = (f"\n   Kelly: `{risk_result['position_size_pct']:.1%}` "
                         f"(${risk_result['position_size_usd']:,.0f})")

        # Telegram message (emoji OK here - rendered in Telegram client)
        emoji_char = "BUY" if action == "BUY" else "SELL"
        msg = (
            f"*{emoji_char}: {name}* (`{current_price:,.2f}`)\n"
            f"   _{signal_type}_ | Prob:`{prob:.0%}` CB:`{cb_prob:.0%}` "
            f"Mode:`{mode}` Risk:`{taleb_val:.1f}`"
            f"{portfolio_line}{size_line}"
        )

        logger.info("%s %s | prob=%.0f%% cb=%.0f%% mode=%s taleb=%.1f",
                    action, name, prob * 100, cb_prob * 100, mode, taleb_val)
        return {"action": action, "confidence": confidence, "message": msg}

    except Exception as exc:
        logger.error("analyze_asset failed for %s: %s", name, exc)
        return None


# ==============================================================================
# TELEGRAM SENDER
# ==============================================================================

def send_telegram(messages, use_proxy):
    if not TELEGRAM_TOKEN or not TELEGRAM_USER_ID:
        logger.error("TELEGRAM_TOKEN / TELEGRAM_USER_ID not set in .env "
                     "(copy .env.example to .env and fill them in)")
        return False

    if use_proxy:
        apihelper.proxy = {"https": SOCKS5_PROXY}
        logger.info("Telegram using proxy: %s", SOCKS5_PROXY)
    else:
        apihelper.proxy = None

    bot = TeleBot(TELEGRAM_TOKEN)
    header = (f"*G-TRADE REPORT*\n"
              f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")
    full_msg = header + "\n\n".join(messages)

    try:
        if len(full_msg) > 4000:
            for chunk in [full_msg[i:i+4000] for i in range(0, len(full_msg), 4000)]:
                bot.send_message(TELEGRAM_USER_ID, chunk, parse_mode="Markdown")
        else:
            bot.send_message(TELEGRAM_USER_ID, full_msg, parse_mode="Markdown")
        logger.info("Telegram: sent %d signals", len(messages))
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


# ==============================================================================
# COMMANDS, DIGEST, DEGRADATION
# ==============================================================================

DIGEST_HOUR = int(os.getenv("GTRADE_DIGEST_HOUR", "9"))
STALE_MAX_DAYS = int(os.getenv("GTRADE_STALE_MAX_DAYS", "7"))
ACC_ALERT_FLOOR = 0.40   # accuracy below this triggers a warning
ACC_ALERT_MIN_N = 10     # minimum verified signals required before reporting
ALERT_STATE_PATH = os.path.join(MODEL_DIR, "alert_state.json")
RISK_STATE_PATH = os.path.join(MODEL_DIR, "risk_state.json")


def _risk_state():
    if os.path.exists(RISK_STATE_PATH):
        try:
            with open(RISK_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _load_alert_state():
    if os.path.exists(ALERT_STATE_PATH):
        try:
            with open(ALERT_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _save_alert_state(state):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ALERT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _send_plain(text, use_proxy):
    """Send without Markdown (report text contains special characters)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_USER_ID:
        return False
    apihelper.proxy = {"https": SOCKS5_PROXY} if use_proxy else None
    bot = TeleBot(TELEGRAM_TOKEN)
    try:
        for chunk in [text[i:i + 4000] for i in range(0, len(text), 4000)]:
            bot.send_message(TELEGRAM_USER_ID, chunk)
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


def _compose_digest():
    signals = track_record.latest_signals()
    stale = track_record.stale_assets(max_age_days=STALE_MAX_DAYS)
    return reports.build_digest(
        signals=signals,
        stale=stale,
        risk=_risk_state(),
        date_str=datetime.now().strftime("%Y-%m-%d"),
    )


def maybe_send_digest(use_proxy):
    """Once a day, starting from DIGEST_HOUR."""
    now = datetime.now()
    if now.hour < DIGEST_HOUR:
        return
    state = _load_alert_state()
    today = now.strftime("%Y-%m-%d")
    if state.get("digest_sent") == today:
        return
    if _send_plain(_compose_digest(), use_proxy):
        state["digest_sent"] = today
        _save_alert_state(state)
        logger.info("Digest sent")


def check_degradation(use_proxy):
    """Stale data and dropped accuracy; no more than once a day per asset."""
    state = _load_alert_state()
    sent = state.setdefault("degradation_sent", {})
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []

    for s in track_record.stale_assets(max_age_days=STALE_MAX_DAYS):
        key = f"stale:{s['asset']}"
        if sent.get(key) == today:
            continue
        if s["last_date"] is None:
            lines.append(f"{s['asset']}: no data in market.db")
        else:
            lines.append(f"{s['asset']}: data from {s['last_date']} ({s['age_days']} days ago)")
        sent[key] = today

    for sig in track_record.latest_signals():
        acc = sig["acc"]
        if acc["n"] >= ACC_ALERT_MIN_N and acc["acc"] is not None and acc["acc"] < ACC_ALERT_FLOOR:
            key = f"acc:{sig['asset']}"
            if sent.get(key) == today:
                continue
            lines.append(
                f"{sig['asset']}: accuracy {acc['acc']:.0%} "
                f"({acc['correct']}/{acc['n']}) - model has degraded"
            )
            sent[key] = today

    if lines:
        text = "Degradation:\n" + "\n".join(lines)
        if _send_plain(text, use_proxy):
            logger.info("Degradation alert: %d items", len(lines))
        _save_alert_state(state)


def start_command_listener(use_proxy):
    """Polling thread: /top, /signal X, /risk, /digest. Owner only."""
    if not TELEGRAM_TOKEN or not TELEGRAM_USER_ID:
        return None

    apihelper.proxy = {"https": SOCKS5_PROXY} if use_proxy else None
    bot = TeleBot(TELEGRAM_TOKEN)

    def _own(message):
        return str(message.chat.id) == str(TELEGRAM_USER_ID)

    @bot.message_handler(commands=["top"])
    def _top(message):
        if not _own(message):
            return
        bot.reply_to(message, reports.build_top_message(track_record.latest_signals()))

    @bot.message_handler(commands=["signal"])
    def _signal(message):
        if not _own(message):
            return
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "Format: /signal BTC")
            return
        asset = parts[1].upper()
        if asset not in FULL_ASSET_MAP:
            bot.reply_to(message, f"Unknown asset {asset}")
            return
        bot.reply_to(message, reports.build_signal_message(
            asset,
            track_record.asset_track(asset, limit=10),
            track_record.asset_accuracy(asset),
        ))

    @bot.message_handler(commands=["risk"])
    def _risk(message):
        if not _own(message):
            return
        bot.reply_to(message, reports.build_risk_message(_risk_state(), RISK_CONFIG))

    @bot.message_handler(commands=["digest"])
    def _digest(message):
        if not _own(message):
            return
        bot.reply_to(message, _compose_digest())

    @bot.message_handler(commands=["start", "help"])
    def _help(message):
        if not _own(message):
            return
        bot.reply_to(message, "Commands:\n/top - best signals\n/signal BTC - asset history\n"
                              "/risk - risk status\n/digest - digest now")

    def _poll():
        while True:
            try:
                bot.infinity_polling(timeout=30, long_polling_timeout=25)
            except Exception as exc:
                logger.error("Polling error: %s", exc)
                time.sleep(30)

    thread = threading.Thread(target=_poll, daemon=True, name="tg-commands")
    thread.start()
    logger.info("Command listener started")
    return thread


# ==============================================================================
# MAIN SCAN CYCLE
# ==============================================================================

def run_cycle():
    registry = _load_json(REGISTRY_PATH)
    thresholds = _load_json(THRESHOLDS_PATH)

    print(f"\n{'='*55}")
    print(f"  G-TRADE SENTINEL  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*55}")
    logger.info("=== New scan cycle started ===")

    yahoo_needs_proxy = detect_proxy_needed()
    tg_needs_proxy = detect_telegram_proxy()

    all_signals = []

    # World markets (Yahoo Finance)
    print("\n[1/2] Scanning world markets...")
    for name, symbol in FULL_ASSET_MAP.items():
        if name in MOEX_TARGETS:
            continue
        df = fetch_world(symbol, use_proxy=yahoo_needs_proxy)
        if df is None:
            print(f"   [WARN] {name:<8} -- no data")
            continue
        result = analyze_asset(df, name, registry, thresholds)
        if result:
            all_signals.append(result["message"])
            print(f"   {result['action']:<4} {name}")
        else:
            print(f"   WAIT {name}")

    # MOEX (no proxy)
    print("\n[2/2] Scanning MOEX...")
    for name in MOEX_TARGETS:
        symbol = FULL_ASSET_MAP.get(name, name)
        df = fetch_moex(symbol, name)
        if df is None:
            print(f"   [WARN] {name:<8} -- no data")
            continue
        result = analyze_asset(df, name, registry, thresholds)
        if result:
            all_signals.append(result["message"])
            print(f"   {result['action']:<4} {name}")
        else:
            print(f"   WAIT {name}")

    # Send to Telegram
    print(f"\n[-] {len(all_signals)} signal(s) found.")
    if all_signals:
        ok = send_telegram(all_signals, use_proxy=tg_needs_proxy)
        if ok:
            print("   [OK] Sent to Telegram.")
        else:
            print("   [ERR] Telegram send failed -- check logs.")
    else:
        print("   No actionable signals this cycle.")

    logger.info("Cycle complete -- %d signals sent", len(all_signals))


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    _tg_proxy = detect_telegram_proxy()
    start_command_listener(_tg_proxy)
    while True:
        try:
            run_cycle()
            maybe_send_digest(_tg_proxy)
            check_degradation(_tg_proxy)
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            logger.info("Bot stopped by user")
            break
        except Exception as exc:
            logger.error("Unexpected cycle error: %s", exc, exc_info=True)
            print(f"[ERR] Unexpected error: {exc}")

        print(f"\nNext scan in {CHECK_INTERVAL // 60} minutes. Press Ctrl+C to stop.")
        try:
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
