"""Asset profiles and trading parameters.

Maps assets to their optimal trading configurations based on behavior type.
"""

# Asset category sets
TRENDY = {
    "BTC", "ETH", "SOL", "DOGE", "BNB", "NVDA", "TSLA", "PLTR", "COIN", "MSTR", "AMD",
    "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR",
}

MEANREV = {"VIX", "DXY", "TNX"}

RUS = {
    "IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "YNDX", "TCSG", "OZON", "VKCO",
    "POSI", "AFLT", "CBOM", "HHRU", "SOFL", "ASTR", "WUSH", "TRMK", "MTLR", "RASP",
    "FEES", "UPRO", "MSNG", "NMTP", "PHOR", "SGZH",
    "FIVE", "FIXP", "LENT", "MVID", "SMLT", "LSRG",
}

FOREX = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDRUB",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADJPY", "CHFJPY", "NZDJPY",
    "USDTRY", "USDMXN", "USDZAR", "USDSGD", "USDNOK", "USDSEK", "USDPLN", "USDCNH",
}

# Profile configurations
PROFILE_DEFAULT = {
    "lookback": 45,
    "thr_buy_grid": [0.52, 0.54, 0.56, 0.58],
    "thr_sell_grid": [0.48, 0.46, 0.44],
    "no_trade_band": 0.02,
    "regime_risk_cap": 5.0,
    "trend_gate": 0.01,
    "top_k_features": 12,
}

PROFILE_TRENDY = {
    **PROFILE_DEFAULT,
    "lookback": 55,
    "thr_buy_grid": [0.51, 0.53, 0.55, 0.57],
    "thr_sell_grid": [0.49, 0.47, 0.45],
    "trend_gate": 0.008,
}

PROFILE_MEANREV = {
    **PROFILE_DEFAULT,
    "lookback": 40,
    "thr_buy_grid": [0.54, 0.56, 0.58, 0.60],
    "thr_sell_grid": [0.46, 0.44, 0.42],
    "no_trade_band": 0.03,
    "trend_gate": 0.015,
}

PROFILE_RUS = {
    **PROFILE_DEFAULT,
    "lookback": 45,
    "thr_buy_grid": [0.53, 0.55, 0.57],
    "thr_sell_grid": [0.47, 0.45, 0.43],
    "regime_risk_cap": 4.5,
}

PROFILE_FOREX = {
    **PROFILE_DEFAULT,
    "lookback": 50,
    "thr_buy_grid": [0.53, 0.55, 0.57, 0.59],
    "thr_sell_grid": [0.47, 0.45, 0.43],
    "no_trade_band": 0.025,
    "trend_gate": 0.005,
}


def get_profile(asset: str) -> dict:
    """Return trading profile for the given asset based on its category."""
    if asset in TRENDY:
        return PROFILE_TRENDY
    if asset in MEANREV:
        return PROFILE_MEANREV
    if asset in RUS:
        return PROFILE_RUS
    if asset in FOREX:
        return PROFILE_FOREX
    return PROFILE_DEFAULT
