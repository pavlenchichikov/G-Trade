"""
Portfolio Manager V1 — G-Trade
=================================================
Cross-asset correlation, sector exposure tracking, diversification scoring,
and signal ranking that accounts for portfolio concentration.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Sector definitions ────────────────────────────────────────────────────────
SECTOR_MAP: Dict[str, List[str]] = {
    "CRYPTO":      ["BTC", "ETH", "SOL", "XRP", "TON", "DOGE", "BNB",
                    "ADA", "AVAX", "DOT", "LINK", "SHIB", "ATOM", "UNI", "NEAR"],
    "US_TECH":     ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN",
                    "META", "AMD", "PLTR", "COIN", "MSTR"],
    "US_HEALTH":   ["JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK"],
    "US_FINANCE":  ["JPM", "BAC", "GS", "V", "MA", "WFC"],
    "US_CONSUMER": ["WMT", "KO", "PEP", "MCD", "NKE", "DIS", "NFLX", "SBUX"],
    "US_INDUSTRY": ["BA", "CAT", "XOM", "CVX", "COP", "INTC", "QCOM", "AVGO",
                    "MU", "CRM", "ORCL", "ADBE", "UBER", "PYPL"],
    "COMMODITIES": ["GOLD", "SILVER", "OIL", "GAS"],
    "INDICES":     ["SP500", "NASDAQ", "DOW"],
    "MACRO":       ["VIX", "DXY", "TNX"],
    "RUS":         ["IMOEX", "SBER", "GAZP", "LKOH", "ROSN", "NVTK",
                    "TATN", "SNGS", "PLZL", "SIBN", "MGNT",
                    "TCSG", "VTBR", "BSPB", "MOEX_EX", "CBOM",
                    "YNDX", "OZON", "VKCO", "POSI", "MTSS", "RTKM",
                    "HHRU", "SOFL", "ASTR", "WUSH",
                    "CHMF", "NLMK", "MAGN", "RUAL", "ALRS", "TRMK", "MTLR", "RASP",
                    "IRAO", "HYDR", "FLOT", "AFLT", "PIKK",
                    "FEES", "UPRO", "MSNG", "NMTP",
                    "PHOR", "SGZH", "FIVE", "FIXP", "LENT", "MVID",
                    "SMLT", "LSRG"],
    "FOREX":       ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "USDRUB",
                    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
                    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
                    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADJPY", "CHFJPY", "NZDJPY",
                    "USDTRY", "USDMXN", "USDZAR", "USDSGD", "USDNOK", "USDSEK", "USDPLN", "USDCNH"],
}

# Maximum fraction of portfolio per sector
SECTOR_LIMITS: Dict[str, float] = {
    "CRYPTO":      0.35,
    "US_TECH":     0.40,
    "COMMODITIES": 0.25,
    "INDICES":     0.30,
    "MACRO":       0.15,
    "RUS":         0.35,
    "FOREX":       0.30,
    "OTHER":       0.20,
}

CORRELATION_HIGH = 0.70    # Threshold for "highly correlated"
LOOKBACK_DAYS    = 120     # Days of history for correlation


class PortfolioManager:
    """
    Manages multi-asset portfolio analytics.

    Usage::

        from config import FULL_ASSET_MAP
        pm = PortfolioManager(FULL_ASSET_MAP)
        corr = pm.get_correlation_matrix()
        score = pm.get_diversification_score({"BTC": 0.10, "ETH": 0.08})
    """

    def __init__(self, asset_map: dict, lookback_days: int = LOOKBACK_DAYS):
        self.asset_map = asset_map
        self.lookback_days = lookback_days
        self._engine = create_engine(f"sqlite:///{os.path.join(BASE_DIR, 'market.db')}")
        self._returns_cache: Optional[pd.DataFrame] = None
        self._corr_cache:    Optional[pd.DataFrame] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _table_name(self, asset: str) -> str:
        return asset.lower().replace("^", "").replace(".", "").replace("-", "")

    def _get_returns(self) -> pd.DataFrame:
        """Load daily close prices → compute returns for all assets."""
        if self._returns_cache is not None:
            return self._returns_cache

        series = {}
        for name in self.asset_map:
            table = self._table_name(name)
            try:
                df = pd.read_sql(
                    f"SELECT Date, close FROM \"{table}\" "
                    f"ORDER BY Date DESC LIMIT {self.lookback_days + 5}",
                    self._engine, index_col="Date",
                )
                if len(df) >= 20:
                    df.index = pd.to_datetime(df.index)
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    series[name] = df["close"].pct_change().dropna()
            except Exception as exc:
                logger.debug("Could not load %s: %s", name, exc)

        if series:
            self._returns_cache = pd.DataFrame(series).dropna(how="all")
        else:
            self._returns_cache = pd.DataFrame()
        return self._returns_cache

    def invalidate_cache(self) -> None:
        """Force re-computation of returns and correlation on next call."""
        self._returns_cache = None
        self._corr_cache    = None

    # ── Correlation ───────────────────────────────────────────────────────────

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation matrix for all available assets."""
        if self._corr_cache is not None:
            return self._corr_cache
        returns = self._get_returns()
        if returns.empty:
            return pd.DataFrame()
        self._corr_cache = returns.corr(method="pearson")
        return self._corr_cache

    def get_correlated_assets(
        self,
        asset: str,
        threshold: float = CORRELATION_HIGH,
        open_only: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Return assets correlated with *asset* above *threshold*.
        If *open_only* is provided, restrict to that subset.
        """
        corr = self.get_correlation_matrix()
        if corr.empty or asset not in corr.columns:
            return []
        col = corr[asset].abs().drop(asset, errors="ignore")
        candidates = col[col >= threshold].index.tolist()
        if open_only is not None:
            candidates = [a for a in candidates if a in open_only]
        return candidates

    # ── Sector helpers ────────────────────────────────────────────────────────

    def get_sector(self, asset: str) -> str:
        for sector, members in SECTOR_MAP.items():
            if asset in members:
                return sector
        return "OTHER"

    # ── Portfolio heat ────────────────────────────────────────────────────────

    def get_portfolio_heat(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate position sizes by sector.

        positions: {asset: fraction_of_capital}  e.g. {"BTC": 0.10, "ETH": 0.08}
        Returns: {sector: total_fraction}
        """
        heat: Dict[str, float] = {s: 0.0 for s in list(SECTOR_MAP) + ["OTHER"]}
        for asset, frac in positions.items():
            sector = self.get_sector(asset)
            heat[sector] = heat.get(sector, 0.0) + frac
        return heat

    def check_sector_limit(
        self,
        asset: str,
        proposed_size: float,
        current_positions: Dict[str, float],
    ) -> tuple:
        """
        Check whether adding *proposed_size* would exceed the sector limit.
        Returns (allowed: bool, current_exposure: float, limit: float).
        """
        sector = self.get_sector(asset)
        limit = SECTOR_LIMITS.get(sector, 0.50)
        current = sum(v for a, v in current_positions.items()
                      if self.get_sector(a) == sector)
        return (current + proposed_size <= limit), current, limit

    # ── Diversification ───────────────────────────────────────────────────────

    def get_diversification_score(self, positions: Dict[str, float]) -> float:
        """
        Score 0–100 measuring portfolio diversification.
        Uses the Herfindahl–Hirschman Index (HHI) over sector exposures.
        100 = perfectly spread, 0 = all in one sector.
        """
        if not positions:
            return 100.0
        heat = self.get_portfolio_heat(positions)
        active = {s: v for s, v in heat.items() if v > 0}
        if not active:
            return 100.0
        total = sum(active.values())
        if total == 0:
            return 100.0
        shares = [v / total for v in active.values()]
        hhi = sum(s ** 2 for s in shares)
        return round((1.0 - hhi) * 100, 1)

    # ── Per-asset statistics ──────────────────────────────────────────────────

    def get_asset_stats(self, asset: str) -> dict:
        """
        Compute risk statistics for a single asset.
        Returns dict with volatility, Sharpe, max drawdown, skewness, kurtosis.
        """
        returns = self._get_returns()
        if returns.empty or asset not in returns.columns:
            return {}
        r = returns[asset].dropna()
        if len(r) < 10:
            return {}

        def _max_dd(r_: pd.Series) -> float:
            cum = (1 + r_).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / (peak + 1e-9)
            return float(dd.min())

        annualise = np.sqrt(252)
        return {
            "vol_annual":   float(r.std() * annualise),
            "sharpe_90d":   float((r.tail(90).mean() / (r.tail(90).std() + 1e-9)) * annualise),
            "max_dd_30d":   _max_dd(r.tail(30)),
            "max_dd_full":  _max_dd(r),
            "skewness":     float(r.skew()),
            "kurtosis":     float(r.kurt()),
            "n_days":       len(r),
        }

    def get_all_stats(self) -> pd.DataFrame:
        """Return stats for every asset as a DataFrame."""
        rows = []
        for name in self.asset_map:
            stats = self.get_asset_stats(name)
            if stats:
                stats["asset"] = name
                stats["sector"] = self.get_sector(name)
                rows.append(stats)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("asset")

    # ── Signal ranking ────────────────────────────────────────────────────────

    def rank_signals(
        self,
        signals: Dict[str, dict],
        current_positions: Dict[str, float] | None = None,
        top_n: int = 5,
    ) -> List[dict]:
        """
        Rank BUY/SELL signals by a composite score:
            score = confidence − correlation_penalty − sector_crowding_penalty

        signals: {asset: {"signal": "BUY"|"SELL"|"WAIT", "confidence": float}}
        Returns top_n ranked list of dicts.
        """
        if current_positions is None:
            current_positions = {}
        open_assets = list(current_positions.keys())
        ranked = []

        for asset, data in signals.items():
            if data.get("signal", "WAIT") == "WAIT":
                continue
            confidence = float(data.get("confidence", 0.5))

            # Penalty: each highly-correlated OPEN position reduces score
            correlated_open = self.get_correlated_assets(asset, open_only=open_assets)
            corr_penalty = len(correlated_open) * 0.05

            # Penalty: sector crowding
            sector = self.get_sector(asset)
            sector_exposure = sum(v for a, v in current_positions.items()
                                  if self.get_sector(a) == sector)
            sector_limit = SECTOR_LIMITS.get(sector, 0.50)
            crowd_penalty = min(0.20, sector_exposure / max(sector_limit, 0.01) * 0.15)

            score = confidence - corr_penalty - crowd_penalty

            ranked.append({
                "asset":             asset,
                "signal":            data["signal"],
                "confidence":        confidence,
                "score":             round(score, 4),
                "correlated_open":   correlated_open,
                "sector":            sector,
                "sector_exposure":   sector_exposure,
                "corr_penalty":      corr_penalty,
                "crowd_penalty":     crowd_penalty,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]

    # ── Context summary for alert / dashboard ─────────────────────────────────

    def get_signal_context(
        self,
        asset: str,
        signal: str,
        current_positions: Dict[str, float] | None = None,
    ) -> dict:
        """Return a full portfolio context dict for a signal."""
        if current_positions is None:
            current_positions = {}
        return {
            "asset":              asset,
            "signal":             signal,
            "sector":             self.get_sector(asset),
            "correlated_assets":  self.get_correlated_assets(asset),
            "diversification":    self.get_diversification_score(current_positions),
            "portfolio_heat":     self.get_portfolio_heat(current_positions),
            "asset_stats":        self.get_asset_stats(asset),
        }


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from config import FULL_ASSET_MAP
    except ImportError:
        FULL_ASSET_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "GOLD": "GC=F"}

    pm = PortfolioManager(FULL_ASSET_MAP)

    corr = pm.get_correlation_matrix()
    if not corr.empty:
        print(f"\nCorrelation matrix: {corr.shape[0]} assets × {corr.shape[1]} assets")

    mock_positions = {"BTC": 0.10, "ETH": 0.08, "GOLD": 0.05}
    heat  = pm.get_portfolio_heat(mock_positions)
    score = pm.get_diversification_score(mock_positions)
    print(f"\nPortfolio heat: { {k: f'{v:.0%}' for k, v in heat.items() if v > 0} }")
    print(f"Diversification score: {score:.1f}/100")

    mock_signals = {
        "BTC":  {"signal": "BUY",  "confidence": 0.63},
        "ETH":  {"signal": "BUY",  "confidence": 0.60},
        "GOLD": {"signal": "BUY",  "confidence": 0.58},
        "SOL":  {"signal": "SELL", "confidence": 0.55},
    }
    top = pm.rank_signals(mock_signals, mock_positions, top_n=3)
    print("\nTop ranked signals (portfolio-adjusted):")
    for r in top:
        print(f"  {r['asset']:<6} {r['signal']:<4}  conf={r['confidence']:.0%}  "
              f"score={r['score']:.3f}  sector={r['sector']}")
