"""
Risk Manager V1 — G-Trade
============================================
Full risk management: Kelly Criterion, position sizing, drawdown protection,
daily loss limits, Taleb risk gating, and persistent state tracking.
"""

import json
import logging
import os
from datetime import date, datetime

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_STATE_PATH = os.path.join(BASE_DIR, "models", "risk_state.json")

# ── Risk configuration ────────────────────────────────────────────────────────
RISK_CONFIG = {
    # Position limits
    "max_portfolio_exposure": 0.30,   # Max 30 % of capital deployed at once
    "max_single_position":    0.10,   # Max 10 % per single trade
    # Circuit breakers
    "max_daily_loss":         0.05,   # Halt trading if today's loss > 5 %
    "max_drawdown_halt":      0.15,   # Halt if drawdown from peak > 15 %
    # Kelly tuning
    "kelly_fraction":         0.25,   # Fractional Kelly (quarter-Kelly)
    "min_kelly_threshold":    0.005,  # Ignore signals with Kelly < 0.5 %
    # Risk adjustments
    "taleb_risk_cap":         5.0,    # Block BUY if Taleb risk > 5.0
    "taleb_soft_cap":         2.5,    # Reduce size above 2.5
    "correlation_penalty":    0.15,   # Size reduction per correlated open position
    # Default trade expectations (used when historical stats are unavailable)
    "default_avg_win":        0.025,  # 2.5 % average winning trade
    "default_avg_loss":       0.012,  # 1.2 % average losing trade
}


class RiskManager:
    """
    Manages position sizing, portfolio exposure, and trading halt conditions.

    Usage::

        rm = RiskManager(initial_capital=10_000)
        result = rm.check_signal("BTC", "BUY", confidence=0.62, taleb_risk=1.2)
        if result["approved"]:
            print(f"Trade size: ${result['position_size_usd']:.2f}")
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.daily_start_date: str = date.today().isoformat()
        # {asset: {"size_usd": float, "entry_price": float, "direction": str}}
        self.open_positions: dict = {}
        self.trade_history: list = []
        self._load_state()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not os.path.exists(RISK_STATE_PATH):
            return
        try:
            with open(RISK_STATE_PATH, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            self.current_capital = float(state.get("current_capital", self.initial_capital))
            self.peak_capital = float(state.get("peak_capital", self.initial_capital))
            self.daily_start_capital = float(state.get("daily_start_capital", self.initial_capital))
            self.daily_start_date = state.get("daily_start_date", date.today().isoformat())
            self.open_positions = state.get("open_positions", {})
            # Reset daily stats if it's a new day
            if self.daily_start_date != date.today().isoformat():
                self.daily_start_capital = self.current_capital
                self.daily_start_date = date.today().isoformat()
            logger.info("Risk state loaded: capital=%.2f, drawdown=%.2f%%",
                        self.current_capital, self.current_drawdown * 100)
        except Exception as exc:
            logger.warning("Could not load risk state: %s", exc)

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(RISK_STATE_PATH), exist_ok=True)
        state = {
            "current_capital":     self.current_capital,
            "peak_capital":        self.peak_capital,
            "daily_start_capital": self.daily_start_capital,
            "daily_start_date":    self.daily_start_date,
            "open_positions":      self.open_positions,
            "last_updated":        datetime.now().isoformat(),
        }
        try:
            with open(RISK_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            logger.error("Could not save risk state: %s", exc)

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def current_drawdown(self) -> float:
        """Fraction of capital lost from all-time peak (0–1)."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.current_capital) / self.peak_capital)

    @property
    def daily_pnl(self) -> float:
        """Today's P&L as a fraction of the day-start capital."""
        if self.daily_start_capital <= 0:
            return 0.0
        return (self.current_capital - self.daily_start_capital) / self.daily_start_capital

    @property
    def total_exposure(self) -> float:
        """Fraction of current capital currently deployed."""
        deployed = sum(p.get("size_usd", 0.0) for p in self.open_positions.values())
        return deployed / max(self.current_capital, 1.0)

    @property
    def remaining_capacity(self) -> float:
        """How much more exposure is allowed before hitting the portfolio limit."""
        return max(0.0, RISK_CONFIG["max_portfolio_exposure"] - self.total_exposure)

    # ── Circuit breakers ─────────────────────────────────────────────────────

    def is_trading_halted(self) -> tuple:
        """
        Returns (halted: bool, reason: str).
        If halted is True, no new trades should be opened.
        """
        if self.current_drawdown >= RISK_CONFIG["max_drawdown_halt"]:
            return True, (f"MAX DRAWDOWN REACHED "
                          f"({self.current_drawdown:.1%} >= "
                          f"{RISK_CONFIG['max_drawdown_halt']:.0%})")
        if self.daily_pnl <= -RISK_CONFIG["max_daily_loss"]:
            return True, (f"DAILY LOSS LIMIT HIT "
                          f"({self.daily_pnl:.1%} <= "
                          f"-{RISK_CONFIG['max_daily_loss']:.0%})")
        return False, ""

    # ── Kelly Criterion ───────────────────────────────────────────────────────

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        taleb_risk: float = 0.0,
        n_correlated: int = 0,
    ) -> float:
        """
        Compute the recommended position size as a fraction of current capital.

        Formula: Kelly = (p·b − q) / b   where b = avg_win / avg_loss
        Then applies:
          • Fractional Kelly  (× kelly_fraction)
          • Taleb risk penalty (exponential decay above soft cap)
          • Correlation penalty (linear reduction per correlated position)
          • Hard cap at max_single_position and remaining_capacity

        Returns 0.0 if the edge is negative or position is too small.
        """
        avg_win = avg_win or RISK_CONFIG["default_avg_win"]
        avg_loss = avg_loss or RISK_CONFIG["default_avg_loss"]

        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss

        kelly_full = (p * b - q) / b
        if kelly_full <= 0:
            return 0.0

        # 1. Fractional Kelly (quarter-Kelly by default)
        size = kelly_full * RISK_CONFIG["kelly_fraction"]

        # 2. Taleb risk penalty — starts reducing size above soft_cap
        soft_cap = RISK_CONFIG["taleb_soft_cap"]
        if taleb_risk > soft_cap:
            penalty = np.exp(-0.4 * (taleb_risk - soft_cap))
            size *= penalty

        # 3. Correlation penalty — each correlated open position reduces size
        if n_correlated > 0:
            size *= max(0.1, 1.0 - n_correlated * RISK_CONFIG["correlation_penalty"])

        # 4. Hard caps
        size = min(size, RISK_CONFIG["max_single_position"], self.remaining_capacity)

        if size < RISK_CONFIG["min_kelly_threshold"]:
            return 0.0

        return size

    # ── Main signal check ─────────────────────────────────────────────────────

    def check_signal(
        self,
        asset: str,
        signal: str,          # "BUY" | "SELL" | "WAIT"
        confidence: float,    # model probability (0–1)
        taleb_risk: float = 0.0,
        n_correlated: int = 0,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
    ) -> dict:
        """
        Full risk gate for a trading signal.

        Returns a dict::

            {
                "approved":           bool,
                "reason":             str,
                "position_size_pct":  float,   # fraction of capital
                "position_size_usd":  float,
                "kelly_raw":          float,   # before caps
                "drawdown":           float,
                "daily_pnl":          float,
            }
        """
        result = {
            "approved": False,
            "reason": "",
            "position_size_pct": 0.0,
            "position_size_usd": 0.0,
            "kelly_raw": 0.0,
            "drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
        }

        # Always allow WAIT
        if signal == "WAIT":
            result["approved"] = True
            result["reason"] = "WAIT — no capital required"
            return result

        # Circuit breaker
        halted, halt_reason = self.is_trading_halted()
        if halted:
            result["reason"] = f"TRADING HALTED — {halt_reason}"
            return result

        # Block BUY in high-risk regime
        if signal == "BUY" and taleb_risk > RISK_CONFIG["taleb_risk_cap"]:
            result["reason"] = (f"HIGH TAIL RISK — Taleb={taleb_risk:.2f} "
                                f"> cap={RISK_CONFIG['taleb_risk_cap']}")
            return result

        # Map signal to win_rate
        effective_wr = win_rate if win_rate is not None else (
            confidence if signal == "BUY" else (1.0 - confidence)
        )

        # Kelly sizing
        size_pct = self.kelly_fraction(
            win_rate=effective_wr,
            avg_win=avg_win,
            avg_loss=avg_loss,
            taleb_risk=taleb_risk,
            n_correlated=n_correlated,
        )

        # Compute raw Kelly for reporting (no caps)
        avg_w = avg_win or RISK_CONFIG["default_avg_win"]
        avg_l = avg_loss or RISK_CONFIG["default_avg_loss"]
        b = avg_w / avg_l
        p, q = effective_wr, 1.0 - effective_wr
        result["kelly_raw"] = max(0.0, (p * b - q) / b)

        if size_pct < RISK_CONFIG["min_kelly_threshold"]:
            result["reason"] = (f"Edge too small — Kelly={size_pct:.2%} "
                                f"< min={RISK_CONFIG['min_kelly_threshold']:.2%}")
            return result

        result["approved"] = True
        result["position_size_pct"] = size_pct
        result["position_size_usd"] = self.current_capital * size_pct
        result["reason"] = (
            f"APPROVED — Kelly={size_pct:.1%}, "
            f"Capital=${self.current_capital:,.0f}, "
            f"DD={self.current_drawdown:.1%}"
        )
        return result

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def record_trade(
        self,
        asset: str,
        direction: str,    # "BUY" | "SELL"
        size_usd: float,
        entry_price: float,
    ) -> None:
        """Register an opened position."""
        self.open_positions[asset] = {
            "size_usd":    size_usd,
            "direction":   direction,
            "entry_price": entry_price,
            "entry_ts":    datetime.now().isoformat(),
        }
        self.save_state()

    def close_trade(self, asset: str, exit_price: float) -> float:
        """
        Close a position and update capital.
        Returns the realised P&L in dollars.
        """
        pos = self.open_positions.pop(asset, None)
        if pos is None:
            return 0.0

        entry = pos["entry_price"]
        size = pos["size_usd"]
        direction = pos.get("direction", "BUY")

        ret = (exit_price - entry) / entry if direction == "BUY" else (entry - exit_price) / entry
        pnl = size * ret

        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

        self.trade_history.append({
            "asset":       asset,
            "direction":   direction,
            "entry_price": entry,
            "exit_price":  exit_price,
            "size_usd":    size,
            "pnl":         pnl,
            "closed_ts":   datetime.now().isoformat(),
        })
        self.save_state()
        return pnl

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a snapshot of the current risk state."""
        halted, halt_reason = self.is_trading_halted()
        return {
            "current_capital":   self.current_capital,
            "initial_capital":   self.initial_capital,
            "peak_capital":      self.peak_capital,
            "total_return_pct":  (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            "current_drawdown":  self.current_drawdown,
            "daily_pnl":         self.daily_pnl,
            "total_exposure":    self.total_exposure,
            "remaining_capacity": self.remaining_capacity,
            "open_positions":    len(self.open_positions),
            "trading_halted":    halted,
            "halt_reason":       halt_reason,
        }

    def print_summary(self) -> None:
        s = self.get_summary()
        print("\n" + "="*55)
        print("  RISK MANAGER STATUS")
        print("="*55)
        print(f"  Capital:      ${s['current_capital']:>12,.2f}")
        print(f"  Total Return: {s['total_return_pct']:>+11.2f}%")
        print(f"  Drawdown:     {s['current_drawdown']:>11.1%}")
        print(f"  Daily P&L:    {s['daily_pnl']:>+11.1%}")
        print(f"  Exposure:     {s['total_exposure']:>11.1%} / {RISK_CONFIG['max_portfolio_exposure']:.0%}")
        print(f"  Open Pos.:    {s['open_positions']:>12}")
        status = "[HALTED]" if s["trading_halted"] else "[ACTIVE]"
        print(f"  Status:       {status}")
        if s["trading_halted"]:
            print(f"  Reason:       {s['halt_reason']}")
        print("="*55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager(initial_capital=10_000)
    rm.print_summary()

    # Demo checks
    for asset, conf, taleb in [("BTC", 0.62, 1.5), ("ETH", 0.58, 6.0), ("GOLD", 0.54, 0.8)]:
        res = rm.check_signal(asset, "BUY", confidence=conf, taleb_risk=taleb)
        status = "[OK]" if res["approved"] else "[NO]"
        print(f"{status} {asset:<6} conf={conf:.0%} taleb={taleb:.1f} → "
              f"size={res['position_size_pct']:.1%} (${res['position_size_usd']:,.0f}) | {res['reason']}")
