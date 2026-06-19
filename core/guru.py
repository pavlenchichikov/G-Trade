"""Guru Council V2 - Fundamental analysis from 4 investment philosophies.

Each guru returns (status, description, score 0-2):
  - Lynch  - GARP: PEG ratio + revenue trend
  - Buffett - Quality moats: ROE, FCF, low debt, margins
  - Graham  - Deep value: NCAV, Graham Number, P/E, balance sheet
  - Munger  - Inversion: what can go wrong? Risk detection

Council aggregation adds three corrections over a plain equal-weight sum:
  - Munger veto: a DANGER from the risk guru blocks a BUY (down to HOLD). The
    inversion voice must not be outvoted by the three bullish value screens.
  - Track-record weighting: per-guru weights can be calibrated from guru_log
    forward hit-rate (council_weights_from_log); equal weights by default.
  - Technical discount: a verdict built only on price/momentum (no fundamentals)
    is less reliable than one built on PEG/ROE/Graham#, so its score is shaved.
  - Sector awareness: financials carry leverage by design, so debt/liquidity
    gates are relaxed for them instead of always firing WEAK/DANGER.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd

# Verdict built on price-only signals (no fundamentals) is shaved by this factor.
TECH_CONFIDENCE_DISCOUNT = 0.85

# Sectors where high debt / low current ratio is structural, not distress.
_FINANCIAL_SECTORS = {"US FINANCE", "RUS FINANCE"}


def _is_financial(sector: str | None) -> bool:
    return bool(sector) and sector in _FINANCIAL_SECTORS


def calc_peg(pe: float, growth: float) -> float | None:
    """PEG ratio: P/E divided by earnings growth rate (%). Lynch's key metric."""
    if pe <= 0 or growth <= 0:
        return None
    return pe / (growth * 100)


def calc_graham_number(eps: float, book_value: float) -> float | None:
    """Graham Number = sqrt(22.5 * EPS * Book Value). Intrinsic value proxy."""
    if eps <= 0 or book_value <= 0:
        return None
    return math.sqrt(22.5 * eps * book_value)


def technical_context(df: pd.DataFrame) -> dict[str, Any] | None:
    """Extract technical indicators from price DataFrame."""
    if df is None or len(df) < 50:
        return None
    last = df.iloc[-1]
    close = last['close']

    above_50 = close > last.get('SMA_50', close)
    above_200 = close > last.get('SMA_200', close)
    rsi = last.get('RSI', 50)

    window = min(252, len(df))
    series = df['close'].iloc[-window:]
    p_min, p_max = series.min(), series.max()
    pct_52w = (close - p_min) / (p_max - p_min) * 100 if p_max > p_min else 50

    rets = df['close'].pct_change().dropna()
    vol_30d = float(rets.iloc[-30:].std() * (252 ** 0.5)) if len(rets) >= 30 else 0

    macd_bull = last.get('MACD_hist', 0) > 0
    sma20_now = last.get('SMA_20', close)
    sma20_prev = df['SMA_20'].iloc[-5] if 'SMA_20' in df.columns and len(df) >= 5 else sma20_now
    sma_rising = sma20_now > sma20_prev

    return {
        'close': close, 'rsi': rsi, 'above_50': above_50, 'above_200': above_200,
        'pct_52w': pct_52w, 'vol_30d': vol_30d, 'macd_bull': macd_bull,
        'sma_rising': sma_rising,
    }


# -- Individual Guru Algorithms -----------------------------------------------

def lynch_analysis(fund: dict | None, tech: dict | None) -> tuple[str, str, int]:
    """Peter Lynch - Growth At Reasonable Price (GARP)."""
    if fund:
        pe = fund.get('pe', 0)
        growth = fund.get('growth', 0)

        peg = fund.get('peg_ratio') or None
        if not peg or peg <= 0:
            peg = calc_peg(pe, growth)

        rev_bonus = ""
        rev_qoq = fund.get('revenue_qoq', 0)
        if rev_qoq > 0.10:
            rev_bonus = f" | Rev QoQ: +{rev_qoq:.0%} (accelerating)"
        elif rev_qoq > 0:
            rev_bonus = f" | Rev QoQ: +{rev_qoq:.0%}"

        earnings_flag = ""
        if fund.get('earnings_stable') is False:
            earnings_flag = " | WARN: unstable earnings"

        if peg is not None and peg > 0:
            if peg < 1.0:
                return "[OK] BUY", f"PEG: {peg:.2f} (growth underpriced) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}{earnings_flag}", 2
            elif peg < 2.0:
                return "[--] FAIR", f"PEG: {peg:.2f} (fair) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}{earnings_flag}", 1
            else:
                return "[OFF] EXP", f"PEG: {peg:.2f} (overpriced) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}", 0
        elif pe > 0:
            adj = ""
            if rev_qoq > 0.15:
                adj = " (but revenue is growing!)"
            if pe < 12:
                return "[OK] CHEAP", f"P/E: {pe:.1f} (no growth data){adj}{rev_bonus}", 2
            elif pe < 25:
                return "[--] FAIR", f"P/E: {pe:.1f} (no growth data){rev_bonus}", 1
            else:
                return "[OFF] EXP", f"P/E: {pe:.1f} (expensive){rev_bonus}", 0

    if tech:
        score = int(tech['above_50']) + int(tech['above_200'])
        sma_txt = f"SMA50: {'^' if tech['above_50'] else 'v'}  SMA200: {'^' if tech['above_200'] else 'v'}"
        if score == 2:
            return "[OK] MOMENTUM", sma_txt, 2
        elif score == 1:
            return "[--] SIDEWAYS", sma_txt, 1
        else:
            return "[OFF] DOWNTREND", sma_txt, 0

    return "[--] N/A", "No data", 0


def buffett_analysis(fund: dict | None, tech: dict | None,
                     *, sector: str | None = None) -> tuple[str, str, int]:
    """Warren Buffett - Quality moats, ROE, FCF, low debt, retained earnings."""
    if fund:
        roe = fund.get('roe', 0)
        debt = fund.get('debt_equity', 0)
        margin = fund.get('profit_margin', 0)
        gross_margin = fund.get('gross_margin', 0)
        div_yield = fund.get('dividend_yield', 0)
        fcf = fund.get('fcf', 0)

        score = 0
        details = []

        if roe > 0.20:
            score += 2; details.append(f"ROE: {roe:.0%} (excellent)")
        elif roe > 0.15:
            score += 1; details.append(f"ROE: {roe:.0%} (good)")
        else:
            details.append(f"ROE: {roe:.0%} (weak)")

        if _is_financial(sector):
            # Financials carry high debt/equity by the nature of the business -
            # not distress. Award a neutral point instead of a "high debt" penalty.
            score += 1; details.append("Debt: N/A for financials")
        elif debt < 0.5:
            score += 2; details.append(f"Debt: {debt:.1f}x (minimal)")
        elif debt < 1.5:
            score += 1; details.append(f"Debt: {debt:.1f}x (moderate)")
        else:
            details.append(f"Debt: {debt:.1f}x (high)")

        if gross_margin > 0.40:
            score += 1; details.append(f"Gross: {gross_margin:.0%} (moat)")
        elif margin > 0.20:
            score += 1; details.append(f"Margin: {margin:.0%}")
        elif margin > 0:
            details.append(f"Margin: {margin:.0%}")

        if div_yield > 3:
            score += 1; details.append(f"Div: {div_yield:.1f}%")
        elif div_yield > 0:
            details.append(f"Div: {div_yield:.1f}%")

        if fcf > 0:
            score += 1
            fcf_streak = fund.get('fcf_positive_streak', 0)
            if fcf_streak >= 4:
                score += 1; details.append("FCF: 4/4 q positive")
            p2fcf = fund.get('price_to_fcf', 0)
            if 0 < p2fcf < 20:
                details.append(f"P/FCF: {p2fcf:.1f}")

        retained = fund.get('retained_earnings', 0)
        if retained and retained > 0:
            score += 1

        cash = fund.get('cash', 0)
        total_debt = fund.get('total_debt', 0)
        if cash > 0 and total_debt > 0 and cash > total_debt * 0.5:
            details.append(f"Cash covers {cash/total_debt:.0%} of debt")

        desc = " | ".join(details[:4])
        if score >= 6:
            return "[TOP] GEM", desc, 2
        elif score >= 3:
            return "[OK] QUALITY", desc, 1
        else:
            return "[!] WEAK", desc, 0

    if tech:
        rsi_ok = 40 < tech['rsi'] < 70
        trend_ok = tech['above_200']
        s = int(rsi_ok) + int(trend_ok)
        desc = f"SMA200: {'+' if trend_ok else '-'} | RSI: {tech['rsi']:.0f} ({'normal' if rsi_ok else 'extreme'})"
        if s == 2:
            return "[OK] STABLE", desc, 1
        elif s == 1:
            return "[--] MIXED", desc, 0
        else:
            return "[!] WEAK", desc, 0

    return "[--] N/A", "No data", 0


def graham_analysis(fund: dict | None, tech: dict | None,
                    *, sector: str | None = None) -> tuple[str, str, int]:
    """Benjamin Graham - Deep value, margin of safety, Graham Number, NCAV."""
    if fund:
        pe = fund.get('pe', 0)
        eps = fund.get('eps', 0)
        bv = fund.get('book_value', 0)
        price = fund.get('price', 0)
        debt = fund.get('debt_equity', 0)
        current_ratio = fund.get('current_ratio', 0)

        score = 0
        details = []

        ncav_ps = fund.get('ncav_per_share')
        if ncav_ps and price > 0:
            ncav_margin = (ncav_ps - price) / price * 100
            if ncav_margin > 0:
                score += 3; details.append(f"NCAV/sh: ${ncav_ps:.0f} > price (liquidation value higher!)")
            elif ncav_margin > -30:
                score += 1; details.append(f"NCAV/sh: ${ncav_ps:.0f} ({ncav_margin:+.0f}%)")

        gn = calc_graham_number(eps, bv)
        if gn and price > 0:
            margin = (gn - price) / price * 100
            if margin > 20:
                score += 2; details.append(f"Graham#: ${gn:.0f} (margin {margin:.0f}%)")
            elif margin > 0:
                score += 1; details.append(f"Graham#: ${gn:.0f} (margin {margin:.0f}%)")
            else:
                details.append(f"Graham#: ${gn:.0f} (above price by {-margin:.0f}%)")

        tbv_ps = fund.get('tbv_per_share')
        if tbv_ps and price > 0:
            pb_tangible = price / tbv_ps if tbv_ps > 0 else 99
            if pb_tangible < 1.0:
                score += 1; details.append(f"P/TBV: {pb_tangible:.2f} (below assets)")

        if pe > 0:
            if pe < 10:
                score += 2; details.append(f"P/E: {pe:.1f} (cheap)")
            elif pe < 15:
                score += 1; details.append(f"P/E: {pe:.1f} (reasonable)")
            else:
                details.append(f"P/E: {pe:.1f} (expensive)")

        if _is_financial(sector):
            # current ratio / debt<0.5 do not apply to banks (no working capital
            # in the usual sense, leverage is regulatory). Neutral point.
            score += 1; details.append("Liquidity: N/A (financials, regulatory)")
        else:
            if current_ratio > 2:
                score += 1; details.append(f"Liquidity: {current_ratio:.1f}x")
            elif fund.get('quick_ratio', 0) > 1.5:
                score += 1; details.append(f"Quick: {fund['quick_ratio']:.1f}x")
            if debt < 0.5:
                score += 1

        wc = fund.get('working_capital', 0)
        if wc and wc > 0:
            total_debt_val = fund.get('total_debt', 0)
            if total_debt_val > 0 and wc > total_debt_val:
                details.append("WC > Total Debt (safe)")

        desc = " | ".join(details[:4]) if details else "No fundamental data"
        if score >= 5:
            return "[OK] BUY", desc, 2
        elif score >= 2:
            return "[--] FAIR", desc, 1
        else:
            return "[OFF] EXP", desc, 0

    if tech:
        pct = tech['pct_52w']
        if pct < 25:
            return "[OK] CHEAP", f"52W: {pct:.0f}% (near lows - margin of safety)", 2
        elif pct < 60:
            return "[--] FAIR", f"52W: {pct:.0f}% (mid-range)", 1
        else:
            return "[OFF] EXP", f"52W: {pct:.0f}% (near highs)", 0

    return "[--] N/A", "No data", 0


def munger_analysis(fund: dict | None, tech: dict | None,
                    *, sector: str | None = None) -> tuple[str, str, int]:
    """Charlie Munger - Inversion: what can go wrong? Deep risk detection."""
    risks = []
    score = 0
    financial = _is_financial(sector)

    if fund:
        pe = fund.get('pe', 0)
        debt = fund.get('debt_equity', 0)
        roe = fund.get('roe', 0)
        margin = fund.get('profit_margin', 0)

        # For banks high debt/leverage is normal (deposits), not a risk flag.
        if not financial:
            if debt > 3:
                score += 2; risks.append(f"Debt: {debt:.1f}x (dangerous)")
            elif debt > 1.5:
                score += 1; risks.append(f"Debt: {debt:.1f}x (elevated)")

        if roe < 0:
            score += 2; risks.append(f"ROE: {roe:.0%} (loss-making)")
        elif roe < 0.05:
            score += 1; risks.append(f"ROE: {roe:.0%} (low return)")

        if pe > 50:
            score += 1; risks.append(f"P/E: {pe:.0f} (bubble?)")

        if margin < 0:
            score += 2; risks.append(f"Margin: {margin:.0%} (loss)")
        elif 0 < margin < 0.05:
            score += 1; risks.append(f"Margin: {margin:.0%} (thin)")

        rev_qoq = fund.get('revenue_qoq', 0)
        if rev_qoq < -0.10:
            score += 2; risks.append(f"Revenue QoQ: {rev_qoq:.0%} (declining)")
        elif rev_qoq < -0.03:
            score += 1; risks.append(f"Revenue QoQ: {rev_qoq:.0%}")

        ni_qoq = fund.get('net_income_qoq', 0)
        if ni_qoq < -0.30:
            score += 1; risks.append(f"Profit QoQ: {ni_qoq:.0%} (collapse)")

        fcf = fund.get('fcf', 0)
        if fcf and fcf < 0:
            score += 1; risks.append("FCF negative (burning cash)")
        fcf_streak = fund.get('fcf_positive_streak', None)
        if fcf_streak is not None and fcf_streak <= 1:
            score += 1; risks.append(f"FCF positive only {fcf_streak}/4 q")

        total_assets = fund.get('total_assets', 0)
        total_liab = fund.get('total_liabilities', 0)
        if not financial and total_assets > 0 and total_liab > 0:
            # A bank's liabilities are ~90% of assets (deposits) - normal, not risk.
            liab_ratio = total_liab / total_assets
            if liab_ratio > 0.85:
                score += 1; risks.append(f"Liabilities: {liab_ratio:.0%} of assets")

        cash = fund.get('cash', 0)
        total_debt_val = fund.get('total_debt', 0)
        if total_debt_val > 0 and cash > 0 and cash < total_debt_val * 0.1:
            score += 1; risks.append("Cash < 10% of debt")

        payout = fund.get('payout_ratio', 0)
        if payout > 1.0:
            score += 1; risks.append(f"Payout: {payout:.0%} (pays more than earnings)")

    if tech:
        if tech['rsi'] > 80:
            score += 2; risks.append(f"RSI: {tech['rsi']:.0f} (overheated)")
        elif tech['rsi'] > 70:
            score += 1; risks.append(f"RSI: {tech['rsi']:.0f} (overbought)")
        elif tech['rsi'] < 25:
            score += 1; risks.append(f"RSI: {tech['rsi']:.0f} (panic)")

        if tech['vol_30d'] > 0.60:
            score += 1; risks.append(f"Vol: {tech['vol_30d']:.0%} (storm)")

        if tech['pct_52w'] > 90 and tech['rsi'] > 65:
            score += 1; risks.append("52W max + RSI high")

    if score == 0:
        desc = "No risks detected"
        if tech:
            desc += f" | RSI: {tech['rsi']:.0f} | Vol: {tech['vol_30d']:.0%}"
        return "[OK] CLEAN", desc, 2

    desc = " | ".join(risks[:4])
    if score >= 5:
        return "[!!] DANGER", desc, 0
    elif score >= 2:
        return "[!] WARNING", desc, 1
    else:
        return "[OK] MINOR", desc, 2


_EQUAL_WEIGHTS = {"lynch": 1.0, "buffett": 1.0, "graham": 1.0, "munger": 1.0}


def council_weights_from_log(db_path: str | None = None, *,
                             horizon: str = "5d", min_rows: int = 40) -> dict[str, float]:
    """Per-guru weights calibrated from guru_log forward returns.

    For each guru, measure the spread in forward ret_{horizon} between its
    high-conviction calls (score 2) and the rest (score <= 1). A guru whose '2'
    reliably precedes higher returns earns more weight; one with no edge is
    down-weighted. Weights are normalised so the mean is 1.0 (sum 4), preserving
    the 0-8 council scale. Falls back to equal weights if guru_log is missing or
    too thin - so the council degrades gracefully before any track record exists.
    """
    try:
        path = db_path or os.path.join(os.path.dirname(__file__), os.pardir, "market.db")
        if not os.path.exists(path):
            return dict(_EQUAL_WEIGHTS)
        import sqlite3
        col = f"ret_{horizon}"
        con = sqlite3.connect(path)
        try:
            df = pd.read_sql(
                f"SELECT lynch_score, buffett_score, graham_score, munger_score, {col} "
                f"FROM guru_log WHERE {col} IS NOT NULL", con)
        finally:
            con.close()
        if len(df) < min_rows:
            return dict(_EQUAL_WEIGHTS)
        edges = {}
        for g in ("lynch", "buffett", "graham", "munger"):
            s = df[f"{g}_score"]
            hi = df.loc[s >= 2, col].mean()
            lo = df.loc[s <= 1, col].mean()
            edges[g] = max(0.0, hi - lo) if pd.notna(hi) and pd.notna(lo) else 0.0
        total = sum(edges.values())
        if total <= 0:
            return dict(_EQUAL_WEIGHTS)
        return {g: 4.0 * e / total for g, e in edges.items()}
    except Exception:
        return dict(_EQUAL_WEIGHTS)


def get_guru_analysis(
    fund: dict | None,
    tech: dict | None,
    *,
    sector: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """Run all 4 gurus and aggregate the council vote.

    Args:
        fund: Fundamental data dict (from any source); None means technical-only.
        tech: Technical context dict (from technical_context()).
        sector: ASSET_TYPES group (e.g. "US FINANCE") - relaxes debt/liquidity
                gates for financials.
        weights: Per-guru weights (see council_weights_from_log); equal if None.

    Aggregation adds, over a plain sum: weighted consensus, a Munger DANGER veto
    on BUY, and a confidence discount when the verdict rests on price alone.

    Returns dict with lynch/buffett/graham/munger + council + data_source.
    """
    lynch_status, lynch_desc, lynch_score = lynch_analysis(fund, tech)
    buffett_status, buffett_desc, buffett_score = buffett_analysis(fund, tech, sector=sector)
    graham_status, graham_desc, graham_score = graham_analysis(fund, tech, sector=sector)
    munger_status, munger_desc, munger_score = munger_analysis(fund, tech, sector=sector)

    w = weights or _EQUAL_WEIGHTS
    scores = {"lynch": lynch_score, "buffett": buffett_score,
              "graham": graham_score, "munger": munger_score}
    weighted = sum(scores[g] * w.get(g, 1.0) for g in scores)
    max_score = 2.0 * sum(w.get(g, 1.0) for g in scores)
    consensus_pct = (weighted / max_score * 100) if max_score > 0 else 0.0

    # Technical-only verdict (no fundamentals) is less reliable than one built on
    # PEG/ROE/Graham#, so its confidence is shaved before the thresholds.
    is_technical = not fund
    if is_technical:
        consensus_pct *= TECH_CONFIDENCE_DISCOUNT

    if consensus_pct >= 75:
        verdict, color, label = "BUY", "green", "STRONG BUY"
    elif consensus_pct >= 50:
        verdict, color, label = "HOLD", "orange", "HOLD"
    else:
        verdict, color, label = "AVOID", "red", "AVOID"

    # Munger veto: the inversion (risk) guru must not be outvoted by the three
    # bullish value screens. A DANGER call blocks a BUY (caps it at HOLD).
    vetoed = munger_status.startswith("[!!]")
    if vetoed and verdict == "BUY":
        verdict, color, label = "HOLD", "orange", "HOLD (Munger veto: DANGER)"

    note = " | tech-only" if is_technical else ""
    council = {"verdict": verdict, "pct": consensus_pct, "color": color,
               "vetoed": vetoed,
               "text": f"{label} ({consensus_pct:.0f}%){note}"}

    data_source = fund.get('source', 'technical') if fund else 'technical'

    return {
        'lynch':   {'status': lynch_status,   'desc': lynch_desc},
        'buffett': {'status': buffett_status, 'desc': buffett_desc},
        'graham':  {'status': graham_status,  'desc': graham_desc},
        'munger':  {'status': munger_status,  'desc': munger_desc},
        'council': council,
        'data_source': data_source,
    }
