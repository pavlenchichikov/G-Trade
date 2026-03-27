"""Guru Council V2 — Fundamental analysis from 4 investment philosophies.

Each guru returns (status, description, score 0-2):
  • Lynch  — GARP: PEG ratio + revenue trend
  • Buffett — Quality moats: ROE, FCF, low debt, margins
  • Graham  — Deep value: NCAV, Graham Number, P/E, balance sheet
  • Munger  — Inversion: what can go wrong? Risk detection
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


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


# ── Individual Guru Algorithms ───────────────────────────────────────────────

def lynch_analysis(fund: dict | None, tech: dict | None) -> tuple[str, str, int]:
    """Peter Lynch — Growth At Reasonable Price (GARP)."""
    if fund:
        pe = fund.get('pe', 0)
        growth = fund.get('growth', 0)

        peg = fund.get('peg_ratio') or None
        if not peg or peg <= 0:
            peg = calc_peg(pe, growth)

        rev_bonus = ""
        rev_qoq = fund.get('revenue_qoq', 0)
        if rev_qoq > 0.10:
            rev_bonus = f" | Rev QoQ: +{rev_qoq:.0%} (ускорение)"
        elif rev_qoq > 0:
            rev_bonus = f" | Rev QoQ: +{rev_qoq:.0%}"

        earnings_flag = ""
        if fund.get('earnings_stable') is False:
            earnings_flag = " | WARN: нестаб. прибыль"

        if peg is not None and peg > 0:
            if peg < 1.0:
                return "[OK] BUY", f"PEG: {peg:.2f} (рост недооценён) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}{earnings_flag}", 2
            elif peg < 2.0:
                return "[--] FAIR", f"PEG: {peg:.2f} (справедливо) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}{earnings_flag}", 1
            else:
                return "[OFF] EXP", f"PEG: {peg:.2f} (переоценён) | P/E: {pe:.1f} | Growth: {growth:.0%}{rev_bonus}", 0
        elif pe > 0:
            adj = ""
            if rev_qoq > 0.15:
                adj = " (но выручка растёт!)"
            if pe < 12:
                return "[OK] CHEAP", f"P/E: {pe:.1f} (нет данных роста){adj}{rev_bonus}", 2
            elif pe < 25:
                return "[--] FAIR", f"P/E: {pe:.1f} (нет данных роста){rev_bonus}", 1
            else:
                return "[OFF] EXP", f"P/E: {pe:.1f} (дорого){rev_bonus}", 0

    if tech:
        score = int(tech['above_50']) + int(tech['above_200'])
        sma_txt = f"SMA50: {'▲' if tech['above_50'] else '▼'}  SMA200: {'▲' if tech['above_200'] else '▼'}"
        if score == 2:
            return "[OK] MOMENTUM", sma_txt, 2
        elif score == 1:
            return "[--] SIDEWAYS", sma_txt, 1
        else:
            return "[OFF] DOWNTREND", sma_txt, 0

    return "[--] N/A", "Нет данных", 0


def buffett_analysis(fund: dict | None, tech: dict | None) -> tuple[str, str, int]:
    """Warren Buffett — Quality moats, ROE, FCF, low debt, retained earnings."""
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
            score += 2; details.append(f"ROE: {roe:.0%} (отлично)")
        elif roe > 0.15:
            score += 1; details.append(f"ROE: {roe:.0%} (хорошо)")
        else:
            details.append(f"ROE: {roe:.0%} (слабо)")

        if debt < 0.5:
            score += 2; details.append(f"Долг: {debt:.1f}x (минимальный)")
        elif debt < 1.5:
            score += 1; details.append(f"Долг: {debt:.1f}x (умеренный)")
        else:
            details.append(f"Долг: {debt:.1f}x (высокий)")

        if gross_margin > 0.40:
            score += 1; details.append(f"Gross: {gross_margin:.0%} (моат)")
        elif margin > 0.20:
            score += 1; details.append(f"Маржа: {margin:.0%}")
        elif margin > 0:
            details.append(f"Маржа: {margin:.0%}")

        if div_yield > 3:
            score += 1; details.append(f"Дивы: {div_yield:.1f}%")
        elif div_yield > 0:
            details.append(f"Дивы: {div_yield:.1f}%")

        if fcf > 0:
            score += 1
            fcf_streak = fund.get('fcf_positive_streak', 0)
            if fcf_streak >= 4:
                score += 1; details.append("FCF: 4/4 кв. полож.")
            p2fcf = fund.get('price_to_fcf', 0)
            if 0 < p2fcf < 20:
                details.append(f"P/FCF: {p2fcf:.1f}")

        retained = fund.get('retained_earnings', 0)
        if retained and retained > 0:
            score += 1

        cash = fund.get('cash', 0)
        total_debt = fund.get('total_debt', 0)
        if cash > 0 and total_debt > 0 and cash > total_debt * 0.5:
            details.append(f"Cash покрывает {cash/total_debt:.0%} долга")

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
        desc = f"SMA200: {'+' if trend_ok else '-'} | RSI: {tech['rsi']:.0f} ({'норма' if rsi_ok else 'экстрем'})"
        if s == 2:
            return "[OK] STABLE", desc, 1
        elif s == 1:
            return "[--] MIXED", desc, 0
        else:
            return "[!] WEAK", desc, 0

    return "[--] N/A", "Нет данных", 0


def graham_analysis(fund: dict | None, tech: dict | None) -> tuple[str, str, int]:
    """Benjamin Graham — Deep value, margin of safety, Graham Number, NCAV."""
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
                score += 3; details.append(f"NCAV/sh: ${ncav_ps:.0f} > цена (ликвидац. стоимость выше!)")
            elif ncav_margin > -30:
                score += 1; details.append(f"NCAV/sh: ${ncav_ps:.0f} ({ncav_margin:+.0f}%)")

        gn = calc_graham_number(eps, bv)
        if gn and price > 0:
            margin = (gn - price) / price * 100
            if margin > 20:
                score += 2; details.append(f"Graham#: ${gn:.0f} (запас {margin:.0f}%)")
            elif margin > 0:
                score += 1; details.append(f"Graham#: ${gn:.0f} (запас {margin:.0f}%)")
            else:
                details.append(f"Graham#: ${gn:.0f} (выше цены на {-margin:.0f}%)")

        tbv_ps = fund.get('tbv_per_share')
        if tbv_ps and price > 0:
            pb_tangible = price / tbv_ps if tbv_ps > 0 else 99
            if pb_tangible < 1.0:
                score += 1; details.append(f"P/TBV: {pb_tangible:.2f} (ниже актива)")

        if pe > 0:
            if pe < 10:
                score += 2; details.append(f"P/E: {pe:.1f} (дешёво)")
            elif pe < 15:
                score += 1; details.append(f"P/E: {pe:.1f} (разумно)")
            else:
                details.append(f"P/E: {pe:.1f} (дорого)")

        if current_ratio > 2:
            score += 1; details.append(f"Ликвидность: {current_ratio:.1f}x")
        elif fund.get('quick_ratio', 0) > 1.5:
            score += 1; details.append(f"Quick: {fund['quick_ratio']:.1f}x")
        if debt < 0.5:
            score += 1

        wc = fund.get('working_capital', 0)
        if wc and wc > 0:
            total_debt_val = fund.get('total_debt', 0)
            if total_debt_val > 0 and wc > total_debt_val:
                details.append("WC > Total Debt (безопасно)")

        desc = " | ".join(details[:4]) if details else "Нет фундам. данных"
        if score >= 5:
            return "[OK] BUY", desc, 2
        elif score >= 2:
            return "[--] FAIR", desc, 1
        else:
            return "[OFF] EXP", desc, 0

    if tech:
        pct = tech['pct_52w']
        if pct < 25:
            return "[OK] CHEAP", f"52W: {pct:.0f}% (у минимумов — запас прочности)", 2
        elif pct < 60:
            return "[--] FAIR", f"52W: {pct:.0f}% (середина)", 1
        else:
            return "[OFF] EXP", f"52W: {pct:.0f}% (у максимумов)", 0

    return "[--] N/A", "Нет данных", 0


def munger_analysis(fund: dict | None, tech: dict | None) -> tuple[str, str, int]:
    """Charlie Munger — Inversion: what can go wrong? Deep risk detection."""
    risks = []
    score = 0

    if fund:
        pe = fund.get('pe', 0)
        debt = fund.get('debt_equity', 0)
        roe = fund.get('roe', 0)
        margin = fund.get('profit_margin', 0)

        if debt > 3:
            score += 2; risks.append(f"Долг: {debt:.1f}x (опасно)")
        elif debt > 1.5:
            score += 1; risks.append(f"Долг: {debt:.1f}x (повышенный)")

        if roe < 0:
            score += 2; risks.append(f"ROE: {roe:.0%} (убыточность)")
        elif roe < 0.05:
            score += 1; risks.append(f"ROE: {roe:.0%} (низкая отдача)")

        if pe > 50:
            score += 1; risks.append(f"P/E: {pe:.0f} (пузырь?)")

        if margin < 0:
            score += 2; risks.append(f"Маржа: {margin:.0%} (убыток)")
        elif 0 < margin < 0.05:
            score += 1; risks.append(f"Маржа: {margin:.0%} (тонкая)")

        rev_qoq = fund.get('revenue_qoq', 0)
        if rev_qoq < -0.10:
            score += 2; risks.append(f"Выручка QoQ: {rev_qoq:.0%} (падение)")
        elif rev_qoq < -0.03:
            score += 1; risks.append(f"Выручка QoQ: {rev_qoq:.0%}")

        ni_qoq = fund.get('net_income_qoq', 0)
        if ni_qoq < -0.30:
            score += 1; risks.append(f"Прибыль QoQ: {ni_qoq:.0%} (обвал)")

        fcf = fund.get('fcf', 0)
        if fcf and fcf < 0:
            score += 1; risks.append("FCF отриц. (жжёт кэш)")
        fcf_streak = fund.get('fcf_positive_streak', None)
        if fcf_streak is not None and fcf_streak <= 1:
            score += 1; risks.append(f"FCF полож. только {fcf_streak}/4 кв.")

        total_assets = fund.get('total_assets', 0)
        total_liab = fund.get('total_liabilities', 0)
        if total_assets > 0 and total_liab > 0:
            liab_ratio = total_liab / total_assets
            if liab_ratio > 0.85:
                score += 1; risks.append(f"Обязательства: {liab_ratio:.0%} активов")

        cash = fund.get('cash', 0)
        total_debt_val = fund.get('total_debt', 0)
        if total_debt_val > 0 and cash > 0 and cash < total_debt_val * 0.1:
            score += 1; risks.append("Кэш < 10% долга")

        payout = fund.get('payout_ratio', 0)
        if payout > 1.0:
            score += 1; risks.append(f"Payout: {payout:.0%} (платит больше прибыли)")

    if tech:
        if tech['rsi'] > 80:
            score += 2; risks.append(f"RSI: {tech['rsi']:.0f} (перегрев)")
        elif tech['rsi'] > 70:
            score += 1; risks.append(f"RSI: {tech['rsi']:.0f} (перекуплен)")
        elif tech['rsi'] < 25:
            score += 1; risks.append(f"RSI: {tech['rsi']:.0f} (паника)")

        if tech['vol_30d'] > 0.60:
            score += 1; risks.append(f"Vol: {tech['vol_30d']:.0%} (шторм)")

        if tech['pct_52w'] > 90 and tech['rsi'] > 65:
            score += 1; risks.append("52W max + RSI высокий")

    if score == 0:
        desc = "Рисков не обнаружено"
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


def get_guru_analysis(
    fund: dict | None,
    tech: dict | None,
) -> dict:
    """Run all 4 gurus and aggregate council vote.

    Args:
        fund: Fundamental data dict (from any source)
        tech: Technical context dict (from technical_context())

    Returns dict with lynch/buffett/graham/munger + council_vote.
    """
    lynch_status, lynch_desc, lynch_score = lynch_analysis(fund, tech)
    buffett_status, buffett_desc, buffett_score = buffett_analysis(fund, tech)
    graham_status, graham_desc, graham_score = graham_analysis(fund, tech)
    munger_status, munger_desc, munger_score = munger_analysis(fund, tech)

    total_score = lynch_score + buffett_score + graham_score + munger_score
    max_score = 8
    consensus_pct = total_score / max_score * 100

    if consensus_pct >= 75:
        council = {"verdict": "BUY", "pct": consensus_pct, "color": "green",
                   "text": f"STRONG BUY ({consensus_pct:.0f}% — {total_score}/{max_score})"}
    elif consensus_pct >= 50:
        council = {"verdict": "HOLD", "pct": consensus_pct, "color": "orange",
                   "text": f"HOLD ({consensus_pct:.0f}% — {total_score}/{max_score})"}
    else:
        council = {"verdict": "AVOID", "pct": consensus_pct, "color": "red",
                   "text": f"AVOID ({consensus_pct:.0f}% — {total_score}/{max_score})"}

    data_source = fund.get('source', 'technical') if fund else 'technical'

    return {
        'lynch':   {'status': lynch_status,   'desc': lynch_desc},
        'buffett': {'status': buffett_status, 'desc': buffett_desc},
        'graham':  {'status': graham_status,  'desc': graham_desc},
        'munger':  {'status': munger_status,  'desc': munger_desc},
        'council': council,
        'data_source': data_source,
    }
