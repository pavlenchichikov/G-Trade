"""Builds text for Telegram: digest and command replies.

Pure functions only: data comes in as arguments, ready-made text goes out.
"""


def _fmt_acc(acc: dict) -> str:
    if not acc or not acc.get("n"):
        return "no stats yet"
    return f"{acc['acc']:.0%} ({acc['correct']}/{acc['n']})"


def _sorted_actionable(signals: list) -> list:
    """BUY/SELL signals, sorted by confidence (further from 0.5 = higher)."""
    actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    return sorted(actionable, key=lambda s: abs((s.get("probability") or 0.5) - 0.5),
                  reverse=True)


def build_top_message(signals: list, n: int = 5) -> str:
    top = _sorted_actionable(signals)[:n]
    if not top:
        return "No active signals - everything is WAIT."
    lines = [f"Top {len(top)} signals:"]
    for s in top:
        lines.append(
            f"{s['signal']:<4} {s['asset']:<8} p={s['probability']:.2f}"
            f"  accuracy: {_fmt_acc(s.get('acc'))}"
        )
    return "\n".join(lines)


def build_signal_message(asset: str, track: list, acc: dict) -> str:
    if not track:
        return f"{asset}: no signal history yet. Run predict.py."
    cur = track[0]
    lines = [
        f"{asset}: {cur['signal']} (p={cur['probability']:.2f}) from {cur['date']}",
        f"Accuracy of last {acc['n']}: {_fmt_acc(acc)}",
        "",
        "History:",
    ]
    for t in track[:10]:
        if t["correct"] is None:
            outcome = "-"
        else:
            outcome = "+" if t["correct"] else "x"
        ret = f" {t['actual_next_ret']:+.2%}" if t["actual_next_ret"] is not None else ""
        lines.append(f"{t['date']}  {t['signal']:<4} p={t['probability']:.2f}  {outcome}{ret}")
    return "\n".join(lines)


def build_risk_message(risk_state, config: dict) -> str:
    lines = ["Risk status"]
    if risk_state:
        cap = risk_state.get("current_capital", 0.0)
        peak = risk_state.get("peak_capital", cap) or cap
        dd = (cap - peak) / peak if peak else 0.0
        positions = risk_state.get("open_positions") or {}
        lines += [
            f"Capital: ${cap:,.2f}",
            f"Drawdown from peak: {dd:.1%}",
            f"Open positions: {len(positions)}"
            + (f" ({', '.join(positions)})" if positions else ""),
        ]
    else:
        lines.append("State not saved (risk_state.json missing) - default limits:")
    lines += [
        f"Daily loss limit: {config.get('max_daily_loss', 0):.0%}",
        f"Drawdown halt: {config.get('max_drawdown_halt', 0):.0%}",
    ]
    return "\n".join(lines)


def build_digest(signals: list, stale: list, risk, date_str: str) -> str:
    """Morning digest: top signals, risk, stale data."""
    parts = [f"Digest {date_str}", ""]

    top = _sorted_actionable(signals)[:5]
    if top:
        parts.append("Signals:")
        for s in top:
            parts.append(
                f"{s['signal']:<4} {s['asset']:<8} p={s['probability']:.2f}"
                f"  accuracy: {_fmt_acc(s.get('acc'))}"
            )
    else:
        parts.append("No active signals.")

    parts.append("")
    if risk:
        cap = risk.get("current_capital", 0.0)
        peak = risk.get("peak_capital", cap) or cap
        dd = (cap - peak) / peak if peak else 0.0
        parts.append(f"Capital: ${cap:,.2f} (drawdown {dd:.1%}), "
                     f"positions: {len(risk.get('open_positions') or {})}")
    else:
        parts.append("Risk status: state not saved.")

    if stale:
        parts.append("")
        parts.append("Stale data:")
        for s in stale[:10]:
            if s["last_date"] is None:
                parts.append(f"{s['asset']}: no data")
            else:
                parts.append(f"{s['asset']}: last update {s['last_date']} ({s['age_days']} days ago)")
        if len(stale) > 10:
            parts.append(f"...and {len(stale) - 10} more")

    return "\n".join(parts)
