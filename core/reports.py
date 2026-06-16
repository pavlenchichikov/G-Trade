"""Сборка текстов для Telegram: дайджест и ответы на команды.

Только чистые функции: данные приходят аргументами, наружу - готовый текст.
"""


def _fmt_acc(acc: dict) -> str:
    if not acc or not acc.get("n"):
        return "нет статистики"
    return f"{acc['acc']:.0%} ({acc['correct']}/{acc['n']})"


def _sorted_actionable(signals: list) -> list:
    """BUY/SELL, отсортированные по уверенности (дальше от 0.5 - выше)."""
    actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    return sorted(actionable, key=lambda s: abs((s.get("probability") or 0.5) - 0.5),
                  reverse=True)


def build_top_message(signals: list, n: int = 5) -> str:
    top = _sorted_actionable(signals)[:n]
    if not top:
        return "Активных сигналов нет - везде WAIT."
    lines = [f"Топ-{len(top)} сигналов:"]
    for s in top:
        lines.append(
            f"{s['signal']:<4} {s['asset']:<8} p={s['probability']:.2f}"
            f"  точность: {_fmt_acc(s.get('acc'))}"
        )
    return "\n".join(lines)


def build_signal_message(asset: str, track: list, acc: dict) -> str:
    if not track:
        return f"{asset}: истории сигналов нет. Запусти predict.py."
    cur = track[0]
    lines = [
        f"{asset}: {cur['signal']} (p={cur['probability']:.2f}) от {cur['date']}",
        f"Точность последних {acc['n']}: {_fmt_acc(acc)}",
        "",
        "История:",
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
    lines = ["Риск-статус"]
    if risk_state:
        cap = risk_state.get("current_capital", 0.0)
        peak = risk_state.get("peak_capital", cap) or cap
        dd = (cap - peak) / peak if peak else 0.0
        positions = risk_state.get("open_positions") or {}
        lines += [
            f"Капитал: ${cap:,.2f}",
            f"Просадка от пика: {dd:.1%}",
            f"Открытых позиций: {len(positions)}"
            + (f" ({', '.join(positions)})" if positions else ""),
        ]
    else:
        lines.append("Состояние не сохранено (risk_state.json нет) - лимиты по умолчанию:")
    lines += [
        f"Лимит дневного убытка: {config.get('max_daily_loss', 0):.0%}",
        f"Стоп по просадке: {config.get('max_drawdown_halt', 0):.0%}",
    ]
    return "\n".join(lines)


def build_digest(signals: list, stale: list, risk, date_str: str) -> str:
    """Утренний дайджест: топ сигналов, риск, протухшие данные."""
    parts = [f"Дайджест {date_str}", ""]

    top = _sorted_actionable(signals)[:5]
    if top:
        parts.append("Сигналы:")
        for s in top:
            parts.append(
                f"{s['signal']:<4} {s['asset']:<8} p={s['probability']:.2f}"
                f"  точность: {_fmt_acc(s.get('acc'))}"
            )
    else:
        parts.append("Активных сигналов нет.")

    parts.append("")
    if risk:
        cap = risk.get("current_capital", 0.0)
        peak = risk.get("peak_capital", cap) or cap
        dd = (cap - peak) / peak if peak else 0.0
        parts.append(f"Капитал: ${cap:,.2f} (просадка {dd:.1%}), "
                     f"позиций: {len(risk.get('open_positions') or {})}")
    else:
        parts.append("Риск-статус: состояние не сохранено.")

    if stale:
        parts.append("")
        parts.append("Протухшие данные:")
        for s in stale[:10]:
            if s["last_date"] is None:
                parts.append(f"{s['asset']}: данных нет")
            else:
                parts.append(f"{s['asset']}: последние от {s['last_date']} ({s['age_days']} дн.)")
        if len(stale) > 10:
            parts.append(f"...и ещё {len(stale) - 10}")

    return "\n".join(parts)
