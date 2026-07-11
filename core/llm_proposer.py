"""The LLM layer of the auto-research agent: prompts, providers, parsing,
retries. auto_research.py only calls the public functions here; this module
never imports auto_research (no cycle) and returns plain dicts.

Providers (GTRADE_AR_LLM): anthropic (default), openai (or any OpenAI-
compatible endpoint via GTRADE_AR_LLM_BASE_URL), ollama (local; added in the
ollama task). SDK imports happen inside the call functions so the module
imports cleanly without them."""

import json
import os


DSL_MENU = (
    "ops: zscore(window 2-200), ratio(a,b), lag(k 1-20), diff(k 1-20), "
    "rolling(window,agg in mean|std|sum), interaction(a,b), lead_lag(leader in "
    "sp500|vix|btc|gold|dxy|tnx, horizon 1-20). Each spec: "
    '{"name": lower_snake, "op": ..., "inputs": [...], "params": {...}}.'
)


def llm_selected():
    """Whether the user picked the LLM proposer (GTRADE_AR_PROPOSER=llm)."""
    return (os.getenv("GTRADE_AR_PROPOSER") or "evolutionary").strip().lower() == "llm"


def _proposer_prompt(log, base_features):
    """The shared features-axis prompt for any LLM provider."""
    history = json.dumps(log[-8:], ensure_ascii=True)
    return (
        "You are proposing engineered features for a trading model to revive weak "
        "neural members. Use ONLY this DSL.\n" + DSL_MENU +
        "\nBase columns you can reference: " + ",".join(base_features) +
        "\nPast experiments (spec + held-back selection Score deltas):\n" + history +
        "\nReturn STRICT JSON: a list of 1-2 new spec dicts, no prose."
    )


def _avoid_clause(avoid):
    """A prompt line listing already-tried candidates so the model proposes something
    novel. Empty string when there is nothing to avoid, so the prompt is unchanged."""
    if not avoid:
        return ""
    return ("\nAlready tried (do NOT repeat these - propose something genuinely "
            "different): " + json.dumps(list(avoid)[-40:], ensure_ascii=True))


def _parse_specs(text):
    """Extract the JSON list of specs from a model reply, tolerant of stray prose."""
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        specs = json.loads(text[start:end + 1])
    except Exception:
        return []
    return specs if isinstance(specs, list) else []


def _call_anthropic(prompt):
    """Anthropic SDK. Model via GTRADE_AR_LLM_MODEL (default claude-opus-4-8)."""
    import anthropic
    client = anthropic.Anthropic()
    model = os.getenv("GTRADE_AR_LLM_MODEL") or "claude-opus-4-8"
    last_err = None
    for _attempt in range(3):
        try:
            msg = client.messages.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError("anthropic proposer failed after 3 attempts: %s" % last_err)


def _call_openai(prompt):
    """OpenAI-compatible chat API. Works with OpenAI and any compatible endpoint
    (Mistral, LM Studio, etc.) via GTRADE_AR_LLM_BASE_URL. Model via
    GTRADE_AR_LLM_MODEL (default gpt-4o)."""
    import openai
    client = openai.OpenAI(base_url=os.getenv("GTRADE_AR_LLM_BASE_URL") or None)
    model = os.getenv("GTRADE_AR_LLM_MODEL") or "gpt-4o"
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError("openai proposer failed after 3 attempts: %s" % last_err)


def _ollama_base_url():
    return os.getenv("GTRADE_AR_LLM_BASE_URL") or "http://localhost:11434/v1"


def list_ollama_models():
    """Every model installed in the local Ollama, newest-first as Ollama returns
    them, via its native tags endpoint (the OpenAI-compatible /v1 API has no model
    listing). Raises RuntimeError if Ollama is unreachable."""
    import urllib.request
    base = _ollama_base_url()
    host = base[:-3] if base.endswith("/v1") else base
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        raise RuntimeError(
            "cannot reach Ollama at %s (is Ollama running?): %s" % (url, exc))
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def _detect_ollama_model():
    """The installed model to use when GTRADE_AR_LLM_MODEL is not set: the first
    gemma* model, else the first installed model (any local model works)."""
    names = list_ollama_models()
    if not names:
        raise RuntimeError("no Ollama models installed; run: ollama pull gemma3")
    gemma = [n for n in names if n.lower().startswith("gemma")]
    return gemma[0] if gemma else names[0]


def _print_ollama_models():
    """Print installed Ollama models as a numbered list for the launcher menu.
    Never raises: an unreachable Ollama prints a friendly note instead."""
    try:
        names = list_ollama_models()
    except RuntimeError as exc:
        print("  (could not list local models: %s)" % exc)
        return
    if not names:
        print("  (no Ollama models installed; run: ollama pull gemma3)")
        return
    for i, name in enumerate(names, 1):
        print("  [%d] %s" % (i, name))


def _call_ollama(prompt):
    """Local Ollama via its OpenAI-compatible API. Base URL via
    GTRADE_AR_LLM_BASE_URL (default localhost:11434/v1); model via
    GTRADE_AR_LLM_MODEL or auto-detected (gemma preferred)."""
    import openai
    base = _ollama_base_url()
    model = os.getenv("GTRADE_AR_LLM_MODEL") or _detect_ollama_model()
    # Reasoning models (e.g. gemma) spend tokens on an internal reasoning trace before
    # the answer; a small cap gets fully consumed by reasoning and returns EMPTY content
    # (the silent cause of a wiki/proposer that "runs" but produces nothing). Budget
    # generously; override with GTRADE_AR_LLM_MAX_TOKENS.
    try:
        max_toks = int(os.getenv("GTRADE_AR_LLM_MAX_TOKENS") or "8000")
    except ValueError:
        max_toks = 8000
    client = openai.OpenAI(base_url=base, api_key="ollama")
    last_err = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_toks,
                messages=[{"role": "user", "content": prompt}])
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        "ollama proposer failed after 3 attempts (is Ollama running at %s?): %s"
        % (base, last_err))


def _backend():
    """The provider call function for GTRADE_AR_LLM, resolved at call time so
    tests can monkeypatch the _call_* functions."""
    provider = (os.getenv("GTRADE_AR_LLM") or "anthropic").strip().lower()
    backends = {"anthropic": _call_anthropic, "openai": _call_openai,
                "ollama": _call_ollama}
    fn = backends.get(provider)
    if fn is None:
        raise RuntimeError(
            "unknown GTRADE_AR_LLM %r (use anthropic, openai or ollama)" % provider)
    return fn


def reflect_on():
    """GTRADE_AR_REFLECT: run a 'reflect then propose' step on the LLM path (default OFF)."""
    return (os.getenv("GTRADE_AR_REFLECT") or "").strip() in ("1", "true", "True")


def _wiki_preamble():
    """The compounding research wiki as a prompt preamble when GTRADE_AR_WIKI is on;
    '' otherwise (so the prompt is unchanged). Any error yields ''."""
    try:
        from core import ar_wiki
        if not ar_wiki.wiki_on():
            return ""
        text = ar_wiki.wiki_summary()
        if not text:
            return ""
        return "Accumulated research wiki (distilled prior findings):\n" + text + "\n"
    except Exception:
        return ""


def _reflect_hypothesis():
    """One-line hypothesis of why recent experiments did not clear the gate, from the
    findings journal. Empty string when reflection is off, the journal is empty, or any
    error - so the caller's prompt is unchanged in those cases."""
    if not reflect_on():
        return ""
    try:
        from core import ar_memory
        recent = ar_memory.findings_recent(5)
        if not recent:
            return ""
        prompt = (
            "Here are recent auto-research experiments and whether they cleared the "
            "held-out gate:\n" + json.dumps(recent, ensure_ascii=True)[:4000] +
            "\nIn ONE sentence, hypothesize why they did not improve the model. No prose.")
        return (_backend()(prompt) or "").strip()
    except Exception:
        return ""


GENOME_MENU = (
    'A genome is JSON: {"drops": [features to drop], "extra": [spec dicts], '
    '"label_mode": "direction" or "rel_median", "label_window": 20 or 30 or 60}. '
    "extra specs use this DSL: " + DSL_MENU
)


def _parse_obj(text):
    """Extract ONE JSON object from a model reply, tolerant of stray prose."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def propose_genome(parent, elites, active, base_features, avoid=None):
    """Ask the LLM for ONE modified experiment genome (the QD agent's
    LLM-guided mutation). Returns a plain dict or None on any parse/shape
    problem; the caller validates and falls back to evolutionary operators.
    Retries once (the QD loop has a cheap fallback, unlike the features axis).
    avoid: already-tried genome signatures the model must not repeat (default None,
    so the prompt is unchanged)."""
    prompt = (
        "You are evolving experiment genomes for a trading-model search "
        "(MAP-Elites). Propose ONE child genome likely to beat the elites.\n"
        + GENOME_MENU +
        "\nDroppable features: " + ",".join(active) +
        "\nBase columns for specs: " + ",".join(base_features) +
        "\nParent genome: " + json.dumps(parent, ensure_ascii=True) +
        "\nCurrent elites (genome + fitness): " + json.dumps(elites, ensure_ascii=True) +
        _avoid_clause(avoid) +
        "\nReturn STRICT JSON: one genome object, no prose."
    )
    prompt = _wiki_preamble() + prompt
    hyp = _reflect_hypothesis()
    if hyp:
        prompt = "Reflection: " + hyp + "\n" + prompt
    backend = _backend()
    for _attempt in range(2):
        obj = _parse_obj(backend(prompt))
        if obj is not None:
            return obj
    return None


def propose_specs(log, base_features, avoid=None):
    """Ask the selected LLM for the next 1-2 feature specs. The backend retries
    a few times then raises cleanly; a non-JSON reply yields no specs (that
    iteration is skipped). avoid: already-tried spec signatures the model must not
    repeat (default None, so the prompt is unchanged)."""
    prompt = _proposer_prompt(log, base_features) + _avoid_clause(avoid)
    prompt = _wiki_preamble() + prompt
    hyp = _reflect_hypothesis()
    if hyp:
        prompt = "Reflection: " + hyp + "\n" + prompt
    return _parse_specs(_backend()(prompt))


if __name__ == "__main__":
    import sys
    if "--list-ollama" in sys.argv:
        _print_ollama_models()
