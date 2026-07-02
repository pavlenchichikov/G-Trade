import json
import sys
import types

import pytest

from core import llm_proposer as lp


def test_parse_specs_tolerates_prose():
    text = ('Sure! [{"name": "f", "op": "lag", "inputs": ["ret_1"],'
            ' "params": {"k": 1}}] hope this helps')
    specs = lp._parse_specs(text)
    assert specs and specs[0]["name"] == "f"
    assert lp._parse_specs("no json here") == []
    assert lp._parse_specs("") == []
    assert lp._parse_specs('{"a": 1}') == []      # dict, not a list


def test_backend_selection_unknown_raises(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "bogus")
    with pytest.raises(RuntimeError):
        lp.propose_specs([], ["ret_1"])


def test_propose_specs_uses_selected_backend(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: (
        '[{"name": "f", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]'))
    specs = lp.propose_specs([], ["ret_1"])
    assert specs == [{"name": "f", "op": "lag", "inputs": ["ret_1"], "params": {"k": 1}}]


def test_propose_specs_prompt_carries_history(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai",
                        lambda prompt: seen.setdefault("prompt", prompt) and "[]" or "[]")
    lp.propose_specs([{"iter": 0, "score": 0.5}], ["ret_1", "rsi"])
    assert "ret_1,rsi" in seen["prompt"]
    assert '"score": 0.5' in seen["prompt"]


def test_llm_selected(monkeypatch):
    monkeypatch.delenv("GTRADE_AR_PROPOSER", raising=False)
    assert lp.llm_selected() is False
    monkeypatch.setenv("GTRADE_AR_PROPOSER", "llm")
    assert lp.llm_selected() is True


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def read(self):
        return json.dumps(self._p).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_detect_ollama_prefers_gemma(monkeypatch):
    import urllib.request
    monkeypatch.delenv("GTRADE_AR_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(
        {"models": [{"name": "llama3:8b"}, {"name": "gemma4:26b"}]}))
    assert lp._detect_ollama_model() == "gemma4:26b"


def test_detect_ollama_first_model_fallback(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(
        {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}))
    assert lp._detect_ollama_model() == "llama3:8b"


def test_detect_ollama_no_models_raises(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=5: _FakeResp({"models": []}))
    with pytest.raises(RuntimeError, match="ollama pull"):
        lp._detect_ollama_model()


def test_detect_ollama_unreachable_raises(monkeypatch):
    import urllib.request

    def dead(url, timeout=5):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", dead)
    with pytest.raises(RuntimeError, match="running"):
        lp._detect_ollama_model()


def test_call_ollama_defaults(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, model, max_tokens, messages):
            captured["model"] = model
            msg = types.SimpleNamespace(content=" [] ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.delenv("GTRADE_AR_LLM_MODEL", raising=False)
    monkeypatch.delenv("GTRADE_AR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(lp, "_detect_ollama_model", lambda: "gemma4:26b")
    assert lp._call_ollama("hi") == "[]"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "ollama"
    assert captured["model"] == "gemma4:26b"


def test_call_ollama_model_env_override(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, model, max_tokens, messages):
            captured["model"] = model
            msg = types.SimpleNamespace(content="[]")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.setenv("GTRADE_AR_LLM_MODEL", "gemma3:latest")
    lp._call_ollama("hi")
    assert captured["model"] == "gemma3:latest"


def test_backend_knows_ollama(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "ollama")
    monkeypatch.setattr(lp, "_call_ollama", lambda prompt: "[]")
    assert lp.propose_specs([], ["ret_1"]) == []


def test_propose_genome_good_json(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    reply = ('Here you go: {"drops": ["rsi"], "extra": [], '
             '"label_mode": "rel_median", "label_window": 20} enjoy')
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: reply)
    obj = lp.propose_genome({"drops": [], "extra": [], "label_mode": "direction",
                             "label_window": 30}, [], ["rsi", "atr"], ["ret_1"])
    assert obj == {"drops": ["rsi"], "extra": [],
                   "label_mode": "rel_median", "label_window": 20}


def test_propose_genome_garbage_returns_none(monkeypatch):
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")
    monkeypatch.setattr(lp, "_call_openai", lambda prompt: "I cannot help with that.")
    assert lp.propose_genome({}, [], ["rsi"], ["ret_1"]) is None


def test_propose_genome_prompt_mentions_parent_and_elites(monkeypatch):
    seen = {}
    monkeypatch.setenv("GTRADE_AR_LLM", "openai")

    def capture(prompt):
        seen["prompt"] = prompt
        return "{}"

    monkeypatch.setattr(lp, "_call_openai", capture)
    lp.propose_genome({"drops": ["atr"]}, [{"genome": {"drops": []}, "fitness": 1.5}],
                      ["rsi", "atr"], ["ret_1"])
    assert '"drops": ["atr"]' in seen["prompt"]
    assert '"fitness": 1.5' in seen["prompt"]
    assert "rsi,atr" in seen["prompt"]
