"""Shared fixtures. ar_memory state files are per-test isolated so no test
pollutes (or depends on) the real project-root research memory."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_ar_memory(tmp_path, monkeypatch):
    from core import ar_memory
    monkeypatch.setattr(ar_memory, "TRIED_PATH", str(tmp_path / "_ar_tried.json"))
    monkeypatch.setattr(ar_memory, "CACHE_PATH", str(tmp_path / "_ar_eval_cache.json"))
    monkeypatch.setattr(ar_memory, "FINDINGS_PATH", str(tmp_path / "_ar_findings.json"))
    monkeypatch.setattr(ar_memory, "DB_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(ar_memory, "REPLICATION_PATH", str(tmp_path / "_ar_replication.json"))
    import core.ar_wiki as _ar_wiki
    monkeypatch.setattr(_ar_wiki, "WIKI_DIR", str(tmp_path / "_ar_wiki"))
    # Tests must never talk to a real LLM, no matter what the local .env enables:
    # GTRADE_AR_WIKI=1 + GTRADE_AR_LLM=ollama would otherwise make any test that
    # walks a run_qd/regate path fire compile_wiki() and load a real local model
    # (observed 2026-07-16: a suite run started Ollama). Tests that exercise these
    # paths inject fakes and set the vars explicitly.
    for var in ("GTRADE_AR_WIKI", "GTRADE_AR_PROPOSER", "GTRADE_AR_REFLECT",
                "GTRADE_AR_LLM", "GTRADE_AR_LLM_MODEL", "GTRADE_AR_RL"):
        monkeypatch.delenv(var, raising=False)
