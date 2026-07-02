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
