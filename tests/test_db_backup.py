"""Tests for core.db_backup — SQLite backup utility."""

import os
import tempfile

from core.db_backup import backup_db, _prune_old_backups


class TestBackupDb:
    def test_creates_backup(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("fake db content")
        backup_dir = tmp_path / "backups"

        result = backup_db(str(db), str(backup_dir))
        assert result is not None
        assert os.path.exists(result)
        assert backup_dir.exists()

    def test_missing_db_returns_none(self, tmp_path):
        result = backup_db(str(tmp_path / "nonexistent.db"), str(tmp_path / "backups"))
        assert result is None

    def test_backup_content_matches(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_bytes(b"binary content here")
        backup_dir = tmp_path / "backups"

        result = backup_db(str(db), str(backup_dir))
        assert open(result, "rb").read() == b"binary content here"


class TestPruneOldBackups:
    def test_prunes_excess(self, tmp_path):
        for i in range(10):
            (tmp_path / f"market_2026030{i}_120000.db").write_text("x")
        _prune_old_backups(str(tmp_path), max_keep=3)
        remaining = [f for f in os.listdir(tmp_path) if f.startswith("market_")]
        assert len(remaining) == 3

    def test_keeps_all_if_under_limit(self, tmp_path):
        for i in range(2):
            (tmp_path / f"market_2026030{i}_120000.db").write_text("x")
        _prune_old_backups(str(tmp_path), max_keep=5)
        remaining = [f for f in os.listdir(tmp_path) if f.startswith("market_")]
        assert len(remaining) == 2
