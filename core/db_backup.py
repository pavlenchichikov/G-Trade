"""SQLite database backup utility.

Usage:
    python -m core.db_backup              # one-time backup
    python -m core.db_backup --schedule   # run forever, backup every 24h
"""

import os
import shutil
import time
from datetime import datetime

from core.logger import get_logger

logger = get_logger("db_backup")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
MAX_BACKUPS = 7  # keep last 7 backups


def backup_db(db_path: str = DB_PATH, backup_dir: str = BACKUP_DIR) -> str | None:
    """Create a timestamped copy of the SQLite database.

    Returns the backup file path on success, None on failure.
    """
    if not os.path.exists(db_path):
        logger.warning("Database not found: %s", db_path)
        return None

    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"market_{ts}.db")

    try:
        shutil.copy2(db_path, backup_path)
        size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        logger.info("Backup created: %s (%.1f MB)", backup_path, size_mb)
        _prune_old_backups(backup_dir)
        return backup_path
    except OSError as e:
        logger.error("Backup failed: %s", e)
        return None


def _prune_old_backups(backup_dir: str, max_keep: int = MAX_BACKUPS) -> None:
    """Delete oldest backups if more than max_keep exist."""
    backups = sorted(
        [f for f in os.listdir(backup_dir) if f.startswith("market_") and f.endswith(".db")],
    )
    while len(backups) > max_keep:
        old = backups.pop(0)
        path = os.path.join(backup_dir, old)
        try:
            os.remove(path)
            logger.info("Pruned old backup: %s", old)
        except OSError:
            pass


def run_scheduled(interval_hours: int = 24) -> None:
    """Run backup in a loop every interval_hours."""
    logger.info("Scheduled backup every %d hours", interval_hours)
    while True:
        backup_db()
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    import sys
    if "--schedule" in sys.argv:
        run_scheduled()
    else:
        result = backup_db()
        if result:
            print(f"Backup saved: {result}")
        else:
            print("Backup failed — check logs")
