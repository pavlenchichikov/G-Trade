"""
DB BACKUP -- Backup and restore utility for market.db
Creates timestamped backups, supports rotation, restore, and listing.
Uses only stdlib: shutil, glob, os, sys, argparse.
"""

import os
import shutil
import glob
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
DEFAULT_KEEP = 5


def _now_str():
    return datetime.now().strftime("%H:%M:%S")


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_size(size_bytes):
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)


def _get_backups():
    """Return list of backup files sorted by modification time (newest first)."""
    pattern = os.path.join(BACKUP_DIR, "market_backup_*.db")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def _print_header():
    print()
    print("=" * 60)
    print("  DB BACKUP  |  market.db")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 60)
    print()


def create_backup(silent=False):
    """Create a timestamped backup of market.db. Returns backup path or None."""
    if not os.path.exists(DB_PATH):
        if not silent:
            print(f"[{_now_str()}] [ERR] Database not found: {DB_PATH}")
        return None

    _ensure_backup_dir()
    backup_name = f"market_backup_{_timestamp()}.db"
    backup_path = os.path.join(BACKUP_DIR, backup_name)

    try:
        shutil.copy2(DB_PATH, backup_path)
        size = os.path.getsize(backup_path)
        if not silent:
            _print_header()
            print(f"[BACKUP] Created: backups/{backup_name} ({_format_size(size)})")
            print()
            list_backups(header=False)
        else:
            # Auto mode: minimal output
            print(f"[{_now_str()}] [BACKUP] backups/{backup_name} ({_format_size(size)})")
        return backup_path
    except Exception as e:
        msg = f"[{_now_str()}] [ERR] Backup failed: {e}"
        print(msg)
        return None


def list_backups(header=True):
    """Show existing backups with sizes and dates."""
    if header:
        _print_header()

    _ensure_backup_dir()
    backups = _get_backups()

    if not backups:
        print("  No backups found.")
        print()
        return

    print("  Existing backups:")
    print(f"  {'#':<4} {'Date':<22} {'Size':<12} {'Filename'}")
    print("  " + "-" * 60)

    for i, path in enumerate(backups, 1):
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        date_str = mtime.strftime("%Y-%m-%d %H:%M")
        size = _format_size(os.path.getsize(path))
        name = os.path.basename(path)
        print(f"  {i:<4} {date_str:<22} {size:<12} {name}")

    print()


def restore_backup(backup_name, force=False):
    """Restore market.db from a backup file."""
    _print_header()

    # Find the backup file
    backup_path = None
    if os.path.isabs(backup_name):
        backup_path = backup_name
    else:
        candidate = os.path.join(BACKUP_DIR, backup_name)
        if os.path.exists(candidate):
            backup_path = candidate
        else:
            # Try partial match
            _ensure_backup_dir()
            for f in _get_backups():
                if backup_name in os.path.basename(f):
                    backup_path = f
                    break

    if not backup_path or not os.path.exists(backup_path):
        print(f"[ERR] Backup not found: {backup_name}")
        print()
        print("Available backups:")
        list_backups(header=False)
        return False

    backup_size = _format_size(os.path.getsize(backup_path))
    backup_date = datetime.fromtimestamp(os.path.getmtime(backup_path)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Source:  {os.path.basename(backup_path)}")
    print(f"  Size:    {backup_size}")
    print(f"  Date:    {backup_date}")
    print()

    if not force:
        try:
            answer = input("  Restore this backup? Current market.db will be overwritten. [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False
        if answer not in ("y", "yes"):
            print("  Cancelled.")
            return False

    # Create safety backup of current DB before restoring
    if os.path.exists(DB_PATH):
        _ensure_backup_dir()
        safety_name = f"market_backup_pre_restore_{_timestamp()}.db"
        safety_path = os.path.join(BACKUP_DIR, safety_name)
        try:
            shutil.copy2(DB_PATH, safety_path)
            print(f"  [SAFETY] Pre-restore backup: backups/{safety_name}")
        except Exception as e:
            print(f"  [WARN] Could not create safety backup: {e}")

    try:
        shutil.copy2(backup_path, DB_PATH)
        print(f"  [OK] Restored market.db from {os.path.basename(backup_path)}")
        return True
    except Exception as e:
        print(f"  [ERR] Restore failed: {e}")
        return False


def clean_backups(keep=DEFAULT_KEEP):
    """Remove old backups, keeping the last N."""
    _print_header()
    _ensure_backup_dir()
    backups = _get_backups()

    if len(backups) <= keep:
        print(f"  {len(backups)} backup(s) found, keeping all (limit: {keep}).")
        print()
        return

    to_delete = backups[keep:]
    print(f"  Total backups: {len(backups)}")
    print(f"  Keeping: {keep}")
    print(f"  Removing: {len(to_delete)}")
    print()

    for path in to_delete:
        name = os.path.basename(path)
        try:
            os.remove(path)
            print(f"  [DEL] {name}")
        except Exception as e:
            print(f"  [ERR] Could not delete {name}: {e}")

    print()
    print(f"  Done. {min(len(backups), keep)} backup(s) remaining.")
    print()


def auto_backup():
    """Silent backup for scheduler/GUI use. No prompts, minimal output."""
    return create_backup(silent=True)


def main():
    parser = argparse.ArgumentParser(description="DB Backup -- market.db backup and restore")
    parser.add_argument("--list", action="store_true", help="Show existing backups")
    parser.add_argument("--restore", metavar="BACKUP", help="Restore from a backup file")
    parser.add_argument("--clean", action="store_true", help="Remove old backups (keep last 5)")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="Number of backups to keep (default: 5)")
    parser.add_argument("--auto", action="store_true", help="Create backup silently (for scheduler/GUI)")
    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        restore_backup(args.restore)
    elif args.clean:
        clean_backups(keep=args.keep)
    elif args.auto:
        auto_backup()
    else:
        create_backup()


if __name__ == "__main__":
    main()
