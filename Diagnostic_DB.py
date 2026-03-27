# -*- coding: utf-8 -*-
import sqlite3
import os
import sys

# Ïðèíóäèòåëüíî ñòàâèì êîäèðîâêó äëÿ âûâîäà â êîíñîëü Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ïóòü ê áàçå äàííûõ
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "market.db")

def run_audit():
    if not os.path.exists(db_path):
        print(f"--- Error: File not found at {db_path} ---")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ïîëó÷àåì ñïèñîê òàáëèö
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print("\n" + "="*60)
        print(f" DATABASE AUDIT: {len(tables)} tables found")
        print("="*60)
        print(f"{'Table Name':<15} | {'Rows':<8} | {'Status'}")
        print("-" * 60)

        for table in tables:
            name = table[0]
            # Ñ÷èòàåì êîëè÷åñòâî ñòðîê
            cursor.execute(f"SELECT COUNT(*) FROM {name}")
            row_count = cursor.fetchone()[0]

            # Ïðîâåðÿåì ñòðóêòóðó (êîëîíêè)
            cursor.execute(f"PRAGMA table_info({name})")
            [col[1] for col in cursor.fetchall()]

            # Îïðåäåëÿåì ñòàòóñ
            # Äëÿ îáó÷åíèÿ íóæíî ìèíèìóì 100 ñòðîê
            if row_count >= 100:
                status = "OK (Ready)"
            elif row_count > 0:
                status = "Low Data (Skipping)"
            else:
                status = "EMPTY"

            print(f"{name:<15} | {row_count:<8} | {status}")

        conn.close()
        print("="*60)
        print("Tip: If US assets (nvda, gold) are EMPTY, check your Proxy settings in Step 1.")

    except Exception as e:
        print(f"Critical Error during audit: {e}")

if __name__ == "__main__":
    run_audit()
