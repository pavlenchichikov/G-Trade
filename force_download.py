import yfinance as yf
import pandas as pd
import os
import time

try:
    from config import FULL_ASSET_MAP
except ImportError:
    FULL_ASSET_MAP = {"BTC": "BTC-USD", "SP500": "^GSPC", "VIX": "^VIX"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "market_data")
os.makedirs(DATA_DIR, exist_ok=True)

def download():
    print("--- FORCING DATA REFRESH ---")
    for name, symbol in FULL_ASSET_MAP.items():
        print(f"Downloading {name}...", end=" ")
        try:
            df = yf.Ticker(symbol).history(period="2y", interval="1d")
            if df.empty:
                print("Failed")
                continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None)
            df.to_csv(os.path.join(DATA_DIR, f"{name.lower()}_data.csv"))
            print(f"OK ({len(df)} rows)")
            time.sleep(0.5)
        except Exception as e: print(f"Error: {e}")

if __name__ == "__main__": download()
