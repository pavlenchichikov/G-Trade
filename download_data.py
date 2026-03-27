import yfinance as yf
import pandas as pd
import os

try:
    from config import FULL_ASSET_MAP
except ImportError:
    # Fallback
    print("Warning: config.py import failed. Using default minimal set.")
    FULL_ASSET_MAP = {"BTC": "BTC-USD", "SP500": "^GSPC"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "market_data")
os.makedirs(output_dir, exist_ok=True)

for name, symbol in FULL_ASSET_MAP.items():
    print(f"Downloading {name} ({symbol})...")
    try:
        data = yf.download(symbol, start="2020-01-01", progress=False, threads=False)
        if not data.empty:
            # Flatten MultiIndex columns if present (new yfinance behavior)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Save to CSV
            file_path = os.path.join(output_dir, f"{name.lower()}_data.csv")
            data.to_csv(file_path)
            print(f"Saved to {file_path}")
        else:
            print(f"Failed to download {symbol} (Empty Data)")
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
