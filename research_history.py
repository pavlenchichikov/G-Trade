import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
TARGET = "^GSPC" # S&P 500
START_DATE = "2000-01-01"
END_DATE = "2024-01-01"
WINDOW = 60 # 3-month rolling window for metrics

def calculate_tale_risk(series):
    # Our V30 Formula
    kurt = series.kurt()
    skew = series.skew()
    # Risk Score
    risk = (np.maximum(0, kurt) * 1.5) + (np.abs(np.minimum(0, skew)) * 2.0)
    return risk

print(f"Downloading {TARGET} history ({START_DATE} - {END_DATE})...")
df = yf.download(TARGET, start=START_DATE, end=END_DATE, progress=False)

# FIX: Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df['log_ret'] = np.log(df['Close']).diff()

print("Calculating Taleb Metrics...")
# Rolling metrics
df['kurtosis'] = df['log_ret'].rolling(WINDOW).kurt()
df['skew'] = df['log_ret'].rolling(WINDOW).skew()
df['volatility'] = df['log_ret'].rolling(WINDOW).std() * np.sqrt(252)

# Our Index
df['taleb_risk'] = df['log_ret'].rolling(WINDOW).apply(calculate_tale_risk)

# Sigmoid Scaling (0-100%) to match Dashboard
df['risk_index'] = (1 / (1 + np.exp(-(df['taleb_risk'] - 2.5)))) * 100

# Identify CRASHES (Drawdown > 15%)
df['peak'] = df['Close'].cummax()
df['drawdown'] = (df['Close'] - df['peak']) / df['peak']
crashes = df[df['drawdown'] < -0.15].index

# Filter unique crash events (start of crash)
unique_crashes = []
last_crash = pd.Timestamp("1900-01-01")
for date in crashes:
    if (date - last_crash).days > 365: # Separate crises by 1 year
        unique_crashes.append(date)
        last_crash = date

print(f"Found {len(unique_crashes)} Major Crashes (Dot-com, 2008, 2020, etc.)")

# --- PLOTTING ---
plt.figure(figsize=(15, 10))
plt.style.use('dark_background')

# 1. Price & Crashes
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Close'], color='white', label='S&P 500')
for date in unique_crashes:
    plt.axvline(date, color='red', linestyle='--', alpha=0.7)
    plt.text(date, df['Close'].max(), f"CRASH {date.year}", color='red', rotation=90)
plt.title("Market History & Major Crashes")
plt.legend()

# 2. Risk Index
plt.subplot(3, 1, 2)
plt.plot(df.index, df['risk_index'], color='#ffca28', label='V30 Black Swan Index')
plt.axhline(70, color='red', linestyle=':', label='Danger Zone (70%)')
plt.fill_between(df.index, df['risk_index'], 70, where=(df['risk_index'] >= 70), color='red', alpha=0.3)
for date in unique_crashes:
    plt.axvline(date, color='red', linestyle='--', alpha=0.3)
plt.title("V30 Black Swan Probability (Does it spike BEFORE crash?)")
plt.legend()

# 3. Kurtosis (The core metric)
plt.subplot(3, 1, 3)
plt.plot(df.index, df['kurtosis'], color='#58a6ff', label='Kurtosis (Fat Tails)')
plt.axhline(0, color='white', linestyle='--', alpha=0.3)
for date in unique_crashes:
    plt.axvline(date, color='red', linestyle='--', alpha=0.3)
plt.title("Kurtosis (Fat Tails)")
plt.legend()

plt.tight_layout()
output_path = os.path.join(os.getcwd(), "history_analysis.png")
plt.savefig(output_path)
print(f"Analysis complete. Saved to '{output_path}'")
