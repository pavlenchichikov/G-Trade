"""Feature engineering for market data.

Extracted from train_hybrid.py for reusability and testability.
Used by: train_hybrid.py, predict.py, backtest.py, signal_dashboard.py
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data.

    Input: DataFrame with columns [Date, Open, High, Low, Close, Volume]
    Output: DataFrame with 15+ features + target + next_ret, NaN rows dropped.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    returns = df['close'].pct_change()

    # Returns multi-timeframe
    df['ret_1'] = returns
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)

    # Volatility & risk
    df['taleb_risk'] = returns.rolling(window=30).kurt().fillna(0)
    vol = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['vol_z'] = (vol - vol.rolling(30).mean()) / (vol.rolling(30).std() + 1e-9)

    # ATR normalized
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean() / (df['close'] + 1e-9)

    # Trend
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['trend_strength'] = (df['close'] / (df['sma_50'] + 1e-9) - 1.0).abs()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD histogram
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9).mean()

    # Bollinger Bands position (-1..+1)
    std20 = df['close'].rolling(20).std()
    df['bb_pos'] = (df['close'] - df['sma_20']) / (2 * std20 + 1e-9)

    # Volume ratio
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1)

    # Target: next bar direction
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['next_ret'] = df['close'].pct_change().shift(-1)

    df = df.dropna()

    # Preserve Date as column for downstream joins
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    return df


def add_weekly_features(df: pd.DataFrame, table: str, engine) -> pd.DataFrame:
    """Merge weekly RSI/trend indicators into daily dataframe (forward-fill)."""
    weekly_table = table + "_weekly"
    try:
        df_w = pd.read_sql(
            f"SELECT * FROM {weekly_table}", engine,
            index_col="Date", parse_dates=["Date"],
        )
        df_w = df_w[~df_w.index.duplicated(keep='last')].sort_index()
        if df_w.empty or len(df_w) < 10:
            return df
        df_w.columns = [c.lower() for c in df_w.columns]
        df_w['w_ret'] = df_w['close'].pct_change()
        w_delta = df_w['close'].diff()
        w_gain = w_delta.where(w_delta > 0, 0).rolling(14).mean()
        w_loss = (-w_delta.where(w_delta < 0, 0)).rolling(14).mean()
        w_rs = w_gain / (w_loss + 1e-9)
        df_w['w_rsi'] = 100 - (100 / (1 + w_rs))
        w_sma4 = df_w['close'].rolling(4).mean()
        df_w['w_trend'] = df_w['close'] / (w_sma4 + 1e-9) - 1.0
        weekly_cols = ['w_ret', 'w_rsi', 'w_trend']
        df_w = df_w[weekly_cols].dropna()

        # Find date column for join
        date_col = None
        if isinstance(df.index, pd.DatetimeIndex):
            date_col = '_index_'
        elif 'Date' in df.columns:
            date_col = 'Date'
        elif 'date' in df.columns:
            date_col = 'date'
        if date_col is None:
            return df

        if date_col != '_index_':
            df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)
        df = df.join(df_w, how='left')
        for col in weekly_cols:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
        df = df.reset_index()
        if date_col == '_index_' and df.columns[0] not in weekly_cols:
            df = df.rename(columns={df.columns[0]: 'Date'})
    except Exception:
        pass
    return df


_CROSS_REFS = [
    ('btcusd', 'corr_btc'),
    ('gspc', 'corr_sp500'),
    ('dxynyd', 'corr_dxy'),
]
_CROSS_SKIP = {'btcusd', 'gspc', 'dxynyd', 'vix'}


def add_crossasset_features(df: pd.DataFrame, table: str, engine) -> pd.DataFrame:
    """Add 20-day rolling correlation with BTC, SP500, DXY as features."""
    if table in _CROSS_SKIP:
        return df
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col is None:
        return df
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    asset_ret = df['close'].pct_change()
    for ref_table, feat_name in _CROSS_REFS:
        if ref_table == table:
            continue
        try:
            ref = pd.read_sql(
                f"SELECT Date, Close FROM {ref_table}", engine,
                index_col="Date", parse_dates=["Date"],
            )
            ref.index = pd.to_datetime(ref.index).normalize()
            ref.columns = [c.lower() for c in ref.columns]
            ref = ref[~ref.index.duplicated(keep='last')].sort_index()
            ref_ret = ref['close'].pct_change()
            combined = pd.concat([asset_ret.rename('a'), ref_ret.rename('r')], axis=1)
            corr = combined['a'].rolling(20).corr(combined['r'])
            df[feat_name] = corr.reindex(df.index).ffill().fillna(0)
        except Exception:
            df[feat_name] = 0.0
    df = df.reset_index()
    return df


# Feature columns used for training (order matters for model compatibility)
CANDIDATE_FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'taleb_risk', 'vol_z', 'atr',
    'sma_20', 'sma_50', 'sma_200', 'trend_strength',
    'rsi', 'macd_hist', 'bb_pos', 'vol_ratio',
    'w_ret', 'w_rsi', 'w_trend',
    'corr_btc', 'corr_sp500', 'corr_dxy',
]
