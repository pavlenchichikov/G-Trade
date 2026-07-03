"""OHLCV features: indicators, weekly, and cross-asset."""

import hashlib
import os

import numpy as np
import pandas as pd

from core.logger import get_logger

_logger = get_logger(__name__)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """14-period RSI (simple rolling mean of gains/losses, Wilder-style)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def make_target(close: pd.Series, mode: str = "direction", window: int = 30) -> pd.Series:
    """Binary next-bar label.

    - "direction" (default): 1 if the next close is higher than the current one.
      This is the historical label, preserved exactly.
    - "rel_median": 1 if the next return beats the trailing-median return over
      `window` bars. The baseline uses only past returns up to t (no look-ahead),
      so the positive class stays near 50 percent in any trend - which fixes the
      pos_ratio=0.00 degeneracy that flat next-bar-direction shows on downtrending
      or low-volatility assets.

    Warm-up rows (and the final row, whose next bar is unknown) are left as NaN
    here and removed downstream by engineer_features' dropna, regardless of
    `window` (a plain boolean compare against NaN would otherwise evaluate to
    False, then astype(int) to 0, fabricating a label for rows with no real
    meaning). Raises ValueError on an unknown mode."""
    if mode == "direction":
        return (close.shift(-1) > close).astype(int)
    if mode == "rel_median":
        ret = close.pct_change()
        baseline = ret.rolling(window).median()
        next_ret = ret.shift(-1)
        target = (next_ret > baseline).astype(float)
        target[baseline.isna() | next_ret.isna()] = np.nan
        return target
    raise ValueError("unknown GTRADE_LABEL_MODE: %r" % mode)


def compute_taleb_risk(close: pd.Series, window: int = 60, min_periods: int = 30) -> pd.Series:
    """Taleb tail-risk index: rolling kurtosis (4th moment) of log-returns.

    A 60-bar window on log-returns - see engineer_features for the rationale
    (kurtosis is unstable on short windows and symmetric, hence the companion
    skew/var_5 features). Returns the raw rolling series; warm-up rows are NaN.
    """
    ratio = close / close.shift(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        log_ret = np.log(ratio.where(ratio > 0))
    return log_ret.rolling(window=window, min_periods=min_periods).kurt()


def latest_taleb_risk(closes) -> float | None:
    """Latest Taleb tail-risk value from a close-price sequence (for display).

    Returns None when there aren't enough closes to form a single estimate
    (need >= min_periods + 1). Warm-up NaNs are filled with the series median,
    matching how engineer_features fills them, so a short history still yields a
    number rather than NaN.
    """
    s = pd.Series(list(closes), dtype="float64")
    if len(s) < 31:
        return None
    risk = compute_taleb_risk(s)
    risk = risk.fillna(risk.median())
    val = risk.iloc[-1]
    return float(val) if pd.notna(val) else None


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

    # Volatility & tail risk (Taleb)
    # Kurtosis is a 4th moment: a 30-bar window is too small to estimate it
    # stably - one outlier dominates, then mechanically drops out 30 bars later,
    # making the index spike and cliff for no real reason. Use 60 bars on
    # log-returns. Kurtosis is also symmetric, so on its own it cannot tell an
    # upside spike from a crash; for a long-biased model the left tail is what
    # hurts, so we add skew (asymmetry) and var_5 (5% left-tail return, VaR).
    _ratio = df['close'] / df['close'].shift(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        log_ret = np.log(_ratio.where(_ratio > 0))
    tail = log_ret.rolling(window=60, min_periods=30)
    df['taleb_risk'] = compute_taleb_risk(df['close'])
    df['ret_skew'] = tail.skew()
    df['var_5'] = tail.quantile(0.05)
    # Warm-up rows: fill with the column median, NOT 0. A zero kurtosis/skew/VaR
    # reads as "perfectly calm" (the safest value), which would feed the model
    # fake calm at the start of every short-history asset.
    for _c in ('taleb_risk', 'ret_skew', 'var_5'):
        df[_c] = df[_c].fillna(df[_c].median())
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
    df['rsi'] = compute_rsi(df['close'])

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

    # Volatility-normalized returns (stationary scale across regimes)
    df['ret_1_vn'] = (df['ret_1'] / (df['ret_1'].rolling(20).std() + 1e-9)).fillna(0.0)
    df['ret_5_vn'] = (df['ret_5'] / (df['ret_5'].rolling(20).std() + 1e-9)).fillna(0.0)

    # Calendar encodings (weekday, position in month). Prefer a 'date' column;
    # fall back to a DatetimeIndex when there is no 'date' column (the common
    # case here, since the index still carries the dates at this point).
    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        _idx = pd.to_datetime(df.index)
        df['cal_dow'] = (_idx.dayofweek / 6.0)
        df['cal_mpos'] = (_idx.day / _idx.days_in_month)
    elif 'date' in df.columns:
        _d = pd.to_datetime(df['date'])
        df['cal_dow'] = (_d.dt.dayofweek / 6.0).values
        df['cal_mpos'] = (_d.dt.day / _d.dt.days_in_month).values
    else:
        df['cal_dow'] = 0.0
        df['cal_mpos'] = 0.0

    # Target: configurable labeling mode (default reproduces next-bar direction
    # exactly; see make_target). Read here, not at import, so an A/B run that
    # flips GTRADE_LABEL_MODE per subprocess is honored on every call.
    _label_mode = os.getenv("GTRADE_LABEL_MODE", "direction")
    try:
        _label_window = int(os.getenv("GTRADE_LABEL_WINDOW", "30"))
    except ValueError:
        _raw_window = os.getenv("GTRADE_LABEL_WINDOW")
        _logger.warning(
            "invalid GTRADE_LABEL_WINDOW=%r, falling back to 30", _raw_window,
        )
        _label_window = 30
    df['target'] = make_target(df['close'], _label_mode, _label_window)
    df['next_ret'] = df['close'].pct_change().shift(-1)

    # Non-positive or near-zero prices (e.g. SHIB) make ratio/log features blow
    # up to +/-inf, which dropna does NOT remove - they would reach the model as
    # "Input X contains infinity" and kill the whole asset. Coerce infinities to
    # NaN first so the dropna below drops exactly those rows.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df['target'] = df['target'].astype(int)

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
        df_w['w_rsi'] = compute_rsi(df_w['close'])
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


# Reference tables are named after the ASSET KEY (config.py), e.g. SP500 - "sp500",
# not after the Yahoo ticker. Using ticker-derived names here (btcusd/gspc/dxynyd)
# meant every read missed its table and corr_* silently fell back to 0.0.
_CROSS_REFS = [
    ('btc', 'corr_btc'),
    ('sp500', 'corr_sp500'),
    ('dxy', 'corr_dxy'),
]
_CROSS_SKIP = {'btc', 'sp500', 'dxy', 'vix'}


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


# Macro regime features sourced from the tnx/vix/dxy price tables (already
# fetched via Yahoo): 10y Treasury yield (rates), VIX (volatility), the dollar
# index (DXY). These are the dominant market-wide regime signals and they come
# from the same reliable route as the rest of the data - no FRED dependency
# (FRED's CSV host is unreachable from this project's VPN exits). Same value
# across all assets on a date; aligned by as-of forward-fill, no look-ahead
# beyond the same-day close the asset's own features already use.
_MACRO_SOURCE_TABLES = ('tnx', 'vix', 'dxy')
_MACRO_FEATURES = [
    'macro_tnx', 'macro_tnx_chg20', 'macro_vix', 'macro_vix_chg5',
    'macro_dxy_chg20',
]


def add_macro_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Merge macro regime features (10y yield, VIX, dollar) into a daily frame.

    Reads the close series from the tnx/vix/dxy tables, derives level + momentum
    features, and aligns them to the asset's dates with an as-of forward-fill
    (last value known on or before each bar). A missing source table leaves its
    features at 0.0 so the feature vector keeps a stable shape and
    `feature_version()` stays consistent.
    """
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col is None:
        for c in _MACRO_FEATURES:
            df[c] = 0.0
        return df
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    def _close(table):
        try:
            ref = pd.read_sql(f"SELECT Date, Close FROM {table}", engine,
                              index_col="Date", parse_dates=["Date"])
            ref.index = pd.to_datetime(ref.index).normalize()
            ref.columns = [c.lower() for c in ref.columns]
            ref = ref[~ref.index.duplicated(keep='last')].sort_index()
            return ref['close']
        except Exception:
            return None

    cols = {}
    tnx = _close('tnx')
    if tnx is not None:
        cols['macro_tnx'] = tnx
        cols['macro_tnx_chg20'] = tnx.diff(20)
    vix = _close('vix')
    if vix is not None:
        cols['macro_vix'] = vix
        cols['macro_vix_chg5'] = vix.diff(5)
    dxy = _close('dxy')
    if dxy is not None:
        cols['macro_dxy_chg20'] = dxy.pct_change(20)

    feats = pd.concat(cols, axis=1).sort_index() if cols else pd.DataFrame()
    for c in _MACRO_FEATURES:
        if not feats.empty and c in feats.columns:
            df[c] = feats[c].reindex(df.index, method='ffill').fillna(0.0).values
        else:
            df[c] = 0.0
    df = df.reset_index()
    return df


CHRONOS_CACHE_TABLE = "chronos_cache"
CHRONOS_COLS = ["chronos_ret", "chronos_spread", "chronos_dir"]


def _chronos_on():
    return (os.getenv("GTRADE_CHRONOS") or "").strip() in ("1", "true", "True")


def add_chronos_features(df, table, engine):
    """LEFT-JOIN cached Chronos forecast columns onto df by date, when GTRADE_CHRONOS
    is set and a cache exists for `table`. Off / no cache -> df unchanged (so the
    production feature space and feature_version are untouched). The column names
    enter training only via GTRADE_EXTRA_FEATURES."""
    if not _chronos_on():
        return df
    try:
        q = "SELECT date, %s FROM %s WHERE asset = ?" % (
            ",".join(CHRONOS_COLS), CHRONOS_CACHE_TABLE)
        cache = pd.read_sql(q, engine, params=(table,))
    except Exception:
        return df                                   # no cache table -> no-op
    if cache.empty:
        return df
    cache["date"] = pd.to_datetime(cache["date"])
    cache = cache.set_index("date")
    cache = cache[~cache.index.duplicated(keep="last")]   # unique right index -> no row multiplication
    cache.index = cache.index.normalize()
    # Follow the sibling add_* convention: in the pipeline df carries a Date/date
    # COLUMN (each add_* does set_index(date) -> work -> reset_index), so join on
    # that column and restore the frame. Without this, a df that arrives with a
    # RangeIndex after a sibling reset_index would join dates-vs-integers and yield
    # SILENTLY all-NaN Chronos columns. When df is already date-indexed (isolated
    # use), join directly.
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col is None:
        df.index = pd.to_datetime(df.index).normalize()
        return df.join(cache[CHRONOS_COLS], how="left")
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(cache[CHRONOS_COLS], how="left")
    return df.reset_index()


_CROSS_LAG_FEATURES = ['lead_sp500_ret', 'lead_vix_ret', 'lead_btc_ret']


def add_cross_lag_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Merge leader (SP500, VIX, BTC) recent moves as features, aligned with an
    as-of forward-fill (last value known on or before each bar, so no leakage).
    A missing leader table leaves its feature at 0.0 to keep a stable shape."""
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col is None:
        for c in _CROSS_LAG_FEATURES:
            df[c] = 0.0
        return df
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    def _close(table):
        try:
            ref = pd.read_sql(f"SELECT Date, Close FROM {table}", engine,
                              index_col="Date", parse_dates=["Date"])
            ref.index = pd.to_datetime(ref.index).normalize()
            ref.columns = [c.lower() for c in ref.columns]
            ref = ref[~ref.index.duplicated(keep='last')].sort_index()
            return ref['close']
        except Exception:
            return None

    cols = {}
    sp = _close('sp500')
    if sp is not None:
        cols['lead_sp500_ret'] = sp.pct_change(1)
    vx = _close('vix')
    if vx is not None:
        cols['lead_vix_ret'] = vx.diff(1)
    bt = _close('btc')
    if bt is not None:
        cols['lead_btc_ret'] = bt.pct_change(1)

    feats = pd.concat(cols, axis=1).sort_index() if cols else pd.DataFrame()
    for c in _CROSS_LAG_FEATURES:
        if not feats.empty and c in feats.columns:
            df[c] = feats[c].reindex(df.index, method='ffill').fillna(0.0).values
        else:
            df[c] = 0.0
    df = df.reset_index()
    return df


# Feature columns used for training (order matters for model compatibility)
# Base candidate features = the real training list (single source of truth).
# Kept in sync with train_hybrid, which imports active_candidate_features().
CANDIDATE_FEATURES = [
    'close', 'volume', 'vol_z', 'taleb_risk', 'ret_skew', 'var_5',
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'trend_strength', 'rsi',
    'sma_20', 'sma_50', 'macd_hist', 'bb_pos', 'atr', 'vol_ratio',
    'w_ret', 'w_rsi', 'w_trend', 'corr_btc', 'corr_sp500', 'corr_dxy',
]

# Extended experiment set: base minus the raw non-stationary levels close/volume,
# plus the macro regime features (computed but never wired in before), volatility-
# normalized returns, cross-asset lead-lag returns, and calendar encodings.
CANDIDATE_FEATURES_EXT = [
    'vol_z', 'taleb_risk', 'ret_skew', 'var_5',
    'ret_1', 'ret_5', 'ret_10', 'ret_20', 'trend_strength', 'rsi',
    'sma_20', 'sma_50', 'macd_hist', 'bb_pos', 'atr', 'vol_ratio',
    'w_ret', 'w_rsi', 'w_trend', 'corr_btc', 'corr_sp500', 'corr_dxy',
    'macro_tnx', 'macro_tnx_chg20', 'macro_vix', 'macro_vix_chg5', 'macro_dxy_chg20',
    'ret_1_vn', 'ret_5_vn',
    'lead_sp500_ret', 'lead_vix_ret', 'lead_btc_ret',
    'cal_dow', 'cal_mpos',
]


def active_candidate_features():
    """Extended list by default (the adopted set); the base list only when
    GTRADE_FEATURE_SET=base; plus any GTRADE_EXTRA_FEATURES (the auto-research
    agent's variant runs append proposed feature names here); minus any
    GTRADE_DROP_FEATURES (the auto-research pruning axis removes names here).
    GTRADE_DROP_FEATURES defaults to empty, so the list and feature_version are
    unchanged for production."""
    if (os.getenv("GTRADE_FEATURE_SET") or "ext").strip().lower() == "base":
        base = CANDIDATE_FEATURES
    else:
        base = CANDIDATE_FEATURES_EXT
    extra = [f.strip() for f in (os.getenv("GTRADE_EXTRA_FEATURES") or "").split(",") if f.strip()]
    feats = base + [f for f in extra if f not in base]
    drop = {f.strip() for f in (os.getenv("GTRADE_DROP_FEATURES") or "").split(",") if f.strip()}
    return [f for f in feats if f not in drop]


def feature_version() -> str:
    """Short stable id of the current feature space.

    Changes whenever CANDIDATE_FEATURES changes (e.g. the 21 to 23 jump when
    skew/var_5 were added), so prediction_log can tag each forward prediction
    with the model generation that made it and never blend the live track record
    of an old feature set with a new one.
    """
    digest = hashlib.sha1(",".join(active_candidate_features()).encode()).hexdigest()
    return digest[:8]
