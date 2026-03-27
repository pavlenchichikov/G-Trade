# G-Trade

**Multi-asset ML trading signal system covering 300+ instruments across global markets.**

Combines CatBoost, LSTM+Attention, Transformer, and TCN models into a 4-architecture ensemble with walk-forward validation, Kelly-based position sizing, and real-time Telegram alerts.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Features

- **300+ instruments** — crypto, US stocks (tech, healthcare, finance, consumer), MOEX Russia (50+ tickers), commodities, forex (32 pairs), global indices
- **4-architecture ensemble** — CatBoost + LSTM+Attention + Transformer + TCN with soft gating and meta-stacking
- **Walk-forward backtesting** — realistic simulation with commissions, slippage, and Kelly-sized positions
- **Risk management** — fractional Kelly sizing, drawdown circuit breakers, Taleb tail-risk gate, daily loss limits
- **Portfolio optimization** — correlation matrix, sector exposure limits, diversification scoring (HHI)
- **Telegram alerts** — hourly signal scans with confidence, Kelly allocation, and sector context
- **Paper trading** — virtual portfolio for safe strategy validation
- **Streamlit dashboard** — 4-tab UI: AI Sniper, Guru Council, Portfolio, Risk
- **Automated scheduler** — cron-like task runner for data updates, predictions, and maintenance
- **Market regime detection** — global (VIX, SP500, DXY) and per-asset regime classification
- **News sentiment** — NLP-based sentiment analysis for signal confirmation

## Asset Coverage

| Category | Count | Examples |
|---|---|---|
| Global Indices | 6 | SP500, NASDAQ, DOW, VIX, DXY, TNX |
| Commodities | 4 | GOLD, SILVER, OIL, GAS |
| Crypto | 15 | BTC, ETH, SOL, XRP, DOGE, BNB, ADA, AVAX, DOT, LINK... |
| US Stocks | 40+ | NVDA, TSLA, AAPL, MSFT, GOOGL, AMZN, META, JPM, JNJ... |
| Russia (MOEX) | 50+ | SBER, GAZP, LKOH, ROSN, YNDX, OZON, CHMF, NLMK... |
| Forex | 32 | EURUSD, GBPUSD, USDJPY, AUDUSD, EURGBP, GBPJPY... |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/pavlenchichikov/g-trade.git
cd g-trade
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your Telegram token and proxy settings
```

### 3. Download market data

```bash
python data_engine.py
```

### 4. Train models

```bash
python train_hybrid.py
```

### 5. Run predictions

```bash
# Console radar scan
python predict.py

# Streamlit dashboard
streamlit run app.py

# Or use the launcher
python launcher.py
```

## Architecture

```
data_engine.py          Fetch OHLCV (Yahoo Finance + MOEX API)
       |
   market.db            SQLite: daily + weekly bars for all assets
       |
train_hybrid.py         Feature engineering + 4-architecture ensemble training
       |
hybrid_models/          CatBoost .cbm + LSTM .keras + Transformer + TCN + meta-stacking
       |
  +----+----+----+
  |    |    |    |
predict.py  |  alert_bot.py  app.py (Streamlit)
            |
     backtest.py        Walk-forward validation with Kelly sizing
```

### ML Pipeline

```
Raw OHLCV
    -> Feature Engineering (returns, volatility, RSI, MACD, SMA, trend strength, weekly features)
    -> CatBoost Classifier (gradient boosting)
    -> LSTM + Attention (sequence model, V50 architecture)
    -> Transformer Encoder (self-attention)
    -> TCN (dilated temporal convolutions)
    -> Soft Gating (trend_strength alpha) + Meta-Stacking (LogisticRegression)
    -> Per-asset threshold tuning
    -> BUY / SELL / WAIT signal
```

### Risk Pipeline

```
Signal (BUY 62%)
    -> Kelly Criterion: position size = f(win_rate, avg_win/loss)
    -> Fractional Kelly (25%) with tail-risk penalty
    -> Correlation penalty (reduce if correlated assets open)
    -> Sector crowding check (max 35% crypto, 40% tech, etc.)
    -> Drawdown circuit breaker (-15% halt)
    -> Daily loss limit (-5% halt)
    -> Final allocation: $X into asset
```

## Project Structure

```
G-Trade/
|
|-- Core Pipeline
|   |-- config.py              Global settings, asset map, thresholds
|   |-- data_engine.py         Market data fetcher (Yahoo + MOEX)
|   |-- train_hybrid.py        Feature engineering + model training
|   |-- predict.py             Real-time radar scan (300+ assets)
|   |-- backtest.py            Walk-forward backtesting
|
|-- Risk & Portfolio
|   |-- risk_manager.py        Kelly sizing, drawdown halts, Taleb gate
|   |-- portfolio.py           Correlation, sector limits, diversification
|   |-- paper_trading.py       Virtual portfolio (paper.db)
|   |-- auto_trader.py         Automated signal execution
|
|-- UI & Alerts
|   |-- app.py                 Streamlit dashboard (4 tabs)
|   |-- launcher.py            GUI launcher
|   |-- alert_bot.py           Telegram bot (hourly scans)
|   |-- signal_dashboard.py    Signal summary DataFrame
|
|-- Analytics
|   |-- regime_detector.py     Market regime classification
|   |-- news_analyzer.py       NLP sentiment analysis
|   |-- whatif_simulator.py    Hypothetical backtest scenarios
|   |-- model_health.py        Model quality monitoring
|   |-- performance_tracker.py Signal accuracy logging
|   |-- equity_curve.py        Equity visualization
|   |-- correlation_alert.py   Cross-asset correlation alerts
|   |-- sector_rotation.py     Sector momentum tracking
|
|-- Maintenance
|   |-- scheduler.py           Cron-like task runner
|   |-- db_check.py            Database validation & repair
|   |-- db_backup.py           Incremental backups
|   |-- Diagnostic_DB.py       Deep DB diagnostics
|
|-- Config Files
|   |-- .env.example           Environment variables template
|   |-- requirements.txt       Python dependencies
|   |-- scheduler_config.json  Task schedule settings
|   |-- auto_trader_config.json Trading parameters
|   |-- watchlist.json         Custom asset groups
```

## Key Modules

### data_engine.py
Fetches daily and weekly OHLCV bars for all 300+ assets. Yahoo Finance for international markets, MOEX ISS API for Russian stocks. Incremental updates, retry logic with exponential backoff, thread-pooled parallel fetching (12 workers), SOCKS5 proxy support.

### train_hybrid.py
Feature engineering (returns, volatility z-scores, RSI, MACD, SMA crossovers, trend strength, weekly features) and 4-architecture ensemble training. Walk-forward split, scaler fit on training data only, champion/challenger model promotion with registry tracking.

### predict.py
Loads latest 500 bars per asset, engineers features, runs all 4 models, applies soft-gated ensemble with per-asset thresholds. Outputs colored console radar with signal, confidence, price, and mode (STACK/DUAL/CB ONLY).

### backtest.py
Walk-forward validation with realistic costs (0.1% commission + 0.15% slippage, higher for forex). Kelly-sized positions, Sharpe and Calmar ratios, per-asset and portfolio-level metrics.

### risk_manager.py
Fractional Kelly position sizing with tail-risk (Taleb) penalty and correlation adjustments. Circuit breakers: -15% drawdown halt, -5% daily loss halt. Persistent state in `models/risk_state.json`.

### app.py
Streamlit dashboard with 4 tabs:
- **AI Sniper** — real-time signals with confidence and Kelly allocation
- **Guru Council** — multi-model consensus view
- **Portfolio** — correlation heatmap, sector exposure, diversification score
- **Risk** — Kelly calculator, circuit breaker status, tail-risk monitor

## Scheduler

Automated task execution with configurable intervals:

```bash
# Run as daemon
python scheduler.py

# Check status
python scheduler.py --status

# Run specific tasks manually
python scheduler.py --once data_update predict
```

| Task | Default Interval | Description |
|---|---|---|
| data_update | 6 hours | Fetch latest market data |
| predict | 4 hours | Run signal radar |
| db_check | 24 hours | Validate and repair database |
| news_scan | 3 hours | Sentiment analysis |
| regime_check | 6 hours | Market regime classification |
| train | 24 hours (disabled) | Retrain models |

## GPU Support

The system **auto-detects your GPU** and adapts automatically:

| GPU | VRAM | Status |
|---|---|---|
| No GPU (CPU only) | — | Fully functional, training is slower |
| NVIDIA GTX 1650+ | 4 GB | Supported (memory_growth mode) |
| NVIDIA RTX 2050–2080 | 4–8 GB | Supported (fixed pool, fp16) |
| NVIDIA RTX 3060–3090 | 8–24 GB | Supported (fixed pool, fp16) |
| NVIDIA RTX 4060–4090 | 8–24 GB | Supported (fixed pool, fp16) |
| NVIDIA A100 / H100 | 40–80 GB | Supported (fixed pool, fp16) |

### Setup

```bash
# Auto-detect GPU and install matching CUDA/cuDNN
setup_gpu.bat
```

The script will:
1. Detect your NVIDIA GPU (or skip if none)
2. Install matching CUDA toolkit + cuDNN via conda
3. Install TensorFlow with GPU support (or CPU-only fallback)
4. Verify GPU acceleration with a benchmark

### How it works

- **VRAM auto-sizing**: TF memory pool set to ~60% of available VRAM (minimum 1 GB free for cuDNN + OS)
- **Mixed precision (fp16)**: Enabled automatically on GPUs with Tensor Cores (RTX 20xx+), ~2x speedup
- **Graceful fallback**: If GPU detection fails, falls back to `memory_growth=True` or CPU
- **Thread safety**: Single GPU slot with semaphore prevents cuDNN OOM in parallel training

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_TOKEN` | For alerts | Telegram bot token (from @BotFather) |
| `TELEGRAM_USER_ID` | For alerts | Your Telegram user ID |
| `SOCKS5_PROXY` | Optional | SOCKS5 proxy for firewall bypass |

### Trading Thresholds (config.py)

Per-asset buy/sell thresholds tuned via backtesting. Default: 0.55 (BUY if probability > threshold).

### Auto Trader (auto_trader_config.json)

```json
{
  "min_confidence": 0.6,
  "max_positions": 10,
  "default_amount": 500,
  "use_kelly_sizing": true,
  "dry_run": true
}
```

## License

MIT
