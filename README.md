# G-Trade

ML-powered trading signal system. Covers 300+ assets — crypto, US/Russian equities, forex, commodities. Runs an ensemble of four model architectures (CatBoost, LSTM+Attention, Transformer, TCN), does walk-forward backtesting with realistic costs, and manages risk with Kelly sizing and circuit breakers.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)

## How it works

1. `data_engine.py` pulls daily/weekly OHLCV bars from Yahoo Finance and MOEX API into a local SQLite database
2. `train_hybrid.py` engineers features (returns, vol, RSI, MACD, SMA crossovers, trend strength, etc.) and trains a 4-model ensemble with walk-forward splits
3. `predict.py` runs inference on all assets — outputs BUY/SELL/WAIT with confidence and Kelly allocation
4. `risk_manager.py` sizes positions via fractional Kelly, checks drawdown limits (-15% halt), daily loss limits (-5% halt), and a Taleb tail-risk gate
5. `portfolio.py` tracks correlations, sector exposure, and diversification

There's a Streamlit dashboard (`app.py`) with four tabs: signal radar, multi-model consensus, portfolio analytics, and risk monitor. Plus a Telegram bot for hourly alert scans.

## Quick start

```bash
git clone https://github.com/pavlenchichikov/g-trade.git
cd g-trade
pip install -r requirements.txt
cp .env.example .env   # add Telegram token, proxy if needed
```

```bash
python data_engine.py       # fetch market data
python train_hybrid.py      # train models
python predict.py           # console radar scan
streamlit run app.py        # dashboard
```

## GPU

Auto-detects NVIDIA GPUs. Works fine on CPU too (just slower training). Run `setup_gpu.bat` to install matching CUDA/cuDNN automatically. Mixed precision (fp16) kicks in on RTX 20xx+ for ~2x speedup.

## Scheduler

`python scheduler.py` runs as a daemon — fetches data every 6h, runs predictions every 4h, validates the DB daily. Configurable via `scheduler_config.json`.

## Config

- `.env` — Telegram credentials, SOCKS5 proxy (optional, needed for Yahoo/Telegram behind firewalls)
- `config.py` — asset map, per-asset buy/sell thresholds
- `auto_trader_config.json` — paper trading parameters (confidence floor, max positions, Kelly toggle)

## License

MIT
