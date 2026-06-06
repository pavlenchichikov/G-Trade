# G-Trade

ML trading-signal system for ~150 assets (crypto, US and Russian equities, forex, commodities). Each asset gets a 4-model ensemble (CatBoost, LSTM+Attention, Transformer, TCN) selected by walk-forward backtesting with realistic costs, plus Kelly position sizing and drawdown circuit breakers.

## Pipeline

1. `data_engine.py` pulls daily and weekly OHLCV from Yahoo Finance and the MOEX API into a local SQLite DB (`market.db`).
2. `train_hybrid.py` builds features (returns, volatility, RSI, MACD, SMA, ATR, weekly and cross-asset correlations), trains the ensemble on purged walk-forward splits, and saves the champion plus its scaler and probability calibrator.
3. `predict.py` runs inference across all assets and prints BUY/SELL/WAIT with confidence.
4. `backtest.py` evaluates champions on held-out data: PnL, win rate, Sharpe, directional accuracy, Brier score, and alpha vs buy & hold.
5. `risk_manager.py` and `portfolio.py` handle Kelly sizing, drawdown and daily-loss halts, correlation and sector exposure.

`app.py` is a Streamlit dashboard (signal radar, model consensus, portfolio, risk). A Telegram bot delivers hourly alert scans.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # Telegram token, optional SOCKS5 proxy

python data_engine.py         # fetch market data
python train_hybrid.py        # train models
python predict.py             # console signal radar
streamlit run app.py          # dashboard
```

`python scheduler.py` runs as a daemon: data every 6h, predictions every 4h, daily DB validation.

## Data routing

Run from a Russian IP, the two sources need opposite routing: Yahoo Finance is geo-blocked and needs a foreign exit, MOEX works only from a Russian IP. Run AmneziaVPN in SOCKS5-proxy mode (`127.0.0.1:12334`); `net.py` then routes Yahoo through the proxy and MOEX direct. Override with `GTRADE_PROXY_MODE=auto|on|off`.

TLS verification on outbound requests defaults off (the proxy/firewall can intercept TLS). On a trusted network set `GTRADE_SSL_VERIFY=1` to enforce it.

## GPU

Native TensorFlow on Windows is CPU-only since 2.11, so training runs on CPU here (fine for this workload). To use an NVIDIA GPU, run under WSL2 with `pip install tensorflow[and-cuda]`.

## Config

- `.env` - Telegram credentials, optional SOCKS5 proxy.
- `config.py` - asset map and per-asset buy/sell thresholds.
- `auto_trader_config.json` - paper-trading parameters.

## Tests

```bash
pytest -q
ruff check .
```

## License

MIT
