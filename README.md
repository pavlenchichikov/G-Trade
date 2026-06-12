# G-Trade

ML trading signals for ~150 assets: crypto, US and Russian stocks, forex, commodities. Each asset has an ensemble of 4 models (CatBoost, LSTM, Transformer, TCN). The best one is picked by walk-forward backtest with commissions. Position sizing is Kelly-based, with drawdown stops.

## How it works

1. `data_engine.py` downloads daily and weekly quotes from Yahoo Finance and MOEX into `market.db` (SQLite).
2. `train_hybrid.py` builds features: returns, volatility, RSI, MACD, SMA, ATR, weekly and cross-asset correlations. Trains the ensemble and saves the champion together with its scaler and probability calibrator.
3. `predict.py` prints BUY/SELL/WAIT with confidence for all assets.
4. `backtest.py` checks champions on held-out data: PnL, win rate, Sharpe, directional accuracy, Brier, alpha vs buy & hold.
5. `risk_manager.py` and `portfolio.py` do position sizing, loss limits and correlations.

`app.py` is a Streamlit dashboard. The Telegram bot sends signals every hour.

## Web UI

```
uvicorn webapp:app --host 0.0.0.0 --port 8000
```

Lightweight web interface, no TensorFlow needed, reads predictions from the database. `/` is the signal radar with per-asset accuracy, `/asset/BTC` is signal history, `/risk` is limits. Same data as JSON under `/api/...`. Works from a phone on the same network.

## Telegram bot

`python alert_bot.py` runs the hourly scan and also:

- commands /top, /signal BTC, /risk, /digest (owner only)
- morning digest, hour is set by GTRADE_DIGEST_HOUR, default 9
- degradation warnings: data older than 7 days or accuracy below 40% on the last 20 verified signals

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # telegram token, proxy if needed

python data_engine.py         # download market data
python train_hybrid.py        # train models
python predict.py             # console signals
streamlit run app.py          # dashboard
```

`python scheduler.py` runs as a daemon: data every 6h, predictions every 4h, daily DB check.

## Network

If SOCKS5_PROXY is set in `.env`, outbound requests go through it. `net.py` checks if the proxy is alive and falls back to a direct connection.

- GTRADE_PROXY_MODE=auto|on|off, default auto
- GTRADE_SSL_VERIFY=0 disables TLS certificate checks. Verification is on by default, turn it off only if your proxy intercepts TLS

## GPU

TensorFlow on Windows is CPU-only since 2.11, so training runs on CPU. Good enough for daily data. For a GPU use WSL2 and `pip install tensorflow[and-cuda]`.

## Config

- `.env` - telegram credentials, proxy
- `config.py` - asset list and buy/sell thresholds
- `auto_trader_config.json` - paper trading settings

## Tests

```bash
pytest -q
ruff check .
```

## License

MIT
