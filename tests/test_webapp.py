"""Тесты веб-интерфейса через TestClient (без TensorFlow и реальной БД)."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from core import track_record
import webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE prediction_log (
            date TEXT, asset TEXT, signal TEXT, probability REAL,
            actual_next_ret REAL, correct INTEGER, cb_prob REAL, lstm_prob REAL
        )
    """)
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-10','BTC','BUY',0.62,NULL,NULL,0.62,NULL)")
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-09','BTC','SELL',0.58,0.004,0,0.58,NULL)")
    con.commit()
    con.close()
    monkeypatch.setattr(track_record, "DB_PATH", path)
    return TestClient(webapp.app)


def test_radar_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "BTC" in r.text


def test_asset_page(client):
    r = client.get("/asset/BTC")
    assert r.status_code == 200
    assert "2026-06-10" in r.text


def test_asset_page_unknown_404(client):
    assert client.get("/asset/NOPE").status_code == 404


def test_risk_page(client):
    r = client.get("/risk")
    assert r.status_code == 200


def test_market_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "global_regime", lambda: {
        "status": "RISK-ON", "vix_level": "CALM", "vix_value": 13.2,
        "sp500_trend": "BULLISH", "sp500_detail": "Above SMA50, Above SMA200",
        "dxy_trend": "FALLING", "dxy_label": "WEAK DOLLAR",
        "description": "favorable"})
    r = client.get("/market")
    assert r.status_code == 200
    assert "RISK-ON" in r.text          # bull/bear regime status
    assert "Настроение" in r.text       # fear/greed sentiment panel


def test_api_regime(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "global_regime", lambda: {
        "status": "CRISIS", "vix_level": "PANIC", "vix_value": 40.0,
        "sp500_trend": "BEARISH", "sp500_detail": "", "dxy_trend": "RISING",
        "dxy_label": "STRONG DOLLAR", "description": "caution"})
    r = client.get("/api/regime")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "CRISIS"
    assert body["score"] < 25


def test_api_ohlc(client):
    r = client.get("/api/ohlc/BTC")
    assert r.status_code == 200
    body = r.json()
    assert body["asset"] == "BTC" and "series" in body


def test_api_ohlc_unknown_404(client):
    assert client.get("/api/ohlc/NOPE").status_code == 404


def test_guru_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "guru_latest", lambda: [
        {"asset": "BTC", "date": "2026-06-12", "verdict": "AVOID", "pct": 37.5,
         "lynch": 1, "buffett": 1, "graham": 0, "munger": 1,
         "source": "yfinance", "correct_5d": 0}])
    monkeypatch.setattr(dash, "guru_accuracy", lambda: {
        "council": {"accuracy": 0.6, "total": 10, "correct": 6, "avg_return": 0.3,
                    "by_verdict": {}, "horizon": "5d"},
        "individual": {"Lynch": {"accuracy": 0.7, "bullish_calls": 5,
                                 "bullish_correct": 4, "avg_ret_when_bullish": 1.2}}})
    r = client.get("/guru")
    assert r.status_code == 200
    assert "BTC" in r.text and "Lynch" in r.text


def test_models_page_health(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "models_health", lambda: {
        "avg_age_days": 11.2, "oldest_asset": "ASTR", "oldest_age_days": 91.7})
    monkeypatch.setattr(dash, "models_stale", lambda: [
        {"asset": "ASTR", "cb_age_days": 91.7, "score": 1.4, "status": "RETRAIN"}])
    r = client.get("/models")
    assert r.status_code == 200
    assert "Требуют переобучения" in r.text and "ASTR" in r.text


def test_sectors_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "sector_momentum", lambda: [
        {"Sector": "Crypto", "Momentum_Score": -3.8, "Trend": "RISING",
         "Best_Asset": "UNI", "Worst_Asset": "ADA"}])
    monkeypatch.setattr(dash, "sector_heatmap", lambda: {
        "xLabels": ["W1"], "yLabels": ["Crypto"], "data": [[0, 0, 2.0]],
        "min": -2, "max": 2})
    r = client.get("/sectors")
    assert r.status_code == 200
    assert "Crypto" in r.text and "sector-heatmap" in r.text


def test_correlations_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "correlation_stress", lambda: {
        "avg_corr": 0.8, "label": "HIGH", "min_corr": -0.5, "max_corr": 0.95,
        "score": 80, "zone": "g-bad"})
    monkeypatch.setattr(dash, "correlation_heatmap", lambda: {
        "xLabels": ["BTC"], "yLabels": ["BTC"], "data": [[0, 0, 1.0]],
        "min": -1, "max": 1})
    r = client.get("/correlations")
    assert r.status_code == 200
    assert "corr-heatmap" in r.text and "HIGH" in r.text


def test_performance_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "accuracy_timeseries", lambda: {
        "current": [{"date": "2026-06-18", "rolling_acc": 0.6, "predictions_count": 5}],
        "all": []})
    monkeypatch.setattr(dash, "top_leaderboard", lambda limit=20: [
        {"Asset": "BTC", "Accuracy": 0.6, "Predictions": 10, "Correct": 6}])
    r = client.get("/performance")
    assert r.status_code == 200
    assert "perf-chart" in r.text and "BTC" in r.text


def test_news_page(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "news_digest", lambda lang="all", category="all": [
        {"title": "Fed holds rates", "link": "http://x", "source": "Reuters",
         "credibility": 0.9, "sentiment_label": "Neutral", "weighted_score": 0.1,
         "category": "macro", "published": "2026-06-18", "description": "body"}])
    r = client.get("/news")
    assert r.status_code == 200
    assert "Fed holds rates" in r.text
    assert "Reuters" in r.text


def test_home_has_command_center(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "global_regime", lambda: {
        "status": "RISK-ON", "vix_level": "CALM", "vix_value": 12.0,
        "sp500_trend": "BULLISH", "sp500_detail": "", "dxy_trend": "FALLING",
        "dxy_label": "WEAK DOLLAR", "description": "ok"})
    monkeypatch.setattr(dash, "market_breadth", lambda: {
        "BUY": 3, "SELL": 1, "WAIT": 2, "total": 6, "actionable": 4})
    monkeypatch.setattr(dash, "top_leaderboard", lambda days=30, limit=8: [])
    r = client.get("/")
    assert r.status_code == 200
    assert "Рынок" in r.text            # nav link to /market
    assert "command-center" in r.text   # command-center section
    assert "Настроение" in r.text       # sentiment card


def test_api_signals(client):
    data = client.get("/api/signals").json()
    assert data[0]["asset"] == "BTC"
    assert data[0]["signal"] == "BUY"


def test_api_track(client):
    data = client.get("/api/track/BTC").json()
    assert data["asset"] == "BTC"
    assert len(data["track"]) == 2


def test_api_risk(client):
    data = client.get("/api/risk").json()
    assert "config" in data


def test_api_prices(client):
    data = client.get("/api/prices/BTC?days=30").json()
    assert data["asset"] == "BTC"
    assert isinstance(data["series"], list)


def test_api_prices_unknown_404(client):
    assert client.get("/api/prices/NOPE").status_code == 404


def test_models_page(client):
    r = client.get("/models")
    assert r.status_code == 200
