"""Web UI tests via TestClient (no TensorFlow, no real DB)."""

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


def test_asset_page_shows_guru_verdict(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "guru_for_asset", lambda asset: {
        "asset": "BTC", "date": "2026-06-12", "verdict": "AVOID", "pct": 37.5,
        "lynch": 1, "buffett": 1, "graham": 0, "munger": 1,
        "source": "yfinance_live", "correct_5d": 0})
    r = client.get("/asset/BTC")
    assert r.status_code == 200
    assert "Guru Council" in r.text
    assert "AVOID" in r.text


def test_asset_page_guru_empty_state(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "guru_for_asset", lambda asset: None)
    r = client.get("/asset/BTC")
    assert r.status_code == 200
    assert "No verdict yet" in r.text


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
    assert "Sentiment" in r.text        # fear/greed sentiment panel


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
    assert "Retrain due" in r.text and "ASTR" in r.text


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
    assert "Market" in r.text           # nav link to /market
    assert "command-center" in r.text   # command-center section
    assert "Sentiment" in r.text        # sentiment card


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


def test_radar_page_has_taleb_column(client):
    import core.dashboard as dash
    dash.cache_clear()
    r = client.get("/")
    assert r.status_code == 200
    assert "Taleb" in r.text


def test_risk_page_has_tail_risk_panel(client):
    import core.dashboard as dash
    dash.cache_clear()
    r = client.get("/risk")
    assert r.status_code == 200
    assert "Tail risk (Taleb)" in r.text


def test_risk_page_has_risk_alerts_panel(client):
    """The Risk Alerts panel (surfacing the HTML report's alert logic) is present
    on /risk and loads lazily (no heavy asset scan on page render)."""
    r = client.get("/risk")
    assert r.status_code == 200
    assert "<h2>Risk Alerts</h2>" in r.text
    assert 'id="risk-alerts-list"' in r.text


def test_api_risk_alerts_returns_report_alerts(client, monkeypatch):
    """/api/risk/alerts serves the same structured records performance_report builds,
    without recomputing on every request (heavy asset scan is done in the report)."""
    import performance_report
    import webapp

    fake = [
        {"level": "overbought", "message": "BTC: RSI=80 (OVERBOUGHT)"},
        {"level": "regime", "message": "VIX: 34 (FEAR)"},
    ]
    monkeypatch.setattr(performance_report, "collect_risk_alerts", lambda: fake)
    webapp._ALERTS_CACHE.update(ts=0.0, alerts=None)  # start from a cold cache

    data = client.get("/api/risk/alerts?force=1").json()
    assert data["alerts"] == fake
    assert data["cached"] is False
    # a second (non-forced) call is served from cache, not a fresh scan
    assert client.get("/api/risk/alerts").json()["cached"] is True


def test_asset_page_shows_taleb_value(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "taleb_for_asset", lambda asset: 3.4)
    r = client.get("/asset/BTC")
    assert r.status_code == 200
    assert "Taleb tail risk" in r.text
    assert "3.4" in r.text


def test_api_risk_open_and_close_position(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))

    r = client.post("/api/risk/position",
                     json={"asset": "BTC", "direction": "BUY",
                           "size_usd": 500, "entry_price": 100.0})
    assert r.status_code == 200
    assert "BTC" in r.json()["state"]["open_positions"]

    r2 = client.post("/api/risk/position/BTC/close", json={"exit_price": 110.0})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["pnl"] > 0
    assert "BTC" not in body2["state"]["open_positions"]


def test_api_risk_open_position_unknown_asset_404(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))
    r = client.post("/api/risk/position",
                     json={"asset": "NOPE", "direction": "BUY",
                           "size_usd": 500, "entry_price": 100.0})
    assert r.status_code == 404


def test_api_risk_open_position_bad_direction_400(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))
    r = client.post("/api/risk/position",
                     json={"asset": "BTC", "direction": "HOLD",
                           "size_usd": 500, "entry_price": 100.0})
    assert r.status_code == 400


def test_api_risk_close_nonexistent_position_404(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))
    r = client.post("/api/risk/position/BTC/close", json={"exit_price": 100.0})
    assert r.status_code == 404


def test_api_risk_config_update(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_CONFIG_OVERRIDE_PATH",
                         str(tmp_path / "risk_config_override.json"))
    original = dict(risk_manager.RISK_CONFIG)
    try:
        r = client.post("/api/risk/config", json={"max_single_position": 0.2})
        assert r.status_code == 200
        assert r.json()["config"]["max_single_position"] == 0.2
    finally:
        risk_manager.RISK_CONFIG.clear()
        risk_manager.RISK_CONFIG.update(original)


def test_api_risk_config_unknown_key_400(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_CONFIG_OVERRIDE_PATH",
                         str(tmp_path / "risk_config_override.json"))
    r = client.post("/api/risk/config", json={"not_a_key": 1})
    assert r.status_code == 400


def test_api_risk_config_out_of_range_400(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_CONFIG_OVERRIDE_PATH",
                         str(tmp_path / "risk_config_override.json"))
    r = client.post("/api/risk/config", json={"max_portfolio_exposure": 1.5})
    assert r.status_code == 400


def test_api_risk_halt_and_resume(client, monkeypatch, tmp_path):
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))

    r = client.post("/api/risk/halt")
    assert r.status_code == 200
    assert r.json()["manual_halt"] is True

    r2 = client.get("/api/risk")
    assert r2.json()["manual_halt"] is True

    r3 = client.post("/api/risk/resume")
    assert r3.json()["manual_halt"] is False


def test_api_guru_recalculate_no_fundamentals_na(client, monkeypatch):
    """No real fundamentals: honest N/A, and nothing enters the track record."""
    import guru_report
    import guru_tracker
    monkeypatch.setattr(guru_report, "fetch_smartlab_data", lambda: {})
    monkeypatch.setattr(guru_report, "fetch_yf_deep", lambda symbol: None)
    monkeypatch.setattr(guru_report, "get_technical", lambda name: None)
    calls = {}
    monkeypatch.setattr(guru_tracker, "log_guru_verdict",
                        lambda *a, **k: calls.setdefault("logged", True))

    r = client.post("/api/guru/BTC/recalculate")
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "N/A"
    assert body["no_fundamentals"] is True
    assert "logged" not in calls  # not logged to the accuracy track record


def test_api_guru_recalculate_with_fundamentals_logs(client, monkeypatch):
    """Real fundamentals: a value verdict that IS logged."""
    import core.guru as cg
    import guru_report
    import guru_tracker
    monkeypatch.setattr(guru_report, "fetch_smartlab_data", lambda: {})
    monkeypatch.setattr(guru_report, "resolve_fundamentals",
                        lambda *a: {"_source": "yfinance_live", "price": 100.0})
    monkeypatch.setattr(guru_report, "get_technical", lambda name: None)
    fake = {"data_source": "yfinance_live",
            "council": {"pct": 75.0, "verdict": "BUY"},
            "lynch": {"_score": 2, "status": "x", "desc": "y"},
            "buffett": {"_score": 1, "status": "x", "desc": "y"},
            "graham": {"_score": 2, "status": "x", "desc": "y"},
            "munger": {"_score": 1, "status": "x", "desc": "y"}}
    monkeypatch.setattr(cg, "get_guru_analysis", lambda fund, tech: fake)
    calls = {}
    monkeypatch.setattr(guru_tracker, "log_guru_verdict",
                        lambda *a, **k: calls.setdefault("logged", True))

    r = client.post("/api/guru/AAPL/recalculate")
    assert r.status_code == 200
    assert r.json()["verdict"] == "BUY"
    assert calls.get("logged") is True


def test_guru_page_value_overlay(client):
    """Guru page renders the 60d value-horizon labels (overlay redesign)."""
    r = client.get("/guru")
    assert r.status_code == 200
    assert "Council accuracy (60d)" in r.text
    assert "ML signal" in r.text or "value overlay" in r.text


def test_api_guru_recalculate_unknown_404(client):
    assert client.post("/api/guru/NOPE/recalculate").status_code == 404


def test_api_prices(client):
    data = client.get("/api/prices/BTC?days=30").json()
    assert data["asset"] == "BTC"
    assert isinstance(data["series"], list)


def test_api_prices_unknown_404(client):
    assert client.get("/api/prices/NOPE").status_code == 404


def test_models_page(client):
    r = client.get("/models")
    assert r.status_code == 200


def test_portfolio_page_empty(client, monkeypatch, tmp_path):
    """No positions: page renders with empty analytics (pm unavailable too)."""
    import core.dashboard as dash
    import risk_manager
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(tmp_path / "risk_state.json"))
    monkeypatch.setattr(dash, "portfolio_manager", lambda: None)
    r = client.get("/portfolio")
    assert r.status_code == 200
    assert "Diversification" in r.text and "Sector exposure" in r.text
    d = client.get("/api/portfolio").json()
    assert d["holdings"] == [] and d["diversification"] == 100.0


def test_portfolio_analytics_with_positions(client, monkeypatch, tmp_path):
    """Positions + a stub manager: holdings, sector heat, diversification,
    and correlated-open warnings are all surfaced."""
    import json

    import pandas as pd

    import core.dashboard as dash
    import risk_manager
    rs = tmp_path / "risk_state.json"
    rs.write_text(json.dumps({"current_capital": 10000, "open_positions": {
        "BTC": {"size_usd": 1000, "direction": "BUY", "entry_price": 50000},
        "ETH": {"size_usd": 500, "direction": "BUY", "entry_price": 3000}}}))
    monkeypatch.setattr(risk_manager, "RISK_STATE_PATH", str(rs))

    class StubPM:
        def get_sector(self, a):
            return "CRYPTO"

        def get_correlated_assets(self, a, open_only=None):
            return ["ETH"] if a == "BTC" else ["BTC"]

        def get_diversification_score(self, fractions):
            return 42.0

        def get_portfolio_heat(self, fractions):
            return {"CRYPTO": sum(fractions.values())}

        def get_correlation_matrix(self):
            return pd.DataFrame([[1.0, 0.8], [0.8, 1.0]],
                                index=["BTC", "ETH"], columns=["BTC", "ETH"])

    monkeypatch.setattr(dash, "portfolio_manager", lambda: StubPM())

    d = client.get("/api/portfolio").json()
    assert len(d["holdings"]) == 2
    assert d["diversification"] == 42.0
    # 1500 / 10000 = 0.15 CRYPTO exposure, under the 0.35 limit
    crypto = next(h for h in d["heat"] if h["sector"] == "CRYPTO")
    assert round(crypto["exposure"], 3) == 0.15 and crypto["over"] is False
    btc = next(h for h in d["holdings"] if h["asset"] == "BTC")
    assert "ETH" in btc["correlated_open"]

    # render the HTML too - exercises the sector-heat bar + correlation heatmap
    # template paths (not just the JSON snapshot)
    html = client.get("/portfolio")
    assert html.status_code == 200
    assert "CRYPTO" in html.text and "BTC" in html.text


def test_whatif_page_renders(client):
    r = client.get("/whatif")
    assert r.status_code == 200
    assert "What-If Simulator" in r.text


def test_api_whatif_assets_mode(client, monkeypatch):
    """Assets mode forwards only known tickers and the chosen strategy."""
    import whatif_simulator
    captured = {}

    def fake_simulate(assets, capital=10000.0, days_back=90, strategy="equal"):
        captured["assets"] = list(assets)
        captured["strategy"] = strategy
        return {"initial": 1000, "final": 1100, "return_pct": 10.0, "max_drawdown": 2.0,
                "sharpe": 1.2, "trades": 3,
                "equity_curve": [["2026-01-01", 1000], ["2026-01-02", 1100]],
                "per_asset": {"BTC": {"return_pct": 10.0, "trades": 3, "max_dd": 2.0}}}

    monkeypatch.setattr(whatif_simulator, "simulate", fake_simulate)
    r = client.post("/api/whatif", json={
        "mode": "assets", "assets": ["BTC", "NOPE"], "capital": 1000,
        "days_back": 90, "strategy": "kelly"})
    assert r.status_code == 200
    d = r.json()
    assert d["final"] == 1100
    assert captured["assets"] == ["BTC"]      # NOPE filtered out
    assert captured["strategy"] == "kelly"


def test_api_whatif_top_mode(client, monkeypatch):
    import whatif_simulator

    def fake_top(n=5, capital=10000.0, days_back=90):
        return {"initial": 5000, "final": 5500, "return_pct": 10.0, "max_drawdown": 1.0,
                "sharpe": 1.0, "trades": 5, "equity_curve": [["d", 5000], ["e", 5500]],
                "per_asset": {}}

    monkeypatch.setattr(whatif_simulator, "simulate_top_n", fake_top)
    r = client.post("/api/whatif", json={"mode": "top", "top_n": 3, "capital": 5000, "days_back": 60})
    assert r.status_code == 200 and r.json()["final"] == 5500


def test_api_whatif_no_valid_assets(client):
    r = client.post("/api/whatif", json={"mode": "assets", "assets": ["NOPE"], "capital": 1000})
    assert r.status_code == 200 and "error" in r.json()


def test_loop_page_and_api(client, monkeypatch, tmp_path):
    import json
    import webapp
    p = tmp_path / "loop_state.json"
    p.write_text(json.dumps({
        "last_run": "2026-06-24T10:00:00", "steps": {"predict": {"status": "ok", "msg": ""}},
        "assets": [{"asset": "AAPL", "status": "propose", "reasons": ["acc 0.40 below floor 0.50"],
                    "acc": 0.40, "baseline_acc": 0.62, "age_days": 5, "stale": False,
                    "miss_streak": 3, "acc_trend": -0.2}],
        "proposed": ["AAPL"], "approved": [], "history": []}))
    monkeypatch.setattr(webapp, "LOOP_STATE_PATH", str(p))

    r = client.get("/loop")
    assert r.status_code == 200 and "AAPL" in r.text

    d = client.get("/api/loop").json()
    assert d["proposed"] == ["AAPL"]

    a = client.post("/api/loop/approve", json={"assets": ["AAPL"]})
    assert a.status_code == 200 and a.json()["approved"] == ["AAPL"]

    dz = client.post("/api/loop/dismiss", json={"asset": "AAPL"})
    assert dz.status_code == 200 and dz.json()["proposed"] == []


def test_api_reconcile(client, monkeypatch):
    import performance_tracker
    monkeypatch.setattr(performance_tracker, "update_actuals",
                        lambda: {"pending": 5, "reconciled": 3})
    r = client.post("/api/reconcile")
    assert r.status_code == 200
    assert r.json() == {"pending": 5, "reconciled": 3}


def test_api_reconcile_error(client, monkeypatch):
    import performance_tracker

    def boom():
        raise RuntimeError("db locked")

    monkeypatch.setattr(performance_tracker, "update_actuals", boom)
    r = client.post("/api/reconcile")
    assert r.status_code == 200
    assert "error" in r.json()


def test_performance_page_has_reconcile_button(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    monkeypatch.setattr(dash, "accuracy_timeseries", lambda: {"current": [], "all": []})
    monkeypatch.setattr(dash, "top_leaderboard", lambda limit=20: [])
    r = client.get("/performance")
    assert r.status_code == 200
    assert "reconcile-btn" in r.text


def test_performance_page_has_meta_shadow_panel(client):
    r = client.get("/performance")
    assert r.status_code == 200
    assert "Meta-sizing shadow" in r.text


def test_performance_page_renders_meta_shadow_sweep(client, monkeypatch):
    import performance_tracker
    monkeypatch.setattr(performance_tracker, "meta_shadow_report", lambda **k: {
        "rows": 40, "baseline_accuracy": 0.52,
        "sweep": [{"thr": 0.55, "kept": 10, "coverage": 0.25, "accuracy": 0.70, "lift": 0.18}],
        "discrimination": {"high_meta_acc": 0.64, "high_meta_n": 22,
                           "low_meta_acc": 0.38, "low_meta_n": 18},
    })
    r = client.get("/performance")
    assert r.status_code == 200
    assert "70.0%" in r.text and "+18.0%" in r.text


def test_research_page_empty(client):
    r = client.get("/research")
    assert r.status_code == 200
    assert "Research" in r.text
    assert "No research runs yet" in r.text        # empty-state


def test_research_page_with_findings(client):
    from core import ar_memory
    ar_memory.findings_append({"ts": "2026-07-03T21:00:00", "mode": "qd",
        "winners": [{"axis": "qd", "adoptable": True, "replicated": True,
                     "clears": 2, "neural_lift": 0.31, "tag": "min dScore 1.20"}]})
    r = client.get("/research")
    assert r.status_code == 200
    assert "REPLICATED" in r.text and "min dScore 1.20" in r.text


def test_api_research(client):
    from core import ar_memory
    ar_memory.findings_append({"ts": "t1", "mode": "axes",
        "winners": [{"axis": "features", "adoptable": False, "tag": "x"}]})
    d = client.get("/api/research").json()
    assert "summary" in d and "rows" in d
    assert any(row["axis"] == "features" for row in d["rows"])


def test_palette_has_research(client):
    d = client.get("/api/palette").json()
    assert any(p[0] == "Research" for p in d["pages"])


def test_radar_shows_live_gate_badge(client, monkeypatch):
    import core.dashboard as dash
    dash.cache_clear()
    sig = {"asset": "EURUSD", "date": "2026-07-16", "signal": "WAIT",
           "signal_raw": "BUY", "gate_reason": "live-gate: FOREX MAJORS 34% (n=126)",
           "probability": 0.9, "acc": {"n": 20, "correct": 6, "acc": 0.3}}
    monkeypatch.setattr(track_record, "latest_signals", lambda *a, **k: [sig])
    html = client.get("/").text
    assert "gated" in html and "live-gate" in html


def test_api_guru_recalculate_all_starts_and_reports(client, monkeypatch):
    """One POST kicks off the batch in a real (daemon) thread; polling status
    reflects the finished counts. The stub returns instantly - no threading
    internals are patched (that would clobber the global module TestClient uses)."""
    import time

    import guru_report
    import webapp

    with webapp._guru_recalc_lock:
        webapp._guru_recalc["running"] = False  # isolate from other tests

    def _stub(progress=None):
        if progress:
            progress(0, 3, None)
        return {"total": 3, "updated": 2, "skipped": 1, "errors": 0}

    monkeypatch.setattr(guru_report, "recalc_all_stocks", _stub)

    assert client.post("/api/guru/recalculate-all").status_code == 200
    st = {}
    for _ in range(60):
        st = client.get("/api/guru/recalculate-all/status").json()
        if not st["running"]:
            break
        time.sleep(0.05)
    assert st["running"] is False
    assert st["total"] == 3 and st["updated"] == 2 and st["skipped"] == 1


def test_api_guru_recalculate_all_no_double_start(client, monkeypatch):
    """A POST while a batch is running is a no-op: it returns the live status
    and does NOT launch a second batch."""
    import guru_report
    import webapp

    ran = {"n": 0}
    monkeypatch.setattr(guru_report, "recalc_all_stocks",
                        lambda progress=None: ran.__setitem__("n", ran["n"] + 1) or {
                            "total": 0, "updated": 0, "skipped": 0, "errors": 0})
    with webapp._guru_recalc_lock:
        webapp._guru_recalc["running"] = True
    try:
        r = client.post("/api/guru/recalculate-all")
        assert r.status_code == 200
        assert r.json()["running"] is True
        assert ran["n"] == 0  # no second batch spawned
    finally:
        with webapp._guru_recalc_lock:
            webapp._guru_recalc["running"] = False


def _client_with_timing(tmp_path, monkeypatch, action, reason):
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE prediction_log (date TEXT, asset TEXT, signal TEXT, "
        "probability REAL, actual_next_ret REAL, correct INTEGER, cb_prob REAL, "
        "lstm_prob REAL, sig_shown TEXT, gate_reason TEXT, timing_action TEXT, "
        "timing_reason TEXT)")
    con.execute("INSERT INTO prediction_log VALUES "
                "('2026-06-10','BTC','BUY',0.62,NULL,NULL,0.62,NULL,'BUY',NULL,?,?)",
                (action, reason))
    con.commit()
    con.close()
    con2 = sqlite3.connect(path)
    con2.execute("CREATE TABLE btc (Date TEXT, Close REAL)")
    con2.execute("INSERT INTO btc VALUES ('2026-06-10', 100.0)")
    con2.commit()
    con2.close()
    monkeypatch.setattr(track_record, "DB_PATH", path)
    from core import timing_policy
    monkeypatch.setattr(timing_policy, "timing_on", lambda: True)
    monkeypatch.setattr(timing_policy, "load_policy", lambda path=None: object())
    return TestClient(webapp.app)


def test_radar_shows_timing_badge_on_divergence(tmp_path, monkeypatch):
    client = _client_with_timing(tmp_path, monkeypatch, "STAY_OUT", "confirm")
    r = client.get("/")
    assert "waiting for confirmation" in r.text


def test_radar_no_badge_when_aligned(tmp_path, monkeypatch):
    client = _client_with_timing(tmp_path, monkeypatch, "HOLD", "ok")
    r = client.get("/")
    assert "policy:" not in r.text


def test_radar_no_badge_when_flag_off(tmp_path, monkeypatch):
    client = _client_with_timing(tmp_path, monkeypatch, "STAY_OUT", "confirm")
    from core import timing_policy
    monkeypatch.setattr(timing_policy, "timing_on", lambda: False)
    r = client.get("/")
    assert "policy:" not in r.text
