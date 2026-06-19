"""Tests for core.dashboard - cached accessors for the web dashboard."""

import core.dashboard as dash


def test_ttl_cache_returns_cached_value():
    dash.cache_clear()
    calls = {"n": 0}

    @dash.ttl_cache(100)
    def f():
        calls["n"] += 1
        return calls["n"]

    assert f() == 1
    assert f() == 1          # served from cache, not recomputed
    assert calls["n"] == 1


def test_global_regime_fallback_on_failure(monkeypatch):
    dash.cache_clear()
    import regime_detector
    monkeypatch.setattr(regime_detector, "get_global_regime",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    r = dash.global_regime()
    assert r["status"] == "UNKNOWN"
    assert r["vix_value"] is None


def test_regime_score_maps_status():
    assert dash.regime_score({"status": "RISK-ON"}) > 70
    assert dash.regime_score({"status": "CRISIS"}) < 25
    assert dash.regime_score({"status": "UNKNOWN"}) == 50


def test_market_breadth_counts(monkeypatch):
    dash.cache_clear()
    from core import track_record
    monkeypatch.setattr(track_record, "latest_signals",
                        lambda: [{"signal": "BUY"}, {"signal": "BUY"},
                                 {"signal": "SELL"}, {"signal": "WAIT"}])
    b = dash.market_breadth()
    assert b["BUY"] == 2 and b["SELL"] == 1 and b["WAIT"] == 1
    assert b["total"] == 4 and b["actionable"] == 3


def test_top_leaderboard_graceful_empty(monkeypatch):
    dash.cache_clear()
    import performance_tracker as pt
    import pandas as pd
    monkeypatch.setattr(pt, "get_leaderboard",
                        lambda days=30, model_version=None: pd.DataFrame())
    assert dash.top_leaderboard() == []


def test_top_leaderboard_records(monkeypatch):
    dash.cache_clear()
    import performance_tracker as pt
    import pandas as pd
    df = pd.DataFrame([
        {"Asset": "BTC", "Accuracy": 0.6, "Predictions": 10, "Correct": 6},
        {"Asset": "ETH", "Accuracy": 0.5, "Predictions": 8, "Correct": 4},
    ])
    monkeypatch.setattr(pt, "get_leaderboard",
                        lambda days=30, model_version=None: df)
    rows = dash.top_leaderboard(limit=1)
    assert len(rows) == 1 and rows[0]["Asset"] == "BTC"


def test_sentiment_score_extremes():
    fear = dash.sentiment_score(
        {"vix_value": 40, "sp500_trend": "BEARISH"}, {"BUY": 0, "SELL": 10})
    greed = dash.sentiment_score(
        {"vix_value": 12, "sp500_trend": "BULLISH"}, {"BUY": 10, "SELL": 0})
    assert fear < 30
    assert greed > 70


def test_sentiment_label_bands():
    assert dash.sentiment_label(10) == "Extreme Fear"
    assert dash.sentiment_label(50) == "Neutral"
    assert dash.sentiment_label(90) == "Extreme Greed"


def test_market_sentiment_composes(monkeypatch):
    dash.cache_clear()
    monkeypatch.setattr(dash, "global_regime", lambda: {
        "vix_value": 12, "vix_level": "CALM", "sp500_trend": "BULLISH"})
    monkeypatch.setattr(dash, "market_breadth", lambda: {"BUY": 8, "SELL": 2})
    s = dash.market_sentiment()
    assert s["score"] > 60
    assert s["label"] in ("Greed", "Extreme Greed")


def test_gauge_zone():
    assert dash.gauge_zone(80) == ""        # green / bullish / greed
    assert dash.gauge_zone(45) == "g-warn"  # amber / neutral
    assert dash.gauge_zone(20) == "g-bad"   # red / bearish / fear


def test_sector_momentum_records(monkeypatch):
    dash.cache_clear()
    import sector_rotation as sr
    import pandas as pd
    df = pd.DataFrame([{"Sector": "Crypto", "Momentum_Score": -3.8,
                        "Trend": "RISING", "Best_Asset": "UNI", "Worst_Asset": "ADA"}])
    monkeypatch.setattr(sr, "get_sector_momentum", lambda weeks=4: df)
    rows = dash.sector_momentum()
    assert rows[0]["Sector"] == "Crypto" and rows[0]["Trend"] == "RISING"


def test_sector_heatmap_shape(monkeypatch):
    dash.cache_clear()
    import sector_rotation as sr
    import pandas as pd
    df = pd.DataFrame([{"Sector": "Crypto", "W1": 2.0, "W2": -16.0},
                       {"Sector": "US Tech", "W1": 1.0, "W2": 0.5}])
    monkeypatch.setattr(sr, "get_sector_returns", lambda weeks=8: df)
    h = dash.sector_heatmap()
    assert h["yLabels"] == ["Crypto", "US Tech"]
    assert h["xLabels"] == ["W1", "W2"]
    assert len(h["data"]) == 4
    assert h["max"] >= 16.0 and h["min"] <= -16.0  # symmetric range


def test_correlation_stress_zone(monkeypatch):
    dash.cache_clear()
    import correlation_alert as ca
    monkeypatch.setattr(ca, "get_stress_indicator",
                        lambda: {"avg_corr": 0.8, "label": "HIGH"})
    s = dash.correlation_stress()
    assert s["score"] == 80 and s["zone"] == "g-bad"
    dash.cache_clear()
    monkeypatch.setattr(ca, "get_stress_indicator",
                        lambda: {"avg_corr": 0.03, "label": "LOW"})
    assert dash.correlation_stress()["zone"] == ""


def test_correlation_heatmap_square(monkeypatch):
    dash.cache_clear()
    import correlation_alert as ca
    import pandas as pd
    df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]],
                      index=["BTC", "ETH"], columns=["BTC", "ETH"])
    monkeypatch.setattr(ca, "get_key_pairs_matrix", lambda: df)
    h = dash.correlation_heatmap()
    assert h["xLabels"] == ["BTC", "ETH"] and len(h["data"]) == 4


def test_accuracy_timeseries_current_and_all(monkeypatch):
    dash.cache_clear()
    import performance_tracker as pt
    import pandas as pd
    df = pd.DataFrame([{"date": "2026-06-18", "rolling_acc": 0.6, "predictions_count": 10}])
    monkeypatch.setattr(pt, "get_accuracy_history",
                        lambda window=7, model_version=None: df)
    ts = dash.accuracy_timeseries()
    assert ts["current"] and ts["all"]
    assert ts["current"][0]["rolling_acc"] == 0.6


def test_guru_latest_picks_latest_per_asset(tmp_path):
    dash.cache_clear()
    import sqlite3
    db = str(tmp_path / "market.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE guru_log (date TEXT, asset TEXT, lynch_score INT, "
        "buffett_score INT, graham_score INT, munger_score INT, council_pct REAL, "
        "council_verdict TEXT, data_source TEXT, price_at_signal REAL, ret_1d REAL, "
        "ret_5d REAL, ret_20d REAL, correct_1d INT, correct_5d INT, correct_20d INT)")
    con.executemany(
        "INSERT INTO guru_log (date,asset,lynch_score,buffett_score,graham_score,"
        "munger_score,council_pct,council_verdict,data_source,correct_5d) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("2026-06-10", "BTC", 2, 2, 1, 2, 87.5, "BUY", "yfinance", 1),
            ("2026-06-12", "BTC", 1, 1, 0, 1, 37.5, "AVOID", "yfinance", 0),
            ("2026-06-11", "ETH", 0, 0, 0, 2, 25.0, "AVOID", "technical", None),
        ])
    con.commit(); con.close()
    rows = dash.guru_latest(db_path=db)
    by = {r["asset"]: r for r in rows}
    assert by["BTC"]["date"] == "2026-06-12" and by["BTC"]["verdict"] == "AVOID"
    assert by["ETH"]["verdict"] == "AVOID" and by["ETH"]["source"] == "technical"


def test_guru_accuracy_wraps(monkeypatch):
    dash.cache_clear()
    import guru_tracker as gt
    monkeypatch.setattr(gt, "get_guru_accuracy",
                        lambda days=30, horizon="5d": {"accuracy": 0.6, "total": 10,
                                                       "correct": 6, "by_verdict": {}})
    monkeypatch.setattr(gt, "get_guru_individual_accuracy",
                        lambda days=30, horizon="5d": {"Lynch": {"accuracy": 0.7}})
    a = dash.guru_accuracy()
    assert a["council"]["accuracy"] == 0.6 and "Lynch" in a["individual"]


def test_models_health_and_stale(monkeypatch):
    dash.cache_clear()
    import model_health as mh
    monkeypatch.setattr(mh, "get_health_summary",
                        lambda: {"cb_count": 154, "avg_age_days": 11.2})
    monkeypatch.setattr(mh, "get_stale_models",
                        lambda max_age_days=30: [{"asset": "ASTR", "status": "RETRAIN",
                                                  "score": 1.4, "cb_age_days": 91.7}])
    assert dash.models_health()["cb_count"] == 154
    assert dash.models_stale()[0]["asset"] == "ASTR"


def test_news_digest_passes_filters(monkeypatch):
    dash.cache_clear()
    import news_analyzer
    seen = {}

    def fake(max_per_source=5, lang_filter="all", category_filter="all",
             fetch_summaries=True):
        seen["lang"] = lang_filter
        seen["cat"] = category_filter
        return [{"title": "A", "weighted_score": 0.5},
                {"title": "B", "weighted_score": 0.2}]

    monkeypatch.setattr(news_analyzer, "fetch_authority_digest", fake)
    items = dash.news_digest(lang="ru", category="macro", limit=1)
    assert seen == {"lang": "ru", "cat": "macro"}
    assert len(items) == 1 and items[0]["title"] == "A"


def test_news_digest_empty_on_failure(monkeypatch):
    dash.cache_clear()
    import news_analyzer
    monkeypatch.setattr(news_analyzer, "fetch_authority_digest",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("net")))
    assert dash.news_digest() == []
