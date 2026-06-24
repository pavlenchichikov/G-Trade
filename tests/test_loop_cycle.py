import json

import loop_cycle
from loop_cycle import run_step, scan_assets


def test_run_step_ok():
    r = run_step("predict", lambda: None)
    assert r == {"step": "predict", "status": "ok", "msg": ""}


def test_run_step_isolates_failure():
    def boom():
        raise RuntimeError("data feed down")
    r = run_step("data", boom)
    assert r["step"] == "data" and r["status"] == "failed"
    assert "data feed down" in r["msg"]


def test_scan_assets_flags_proposed_and_ok():
    rows = [
        {"asset": "AAPL", "acc": 0.40, "n": 20, "baseline_acc": 0.62,
         "age_days": 5, "is_stale": False, "recent_outcomes": [1, 0] * 10},
        {"asset": "VIX", "acc": 0.60, "n": 20, "baseline_acc": 0.60,
         "age_days": 5, "is_stale": False, "recent_outcomes": [1] * 20},
    ]
    out = {r["asset"]: r for r in scan_assets(rows)}
    assert out["AAPL"]["status"] == "propose"
    assert out["VIX"]["status"] == "ok"


def test_chunk_splits_evenly():
    from loop_retrain import chunk
    assert chunk(["A", "B", "C", "D", "E"], 2) == [["A", "B"], ["C", "D"], ["E"]]
    assert chunk([], 3) == []


def test_build_rows_uses_real_contracts(tmp_path, monkeypatch):
    from core import track_record

    def fake_asset_accuracy(asset, *args, **kwargs):
        return {"n": 20, "correct": 8, "acc": 0.40}

    def fake_asset_track(asset, *args, **kwargs):
        return [{"correct": 1}, {"correct": 0}, {"correct": 1}, {"correct": 1}]

    def fake_stale_assets(*args, **kwargs):
        return [{"asset": "AAPL", "last_date": None, "age_days": None}]

    monkeypatch.setattr(track_record, "asset_accuracy", fake_asset_accuracy)
    monkeypatch.setattr(track_record, "asset_track", fake_asset_track)
    monkeypatch.setattr(track_record, "stale_assets", fake_stale_assets)

    monkeypatch.setattr(loop_cycle, "FULL_ASSET_MAP", {"AAPL": "AAPL", "MSFT": "MSFT"})

    registry_path = tmp_path / "champion_registry.json"
    registry_path.write_text(
        json.dumps({"AAPL": {"score": 5.0, "updated_at": "2020-01-01T00:00:00"}}),
        encoding="utf-8",
    )
    quality_path = tmp_path / "quality_report.json"
    quality_path.write_text(
        json.dumps([{"Asset": "AAPL", "CB_Acc": 0.62}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(loop_cycle, "REGISTRY_PATH", str(registry_path))
    monkeypatch.setattr(loop_cycle, "QUALITY_PATH", str(quality_path))

    rows = loop_cycle._build_rows()

    by_asset = {r["asset"]: r for r in rows}
    aapl = by_asset["AAPL"]
    assert aapl["baseline_acc"] == 0.62
    assert aapl["age_days"] is not None and aapl["age_days"] > 1000
    assert aapl["is_stale"] is True
