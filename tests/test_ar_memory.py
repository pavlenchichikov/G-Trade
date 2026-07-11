from core import ar_memory as am


def test_tried_roundtrip_and_count():
    assert am.tried_seen("spec", "sigA") is False
    am.tried_add("spec", "sigA")
    am.tried_add("spec", "sigA")          # idempotent
    am.tried_add("genome", "sigB")
    assert am.tried_seen("spec", "sigA") is True
    assert am.tried_seen("genome", "sigB") is True
    assert am.tried_seen("genome", "sigA") is False   # kinds are separate
    assert am.tried_count() == 2


def test_tried_corrupt_file_treated_as_empty():
    with open(am.TRIED_PATH, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert am.tried_seen("spec", "x") is False
    am.tried_add("spec", "x")             # recovers by rewriting
    assert am.tried_seen("spec", "x") is True


def test_findings_append_and_summary():
    am.tried_add("spec", "s1")
    am.tried_add("spec", "s2")
    am.findings_append({"ts": "t1", "mode": "axes",
                        "winners": [{"axis": "features", "adoptable": True},
                                    {"axis": "labeling", "adoptable": False}]})
    am.findings_append({"ts": "t2", "mode": "qd", "winners": []})
    s = am.findings_summary()
    assert s == {"experiments": 2, "adoptable": 1, "replicated": 0}


def test_base_key_sensitivity(monkeypatch):
    monkeypatch.setattr(am, "data_fingerprint", lambda subset: "fp1")
    k1 = am.base_key("SP500,NVDA", {})
    assert am.base_key("SP500,NVDA", {}) == k1                       # deterministic
    assert am.base_key("SP500,NVDA", {"GTRADE_SCREEN_ONLY": "1"}) != k1  # env matters
    assert am.base_key("MSFT", {}) != k1                             # subset matters
    monkeypatch.setattr(am, "data_fingerprint", lambda subset: "fp2")
    assert am.base_key("SP500,NVDA", {}) != k1                       # new data invalidates


def test_base_key_feature_version(monkeypatch):
    import core.features
    monkeypatch.setattr(am, "data_fingerprint", lambda subset: "fp")
    k1 = am.base_key("SP500", {})
    monkeypatch.setattr(core.features, "feature_version", lambda: "other-version")
    assert am.base_key("SP500", {}) != k1


def test_cache_roundtrip_and_cap(monkeypatch):
    rows = [{"Asset": "SP500", "Score": 1.5}]
    assert am.cache_get("k1") is None
    am.cache_put("k1", rows)
    assert am.cache_get("k1") == rows
    monkeypatch.setattr(am, "CACHE_CAP", 2)
    am.cache_put("k2", rows)
    am.cache_put("k3", rows)
    assert am.cache_get("k1") is None      # oldest evicted
    assert am.cache_get("k3") == rows


def test_data_fingerprint_reads_max_dates(tmp_path, monkeypatch):
    import sqlite3
    db = str(tmp_path / "fp.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sp500 (Date TEXT, Close REAL)")
    con.execute("INSERT INTO sp500 VALUES ('2026-06-30', 1.0)")
    con.execute("INSERT INTO sp500 VALUES ('2026-07-01', 2.0)")
    con.commit()
    con.close()
    monkeypatch.setattr(am, "DB_PATH", db)
    fp1 = am.data_fingerprint("SP500")
    assert "2026-07-01" in fp1
    con = sqlite3.connect(db)
    con.execute("INSERT INTO sp500 VALUES ('2026-07-02', 3.0)")
    con.commit()
    con.close()
    assert am.data_fingerprint("SP500") != fp1     # new bar changes the fingerprint
    # a missing table is a deterministic marker, not an error
    assert am.data_fingerprint("NOPE") == am.data_fingerprint("NOPE")


def test_replication_seen_and_add():
    assert am.replication_seen("sigX") is False
    n1 = am.replication_add("sigX", "2026-07-03T10:00:00")
    assert n1 == 1
    assert am.replication_seen("sigX") is True          # a prior clear now exists
    n2 = am.replication_add("sigX", "2026-07-03T10:00:00")   # same ts, idempotent
    assert n2 == 1
    n3 = am.replication_add("sigX", "2026-07-04T10:00:00")   # a distinct run
    assert n3 == 2
    assert am.replication_seen("other") is False


def test_replication_corrupt_file_is_empty():
    with open(am.REPLICATION_PATH, "w", encoding="utf-8") as f:
        f.write("{bad")
    assert am.replication_seen("s") is False
    assert am.replication_add("s", "t") == 1


def test_findings_summary_counts_replicated():
    am.tried_add("spec", "s1")
    am.findings_append({"ts": "t1", "winners": [
        {"adoptable": True, "replicated": True},
        {"adoptable": True, "replicated": False},
        {"adoptable": False, "replicated": False}]})
    s = am.findings_summary()
    assert s["adoptable"] == 2
    assert s["replicated"] == 1
    assert s["experiments"] == 1


def test_findings_recent_newest_first_and_cap():
    assert am.findings_recent() == []
    for i in range(3):
        am.findings_append({"ts": "t%d" % i, "winners": []})
    rec = am.findings_recent(2)
    assert [r["ts"] for r in rec] == ["t2", "t1"]      # newest first, capped at 2
    assert len(am.findings_recent()) == 3


def test_tried_recent_returns_last_n():
    for i in range(5):
        am.tried_add("genome", "sig-%d" % i)
    assert am.tried_recent("genome", 2) == ["sig-3", "sig-4"]   # last n, in-order
    assert am.tried_recent("genome", 10) == ["sig-0", "sig-1", "sig-2", "sig-3", "sig-4"]
    assert am.tried_recent("spec", 5) == []                     # empty / unknown kind


def test_findings_all_returns_full_journal(tmp_path, monkeypatch):
    import core.ar_memory as am
    monkeypatch.setattr(am, "FINDINGS_PATH", str(tmp_path / "f.json"))
    am.findings_append({"ts": "t1", "mode": "qd", "winners": []})
    am.findings_append({"ts": "t2", "mode": "regate", "winners": []})
    allf = am.findings_all()
    assert [r["ts"] for r in allf] == ["t1", "t2"]
