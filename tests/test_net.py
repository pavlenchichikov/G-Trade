"""Tests for net.http_get adaptive routing, failover and per-host learning."""
import pytest
import requests

import net


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Fresh route cache, no real sleeps, proxy configured + alive by default."""
    net.reset_route_cache()
    monkeypatch.setattr(net.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(net, "SOCKS5_PROXY", "socks5h://127.0.0.1:12334")
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: True)
    yield
    net.reset_route_cache()


def _record_get(monkeypatch, behavior):
    """Install a fake requests.get. `behavior(proxies)` returns a FakeResp or
    raises. Records the proxies passed on each call in `calls`."""
    calls = []

    def fake_get(url, *, headers=None, proxies=None, timeout=None, verify=None, **kw):
        calls.append({"proxies": proxies, "timeout": timeout})
        return behavior(proxies)

    monkeypatch.setattr(net.requests, "get", fake_get)
    return calls


def _route_name(proxies):
    return "direct" if proxies is None else "proxy"


# --------------------------------------------------------------------------

def test_direct_only_when_no_proxy(monkeypatch):
    monkeypatch.setattr(net, "SOCKS5_PROXY", "")
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: False)
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    r = net.http_get("https://iss.moex.com/x", retries=1)
    assert r.status_code == 200
    assert [_route_name(c["proxies"]) for c in calls] == ["direct"]


def test_failover_to_proxy_on_transport_error(monkeypatch):
    def behavior(proxies):
        if proxies is None:  # direct fails like a blackholed path
            raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=5)")
        return FakeResp(200)
    calls = _record_get(monkeypatch, behavior)
    r = net.http_get("https://iss.moex.com/x", retries=1)
    assert r.status_code == 200
    assert [_route_name(c["proxies"]) for c in calls] == ["direct", "proxy"]
    # winning route remembered
    assert net._learned_route("iss.moex.com") == "proxy"


def test_sticky_route_tried_first(monkeypatch):
    net._remember_route("iss.moex.com", "proxy")
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    net.http_get("https://iss.moex.com/x", retries=1)
    # learned 'proxy' must be attempted before 'direct'
    assert _route_name(calls[0]["proxies"]) == "proxy"


def test_total_failure_clears_cache_and_raises(monkeypatch):
    net._remember_route("iss.moex.com", "proxy")
    _record_get(monkeypatch, lambda proxies: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("reset")))
    with pytest.raises(requests.exceptions.RequestException):
        net.http_get("https://iss.moex.com/x", retries=1)
    assert net._learned_route("iss.moex.com") is None


def test_fast_fail_timeout_tuple_passed(monkeypatch):
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    net.http_get("https://iss.moex.com/x", retries=1)
    assert calls[0]["timeout"] == (net.CONNECT_TIMEOUT, net.READ_TIMEOUT)


def test_solo_route_widens_connect_timeout(monkeypatch):
    # Proxy dead - only the direct route. With nothing to fail over to, the
    # fast-failover connect budget should give way to the longer solo budget so
    # a slow-but-usable path isn't killed prematurely. Read budget unchanged.
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: False)
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    net.http_get("https://query1.finance.yahoo.com/x", retries=1)
    assert calls[0]["timeout"] == (net.SOLO_CONNECT_TIMEOUT, net.READ_TIMEOUT)


def test_explicit_timeout_not_overridden_on_solo_route(monkeypatch):
    monkeypatch.setattr(net, "is_proxy_alive", lambda *a, **k: False)
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    net.http_get("https://query1.finance.yahoo.com/x", retries=1, timeout=(3, 7))
    assert calls[0]["timeout"] == (3, 7)


def test_validate_failure_triggers_failover(monkeypatch):
    # Both routes return HTTP 200, but direct's body is an "error" payload.
    def behavior(proxies):
        if proxies is None:
            return FakeResp(200, {"chart": {"error": "Unauthorized"}})
        return FakeResp(200, {"chart": {"result": [{}]}})

    def validate(r):
        chart = r.json().get("chart", {})
        return not chart.get("error") and bool(chart.get("result"))

    calls = _record_get(monkeypatch, behavior)
    r = net.http_get("https://query1.finance.yahoo.com/x", retries=1, validate=validate)
    assert r.json()["chart"]["result"]  # got the good (proxy) response
    assert [_route_name(c["proxies"]) for c in calls] == ["direct", "proxy"]


def test_5xx_retries_within_route_then_succeeds(monkeypatch):
    seq = [FakeResp(503), FakeResp(200)]
    calls = _record_get(monkeypatch, lambda proxies: seq.pop(0))
    r = net.http_get("https://iss.moex.com/x", retries=3)
    assert r.status_code == 200
    # two attempts on the same (direct) route, no failover needed
    assert [_route_name(c["proxies"]) for c in calls] == ["direct", "direct"]


def test_forced_direct_ignores_proxy(monkeypatch):
    calls = _record_get(monkeypatch, lambda proxies: FakeResp(200))
    net.http_get("https://iss.moex.com/x", route="direct", retries=1)
    assert all(c["proxies"] is None for c in calls)
    assert len(calls) == 1
