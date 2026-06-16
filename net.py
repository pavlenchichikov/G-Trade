"""Сетевые хелперы с адаптивной маршрутизацией (direct / SOCKS5-прокси).

Рабочий маршрут зависит от VPN: MOEX (iss.moex.com) ходит только по
российскому IP, Yahoo лучше через зарубежный выход, а какой из двух
маршрутов сейчас какой - заранее неизвестно и меняется при переключении VPN.
Поэтому http_get не хардкодит маршрут под источник: пробует доступные
маршруты с быстрым failover (короткий connect-таймаут), запоминает per-host
рабочий маршрут (sticky-кэш с TTL) и сбрасывает его при полном отказе, чтобы
смена VPN переучивалась сама. validate-callback позволяет считать "200 с
ошибкой в теле" отказом маршрута.

Env: GTRADE_PROXY_MODE (auto|on|off), GTRADE_CONNECT_TIMEOUT (5),
GTRADE_READ_TIMEOUT (25), GTRADE_ROUTE_TTL (300), GTRADE_HTTP_RETRIES (3).
API: is_proxy_alive(), proxies_for(route), http_get(url, ...), reset_route_cache().
"""
from __future__ import annotations

import os
import socket
import threading
import time
from urllib.parse import urlparse

import requests

try:
    from config import SOCKS5_PROXY
except Exception:
    SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "")

_PROXY_MODE = (os.getenv("GTRADE_PROXY_MODE") or "auto").strip().lower()
_PROBE_TIMEOUT = float(os.getenv("GTRADE_PROXY_PROBE_TIMEOUT", "1.0"))
_PROBE_TTL = 60.0

# Fast-fail connect + generous read. A blackholed route dies on the short
# connect budget and we fail over instead of waiting out the read timeout.
CONNECT_TIMEOUT = float(os.getenv("GTRADE_CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("GTRADE_READ_TIMEOUT", "25"))
# When only one route is available there is nothing to fail over to, so the
# tight CONNECT_TIMEOUT (meant to race to the next route) only kills a slow but
# usable path. Give the lone route a longer connect budget instead.
SOLO_CONNECT_TIMEOUT = float(os.getenv("GTRADE_SOLO_CONNECT_TIMEOUT", "15"))
_ROUTE_TTL = float(os.getenv("GTRADE_ROUTE_TTL", "300"))
_DEFAULT_RETRIES = int(os.getenv("GTRADE_HTTP_RETRIES", "3"))


def ssl_verify() -> bool:
    """Whether outbound HTTPS requests should verify TLS certificates.

    Defaults to True (secure). Set GTRADE_SSL_VERIFY=0 (or false/no/off) only
    when a TLS-intercepting proxy breaks verification.
    """
    return (os.getenv("GTRADE_SSL_VERIFY") or "1").strip().lower() not in ("0", "false", "no", "off")


_alive_cache: bool | None = None
_cache_ts: float = 0.0

# Per-host learned route {host: (route_name, learned_ts)}. Threads share it.
_route_cache: dict[str, tuple[str, float]] = {}
_route_lock = threading.Lock()


def _endpoint(proxy_url: str):
    try:
        u = urlparse(proxy_url)
        if u.hostname and u.port:
            return u.hostname, int(u.port)
    except Exception:
        pass
    return None


def is_proxy_alive(force: bool = False) -> bool:
    """True if the local SOCKS5 endpoint accepts a TCP connection.

    Mode override: GTRADE_PROXY_MODE=on means always True, off means always
    False, auto (default) uses a cached TCP probe.
    """
    global _alive_cache, _cache_ts
    if _PROXY_MODE == "off":
        return False
    if _PROXY_MODE == "on":
        return bool(SOCKS5_PROXY)
    if not SOCKS5_PROXY:
        return False

    now = time.time()
    if (not force) and _alive_cache is not None and (now - _cache_ts) < _PROBE_TTL:
        return _alive_cache

    ep = _endpoint(SOCKS5_PROXY)
    alive = False
    if ep:
        try:
            with socket.create_connection(ep, timeout=_PROBE_TIMEOUT):
                alive = True
        except Exception:
            alive = False
    _alive_cache, _cache_ts = alive, now
    return alive


def proxies_for(route: str = "auto") -> dict | None:
    """Return a requests `proxies=` dict, or None for direct.

    route='proxy' forces proxy (if configured), 'direct' forces None,
    'auto' uses the probe.
    """
    if route == "direct":
        return None
    if not SOCKS5_PROXY:
        return None
    if route == "proxy" or is_proxy_alive():
        return {"http": SOCKS5_PROXY, "https": SOCKS5_PROXY}
    return None


# --------------------------------------------------------------------------
# Sticky per-host route cache
# --------------------------------------------------------------------------
def reset_route_cache() -> None:
    """Forget all learned routes (used by tests and on explicit re-probe)."""
    with _route_lock:
        _route_cache.clear()


def _host(url: str) -> str:
    return urlparse(url).hostname or url


def _learned_route(host: str) -> str | None:
    with _route_lock:
        v = _route_cache.get(host)
        if not v:
            return None
        name, ts = v
        if (time.time() - ts) < _ROUTE_TTL:
            return name
        _route_cache.pop(host, None)  # expired
        return None


def _remember_route(host: str, name: str) -> None:
    with _route_lock:
        _route_cache[host] = (name, time.time())


def _forget_route(host: str) -> None:
    with _route_lock:
        _route_cache.pop(host, None)


def _candidate_routes(route: str) -> list[tuple[str, dict | None]]:
    """Ordered (name, proxies) candidates to try."""
    proxy_dict = {"http": SOCKS5_PROXY, "https": SOCKS5_PROXY} if SOCKS5_PROXY else None
    if route == "direct":
        return [("direct", None)]
    if route == "proxy":
        return [("proxy", proxy_dict)] if proxy_dict else [("direct", None)]
    # auto: direct is always available; proxy only when its endpoint answers.
    routes: list[tuple[str, dict | None]] = [("direct", None)]
    if proxy_dict and is_proxy_alive():
        routes.append(("proxy", proxy_dict))
    return routes


def http_get(url, *, route="auto", headers=None, timeout=None,
             verify=None, retries=None, validate=None, **kwargs):
    """GET with adaptive route selection, failover and per-host learning.

    route='auto' (default): try every available route, remember the one that
    works for this host, and try it first next time. route='direct'/'proxy'
    force a single route.

    timeout: defaults to (CONNECT_TIMEOUT, READ_TIMEOUT) so a blackholed path
    fails fast and we fail over. Pass a tuple/scalar to override.

    validate: optional callable taking the response and returning a bool. When
    it returns False the response is treated as a route failure (e.g. HTTP 200
    with an error body)
    and the next route is tried. Transport errors, 5xx and 429 always fail
    over / retry regardless of validate.
    """
    if verify is None:
        verify = ssl_verify()
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    explicit_timeout = timeout is not None
    if timeout is None:
        timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)
    if retries is None:
        retries = _DEFAULT_RETRIES

    host = _host(url)
    routes = _candidate_routes(route)

    # Single route means no failover to race toward, so widen the connect budget
    # (unless the caller pinned its own timeout). A blackholed path still dies,
    # just after more patience; a slow-but-usable one now gets through.
    if (not explicit_timeout and len(routes) == 1
            and isinstance(timeout, tuple)):
        timeout = (max(timeout[0], SOLO_CONNECT_TIMEOUT), timeout[1])

    # Put the previously-winning route first.
    if route == "auto" and len(routes) > 1:
        learned = _learned_route(host)
        if learned:
            routes.sort(key=lambda r: 0 if r[0] == learned else 1)

    def _attempt(proxies):
        last = (None, "no attempt")
        for i in range(retries):
            try:
                r = requests.get(url, headers=headers, proxies=proxies,
                                 timeout=timeout, verify=verify, **kwargs)
                if (r.status_code >= 500 or r.status_code == 429) and i < retries - 1:
                    time.sleep(2 ** i)
                    continue
                if validate is not None:
                    try:
                        ok = bool(validate(r))
                    except Exception:
                        ok = False
                    if not ok:
                        # Content-level failure won't change on retry; fail over.
                        return None, f"validate failed (status {r.status_code})"
                return r, None
            except requests.exceptions.RequestException as e:
                last = (None, str(e))
                if i < retries - 1:
                    time.sleep(2 ** i)
                    continue
        return last

    last_err = None
    for name, proxies in routes:
        r, err = _attempt(proxies)
        if r is not None:
            if route == "auto":
                _remember_route(host, name)
            return r
        last_err = err

    if route == "auto":
        _forget_route(host)  # re-probe fresh order next time (e.g. after VPN switch)
    raise requests.exceptions.RequestException(
        f"GET {url} failed on all routes ({[n for n, _ in routes]}): {last_err}"
    )


if __name__ == "__main__":
    print(f"proxy={SOCKS5_PROXY}  mode={_PROXY_MODE}  alive={is_proxy_alive(force=True)}")
