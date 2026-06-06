"""
net.py - proxy-aware network helpers for G-Trade.

Why:
  data_engine.py historically tried the SOCKS5 proxy FIRST on every Yahoo
  request, then fell back to direct. When the local SOCKS5 endpoint isn't
  running (system VPN, or no VPN), every request still wasted its full
  timeout waiting on the dead proxy - ~10s x 150 assets = minutes lost.

  This module probes the SOCKS5 endpoint ONCE per process (cached) so callers
  can skip the proxy attempt entirely when it's not alive.

  BETEDGE_PROXY_MODE-style override:
      GTRADE_PROXY_MODE = auto | on | off   (default: auto)

Public API:
  is_proxy_alive()      -> bool (cached TCP probe)
  proxies_for(route)    -> dict | None
  http_get(url, ...)    -> requests.Response   (proxy-aware, with failover)
"""
from __future__ import annotations

import os
import socket
import time
from urllib.parse import urlparse

import requests

try:
    from config import SOCKS5_PROXY
except Exception:
    SOCKS5_PROXY = os.getenv("SOCKS5_PROXY", "socks5h://127.0.0.1:12334")

_PROXY_MODE = (os.getenv("GTRADE_PROXY_MODE") or "auto").strip().lower()
_PROBE_TIMEOUT = float(os.getenv("GTRADE_PROXY_PROBE_TIMEOUT", "1.0"))
_PROBE_TTL = 60.0


def ssl_verify() -> bool:
    """Whether outbound HTTPS requests should verify TLS certificates.

    Defaults to False to preserve working data fetches behind the AmneziaVPN
    SOCKS5 proxy / RU firewall, which can intercept TLS. In a trusted-network
    deployment set GTRADE_SSL_VERIFY=1 (or true/yes/on) to enforce verification.
    """
    return (os.getenv("GTRADE_SSL_VERIFY") or "").strip().lower() in ("1", "true", "yes", "on")

_alive_cache: bool | None = None
_cache_ts: float = 0.0


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

    Mode override: GTRADE_PROXY_MODE=on -> always True, off -> always False,
    auto (default) -> cached TCP probe.
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


def http_get(url, *, route="auto", headers=None, timeout=10,
             verify=None, retries=3, **kwargs):
    """GET with proxy auto-selection + one failover.

    On 'auto': if the proxy is alive it's tried first, then direct on
    transport failure. If the proxy is dead the attempt is skipped entirely.
    5xx triggers exponential-backoff retry within each route.

    verify defaults to ssl_verify() (env GTRADE_SSL_VERIFY) when not passed.
    """
    if verify is None:
        verify = ssl_verify()
    headers = headers or {"User-Agent": "Mozilla/5.0"}

    def _attempt(proxies):
        for i in range(retries):
            try:
                r = requests.get(url, headers=headers, proxies=proxies,
                                  timeout=timeout, verify=verify, **kwargs)
                if r.status_code >= 500 and i < retries - 1:
                    time.sleep(2 ** (i + 1))
                    continue
                return r, None
            except requests.exceptions.RequestException as e:
                if i < retries - 1:
                    time.sleep(2 ** (i + 1))
                    continue
                return None, str(e)
        return None, "max retries exceeded"

    # Ordered routes: proxy first only when alive.
    routes: list[dict | None] = []
    if route == "direct":
        routes = [None]
    elif route == "proxy":
        routes = [proxies_for("proxy")]
    else:  # auto
        if is_proxy_alive():
            routes = [proxies_for("proxy"), None]
        else:
            routes = [None]

    last_err = None
    for proxies in routes:
        r, err = _attempt(proxies)
        if r is not None:
            return r
        last_err = err
    raise requests.exceptions.RequestException(
        f"GET {url} failed on all routes: {last_err}"
    )


if __name__ == "__main__":
    print(f"proxy={SOCKS5_PROXY}  mode={_PROXY_MODE}  alive={is_proxy_alive(force=True)}")
