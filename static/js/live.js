// Shared auto-refresh helpers for dashboard pages that poll an /api/* endpoint
// and patch the DOM in place (no page reload). No build step, no dependencies.

function gtPoll(url, intervalMs, onData) {
  async function tick() {
    if (document.hidden) return;
    let data;
    try {
      const r = await fetch(url);
      if (!r.ok) return;
      data = await r.json();
    } catch (e) {
      return;
    }
    onData(data);
  }
  tick();
  return setInterval(tick, intervalMs);
}

function gtFlash(el, cls) {
  if (!el) return;
  cls = cls || 'flash';
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}
