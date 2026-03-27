"""
Performance Report — генерация HTML-отчёта по состоянию системы.
  python performance_report.py               — создать и открыть
  python performance_report.py --no-open      — без открытия
  python performance_report.py --output f.html
"""

import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

sys.path.insert(0, BASE_DIR)
try:
    from config import FULL_ASSET_MAP, ASSET_TYPES
except ImportError:
    FULL_ASSET_MAP = {}
    ASSET_TYPES = {}

engine = create_engine(f"sqlite:///{DB_PATH}")


def _table_name(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _get_asset_stats(asset, rows=60):
    table = _table_name(asset)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {table} ORDER BY Date DESC LIMIT {rows}",
            engine, index_col="Date", parse_dates=["Date"]
        )
        if df.empty:
            return None
        df = df.sort_index()
        df.columns = [c.lower() for c in df.columns]
        price = float(df["close"].iloc[-1])
        chg_1d = float(df["close"].pct_change().iloc[-1]) if len(df) > 1 else 0
        chg_5d = float((df["close"].iloc[-1] / df["close"].iloc[-6] - 1)) if len(df) > 5 else 0
        chg_20d = float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1)) if len(df) > 20 else 0

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        sma50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else price
        trend = "UP" if price > sma50 else "DOWN"

        return {
            "price": price, "chg_1d": chg_1d, "chg_5d": chg_5d,
            "chg_20d": chg_20d, "rsi": rsi, "trend": trend,
            "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])[:10]
        }
    except Exception:
        return None


def _model_info():
    if not os.path.isdir(MODEL_DIR):
        return {"cb": 0, "lstm": 0, "oldest_days": 0, "avg_age": 0}
    now = datetime.now().timestamp()
    cb_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".cbm")]
    lstm_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    ages = []
    for f in cb_files + lstm_files:
        mtime = os.path.getmtime(os.path.join(MODEL_DIR, f))
        ages.append((now - mtime) / 86400)
    return {
        "cb": len(cb_files), "lstm": len(lstm_files),
        "oldest_days": max(ages) if ages else 0,
        "avg_age": sum(ages) / len(ages) if ages else 0,
    }


def _regime_info():
    try:
        from regime_detector import get_global_regime, get_market_breadth
        return {"regime": get_global_regime(), "breadth": get_market_breadth()}
    except Exception:
        return None


def _chg_color(val):
    if val > 0:
        return "#22c55e"
    elif val < 0:
        return "#ef4444"
    return "#64748b"


def generate_html():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    registry = _load_registry()
    minfo = _model_info()
    regime_data = _regime_info()

    # Asset rows
    asset_rows = []
    for asset in sorted(FULL_ASSET_MAP.keys()):
        stats = _get_asset_stats(asset)
        reg = registry.get(asset, {})
        score = reg.get("score", "")
        policy = reg.get("policy", "")
        if stats:
            asset_rows.append({**stats, "asset": asset, "score": score, "policy": policy})

    # Regime section
    regime_html = ""
    if regime_data:
        r = regime_data["regime"]
        b = regime_data["breadth"]
        regime_html = f"""
        <div class="card">
            <h2>Market Regime</h2>
            <table>
                <tr><td>Status</td><td><strong>{r.get('status','N/A')}</strong></td></tr>
                <tr><td>VIX</td><td>{r.get('vix_value','N/A')} ({r.get('vix_level','N/A')})</td></tr>
                <tr><td>SP500</td><td>{r.get('sp500_trend','N/A')}</td></tr>
                <tr><td>DXY</td><td>{r.get('dxy_trend','N/A')}</td></tr>
                <tr><td>Above SMA50</td><td>{b.get('above_sma50_pct',0):.0%}</td></tr>
                <tr><td>Breadth</td><td>{b.get('score','N/A')}</td></tr>
            </table>
        </div>"""

    # Top signals
    scored = [r for r in asset_rows if isinstance(r.get("score"), (int, float)) and r["score"] > 0]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top10 = scored[:10]
    top_html = "".join(
        f"<tr><td>{r['asset']}</td><td>{r['price']:,.2f}</td>"
        f"<td style='color:{_chg_color(r['chg_1d'])}'>{r['chg_1d']:+.1%}</td>"
        f"<td>{r['rsi']:.0f}</td><td>{r['score']:.1f}</td><td>{r['policy']}</td></tr>"
        for r in top10
    )

    # Risk alerts
    alerts = []
    for r in asset_rows:
        if r["rsi"] > 75:
            alerts.append(f"{r['asset']}: RSI={r['rsi']:.0f} (OVERBOUGHT)")
        elif r["rsi"] < 25:
            alerts.append(f"{r['asset']}: RSI={r['rsi']:.0f} (OVERSOLD)")
    if regime_data and regime_data["regime"].get("vix_level") in ("FEAR", "PANIC"):
        alerts.insert(0, f"VIX: {regime_data['regime'].get('vix_value','')} ({regime_data['regime'].get('vix_level','')})")
    if minfo["oldest_days"] > 7:
        alerts.append(f"Oldest model: {minfo['oldest_days']:.0f} days (consider retraining)")
    alerts_html = "".join(f"<li>{a}</li>" for a in alerts) if alerts else "<li>No alerts</li>"

    # Full asset table
    def _fmt_score(s):
        return f"{s:.1f}" if isinstance(s, (int, float)) else "N/A"

    full_rows = "".join(
        f"<tr><td>{r['asset']}</td><td>{r['price']:,.2f}</td>"
        f"<td style='color:{_chg_color(r['chg_1d'])}'>{r['chg_1d']:+.1%}</td>"
        f"<td style='color:{_chg_color(r['chg_5d'])}'>{r['chg_5d']:+.1%}</td>"
        f"<td style='color:{_chg_color(r['chg_20d'])}'>{r['chg_20d']:+.1%}</td>"
        f"<td>{r['rsi']:.0f}</td><td>{r['trend']}</td>"
        f"<td>{_fmt_score(r['score'])}</td>"
        f"<td>{r['policy']}</td></tr>"
        for r in asset_rows
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>G-Trade Report {now_str}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e17; color:#e2e8f0; font-family:'Segoe UI',Consolas,monospace; padding:24px; }}
  h1 {{ color:#38bdf8; margin-bottom:4px; }}
  h2 {{ color:#38bdf8; font-size:16px; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:6px; }}
  .meta {{ color:#64748b; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); gap:16px; margin-bottom:20px; }}
  .card {{ background:#111827; border:1px solid #1e293b; border-radius:8px; padding:16px; }}
  .card table {{ width:100%; }}
  .card td {{ padding:4px 8px; }}
  .card td:first-child {{ color:#64748b; }}
  .stats {{ display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }}
  .stat {{ background:#111827; border:1px solid #1e293b; border-radius:8px; padding:12px 20px; text-align:center; }}
  .stat .val {{ font-size:24px; font-weight:bold; color:#38bdf8; }}
  .stat .lbl {{ font-size:11px; color:#64748b; }}
  table.data {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.data th {{ background:#1e293b; color:#94a3b8; padding:8px; text-align:left; cursor:pointer; }}
  table.data th:hover {{ color:#38bdf8; }}
  table.data td {{ padding:6px 8px; border-bottom:1px solid #1e293b; }}
  table.data tr:hover {{ background:#1e293b; }}
  .alerts li {{ padding:4px 0; color:#eab308; }}
</style></head><body>
<h1>G-TRADE -- Performance Report</h1>
<p class="meta">Generated: {now_str}</p>

<div class="stats">
  <div class="stat"><div class="val">{len(asset_rows)}</div><div class="lbl">Assets</div></div>
  <div class="stat"><div class="val">{minfo['cb']}</div><div class="lbl">CB Models</div></div>
  <div class="stat"><div class="val">{minfo['lstm']}</div><div class="lbl">LSTM Models</div></div>
  <div class="stat"><div class="val">{minfo['avg_age']:.1f}d</div><div class="lbl">Avg Model Age</div></div>
  <div class="stat"><div class="val">{len(registry)}</div><div class="lbl">Registry</div></div>
</div>

<div class="grid">
  {regime_html}
  <div class="card">
    <h2>Top Signals (by score)</h2>
    <table class="data">
      <tr><th>Asset</th><th>Price</th><th>1d</th><th>RSI</th><th>Score</th><th>Policy</th></tr>
      {top_html}
    </table>
  </div>
  <div class="card">
    <h2>Risk Alerts</h2>
    <ul class="alerts">{alerts_html}</ul>
  </div>
</div>

<div class="card">
  <h2>All Assets ({len(asset_rows)})</h2>
  <table class="data" id="assetTable">
    <thead><tr>
      <th onclick="sortTable(0)">Asset</th><th onclick="sortTable(1)">Price</th>
      <th onclick="sortTable(2)">1d</th><th onclick="sortTable(3)">5d</th>
      <th onclick="sortTable(4)">20d</th><th onclick="sortTable(5)">RSI</th>
      <th onclick="sortTable(6)">Trend</th><th onclick="sortTable(7)">Score</th>
      <th onclick="sortTable(8)">Policy</th>
    </tr></thead>
    <tbody>{full_rows}</tbody>
  </table>
</div>

<script>
function sortTable(col) {{
  const tb = document.getElementById('assetTable');
  const rows = Array.from(tb.tBodies[0].rows);
  const dir = tb.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  tb.dataset.sortDir = dir;
  rows.sort((a,b) => {{
    let va = a.cells[col].textContent.replace(/[,$%+]/g,'');
    let vb = b.cells[col].textContent.replace(/[,$%+]/g,'');
    let na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return dir==='asc' ? na-nb : nb-na;
    return dir==='asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tb.tBodies[0].appendChild(r));
}}
</script>
</body></html>"""
    return html


def export_report(output=None, open_browser=True):
    os.makedirs(REPORT_DIR, exist_ok=True)
    if output is None:
        output = os.path.join(REPORT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    elif not os.path.isabs(output):
        output = os.path.join(REPORT_DIR, output)

    html = generate_html()
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output) / 1024
    print(f"\n{'='*60}")
    print("  PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"  Generated: {output}")
    print(f"  Size: {size_kb:.1f} KB")

    if open_browser:
        print("  Opening in browser...")
        webbrowser.open(f"file:///{output}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Performance Report")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    args = parser.parse_args()
    export_report(output=args.output, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
