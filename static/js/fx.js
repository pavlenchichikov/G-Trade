/* Signature FX for the G-Trade terminal: boot sequence, chart crosshair,
   regime-reactive ambient wash, and odometer count-up on hero numbers.
   Pure cosmetics, no dependencies, honours prefers-reduced-motion. */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- Boot sequence (plays once per browser session) ----
  function boot() {
    var el = document.getElementById("boot");
    if (!el) return;
    if (reduce || sessionStorage.getItem("gt_booted")) { el.remove(); return; }
    sessionStorage.setItem("gt_booted", "1");
    var lines = el.querySelectorAll(".boot-line");
    Array.prototype.forEach.call(lines, function (ln, i) {
      setTimeout(function () { ln.classList.add("show"); }, 140 + i * 175);
    });
    setTimeout(function () {
      el.classList.add("done");
      setTimeout(function () { if (el.parentNode) el.remove(); }, 600);
    }, 1320);
  }

  // ---- Chart crosshair that tracks the cursor (fine pointers only) ----
  function crosshair() {
    if (reduce) return;
    if (window.matchMedia && !window.matchMedia("(pointer: fine)").matches) return;
    var hx = document.querySelector(".xhair-x");
    var hy = document.querySelector(".xhair-y");
    if (!hx || !hy) return;
    var raf = null, mx = 0, my = 0;
    document.addEventListener("mousemove", function (e) {
      mx = e.clientX; my = e.clientY;
      hx.classList.add("on"); hy.classList.add("on");
      if (raf) return;
      raf = requestAnimationFrame(function () {
        hx.style.top = my + "px";
        hy.style.left = mx + "px";
        raf = null;
      });
    });
    document.addEventListener("mouseleave", function () {
      hx.classList.remove("on"); hy.classList.remove("on");
    });
  }

  // ---- Regime-reactive ambient: tint the floor by the bull/bear score ----
  function regimeTint() {
    fetch("/api/regime").then(function (r) { return r.json(); }).then(function (d) {
      var s = (d && typeof d.score === "number") ? d.score : 50;
      var tint;
      if (s >= 60) {
        tint = "rgba(0, 255, 163, " + (0.05 + (s - 60) / 40 * 0.06).toFixed(3) + ")";
      } else if (s <= 40) {
        tint = "rgba(255, 59, 92, " + (0.05 + (40 - s) / 40 * 0.06).toFixed(3) + ")";
      } else {
        tint = "rgba(120, 140, 170, 0.035)";
      }
      document.documentElement.style.setProperty("--regime-tint", tint);
    }).catch(function () {});
  }

  // ---- Odometer count-up for hero stat numbers ----
  function fmt(n, decimals, grouped) {
    var s = n.toFixed(decimals);
    if (grouped) {
      var p = s.split(".");
      p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      s = p.join(".");
    }
    return s;
  }
  function countUp(node) {
    var raw = node.nodeValue;
    var m = raw.match(/^(\s*[-+]?\$?\s*)([\d,]*\.?\d+)(\s*%?\s*)$/);
    if (!m) return;
    var pre = m[1], core = m[2], post = m[3];
    var target = parseFloat(core.replace(/,/g, ""));
    if (!isFinite(target)) return;
    var decimals = (core.split(".")[1] || "").length;
    var grouped = core.indexOf(",") !== -1;
    var dur = 850, t0 = performance.now();
    function step(now) {
      var p = Math.min(1, (now - t0) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      node.nodeValue = pre + fmt(target * eased, decimals, grouped) + post;
      if (p < 1) requestAnimationFrame(step);
      else node.nodeValue = raw;
    }
    requestAnimationFrame(step);
  }
  function odometers() {
    if (reduce) return;
    document.querySelectorAll(".stat-value, .gauge-text b, .breadth-seg b").forEach(function (el) {
      var tn = el.firstChild;
      if (tn && tn.nodeType === 3 && /\d/.test(tn.nodeValue)) countUp(tn);
    });
  }

  // ---- Add a pulsing needle marker to any gauge that lacks one ----
  // Pages with live gauges (Market) ship their own marker + keep it in sync;
  // this only decorates the static gauges (Radar cards, stress) for parity.
  function gaugeMarkers() {
    document.querySelectorAll("svg.gauge").forEach(function (svg) {
      var fill = svg.querySelector(".gauge-fill");
      if (!fill || svg.querySelector(".gauge-marker")) return;
      var m = (fill.style.strokeDasharray || "").match(/([\d.]+)/);
      if (!m) return;
      var f = Math.max(0, Math.min(1, parseFloat(m[1]) / 157));
      var pos = function (frac) {
        var a = Math.PI * (1 - frac);
        return [(60 + 50 * Math.cos(a)).toFixed(2), (65 - 50 * Math.sin(a)).toFixed(2)];
      };
      var cls = fill.getAttribute("class") || "";
      var zone = cls.indexOf("g-bad") !== -1 ? "g-bad" : (cls.indexOf("g-warn") !== -1 ? "g-warn" : "");
      var dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("r", "3.5");
      dot.setAttribute("class", "gauge-marker " + zone);
      var target = pos(f);
      if (reduce) {
        dot.setAttribute("cx", target[0]); dot.setAttribute("cy", target[1]);
        svg.appendChild(dot);
      } else {
        var start = pos(0);
        dot.setAttribute("cx", start[0]); dot.setAttribute("cy", start[1]);
        svg.appendChild(dot);
        requestAnimationFrame(function () {
          requestAnimationFrame(function () {
            dot.setAttribute("cx", target[0]); dot.setAttribute("cy", target[1]);
          });
        });
      }
    });
  }

  // ---- Gauges draw in from zero on load (uses the CSS dasharray transition) ----
  function gauges() {
    if (reduce) return;
    document.querySelectorAll(".gauge-fill").forEach(function (p) {
      var target = p.style.strokeDasharray;
      if (!target) return;
      var total = target.split(/[ ,]+/)[1] || "157";
      p.style.strokeDasharray = "0 " + total;
      void p.getBoundingClientRect();  // force a reflow so the reset is committed
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { p.style.strokeDasharray = target; });
      });
    });
  }

  // ---- Ticker tape of top movers (duplicated content for a seamless loop) ----
  function ticker() {
    var bar = document.getElementById("ticker");
    var track = document.getElementById("ticker-track");
    if (!bar || !track) return;
    fetch("/api/ticker").then(function (r) { return r.json(); }).then(function (d) {
      var m = (d && d.movers) || [];
      if (!m.length) { bar.style.display = "none"; return; }
      var html = m.map(function (x) {
        var pct = x.chg * 100;
        var cls = pct >= 0 ? "up" : "down";
        return '<span class="tk"><b>' + x.asset + '</b>' +
          '<span class="tk-sig ' + x.signal + '">' + x.signal + '</span>' +
          '<span class="' + cls + '">' + (pct >= 0 ? "+" : "") + pct.toFixed(1) + '%</span>' +
          '<span class="tk-sep">|</span></span>';
      }).join("");
      track.innerHTML = html + html;
      track.style.animationDuration = Math.max(30, m.length * 3.2) + "s";
    }).catch(function () { bar.style.display = "none"; });
  }

  // ---- Header status LEDs from data freshness ----
  function leds() {
    fetch("/api/health").then(function (r) { return r.json(); }).then(function (d) {
      var dataLed = document.querySelector('.led[data-led="data"]');
      var modelsLed = document.querySelector('.led[data-led="models"]');
      if (dataLed) {
        var age = (d && typeof d.age_days === "number") ? d.age_days : null;
        dataLed.className = "led " + (age === null ? "off" : (age <= 5 ? "on" : "warn"));
        if (d && d.data_date) dataLed.title = "Latest data: " + d.data_date;
      }
      if (modelsLed) {
        var n = (d && d.models) || 0;
        modelsLed.className = "led " + (n > 0 ? "on" : "off");
        modelsLed.title = n + " trained models";
      }
    }).catch(function () {});
  }

  // ---- Command palette (Cmd/Ctrl-K) ----
  function palette() {
    var pal = document.getElementById("palette");
    var input = document.getElementById("palette-input");
    var list = document.getElementById("palette-list");
    if (!pal || !input || !list) return;
    var items = [], sel = 0;
    fetch("/api/palette").then(function (r) { return r.json(); }).then(function (d) {
      (d.pages || []).forEach(function (p) { items.push({ label: p[0], url: p[1], tag: "page" }); });
      (d.assets || []).forEach(function (a) { items.push({ label: a, url: "/asset/" + a, tag: "asset" }); });
    }).catch(function () {});

    function open() { pal.hidden = false; input.value = ""; render(""); input.focus(); }
    function close() { pal.hidden = true; }
    function render(q) {
      q = q.toLowerCase();
      var matches = items.filter(function (it) {
        return it.label.toLowerCase().indexOf(q) !== -1;
      }).slice(0, 40);
      sel = 0;
      list.innerHTML = matches.map(function (it, i) {
        return '<li data-url="' + it.url + '" class="' + (i === 0 ? "sel" : "") + '">' +
          '<span class="pk">' + (it.tag === "asset" ? ">" : "#") + '</span>' +
          '<span>' + it.label + '</span><span class="tag">' + it.tag + '</span></li>';
      }).join("") || '<li class="tag" style="padding:12px">no match</li>';
    }
    function move(dir) {
      var lis = list.querySelectorAll("li[data-url]");
      if (!lis.length) return;
      if (lis[sel]) lis[sel].classList.remove("sel");
      sel = (sel + dir + lis.length) % lis.length;
      lis[sel].classList.add("sel");
      lis[sel].scrollIntoView({ block: "nearest" });
    }
    function go() {
      var cur = list.querySelector("li.sel[data-url]");
      if (cur) location.href = cur.getAttribute("data-url");
    }
    document.addEventListener("keydown", function (e) {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        if (pal.hidden) open(); else close();
        return;
      }
      if (pal.hidden) return;
      if (e.key === "Escape") close();
      else if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") { e.preventDefault(); go(); }
    });
    input.addEventListener("input", function () { render(input.value); });
    list.addEventListener("click", function (e) {
      var li = e.target.closest("li[data-url]");
      if (li) location.href = li.getAttribute("data-url");
    });
    pal.addEventListener("click", function (e) { if (e.target === pal) close(); });
  }

  // ---- Sparklines draw in from the left with a glowing leading dot ----
  function sparklines() {
    if (reduce) return;
    document.querySelectorAll(".spark polyline").forEach(function (pl) {
      var len;
      try { len = pl.getTotalLength(); } catch (e) { return; }
      if (!len) return;
      pl.style.strokeDasharray = len;
      pl.style.strokeDashoffset = len;
      void pl.getBoundingClientRect();
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { pl.style.strokeDashoffset = "0"; });
      });
      var pts = (pl.getAttribute("points") || "").trim().split(/\s+/);
      var last = pts[pts.length - 1];
      if (last && last.indexOf(",") !== -1 && pl.ownerSVGElement) {
        var xy = last.split(",");
        var dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", xy[0]);
        dot.setAttribute("cy", xy[1]);
        dot.setAttribute("r", "1.6");
        dot.setAttribute("class", "spark-dot");
        dot.style.color = getComputedStyle(pl).stroke;
        pl.ownerSVGElement.appendChild(dot);
      }
    });
  }

  // ---- Live number flip on radar probability / tail-risk cells ----
  function liveFlips() {
    if (reduce || typeof MutationObserver === "undefined") return;
    document.querySelectorAll('tr[data-asset] .c-prob, tr[data-asset] .c-taleb').forEach(function (cell) {
      var prev = parseFloat(cell.textContent);
      new MutationObserver(function () {
        var now = parseFloat(cell.textContent);
        if (isNaN(now) || isNaN(prev) || now === prev) { prev = now; return; }
        cell.classList.remove("flip", "flip-up", "flip-down");
        void cell.offsetWidth;
        cell.classList.add("flip", now > prev ? "flip-up" : "flip-down");
        setTimeout(function () { cell.classList.remove("flip-up", "flip-down"); }, 700);
        prev = now;
      }).observe(cell, { childList: true, characterData: true, subtree: true });
    });
  }

  // ---- Radar sweep motif, Radar page only ----
  function radarSweep() {
    if (reduce || location.pathname !== "/") return;
    var h1 = document.querySelector("main h1");
    if (!h1 || h1.querySelector(".radar-sweep")) return;
    var s = document.createElement("span");
    s.className = "radar-sweep";
    h1.appendChild(s);
  }

  function run() {
    boot(); crosshair(); regimeTint(); odometers(); gaugeMarkers(); gauges();
    ticker(); leds(); palette(); sparklines(); liveFlips(); radarSweep();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
