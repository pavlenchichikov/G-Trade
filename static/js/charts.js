// ECharts helper layer for the Atratus dashboard. Each helper takes a DOM id
// plus plain data and renders one chart. No build step; echarts.min.js loads first.
function gtGauge(id, score, label) {
  var el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return;
  var chart = echarts.init(el);
  chart.setOption({
    series: [{
      type: "gauge", min: 0, max: 100, progress: { show: true, width: 14 },
      axisLine: { lineStyle: { width: 14, color: [
        [0.25, "#e5484d"], [0.5, "#f5a623"], [0.75, "#9aa0a6"], [1, "#30a46c"]] } },
      pointer: { width: 5 },
      detail: { valueAnimation: true, formatter: label, fontSize: 18, offsetCenter: [0, "70%"] },
      data: [{ value: score }],
    }],
  });
  window.addEventListener("resize", function () { chart.resize(); });
}

function gtCandles(id, ohlc) {
  // ohlc: [{date, open, close, low, high}, ...]
  var el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return;
  var chart = echarts.init(el);
  var dates = ohlc.map(function (r) { return r.date; });
  var bars = ohlc.map(function (r) { return [r.open, r.close, r.low, r.high]; });
  chart.setOption({
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: dates, scale: true },
    yAxis: { scale: true },
    dataZoom: [{ type: "inside" }, { type: "slider" }],
    series: [{ type: "candlestick", data: bars,
      itemStyle: { color: "#30a46c", color0: "#e5484d",
        borderColor: "#30a46c", borderColor0: "#e5484d" } }],
  });
  window.addEventListener("resize", function () { chart.resize(); });
}

function gtLine(id, series) {
  // series: {labels: [...], values: [...]}
  var el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return;
  var chart = echarts.init(el);
  chart.setOption({
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: series.labels },
    yAxis: { type: "value" },
    series: [{ type: "line", smooth: true, areaStyle: {}, data: series.values }],
  });
  window.addEventListener("resize", function () { chart.resize(); });
}

function gtHeatmap(id, p) {
  // p: {xLabels, yLabels, data: [[x, y, value], ...], min, max}
  var el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return;
  var chart = echarts.init(el);
  chart.setOption({
    tooltip: { position: "top" },
    grid: { left: 96, right: 16, top: 12, bottom: 78 },
    xAxis: { type: "category", data: p.xLabels, splitArea: { show: true },
      axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "category", data: p.yLabels, splitArea: { show: true },
      axisLabel: { fontSize: 10 } },
    visualMap: {
      min: p.min, max: p.max, calculable: true, orient: "horizontal",
      left: "center", bottom: 4,
      inRange: { color: ["#e25c5c", "#141a24", "#34b873"] },
    },
    series: [{
      type: "heatmap", data: p.data,
      label: { show: false },
      emphasis: { itemStyle: { borderColor: "#e3a23a", borderWidth: 1 } },
    }],
  });
  window.addEventListener("resize", function () { chart.resize(); });
}

function gtBars(id, labels, values) {
  var el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return;
  var chart = echarts.init(el);
  chart.setOption({
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: labels },
    yAxis: { type: "value" },
    series: [{ type: "bar", data: values }],
  });
  window.addEventListener("resize", function () { chart.resize(); });
}
