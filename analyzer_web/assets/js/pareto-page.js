import { PLOT_CONFIG } from "./constants.js";
import { uniq } from "./format-utils.js";
import { escapeHtml } from "./dom-utils.js";
import { isBaselineRow, getRowNumeric } from "./rows.js";
import { plotlyNewPlot, plotlyPurge, academicLayout } from "./plotly-charts.js";

const X_KEY = "compression_ratio";
const Y_KEY = "accuracy";

const METHOD_COLORS = [
  ["#1b6ca8", "#0d47a1"],
  ["#2e7d32", "#1b5e20"],
  ["#c62828", "#b71c1c"],
  ["#6a1b9a", "#4a148c"],
  ["#ef6c00", "#e65100"],
  ["#00838f", "#006064"],
];

function methodBaseColor(method) {
  const m = String(method || "—").trim() || "—";
  let h = 0;
  for (let i = 0; i < m.length; i += 1) h = (h * 31 + m.charCodeAt(i)) >>> 0;
  const pair = METHOD_COLORS[h % METHOD_COLORS.length];
  return pair[0];
}

function median(nums) {
  const s = nums.filter((n) => Number.isFinite(n)).sort((a, b) => a - b);
  if (!s.length) return null;
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Maximize both compression_ratio and accuracy (both higher better). */
function paretoNonDominatedIndices(points) {
  const n = points.length;
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const pi = points[i];
    let dominated = false;
    for (let j = 0; j < n; j += 1) {
      if (j === i) continue;
      const pj = points[j];
      if (pj.x >= pi.x && pj.y >= pi.y && (pj.x > pi.x || pj.y > pi.y)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) out.push(i);
  }
  return out;
}

function splitLines(pts, opts = {}) {
  const accuracyUseMedian = !!opts.accuracyUseMedian;
  const compressionUseMedian = !!opts.compressionUseMedian;
  const accPctRaw = opts.accuracyPctOfBaseline;
  const accPct = Number.isFinite(Number(accPctRaw)) ? Math.min(100, Math.max(1, Number(accPctRaw))) : 97;
  const compThreshRaw = opts.compressionThreshold;
  const compThresh = Number.isFinite(Number(compThreshRaw)) ? Number(compThreshRaw) : 9;

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const medX = median(xs);
  const medY = median(ys);

  let vSplit = medX;
  let vMode = "median";
  if (!compressionUseMedian && Number.isFinite(compThresh) && compThresh >= xMin && compThresh <= xMax) {
    vSplit = compThresh;
    vMode = "fixed";
  }

  const baselineAccs = pts
    .filter((p) => isBaselineRow(p.r))
    .map((p) => p.y)
    .filter((y) => Number.isFinite(y));

  let hSplit = medY;
  let hMode = "median";
  if (!accuracyUseMedian && baselineAccs.length > 0) {
    const bmax = Math.max(...baselineAccs);
    if (Number.isFinite(bmax) && bmax > 0) {
      hSplit = (accPct / 100) * bmax;
      hMode = "baseline_pct";
    }
  }

  return {
    vSplit,
    hSplit,
    xMin,
    xMax,
    yMin,
    yMax,
    medX,
    medY,
    splitMeta: {
      vMode,
      hMode,
      accPct,
      compThresh,
      compressionUseMedian,
      accuracyUseMedian,
    },
  };
}

/**
 * @param {HTMLElement} chartEl
 * @param {object[]} rows
 * @param {object} [splitOpts]
 * @param {number} [splitOpts.accuracyPctOfBaseline] 1–100, default 97 (% of max baseline accuracy for horizontal line)
 * @param {boolean} [splitOpts.accuracyUseMedian] if true, horizontal line at median accuracy
 * @param {number} [splitOpts.compressionThreshold] compression ratio for vertical line when inside data range
 * @param {boolean} [splitOpts.compressionUseMedian] if true, vertical line at median compression
 */
export function renderTradeoffMap(chartEl, rows, splitOpts = {}) {
  if (!chartEl) return;

  const opts = {
    accuracyPctOfBaseline: 97,
    accuracyUseMedian: false,
    compressionThreshold: 9,
    compressionUseMedian: false,
    ...splitOpts,
  };

  const pts = [];
  rows.forEach((r) => {
    const x = getRowNumeric(r, X_KEY);
    const y = getRowNumeric(r, Y_KEY);
    if (x === null || y === null) return;
    pts.push({ x, y, r });
  });

  if (!pts.length) {
    plotlyPurge(chartEl);
    chartEl.innerHTML =
      '<p class="named-empty">No points with numeric compression ratio and accuracy in the combined cohorts.</p>';
    return;
  }

  const { vSplit, hSplit, xMin, xMax, yMin, yMax, splitMeta } = splitLines(pts, opts);
  const xPad = (xMax - xMin) * 0.05 || 0.05;
  const yPad = (yMax - yMin) * 0.05 || 0.5;
  const x0 = xMin - xPad;
  const x1 = xMax + xPad;
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  const shapes = [
    {
      type: "rect",
      xref: "x",
      yref: "y",
      x0: vSplit,
      x1,
      y0: hSplit,
      y1,
      fillcolor: "rgba(187, 222, 251, 0.42)",
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "rect",
      xref: "x",
      yref: "y",
      x0,
      x1: vSplit,
      y0: hSplit,
      y1,
      fillcolor: "rgba(200, 230, 201, 0.38)",
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "rect",
      xref: "x",
      yref: "y",
      x0: vSplit,
      x1,
      y0,
      y1: hSplit,
      fillcolor: "rgba(255, 224, 178, 0.45)",
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "rect",
      xref: "x",
      yref: "y",
      x0,
      x1: vSplit,
      y0,
      y1: hSplit,
      fillcolor: "rgba(224, 224, 224, 0.35)",
      line: { width: 0 },
      layer: "below",
    },
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0: vSplit,
      x1: vSplit,
      y0,
      y1,
      line: { color: "rgba(33, 33, 33, 0.55)", width: 1.5, dash: "solid" },
      layer: "below",
    },
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0,
      x1,
      y0: hSplit,
      y1: hSplit,
      line: { color: "rgba(33, 33, 33, 0.55)", width: 1.5, dash: "solid" },
      layer: "below",
    },
  ];

  const groups = new Map();
  pts.forEach((p) => {
    const method = String(p.r.method || "—").trim() || "—";
    const ft = p.r.fine_tuning_enabled ? "Fine-tuned" : "No fine-tuning";
    const key = `${method}\t${ft}`;
    if (!groups.has(key)) groups.set(key, { method, ft, pts: [], color: methodBaseColor(method) });
    groups.get(key).pts.push(p);
  });

  const traces = [];
  [...groups.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .forEach(([, g]) => {
      const isFt = g.ft === "Fine-tuned";
      const name = `${g.method} · ${isFt ? "FT" : "no FT"}`;
      traces.push({
        name,
        type: "scatter",
        mode: "markers",
        x: g.pts.map((p) => p.x),
        y: g.pts.map((p) => p.y),
        text: g.pts.map((p) => {
          const src = p.r._mapSourceLabel ? `${escapeHtml(String(p.r._mapSourceLabel))}<br>` : "";
          return `${src}<b>${escapeHtml(p.r.experiment_name || "—")}</b><br>${escapeHtml(p.r.method)} · ${escapeHtml(p.r.phase)}<br>run ${escapeHtml(String(p.r.run_id))}<br>rank ${escapeHtml(String(p.r.rank_raw ?? p.r.rank_scalar ?? ""))}`;
        }),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          symbol: isFt ? "circle" : "circle-open",
          size: isFt ? 6 : 7,
          color: g.color,
          opacity: isFt ? 0.92 : 0.88,
          line: isFt
            ? { width: 0.6, color: "rgba(0,0,0,0.22)" }
            : { width: 1.8, color: g.color },
        },
      });
    });

  const ndIdx = paretoNonDominatedIndices(pts);
  const paretoPts = ndIdx.map((i) => pts[i]).sort((a, b) => a.x - b.x);
  if (paretoPts.length >= 2) {
    traces.push({
      name: "Pareto frontier",
      type: "scatter",
      mode: "lines+markers",
      x: paretoPts.map((p) => p.x),
      y: paretoPts.map((p) => p.y),
      line: { color: "#111111", width: 2, shape: "linear" },
      marker: { size: 5, color: "#111111", symbol: "diamond", line: { width: 0 } },
      hoverinfo: "skip",
    });
  } else if (paretoPts.length === 1) {
    traces.push({
      name: "Pareto (single)",
      type: "scatter",
      mode: "markers",
      x: [paretoPts[0].x],
      y: [paretoPts[0].y],
      marker: { size: 7, color: "#111111", symbol: "diamond" },
      hoverinfo: "skip",
    });
  }

  const baseLayout = academicLayout(
    "Compression vs accuracy — trade-off map",
    "Compression ratio (×)",
    "Accuracy"
  );
  let splitNoteV = "Vertical split at median compression.";
  if (splitMeta.compressionUseMedian) {
    splitNoteV = "Vertical split at median compression (option).";
  } else if (splitMeta.vMode === "fixed") {
    splitNoteV = `Vertical split at ${splitMeta.compThresh}× compression.`;
  } else {
    splitNoteV = `Compression threshold ${splitMeta.compThresh}× is outside the data range — using median.`;
  }

  let splitNoteH = "Horizontal split at median accuracy.";
  if (splitMeta.accuracyUseMedian) {
    splitNoteH = "Horizontal split at median accuracy (option).";
  } else if (splitMeta.hMode === "baseline_pct") {
    splitNoteH = `Horizontal split at ${splitMeta.accPct}% of baseline max accuracy.`;
  } else {
    splitNoteH = "No baseline rows in the data — horizontal split uses median accuracy.";
  }

  const qInsetX = (xa, xb) => Math.max((xb - xa) * 0.04, (x1 - x0) * 0.012);
  const qInsetY = (ya, yb) => Math.max((yb - ya) * 0.04, (y1 - y0) * 0.012);
  const capHtml = (title, sub, color) =>
    `<b>${title}</b><br><span style="font-size:9px;font-weight:500;color:${color};opacity:0.92">${sub}</span>`;
  const zoneNote = {
    bgcolor: "rgba(255,255,255,0.72)",
    borderpad: 3,
    bordercolor: "rgba(0,0,0,0.06)",
    borderwidth: 1,
    showarrow: false,
    font: { size: 11, family: "system-ui, sans-serif" },
  };

  const layout = {
    ...baseLayout,
    margin: { ...baseLayout.margin, r: 140 },
    legend: {
      ...baseLayout.legend,
      orientation: "v",
      yanchor: "middle",
      y: 0.5,
      xanchor: "left",
      x: 1.02,
      font: { size: 11 },
    },
    shapes,
    annotations: [
      {
        ...zoneNote,
        x: x1 - qInsetX(vSplit, x1),
        y: y1 - qInsetY(hSplit, y1),
        xref: "x",
        yref: "y",
        xanchor: "right",
        yanchor: "top",
        align: "right",
        text: capHtml("Ideal", "High accuracy and high compression", "#0d47a1"),
        font: { ...zoneNote.font, color: "#0d47a1" },
      },
      {
        ...zoneNote,
        x: x0 + qInsetX(x0, vSplit),
        y: y1 - qInsetY(hSplit, y1),
        xref: "x",
        yref: "y",
        xanchor: "left",
        yanchor: "top",
        align: "left",
        text: capHtml("Accuracy-first", "High accuracy, lower compression", "#1b5e20"),
        font: { ...zoneNote.font, color: "#1b5e20" },
      },
      {
        ...zoneNote,
        x: x1 - qInsetX(vSplit, x1),
        y: y0 + qInsetY(y0, hSplit),
        xref: "x",
        yref: "y",
        xanchor: "right",
        yanchor: "bottom",
        align: "right",
        text: capHtml("Compression-first", "High compression, accuracy trade-off", "#bf360c"),
        font: { ...zoneNote.font, color: "#bf360c" },
      },
      {
        ...zoneNote,
        x: x0 + qInsetX(x0, vSplit),
        y: y0 + qInsetY(y0, hSplit),
        xref: "x",
        yref: "y",
        xanchor: "left",
        yanchor: "bottom",
        align: "left",
        text: capHtml("Weak", "Below both split lines", "#616161"),
        font: { ...zoneNote.font, color: "#616161" },
      },
      {
        x: 0.02,
        y: -0.2,
        xref: "paper",
        yref: "paper",
        xanchor: "left",
        text: `${splitNoteV} ${splitNoteH} Filled markers = fine-tuned; open markers = no FT.`,
        showarrow: false,
        font: { size: 10, family: "system-ui, sans-serif", color: "#5c6370" },
      },
    ],
    xaxis: { ...baseLayout.xaxis, range: [x0, x1], zeroline: false },
    yaxis: { ...baseLayout.yaxis, range: [y0, y1], zeroline: false },
  };

  plotlyNewPlot(chartEl, traces, layout, PLOT_CONFIG);
}
