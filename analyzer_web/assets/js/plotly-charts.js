import { PLOT_CONFIG } from "./constants.js";
import { getRowNumeric, seriesLabel } from "./rows.js";
import { escapeHtml } from "./dom-utils.js";

export function plotlyNewPlot(containerIdOrEl, traces, layout, cfg) {
  const gd =
    typeof containerIdOrEl === "string"
      ? document.getElementById(containerIdOrEl)
      : containerIdOrEl;
  if (!gd || typeof Plotly === "undefined") return false;
  try {
    Plotly.newPlot(gd, traces, layout, cfg != null ? cfg : PLOT_CONFIG);
    return true;
  } catch (err) {
    console.warn("Plotly.newPlot failed:", err);
    return false;
  }
}

export function plotlyPurge(containerIdOrEl) {
  const gd =
    typeof containerIdOrEl === "string" ? document.getElementById(containerIdOrEl) : containerIdOrEl;
  if (!gd || typeof Plotly === "undefined") return;
  try {
    Plotly.purge(gd);
  } catch (_) {
    /* ignore */
  }
}

export function academicLayout(title, xTitle, yTitle) {
  return {
    title: {
      text: title,
      font: { size: 15, family: "Georgia, Times New Roman, serif", color: "#1a1a1a" },
      x: 0.02,
      xanchor: "left",
    },
    margin: { t: 72, l: 72, r: 32, b: 96 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fafafa",
    font: { family: "system-ui, sans-serif", size: 12, color: "#333333" },
    legend: {
      orientation: "h",
      yanchor: "top",
      y: -0.22,
      xanchor: "center",
      x: 0.5,
      bgcolor: "rgba(255,255,255,0.85)",
      bordercolor: "#d8dce3",
      borderwidth: 1,
    },
    hoverlabel: {
      bgcolor: "#ffffff",
      bordercolor: "#d8dce3",
      font: { family: "system-ui, sans-serif", color: "#1a1a1a" },
    },
    xaxis: {
      title: { text: xTitle, standoff: 14, font: { size: 13 } },
      automargin: true,
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.06)",
      zeroline: false,
      linecolor: "#cccccc",
      tickfont: { size: 11 },
      tickangle: -35,
    },
    yaxis: {
      title: { text: yTitle, standoff: 14, font: { size: 13 } },
      automargin: true,
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.06)",
      zeroline: false,
      linecolor: "#cccccc",
      tickfont: { size: 11 },
    },
  };
}

export function buildChartTraces(rows, xKey, yKey, sKey, mode, opts = {}) {
  const multiDataset = Boolean(opts.multiDataset);
  const valid = rows.filter((r) => {
    const x = getRowNumeric(r, xKey);
    const y = getRowNumeric(r, yKey);
    return x !== null && y !== null;
  });
  const groups = {};
  valid.forEach((r) => {
    let name = seriesLabel(r, sKey);
    if (multiDataset && r._cmpSourceLabel) {
      name = `${r._cmpSourceLabel} · ${name}`;
    }
    groups[name] = groups[name] || [];
    groups[name].push(r);
  });

  return Object.entries(groups).map(([name, arr]) => {
    const sorted = [...arr].sort((a, b) => getRowNumeric(a, xKey) - getRowNumeric(b, xKey));
    const x = sorted.map((r) => getRowNumeric(r, xKey));
    const y = sorted.map((r) => getRowNumeric(r, yKey));
    const text = sorted.map((r) => {
      const parts = [
        `<b>${escapeHtml(r.experiment_name)}</b>`,
        escapeHtml(String(r.run_id)),
        `${escapeHtml(r.method)} · ${escapeHtml(r.phase)}`,
      ];
      if (r._cmpSourceLabel) parts.push(`Dataset: ${escapeHtml(r._cmpSourceLabel)}`);
      return parts.join("<br>");
    });
    const base = {
      name,
      type: "scatter",
      x,
      y,
      text,
      hovertemplate: "%{text}<extra></extra>",
    };
    if (mode === "line") {
      return { ...base, mode: "lines+markers", line: { shape: "linear" } };
    }
    return { ...base, mode: "markers", marker: { size: 9, opacity: 0.88 } };
  });
}
