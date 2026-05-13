import { PLOT_CONFIG } from "./constants.js";
import { fmt } from "./format-utils.js";
import { pickDefaultRun } from "./data-loader.js";
import { plotlyNewPlot, academicLayout } from "./plotly-charts.js";

export function initExperimentPage(payload) {
  const DATA = Array.isArray(payload.rows) ? payload.rows : [];
  const meta = payload.meta || {};
  const params = new URLSearchParams(window.location.search);
  const runId = params.get("run") || pickDefaultRun(meta);
  const rows = DATA.filter((r) => r.run_id === runId);
  const runLabelEl = document.getElementById("runLabel");
  if (runLabelEl) runLabelEl.textContent = runId || "(none)";

  const metaLine = document.getElementById("dataMetaLine");
  if (metaLine) {
    metaLine.textContent = `${rows.length} rows for ${runId} · ${DATA.length} rows in dataset`;
  }

  const byMethod = {};
  rows.forEach((r) => {
    byMethod[r.method] = byMethod[r.method] || [];
    byMethod[r.method].push(r);
  });

  const traces = Object.entries(byMethod)
    .filter(([method]) => method !== "None")
    .map(([method, arr]) => {
      const sorted = [...arr].filter((a) => a.rank_scalar !== null).sort((a, b) => a.rank_scalar - b.rank_scalar);
      return {
        name: method,
        type: "scatter",
        mode: "lines+markers",
        x: sorted.map((a) => a.rank_scalar),
        y: sorted.map((a) => a.accuracy),
        text: sorted.map((a) => `${a.experiment_name}<br>${a.phase}`),
        line: { shape: "linear" },
      };
    });

  const overviewEl = document.getElementById("runOverviewPlot");
  if (overviewEl && typeof Plotly !== "undefined") {
    plotlyNewPlot(
      overviewEl,
      traces,
      academicLayout(`Accuracy vs rank — ${runId}`, "Rank", "Accuracy (%)"),
      PLOT_CONFIG
    );
  }

  const body = document.getElementById("runTableBody");
  if (!body) return;
  body.innerHTML = rows
    .sort((a, b) => (a.rank_scalar ?? -1) - (b.rank_scalar ?? -1))
    .map(
      (r) => `<tr>
      <td>${r.experiment_name}</td><td>${r.method}</td><td>${r.phase}</td><td>${r.rank_raw}</td>
      <td>${fmt(r.accuracy, 2)}</td><td>${fmt(r.compression_ratio, 2)}</td><td>${fmt(r.latency_ms, 2)}</td>
      <td>${fmt(r.fine_tuning_time_s, 1)}</td></tr>`
    )
    .join("");
}
