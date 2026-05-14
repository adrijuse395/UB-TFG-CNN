import { SERIES_OPTIONS, PLOT_CONFIG } from "./constants.js";
import { fmt, uniq, fieldLabel } from "./format-utils.js";
import { setOptions, setAxisOptions, escapeHtml, safeSetHtml, updateDashboardFiltersForActiveTab } from "./dom-utils.js";
import { withDerived, isBaselineRow } from "./rows.js";
import { pickDefaultRun } from "./data-loader.js";
import { plotlyNewPlot, plotlyPurge, academicLayout, buildChartTraces } from "./plotly-charts.js";
import { initCompareTab } from "./compare-tab.js";
import { initPredefinedGraphsTab } from "./predefined-graphs-tab.js";

export function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.tab;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) => p.classList.toggle("active", p.id === `panel-${id}`));
      updateDashboardFiltersForActiveTab();
      if (id === "compare" && typeof window.__analyzerRefreshCompareRuns === "function") {
        window.__analyzerRefreshCompareRuns();
      }
      window.dispatchEvent(new Event("resize"));
      requestAnimationFrame(() => {
        if (typeof Plotly === "undefined") return;
        const tryResize = (el) => {
          if (!el) return;
          try {
            Plotly.Plots.resize(el);
          } catch (_) {
            /* element may not host a Plotly figure */
          }
        };
        tryResize(document.getElementById("customChart"));
        tryResize(document.getElementById("paretoChart"));
        tryResize(document.getElementById("idealGapChart"));
        if (id === "predefined-graphs" && typeof window.__analyzerResizePredefinedGraphs === "function") {
          window.__analyzerResizePredefinedGraphs();
        }
      });
    });
  });
}

export function initDashboard(payload) {
  const DATA = Array.isArray(payload.rows) ? payload.rows : [];
  const meta = payload.meta || {};
  const DATA_ALL = withDerived(DATA, DATA);

  const metaLine = document.getElementById("dataMetaLine");
  if (metaLine) {
    const gen = meta.generated_at || "?";
    const byRun = meta.row_count_by_run || {};
    const latest = meta.latest_run || "";
    const src = meta.source || "?";
    const sourceLabel =
      src === "live_csv" ? "live CSV" : String(src ?? "");
    const counts = Object.entries(byRun)
      .map(([k, v]) => `${k}: ${v}`)
      .join(" · ");
    const baseHint = meta.runs_fetch_base ? ` · runs base: ${meta.runs_fetch_base}` : "";
    metaLine.textContent = `Dataset (${sourceLabel}): ${DATA.length} rows · ${gen} · latest run: ${latest}${
      counts ? ` · ${counts}` : ""
    }${baseHint}`;
  }

  initTabs();

  const runSel = document.getElementById("runSelect");
  const methodSel = document.getElementById("methodSelect");
  const phaseSel = document.getElementById("phaseSelect");
  const expSel = document.getElementById("experimentSelect");
  const showBaselineToggle = document.getElementById("showBaselineToggle");
  const tableBody = document.getElementById("tableBody");

  const chartX = document.getElementById("chartXField");
  const chartY = document.getElementById("chartYField");
  const chartType = document.getElementById("chartPlotType");
  const chartSeries = document.getElementById("chartSeriesField");
  const chartTitleIn = document.getElementById("chartTitleInput");
  const chartXLabelIn = document.getElementById("chartXLabelInput");
  const chartYLabelIn = document.getElementById("chartYLabelInput");
  const customChartEl = document.getElementById("customChart");

  const filtersOk = runSel && methodSel && phaseSel && expSel && showBaselineToggle && tableBody;

  const chartReady =
    chartX &&
    chartY &&
    chartType &&
    chartSeries &&
    chartTitleIn &&
    chartXLabelIn &&
    chartYLabelIn &&
    customChartEl;

  if (!filtersOk) {
    const err = document.getElementById("errorBox");
    if (err) err.textContent = "Missing table/filter markup. Hard-refresh (Ctrl+Shift+R).";
    return;
  }

  if (!chartReady) {
    const err = document.getElementById("errorBox");
    if (err) {
      err.textContent = "Missing chart panel (customChart). Hard-refresh (Ctrl+Shift+R).";
    }
  }

  setOptions(runSel, meta.runs || []);
  runSel.value = pickDefaultRun(meta);
  setOptions(methodSel, meta.methods || []);
  setOptions(phaseSel, meta.phases || []);

  if (chartReady) {
    setAxisOptions(chartX);
    setAxisOptions(chartY);
    chartX.value = "rank_scalar";
    chartY.value = "accuracy";
    chartSeries.innerHTML = "";
    SERIES_OPTIONS.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.key;
      o.textContent = s.label;
      chartSeries.appendChild(o);
    });
    chartSeries.value = "fine_tuning_enabled";
  }

  const baseFilter = () =>
    DATA_ALL.filter(
      (r) =>
        (runSel.value === "__all__" || r.run_id === runSel.value) &&
        (methodSel.value === "__all__" || r.method === methodSel.value) &&
        (phaseSel.value === "__all__" || r.phase === phaseSel.value) &&
        (showBaselineToggle.checked || !isBaselineRow(r))
    );

  const refreshExpFilter = () => {
    const rows = baseFilter();
    const current = expSel.value;
    setOptions(
      expSel,
      uniq(
        rows
          .map((r) => r.experiment_name)
          .filter((n) => n !== undefined && n !== null && String(n).trim() !== "")
      )
    );
    if ([...expSel.options].some((o) => o.value === current)) expSel.value = current;
  };

  const filtered = () =>
    withDerived(
      baseFilter().filter((r) => expSel.value === "__all__" || r.experiment_name === expSel.value),
      DATA_ALL
    );

  const renderCustomChart = (rows) => {
    if (!chartReady) return;
    const xKey = chartX.value;
    const yKey = chartY.value;
    const sKey = chartSeries.value;
    const mode = chartType.value;
    const traces = buildChartTraces(rows, xKey, yKey, sKey, mode, { multiDataset: false });
    const xLab = chartXLabelIn.value.trim() || fieldLabel(xKey);
    const yLab = chartYLabelIn.value.trim() || fieldLabel(yKey);
    const title = chartTitleIn.value.trim() || `${yLab} vs ${xLab}`;
    if (!traces.length) {
      plotlyPurge(customChartEl);
      safeSetHtml(
        customChartEl,
        '<p class="named-empty">No numeric X/Y pairs for the current filters — change axes or filters.</p>'
      );
      return;
    }
    plotlyNewPlot(customChartEl, traces, academicLayout(title, xLab, yLab), PLOT_CONFIG);
  };

  const renderTable = (rows) => {
    const sorted = [...rows].sort((a, b) => (b.accuracy ?? -1) - (a.accuracy ?? -1));
    safeSetHtml(
      tableBody,
      sorted
        .map(
          (r) => `<tr>
      <td>${escapeHtml(r.run_id)}</td><td>${escapeHtml(r.experiment_name)}</td><td>${escapeHtml(r.method)}</td><td>${escapeHtml(r.phase)}</td><td>${escapeHtml(r.rank_raw)}</td>
      <td>${fmt(r.accuracy, 2)}</td><td>${fmt(r.delta_accuracy_vs_baseline, 2)}</td><td>${fmt(r.compression_ratio, 2)}</td>
      <td>${fmt(r.latency_ms, 2)}</td><td>${fmt(r.throughput_fps, 2)}</td><td>${fmt(r.fine_tuning_time_s, 1)}</td></tr>`
        )
        .join("")
    );
  };

  const render = () => {
    refreshExpFilter();
    const rows = filtered();
    renderTable(rows);
    renderCustomChart(rows);
  };

  [
    runSel,
    methodSel,
    phaseSel,
    expSel,
    showBaselineToggle,
    chartX,
    chartY,
    chartType,
    chartSeries,
    chartTitleIn,
    chartXLabelIn,
    chartYLabelIn,
  ]
    .filter(Boolean)
    .forEach((el) => el.addEventListener("input", render));
  [runSel, methodSel, phaseSel, expSel, showBaselineToggle, chartX, chartY, chartType, chartSeries]
    .filter(Boolean)
    .forEach((el) => el.addEventListener("change", render));

  window.addEventListener("resize", () => {
    if (typeof Plotly !== "undefined" && customChartEl) {
      Plotly.Plots.resize(customChartEl);
    }
  });

  refreshExpFilter();
  render();

  initCompareTab(DATA_ALL, meta);
  initPredefinedGraphsTab(DATA_ALL, meta);

  if (typeof location !== "undefined") {
    const h = location.hash.slice(1);
    if (h === "compression-map" || h === "predefined-graphs" || h === "predefined-graphs--ideal-gap") {
      const t = document.querySelector('.tab[data-tab="predefined-graphs"]');
      if (t) t.click();
      if (typeof window.__analyzerSetPredefinedGraph === "function") {
        window.__analyzerSetPredefinedGraph(h === "predefined-graphs--ideal-gap" ? "ideal-gap" : "tradeoff");
      }
    }
  }

  updateDashboardFiltersForActiveTab();
}
