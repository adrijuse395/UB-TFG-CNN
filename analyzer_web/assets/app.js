const PLOT_CONFIG = { responsive: true, displaylogo: false };
const LIVE_REFRESH_MS = 8000;

/** Plotly needs a real DOM node; string ids cause null.innerHTML inside Plotly when the div is missing. */
function plotlyNewPlot(containerIdOrEl, traces, layout, cfg) {
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

function plotlyPurge(containerIdOrEl) {
  const gd =
    typeof containerIdOrEl === "string" ? document.getElementById(containerIdOrEl) : containerIdOrEl;
  if (!gd || typeof Plotly === "undefined") return;
  try {
    Plotly.purge(gd);
  } catch (_) {
    /* ignore */
  }
}

const AXIS_FIELDS = [
  { key: "rank_scalar", label: "Rank" },
  { key: "accuracy", label: "Accuracy (%)" },
  { key: "compression_ratio", label: "Compression ratio (x)" },
  { key: "latency_ms", label: "Latency (ms)" },
  { key: "throughput_fps", label: "Throughput (samples/s)" },
  { key: "macs_g", label: "GMACs" },
  { key: "total_parameters", label: "Parameters" },
  { key: "compression_time_s", label: "Compression time (s)" },
  { key: "fine_tuning_time_s", label: "Fine-tune time (s)" },
];

const METRICS = AXIS_FIELDS.filter((f) =>
  ["accuracy", "compression_ratio", "latency_ms", "throughput_fps", "macs_g"].includes(f.key)
).map((f) => f.key);

const SERIES_OPTIONS = [
  { key: "fine_tuning_enabled", label: "Fine-tuning" },
  { key: "phase", label: "Phase" },
  { key: "method", label: "Method" },
  { key: "run_id", label: "Run" },
];

const LS_NAMED_COMPARISON = "analyzer_named_comparison_v1";

const fmt = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d);

const uniq = (arr) =>
  [...new Set(arr.filter((v) => v !== null && v !== undefined && String(v) !== ""))].sort();

const toFloat = (v) => {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
};

const toBool = (v) => ["true", "1", "yes", "y"].includes(String(v).trim().toLowerCase());

function rankScalar(raw) {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s || s === "None" || s === "null") return null;
  const direct = Number(s);
  if (Number.isFinite(direct)) return direct;
  try {
    const parsed = JSON.parse(s);
    if (Array.isArray(parsed) && parsed.length) {
      const n = Number(parsed[0]);
      return Number.isFinite(n) ? n : null;
    }
  } catch (_) {
    return null;
  }
  return null;
}

function fieldLabel(key) {
  const f = AXIS_FIELDS.find((x) => x.key === key);
  return f ? f.label : key;
}

function academicLayout(title, xTitle, yTitle) {
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

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function parseCsvText(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (!lines.length) return [];
  const headers = parseCsvLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const values = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

async function listRunIdsFromHttpDirectory() {
  const runsUrl = new URL("../runs/", window.location.href);
  const res = await fetch(`${runsUrl.href}?cb=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Cannot list runs directory (${res.status}). Serve project root with python3 -m http.server`);
  }
  const html = await res.text();
  const ids = [];
  const re = /href="(run_\d{8}_\d{6}\/?)"/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    ids.push(m[1].replace(/\/$/, ""));
  }
  return uniq(ids);
}

function isBaselineRow(r) {
  const name = (r.experiment_name || "").trim();
  return (
    r.phase === "baseline" ||
    (r.method === "None" && name === "Baseline") ||
    (r.phase === "legacy" && name === "Baseline")
  );
}

function baselineMap(rows) {
  const map = new Map();
  rows.forEach((r) => {
    if (isBaselineRow(r)) map.set(r.run_id, r);
  });
  return map;
}

function withDerived(rows, allRows) {
  const b = baselineMap(allRows);
  return rows.map((r) => {
    const base = b.get(r.run_id);
    const delta =
      base && r.accuracy !== null && base.accuracy !== null ? r.accuracy - base.accuracy : null;
    return { ...r, baseline_accuracy: base ? base.accuracy : null, delta_accuracy_vs_baseline: delta };
  });
}

async function loadDataFromRuns() {
  const runIds = await listRunIdsFromHttpDirectory();
  const allRows = [];
  const rowCountByRun = {};
  for (const runId of runIds) {
    const csvUrl = new URL(`../runs/${runId}/results.csv?cb=${Date.now()}`, window.location.href);
    const res = await fetch(csvUrl.href, { cache: "no-store" });
    if (!res.ok) continue;
    const rawRows = parseCsvText(await res.text());
    rowCountByRun[runId] = rawRows.length;
    rawRows.forEach((row) => {
      allRows.push({
        run_id: runId,
        experiment_name: row.experiment_name || "",
        method: row.method || "",
        phase: row.fine_tuning_phase || "legacy",
        fine_tuning_enabled: toBool(row.fine_tuning_enabled),
        rank_raw: row.rank || "",
        rank_scalar: rankScalar(row.rank),
        accuracy: toFloat(row.accuracy),
        compression_ratio: toFloat(row.compression_ratio),
        latency_ms: toFloat(row.latency_ms),
        throughput_fps: toFloat(row.throughput_fps),
        macs_g: toFloat(row.macs_g),
        total_parameters: toFloat(row.total_parameters),
        compression_time_s: toFloat(row.compression_time_s),
        fine_tuning_time_s: toFloat(row.fine_tuning_time_s),
      });
    });
  }
  return {
    meta: {
      rows: allRows.length,
      runs: runIds,
      methods: uniq(allRows.map((r) => r.method)),
      phases: uniq(allRows.map((r) => r.phase)),
      latest_run: runIds.length ? [...runIds].sort().at(-1) : "",
      row_count_by_run: rowCountByRun,
      generated_at: new Date().toISOString(),
      source: "live_csv",
    },
    rows: allRows,
  };
}

async function loadDataFromJsonFallback() {
  const url = `./data/results.json?cb=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Cannot load live CSV or data/results.json (${res.status})`);
  const payload = await res.json();
  payload.meta = payload.meta || {};
  payload.meta.source = payload.meta.source || "results_json";
  return payload;
}

async function loadData() {
  try {
    return await loadDataFromRuns();
  } catch (_) {
    return loadDataFromJsonFallback();
  }
}

function normalizePayload(raw) {
  const rows = Array.isArray(raw?.rows) ? raw.rows : [];
  const meta = raw?.meta && typeof raw.meta === "object" ? { ...raw.meta } : {};
  const methodsU = uniq(rows.map((r) => r.method).filter((x) => x !== undefined && x !== ""));
  const phasesU = uniq(rows.map((r) => r.phase).filter((x) => x !== undefined && x !== ""));
  const runsU = uniq(rows.map((r) => r.run_id).filter((x) => x !== undefined && x !== ""));
  meta.runs = Array.isArray(meta.runs) && meta.runs.length ? meta.runs : runsU;
  meta.methods =
    Array.isArray(meta.methods) && meta.methods.length ? meta.methods : methodsU;
  meta.phases = Array.isArray(meta.phases) && meta.phases.length ? meta.phases : phasesU;
  meta.row_count_by_run =
    meta.row_count_by_run && typeof meta.row_count_by_run === "object" ? meta.row_count_by_run : {};
  meta.rows = rows.length;
  if (!meta.generated_at) meta.generated_at = new Date().toISOString();
  if (meta.latest_run === undefined || meta.latest_run === null || meta.latest_run === "") {
    meta.latest_run =
      meta.runs.length ? [...meta.runs].sort().at(-1) : runsU.length ? [...runsU].sort().at(-1) : "";
  }
  if (!meta.source) meta.source = rows.length ? "dataset" : "empty";
  return { rows, meta };
}

function pickDefaultRun(meta) {
  const runs = meta.runs || [];
  const latest = meta.latest_run || "";
  if (latest && runs.includes(latest)) return latest;
  const counts = meta.row_count_by_run || {};
  let best = "";
  let bestN = -1;
  for (const rid of runs) {
    const n = counts[rid] ?? 0;
    if (n > bestN) {
      bestN = n;
      best = rid;
    }
  }
  return best || "__all__";
}

function setOptions(select, values, includeAll = true) {
  if (!select) return;
  const vals = Array.isArray(values) ? values : [];
  select.innerHTML = "";
  if (includeAll) {
    const o = document.createElement("option");
    o.value = "__all__";
    o.textContent = "All";
    select.appendChild(o);
  }
  vals.forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    select.appendChild(o);
  });
}

function setAxisOptions(select) {
  if (!select) return;
  select.innerHTML = "";
  AXIS_FIELDS.forEach((f) => {
    const o = document.createElement("option");
    o.value = f.key;
    o.textContent = f.label;
    select.appendChild(o);
  });
}

function getRowNumeric(r, key) {
  const v = r[key];
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return Number(v);
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const id = tab.dataset.tab;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) => p.classList.toggle("active", p.id === `panel-${id}`));
      window.dispatchEvent(new Event("resize"));
      requestAnimationFrame(() => {
        if (typeof Plotly === "undefined") return;
        const customChart = document.getElementById("customChart");
        const cmpChart = document.getElementById("cmpChart");
        if (customChart) Plotly.Plots.resize(customChart);
        if (cmpChart) Plotly.Plots.resize(cmpChart);
      });
    });
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function safeSetHtml(el, html) {
  if (el) el.innerHTML = html;
}

function formatRowOneLine(r) {
  const rk =
    r.rank_raw !== undefined && String(r.rank_raw).trim() !== ""
      ? String(r.rank_raw)
      : String(r.rank_scalar ?? "—");
  const ft = r.fine_tuning_enabled ? "FT" : "no FT";
  return `${r.run_id} · ${r.method} · ${r.experiment_name} · ${r.phase} · r=${rk} · ${ft}`;
}

function seriesLabel(row, key) {
  if (key === "fine_tuning_enabled") return row.fine_tuning_enabled ? "Fine-tuned" : "No FT";
  return String(row[key] ?? "-");
}

function loadNamedEntries() {
  try {
    const raw = localStorage.getItem(LS_NAMED_COMPARISON);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((e, i) => ({
        uid:
          e.uid ||
          (typeof crypto !== "undefined" && crypto.randomUUID
            ? crypto.randomUUID()
            : `legacy_${i}_${Date.now()}`),
        label: e.label != null ? String(e.label) : "Unnamed",
        row: e.row,
      }))
      .filter((e) => e.row);
  } catch (_) {
    return [];
  }
}

function saveNamedEntries(entries) {
  localStorage.setItem(LS_NAMED_COMPARISON, JSON.stringify(entries));
}

function rowFingerprint(r) {
  return `${r.run_id}|${r.experiment_name}|${r.method}|${r.phase}|${r.rank_scalar}|${r.fine_tuning_enabled}`;
}

function initDashboard(payload) {
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
      src === "live_csv" ? "live CSV" : src === "results_json" ? "results.json" : String(src);
    const counts = Object.entries(byRun)
      .map(([k, v]) => `${k}: ${v}`)
      .join(" · ");
    metaLine.textContent = `Dataset (${sourceLabel}): ${DATA.length} rows · ${gen} · latest run: ${latest}${
      counts ? ` · ${counts}` : ""
    }`;
  }

  initTabs();

  const runSel = document.getElementById("runSelect");
  const methodSel = document.getElementById("methodSelect");
  const phaseSel = document.getElementById("phaseSelect");
  const expSel = document.getElementById("experimentSelect");
  const metricSel = document.getElementById("metricSelect");
  const colorSel = document.getElementById("colorBySelect");
  const showBaselineToggle = document.getElementById("showBaselineToggle");
  const kpiGridEl = document.getElementById("kpiGrid");
  const tableBody = document.getElementById("tableBody");

  const chartX = document.getElementById("chartXField");
  const chartY = document.getElementById("chartYField");
  const chartType = document.getElementById("chartPlotType");
  const chartSeries = document.getElementById("chartSeriesField");
  const chartTitleIn = document.getElementById("chartTitleInput");
  const chartXLabelIn = document.getElementById("chartXLabelInput");
  const chartYLabelIn = document.getElementById("chartYLabelInput");
  const customChartEl = document.getElementById("customChart");

  const filtersOk =
    runSel &&
    methodSel &&
    phaseSel &&
    expSel &&
    showBaselineToggle &&
    kpiGridEl &&
    tableBody;

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
  if (metricSel) {
    setOptions(metricSel, METRICS, false);
    metricSel.value = "accuracy";
  }
  if (colorSel) {
    setOptions(colorSel, ["fine_tuning_enabled", "phase", "run_id"], false);
    colorSel.value = "fine_tuning_enabled";
  }

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
    setOptions(expSel, uniq(rows.map((r) => r.experiment_name)));
    if ([...expSel.options].some((o) => o.value === current)) expSel.value = current;
  };

  const filtered = () =>
    withDerived(
      baseFilter().filter((r) => expSel.value === "__all__" || r.experiment_name === expSel.value),
      DATA_ALL
    );

  const renderKpis = (rows) => {
    const avg = (k) => {
      const vals = rows.map((r) => r[k]).filter((x) => x !== null && !Number.isNaN(x));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const best = [...rows]
      .filter((r) => r.accuracy !== null)
      .sort((a, b) => b.accuracy - a.accuracy)[0];
    let scopeNote = "after filters";
    if (runSel.value !== "__all__") {
      const total = (meta.row_count_by_run || {})[runSel.value];
      if (total !== undefined) {
        scopeNote =
          rows.length === total
            ? `all ${total} rows for this run`
            : `${rows.length} of ${total} rows (relax experiment / method / phase)`;
      }
    }
    const cards = [
      ["Rows (filtered)", rows.length, scopeNote],
      ["Mean accuracy", `${fmt(avg("accuracy"), 2)}%`, ""],
      ["Mean Δ vs baseline", `${fmt(avg("delta_accuracy_vs_baseline"), 2)} pp`, ""],
      ["Mean compression", `${fmt(avg("compression_ratio"), 2)}x`, ""],
      ["Best accuracy", best ? `${fmt(best.accuracy, 2)}%` : "-", best ? best.experiment_name : ""],
    ];
    safeSetHtml(
      kpiGridEl,
      cards
        .map(
          ([k, v, n]) =>
            `<div class="panel"><div class="kpi-title">${k}</div><div class="kpi-value">${v}</div><div class="kpi-note">${n}</div></div>`
        )
        .join("")
    );
  };

  const renderCustomChart = (rows) => {
    if (!chartReady) return;
    const xKey = chartX.value;
    const yKey = chartY.value;
    const sKey = chartSeries.value;
    const mode = chartType.value;

    const valid = rows.filter((r) => {
      const x = getRowNumeric(r, xKey);
      const y = getRowNumeric(r, yKey);
      return x !== null && y !== null;
    });

    const groups = {};
    valid.forEach((r) => {
      const name = seriesLabel(r, sKey);
      groups[name] = groups[name] || [];
      groups[name].push(r);
    });

    const traces = Object.entries(groups).map(([name, arr]) => {
      const sorted = [...arr].sort((a, b) => getRowNumeric(a, xKey) - getRowNumeric(b, xKey));
      const x = sorted.map((r) => getRowNumeric(r, xKey));
      const y = sorted.map((r) => getRowNumeric(r, yKey));
      const text = sorted.map((r) => `${r.experiment_name}<br>${r.run_id}<br>${r.method} · ${r.phase}`);
      if (mode === "line") return { name, mode: "lines+markers", x, y, text };
      return { name, mode: "markers", type: "scatter", x, y, text, marker: { size: 9, opacity: 0.88 } };
    });

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
    renderKpis(rows);
    renderTable(rows);
    renderCustomChart(rows);
  };

  [
    runSel,
    methodSel,
    phaseSel,
    expSel,
    metricSel,
    colorSel,
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
  [runSel, methodSel, phaseSel, expSel, metricSel, colorSel, showBaselineToggle, chartX, chartY, chartType, chartSeries]
    .filter(Boolean)
    .forEach((el) => el.addEventListener("change", render));

  window.addEventListener("resize", () => {
    if (typeof Plotly !== "undefined") {
      if (customChartEl) Plotly.Plots.resize(customChartEl);
      const cmp = document.getElementById("cmpChart");
      if (cmp) Plotly.Plots.resize(cmp);
    }
  });

  refreshExpFilter();
  render();

  initCompareTab(DATA_ALL);
}

function initCompareTab(DATA_ALL) {
  const cmpSearch = document.getElementById("cmpSearch");
  const cmpIncludeBaseline = document.getElementById("cmpIncludeBaseline");
  const cmpRowSelect = document.getElementById("cmpRowSelect");
  const cmpLabel = document.getElementById("cmpLabel");
  const cmpAdd = document.getElementById("cmpAdd");
  const cmpClear = document.getElementById("cmpClear");
  const cmpList = document.getElementById("cmpList");
  const cmpCount = document.getElementById("cmpCount");
  const cmpEmpty = document.getElementById("cmpEmpty");
  const cmpMeta = document.getElementById("cmpMeta");
  const cmpX = document.getElementById("cmpX");
  const cmpY = document.getElementById("cmpY");
  const cmpChart = document.getElementById("cmpChart");

  if (!cmpRowSelect || !cmpAdd || !cmpClear || !cmpList || !cmpX || !cmpY || !cmpChart) {
    return;
  }

  let entries = loadNamedEntries();

  const refreshCmpPicker = () => {
    const q = (cmpSearch && cmpSearch.value ? cmpSearch.value : "").trim().toLowerCase();
    const inc = cmpIncludeBaseline ? cmpIncludeBaseline.checked : false;
    let pool = inc ? [...DATA_ALL] : DATA_ALL.filter((r) => !isBaselineRow(r));
    if (q) {
      pool = pool.filter((r) => {
        const blob = `${r.run_id} ${r.method} ${r.experiment_name} ${r.phase} ${r.rank_raw} ${r.rank_scalar}`.toLowerCase();
        return blob.includes(q);
      });
    }
    cmpRowSelect.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = pool.length
      ? `— Select a row (${pool.length} matches) —`
      : "— No matches — adjust search or include baseline";
    opt0.disabled = true;
    cmpRowSelect.appendChild(opt0);
    const seen = new Set();
    pool.forEach((r) => {
      const fp = rowFingerprint(r);
      if (seen.has(fp)) return;
      seen.add(fp);
      const o = document.createElement("option");
      o.value = fp;
      o.textContent = formatRowOneLine(r);
      cmpRowSelect.appendChild(o);
    });
    if (cmpMeta) {
      cmpMeta.textContent = `${pool.length} shown · ${DATA_ALL.length} rows in dataset`;
    }
  };

  setAxisOptions(cmpX);
  setAxisOptions(cmpY);
  cmpX.value = "total_parameters";
  cmpY.value = "accuracy";

  const renderCmpList = () => {
    if (cmpCount) cmpCount.textContent = String(entries.length);
    if (cmpEmpty) cmpEmpty.classList.toggle("hidden", entries.length > 0);
    safeSetHtml(
      cmpList,
      entries
        .map(
          (e) => `<li class="named-row cmp-row" data-uid="${escapeHtml(e.uid)}">
      <div>
        <div class="named-row-label">${escapeHtml(e.label)}</div>
        <div class="named-row-meta">${escapeHtml(formatRowOneLine(e.row))}</div>
      </div>
      <button type="button" class="btn-remove" data-remove="${escapeHtml(e.uid)}">Remove</button>
    </li>`
        )
        .join("")
    );
    cmpList.querySelectorAll("[data-remove]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const uid = btn.getAttribute("data-remove");
        entries = entries.filter((x) => x.uid !== uid);
        saveNamedEntries(entries);
        renderCmpList();
        renderCmpChart();
      });
    });
  };

  const renderCmpChart = () => {
    const xKey = cmpX.value;
    const yKey = cmpY.value;

    const valid = entries.filter((e) => {
      const x = getRowNumeric(e.row, xKey);
      const y = getRowNumeric(e.row, yKey);
      return x !== null && y !== null;
    });

    if (!valid.length) {
      plotlyPurge(cmpChart);
      cmpChart.innerHTML =
        '<p class="named-empty">Add experiments above, or pick X/Y fields that are numeric for those rows.</p>';
      return;
    }

    const usedLegend = new Map();
    const traces = valid.map((e) => {
      const x = getRowNumeric(e.row, xKey);
      const y = getRowNumeric(e.row, yKey);
      let legendName = (e.label && String(e.label).trim()) || "Unnamed";
      const n = (usedLegend.get(legendName) || 0) + 1;
      usedLegend.set(legendName, n);
      if (n > 1) legendName = `${(e.label && String(e.label).trim()) || "Unnamed"} (${n})`;
      const text = `${escapeHtml(e.label)}<br>${escapeHtml(e.row.experiment_name)}<br>${escapeHtml(
        e.row.method
      )} · ${escapeHtml(e.row.phase)}`;
      return {
        name: legendName,
        type: "scatter",
        mode: "markers",
        x: [x],
        y: [y],
        text: [text],
        marker: { size: 12, opacity: 0.92, line: { width: 1, color: "#fff" } },
      };
    });

    const xLab = fieldLabel(xKey);
    const yLab = fieldLabel(yKey);
    const title = `${yLab} vs ${xLab} (named comparison)`;
    plotlyNewPlot(cmpChart, traces, academicLayout(title, xLab, yLab), PLOT_CONFIG);
  };

  cmpAdd.addEventListener("click", () => {
    const fp = cmpRowSelect.value;
    if (!fp) {
      alert("Select a row in the list (below the placeholder).");
      return;
    }
    const rowOrig = DATA_ALL.find((r) => rowFingerprint(r) === fp);
    if (!rowOrig) {
      alert("That row is no longer in the dataset — reload the page.");
      return;
    }
    const row = { ...rowOrig };
    let label = cmpLabel && (cmpLabel.value || "").trim();
    if (!label) label = formatRowOneLine(row).slice(0, 140);

    const uid =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `id_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    entries.push({ uid, label, row });
    saveNamedEntries(entries);
    if (cmpLabel) cmpLabel.value = "";
    renderCmpList();
    renderCmpChart();
  });

  cmpClear.addEventListener("click", () => {
    if (!entries.length) return;
    if (!window.confirm("Remove all experiments from the comparison?")) return;
    entries = [];
    saveNamedEntries(entries);
    renderCmpList();
    renderCmpChart();
  });

  if (cmpSearch) cmpSearch.addEventListener("input", refreshCmpPicker);
  if (cmpIncludeBaseline) cmpIncludeBaseline.addEventListener("change", refreshCmpPicker);

  [cmpX, cmpY].forEach((el) => {
    el.addEventListener("input", renderCmpChart);
    el.addEventListener("change", renderCmpChart);
  });

  refreshCmpPicker();
  renderCmpList();
  renderCmpChart();
}

function initExperimentPage(payload) {
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
        mode: "lines+markers",
        x: sorted.map((a) => a.rank_scalar),
        y: sorted.map((a) => a.accuracy),
        text: sorted.map((a) => `${a.experiment_name}<br>${a.phase}`),
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

async function boot() {
  try {
    const payload = normalizePayload(await loadData());
    const signature = JSON.stringify({
      rows: payload.meta.rows,
      latest: payload.meta.latest_run,
      counts: payload.meta.row_count_by_run || {},
      source: payload.meta.source || "",
    });

    if (document.body.dataset.page === "dashboard") initDashboard(payload);
    if (document.body.dataset.page === "run") initExperimentPage(payload);

    if (document.body.dataset.page === "dashboard") {
      window.setInterval(async () => {
        try {
          const latestPayload = normalizePayload(await loadData());
          const nextSignature = JSON.stringify({
            rows: latestPayload.meta.rows,
            latest: latestPayload.meta.latest_run,
            counts: latestPayload.meta.row_count_by_run || {},
            source: latestPayload.meta.source || "",
          });
          if (nextSignature !== signature) window.location.reload();
        } catch (_) {
          // Keep current view if transient read error occurs.
        }
      }, LIVE_REFRESH_MS);
    }
  } catch (e) {
    const el = document.getElementById("errorBox");
    if (el) el.textContent = `Error loading data: ${e.message}`;
  }
}

boot();
