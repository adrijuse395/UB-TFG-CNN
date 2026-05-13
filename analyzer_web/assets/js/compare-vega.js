import { LS_COMPARE_SOURCES } from "./constants.js";
import { uniq } from "./format-utils.js";

export function loadComparePersisted() {
  try {
    const raw = localStorage.getItem(LS_COMPARE_SOURCES);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((e) => ({
        kind: "run",
        run_id: e.run_id != null ? String(e.run_id) : "",
        label: e.label != null ? String(e.label) : "",
        includeBaseline: e.includeBaseline !== false,
        method: e.method != null && String(e.method).trim() !== "" ? String(e.method) : "__all__",
        phase: e.phase != null && String(e.phase).trim() !== "" ? String(e.phase) : "__all__",
        experiment_name:
          e.experiment_name != null && String(e.experiment_name).trim() !== ""
            ? String(e.experiment_name)
            : "__all__",
      }))
      .filter((e) => e.run_id);
  } catch (_) {
    return [];
  }
}

export function saveComparePersisted(entries) {
  const serializable = entries
    .filter((s) => s.kind === "run" && s.run_id)
    .map((s) => ({
      kind: "run",
      run_id: s.run_id,
      label: s.label,
      includeBaseline: s.includeBaseline !== false,
      method: s.filterMethod && s.filterMethod !== "__all__" ? s.filterMethod : "__all__",
      phase: s.filterPhase && s.filterPhase !== "__all__" ? s.filterPhase : "__all__",
      experiment_name:
        s.filterExperiment && s.filterExperiment !== "__all__" ? s.filterExperiment : "__all__",
    }));
  localStorage.setItem(LS_COMPARE_SOURCES, JSON.stringify(serializable));
}

export function getCompareVegaThemeConfig(themeKey) {
  const clean = {
    background: "#ffffff",
    view: { stroke: null },
    axis: {
      grid: true,
      gridColor: "#e8e8ea",
      gridOpacity: 1,
      domainColor: "#c8c8cc",
      tickColor: "#c8c8cc",
      labelFontSize: 11,
      titleFontSize: 12,
      titleFontWeight: "normal",
    },
    legend: {
      labelFontSize: 11,
      symbolSize: 44,
      orient: "bottom",
      direction: "horizontal",
    },
    title: { anchor: "start", fontSize: 15, offset: 10, fontWeight: 600 },
  };
  const ts = typeof vegaThemes !== "undefined" ? vegaThemes : {};
  const map = {
    ggplot2: ts.ggplot2,
    quartz: ts.quartz,
    urban: ts.urbaninstitute,
    excel: ts.excel,
    dark: ts.dark,
  };
  const picked = map[themeKey];
  if (!picked) return clean;
  try {
    return JSON.parse(JSON.stringify(picked));
  } catch (_) {
    return clean;
  }
}

/** Vega-Lite spec for Compare: one mark per series, line connects samples in X order, optional tiny points. */
export function buildCompareVegaLiteSpec({
  values,
  xTitle,
  yTitle,
  chartTitle,
  vizStyle,
  pointSize,
  themeKey,
}) {
  const config = getCompareVegaThemeConfig(themeKey);
  const ps = Math.max(2, Number(pointSize) || 10);

  const encoding = {
    x: {
      field: "x",
      type: "quantitative",
      title: xTitle,
      scale: { nice: true, zero: false },
    },
    y: {
      field: "y",
      type: "quantitative",
      title: yTitle,
      scale: { nice: true, zero: false },
    },
    color: {
      field: "series",
      type: "nominal",
      title: "Series",
      legend: { orient: "bottom", direction: "horizontal" },
    },
    order: { field: "x", type: "quantitative", sort: "ascending" },
    tooltip: [
      { field: "series", type: "nominal", title: "Series" },
      { field: "experiment", type: "nominal", title: "Experiment" },
      { field: "method", type: "nominal", title: "Method" },
      { field: "phase", type: "nominal", title: "Phase" },
      { field: "x", type: "quantitative", title: xTitle, format: ".6g" },
      { field: "y", type: "quantitative", title: yTitle, format: ".6g" },
    ],
  };

  const interpolate = vizStyle === "step-point" ? "step-after" : "linear";

  let mark;
  if (vizStyle === "line") {
    mark = { type: "line", interpolate, strokeWidth: 1.05 };
  } else if (vizStyle === "point") {
    mark = { type: "point", filled: true, size: ps, stroke: "white", strokeWidth: 0.15, opacity: 0.9 };
  } else {
    mark = {
      type: "line",
      interpolate,
      strokeWidth: 0.8,
      point: { filled: true, size: ps, stroke: "white", strokeWidth: 0.2, opacity: 0.92 },
    };
  }

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: { text: chartTitle, anchor: "start", fontSize: 15, offset: 10 },
    width: "container",
    height: 440,
    autosize: { type: "fit-x", contains: "padding" },
    data: { values },
    mark,
    encoding,
    config,
  };
}

export function newCompareSourceId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `src_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export function collectCompareRunOptions(meta, rows) {
  const fromRows = uniq(
    rows.map((r) => r.run_id).filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
  );
  const fromMeta = Array.isArray(meta?.runs)
    ? meta.runs.filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
    : [];
  return uniq([...fromMeta, ...fromRows]).sort();
}
