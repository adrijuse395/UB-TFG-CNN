import { uniq } from "./format-utils.js";
import { setOptions, escapeHtml, safeSetHtml } from "./dom-utils.js";
import { isBaselineRow, withDerived, datasetRowFromRawCsvRow } from "./rows.js";
import { parseCsvText } from "./csv.js";
import { listRunIdsFromHttpDirectory } from "./data-loader.js";
import { newCompareSourceId, collectCompareRunOptions } from "./compare-vega.js";
import { renderTradeoffMap } from "./pareto-page.js";
import { renderIdealGapPanel } from "./ideal-gap-panel.js";

const cohortSignature = (runId, includeBaseline, fm, fp, fe) =>
  `${runId}\t${includeBaseline}\t${fm}\t${fp}\t${fe}`;

const PREDEF_HASH_TRADEOFF = "predefined-graphs";
const PREDEF_HASH_IDEAL = "predefined-graphs--ideal-gap";

/**
 * Predefined graphs tab: shared multi-cohort loading (same as Compare), then pick
 * a built-in visualization (trade-off map, ideal-vs-achieved, …) without duplicating cohort logic.
 */
export function initPredefinedGraphsTab(DATA_ALL, meta) {
  const mapRunPick = document.getElementById("mapRunPick");
  const mapMethodPick = document.getElementById("mapMethodPick");
  const mapPhasePick = document.getElementById("mapPhasePick");
  const mapExperimentPick = document.getElementById("mapExperimentPick");
  const mapRunLabel = document.getElementById("mapRunLabel");
  const mapIncludeBaselineRuns = document.getElementById("mapIncludeBaselineRuns");
  const mapAddRun = document.getElementById("mapAddRun");
  const mapFile = document.getElementById("mapFile");
  const mapFileLabel = document.getElementById("mapFileLabel");
  const mapAddFile = document.getElementById("mapAddFile");
  const mapSourceList = document.getElementById("mapSourceList");
  const mapSourceCount = document.getElementById("mapSourceCount");
  const mapSourceEmpty = document.getElementById("mapSourceEmpty");
  const mapClearSources = document.getElementById("mapClearSources");
  const mapMetaLine = document.getElementById("mapMetaLine");
  const mapAccPct = document.getElementById("mapAccPct");
  const mapAccMedian = document.getElementById("mapAccMedian");
  const mapCompThreshold = document.getElementById("mapCompThreshold");
  const mapCompMedian = document.getElementById("mapCompMedian");
  const chartEl = document.getElementById("paretoChart");
  const idealGapEl = document.getElementById("idealGapChart");
  const subTradeoff = document.getElementById("predefSubTradeoff");
  const subIdealGap = document.getElementById("predefSubIdealGap");
  const subTabButtons = document.querySelectorAll("[data-predef-graph]");

  if (!mapRunPick || !chartEl) return;

  let sources = [];
  let activePredef = "tradeoff";

  const hydrateRunCohortRows = (runId, includeBaseline, filterMethod, filterPhase, filterExperiment) => {
    const fm = filterMethod || "__all__";
    const fp = filterPhase || "__all__";
    const fe = filterExperiment || "__all__";
    let pool = DATA_ALL.filter((r) => r.run_id === runId);
    if (!includeBaseline) pool = pool.filter((r) => !isBaselineRow(r));
    if (fm !== "__all__") pool = pool.filter((r) => r.method === fm);
    if (fp !== "__all__") pool = pool.filter((r) => r.phase === fp);
    if (fe !== "__all__") pool = pool.filter((r) => r.experiment_name === fe);
    return withDerived([...pool], DATA_ALL);
  };

  const refillMapCohortFilters = () => {
    if (!mapMethodPick || !mapPhasePick || !mapExperimentPick || !mapRunPick) return;
    const runId = mapRunPick.value;
    const preserve = (sel) => sel.value;
    const curM = preserve(mapMethodPick);
    const curP = preserve(mapPhasePick);
    const curE = preserve(mapExperimentPick);
    if (!runId) {
      setOptions(mapMethodPick, [], true);
      setOptions(mapPhasePick, [], true);
      setOptions(mapExperimentPick, [], true);
      return;
    }
    const pool = DATA_ALL.filter((r) => r.run_id === runId);
    const methods = uniq(pool.map((r) => r.method).filter((x) => x !== undefined && String(x).trim() !== ""));
    const phases = uniq(pool.map((r) => r.phase).filter((x) => x !== undefined && String(x).trim() !== ""));
    const experiments = uniq(
      pool
        .map((r) => r.experiment_name)
        .filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
    );
    setOptions(mapMethodPick, methods, true);
    setOptions(mapPhasePick, phases, true);
    setOptions(mapExperimentPick, experiments, true);
    if ([...mapMethodPick.options].some((o) => o.value === curM)) mapMethodPick.value = curM;
    if ([...mapPhasePick.options].some((o) => o.value === curP)) mapPhasePick.value = curP;
    if ([...mapExperimentPick.options].some((o) => o.value === curE)) mapExperimentPick.value = curE;
  };

  const mergeRowsForMap = () => {
    const out = [];
    sources.forEach((src) => {
      src.rows.forEach((r) => {
        out.push({ ...r, _mapSourceLabel: src.label });
      });
    });
    return out;
  };

  const updateMetaLine = () => {
    if (!mapMetaLine) return;
    const n = sources.length;
    const rows = mergeRowsForMap();
    if (!n) {
      mapMetaLine.textContent = "Add one or more cohorts below — same workflow as the Compare tab.";
      return;
    }
    mapMetaLine.textContent = `${n} cohort(s) · ${rows.length} rows combined`;
  };

  const readMapSplitOpts = () => {
    const accPct = mapAccPct && mapAccPct.value !== "" ? Number(mapAccPct.value) : 97;
    const compTh = mapCompThreshold && mapCompThreshold.value !== "" ? Number(mapCompThreshold.value) : 9;
    return {
      accuracyPctOfBaseline: accPct,
      accuracyUseMedian: !!(mapAccMedian && mapAccMedian.checked),
      compressionThreshold: compTh,
      compressionUseMedian: !!(mapCompMedian && mapCompMedian.checked),
    };
  };

  const syncSplitInputsDisabled = () => {
    if (mapAccPct && mapAccMedian) mapAccPct.disabled = mapAccMedian.checked;
    if (mapCompThreshold && mapCompMedian) mapCompThreshold.disabled = mapCompMedian.checked;
  };

  const renderActiveGraph = () => {
    updateMetaLine();
    if (activePredef === "ideal-gap") {
      renderIdealGapPanel(idealGapEl, mergeRowsForMap());
    } else {
      renderTradeoffMap(chartEl, mergeRowsForMap(), readMapSplitOpts());
    }
  };

  const setPredefGraph = (kind) => {
    activePredef = kind === "ideal-gap" ? "ideal-gap" : "tradeoff";
    subTabButtons.forEach((btn) => {
      const sel = btn.getAttribute("data-predef-graph") === activePredef;
      btn.classList.toggle("active", sel);
      btn.setAttribute("aria-selected", sel ? "true" : "false");
    });
    if (subTradeoff) {
      subTradeoff.classList.toggle("active", activePredef === "tradeoff");
    }
    if (subIdealGap) {
      subIdealGap.classList.toggle("active", activePredef === "ideal-gap");
    }
    renderActiveGraph();
    if (typeof history !== "undefined" && history.replaceState) {
      const suffix = activePredef === "ideal-gap" ? `#${PREDEF_HASH_IDEAL}` : `#${PREDEF_HASH_TRADEOFF}`;
      history.replaceState(null, "", `${location.pathname}${location.search}${suffix}`);
    }
    window.dispatchEvent(new Event("resize"));
  };

  window.__analyzerSetPredefinedGraph = setPredefGraph;

  const renderSourceList = () => {
    if (mapSourceCount) mapSourceCount.textContent = String(sources.length);
    if (mapSourceEmpty) mapSourceEmpty.classList.toggle("hidden", sources.length > 0);
    if (!mapSourceList) return;
    safeSetHtml(
      mapSourceList,
      sources
        .map((s) => {
          let cohortExtra = "";
          if (s.kind === "run") {
            const bits = [];
            if (s.filterMethod && s.filterMethod !== "__all__") bits.push(`method=${escapeHtml(s.filterMethod)}`);
            if (s.filterPhase && s.filterPhase !== "__all__") bits.push(`phase=${escapeHtml(s.filterPhase)}`);
            if (s.filterExperiment && s.filterExperiment !== "__all__")
              bits.push(`experiment=${escapeHtml(s.filterExperiment)}`);
            if (bits.length) cohortExtra = ` · ${bits.join(", ")}`;
          }
          const metaLine =
            s.kind === "run"
              ? `${s.rows.length} rows · ${escapeHtml(s.run_id)}${
                  s.includeBaseline === false ? " · baseline hidden" : ""
                }${cohortExtra}`
              : `${s.rows.length} rows · file`;
          return `<li class="cmp-source-item">
        <div class="cmp-source-body">
          <div class="cmp-source-label">${escapeHtml(s.label)}</div>
          <div class="cmp-source-meta">${metaLine}</div>
        </div>
        <button type="button" class="btn-remove" data-map-remove="${escapeHtml(s.id)}">Remove</button>
      </li>`;
        })
        .join("")
    );
    mapSourceList.querySelectorAll("[data-map-remove]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-map-remove");
        sources = sources.filter((x) => x.id !== id);
        renderSourceList();
        renderActiveGraph();
      });
    });
  };

  let runOptions = collectCompareRunOptions(meta, DATA_ALL);
  setOptions(mapRunPick, runOptions, false);
  if (!runOptions.length) {
    listRunIdsFromHttpDirectory()
      .then((ids) => {
        if (!ids.length) {
          refillMapCohortFilters();
          return;
        }
        const merged = uniq([...ids, ...collectCompareRunOptions(meta, DATA_ALL)]).filter(
          (x) => x !== undefined && x !== null && String(x).trim() !== ""
        ).sort();
        if (merged.length) setOptions(mapRunPick, merged, false);
        refillMapCohortFilters();
      })
      .catch(() => {
        refillMapCohortFilters();
      });
  }
  if (!runOptions.length) {
    const o = document.createElement("option");
    o.value = "";
    o.disabled = true;
    o.selected = true;
    o.textContent = "No runs found — serve repo root (see README)";
    mapRunPick.appendChild(o);
  }
  refillMapCohortFilters();

  mapRunPick.addEventListener("change", refillMapCohortFilters);

  subTabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = btn.getAttribute("data-predef-graph");
      if (k) setPredefGraph(k);
    });
  });

  [mapAccPct, mapCompThreshold].forEach((el) => {
    if (el) el.addEventListener("input", () => renderActiveGraph());
  });
  [mapAccMedian, mapCompMedian].forEach((el) => {
    if (!el) return;
    el.addEventListener("change", () => {
      syncSplitInputsDisabled();
      renderActiveGraph();
    });
  });
  syncSplitInputsDisabled();

  if (mapAddRun) {
    mapAddRun.addEventListener("click", () => {
      const runId = mapRunPick.value;
      if (!runId) {
        alert("Select a run.");
        return;
      }
      const fm = mapMethodPick ? mapMethodPick.value : "__all__";
      const fp = mapPhasePick ? mapPhasePick.value : "__all__";
      const fe = mapExperimentPick ? mapExperimentPick.value : "__all__";
      const includeBaseline = mapIncludeBaselineRuns ? mapIncludeBaselineRuns.checked : true;
      const sig = cohortSignature(runId, includeBaseline, fm, fp, fe);
      if (
        sources.some((s) => {
          if (s.kind !== "run") return false;
          return (
            cohortSignature(
              s.run_id,
              s.includeBaseline,
              s.filterMethod || "__all__",
              s.filterPhase || "__all__",
              s.filterExperiment || "__all__"
            ) === sig
          );
        })
      ) {
        alert("This cohort is already in the list.");
        return;
      }
      const label = (mapRunLabel && mapRunLabel.value.trim()) || runId;
      const rows = hydrateRunCohortRows(runId, includeBaseline, fm, fp, fe);
      if (!rows.length) {
        alert("No rows match that run and filter selection.");
        return;
      }
      sources.push({
        id: newCompareSourceId(),
        kind: "run",
        run_id: runId,
        label,
        includeBaseline,
        filterMethod: fm,
        filterPhase: fp,
        filterExperiment: fe,
        rows,
      });
      if (mapRunLabel) mapRunLabel.value = "";
      renderSourceList();
      renderActiveGraph();
    });
  }

  if (mapAddFile && mapFile) {
    mapAddFile.addEventListener("click", () => {
      if (!mapFile.files || !mapFile.files[0]) {
        alert("Choose a results.csv file first.");
        return;
      }
      const file = mapFile.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        const rawRows = parseCsvText(text);
        if (!rawRows.length) {
          alert("No data rows in that CSV.");
          return;
        }
        const uploadRunId = `upload_${newCompareSourceId().replace(/-/g, "")}`;
        const label =
          (mapFileLabel && mapFileLabel.value.trim()) || file.name.replace(/\.csv$/i, "") || uploadRunId;
        const mapped = rawRows.map((row) => datasetRowFromRawCsvRow(row, uploadRunId));
        const rows = withDerived(mapped, mapped);
        sources.push({
          id: newCompareSourceId(),
          kind: "file",
          label,
          rows,
        });
        mapFile.value = "";
        if (mapFileLabel) mapFileLabel.value = "";
        renderSourceList();
        renderActiveGraph();
      };
      reader.onerror = () => alert("Could not read the file.");
      reader.readAsText(file);
    });
  }

  if (mapClearSources) {
    mapClearSources.addEventListener("click", () => {
      if (!sources.length) return;
      if (!window.confirm("Remove all cohorts from predefined graphs?")) return;
      sources = [];
      renderSourceList();
      renderActiveGraph();
    });
  }

  const resizeIfPlotly = (el) => {
    if (!el || typeof Plotly === "undefined") return;
    if (!el.querySelector || !el.querySelector(".js-plotly-plot")) return;
    try {
      Plotly.Plots.resize(el);
    } catch (_) {
      /* ignore */
    }
  };

  window.__analyzerResizePredefinedGraphs = () => {
    resizeIfPlotly(chartEl);
    resizeIfPlotly(idealGapEl);
  };
  window.__analyzerResizeCompressionMap = window.__analyzerResizePredefinedGraphs;

  renderSourceList();
  renderActiveGraph();
}
