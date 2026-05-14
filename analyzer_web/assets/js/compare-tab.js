import { SERIES_OPTIONS } from "./constants.js";
import { uniq, fieldLabel } from "./format-utils.js";
import { setOptions, setAxisOptions, escapeHtml, safeSetHtml } from "./dom-utils.js";
import {
  getRowNumeric,
  seriesLabel,
  isBaselineRow,
  withDerived,
  datasetRowFromRawCsvRow,
  rawResultsCsvSampleHasRequiredColumns,
  MIN_RESULTS_CSV_COLUMNS,
} from "./rows.js";
import { parseCsvText } from "./csv.js";
import { listRunIdsFromHttpDirectory } from "./data-loader.js";
import {
  loadComparePersisted,
  saveComparePersisted,
  buildCompareVegaLiteSpec,
  newCompareSourceId,
  collectCompareRunOptions,
} from "./compare-vega.js";

export function initCompareTab(DATA_ALL, meta) {
  const cmpRunPick = document.getElementById("cmpRunPick");
  const updateCmpRunStatus = () => {};

  let sources = [];

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

  const cohortSignature = (runId, includeBaseline, fm, fp, fe) =>
    `${runId}\t${includeBaseline}\t${fm}\t${fp}\t${fe}`;

  const cmpMethodPick = document.getElementById("cmpMethodPick");
  const cmpPhasePick = document.getElementById("cmpPhasePick");
  const cmpExperimentPick = document.getElementById("cmpExperimentPick");

  const refillCompareCohortFilters = () => {
    if (!cmpMethodPick || !cmpPhasePick || !cmpExperimentPick || !cmpRunPick) return;
    const runId = cmpRunPick.value;
    const preserve = (sel) => sel.value;
    const curM = preserve(cmpMethodPick);
    const curP = preserve(cmpPhasePick);
    const curE = preserve(cmpExperimentPick);
    if (!runId) {
      setOptions(cmpMethodPick, [], true);
      setOptions(cmpPhasePick, [], true);
      setOptions(cmpExperimentPick, [], true);
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
    setOptions(cmpMethodPick, methods, true);
    setOptions(cmpPhasePick, phases, true);
    setOptions(cmpExperimentPick, experiments, true);
    if ([...cmpMethodPick.options].some((o) => o.value === curM)) cmpMethodPick.value = curM;
    if ([...cmpPhasePick.options].some((o) => o.value === curP)) cmpPhasePick.value = curP;
    if ([...cmpExperimentPick.options].some((o) => o.value === curE)) cmpExperimentPick.value = curE;
  };

  const rebuildFromStorage = () => {
    const persisted = loadComparePersisted();
    const next = [];
    persisted.forEach((p) => {
      if (p.kind !== "run" || !p.run_id) return;
      const fm = p.method || "__all__";
      const fp = p.phase || "__all__";
      const fe = p.experiment_name || "__all__";
      const rows = hydrateRunCohortRows(p.run_id, p.includeBaseline !== false, fm, fp, fe);
      if (!rows.length) return;
      next.push({
        id: newCompareSourceId(),
        kind: "run",
        run_id: p.run_id,
        label: (p.label && String(p.label).trim()) || p.run_id,
        includeBaseline: p.includeBaseline !== false,
        filterMethod: fm,
        filterPhase: fp,
        filterExperiment: fe,
        rows,
      });
    });
    sources = next;
  };

  rebuildFromStorage();

  if (cmpRunPick) {
    let runOptions = collectCompareRunOptions(meta, DATA_ALL);
    setOptions(cmpRunPick, runOptions, false);
    if (!runOptions.length) {
      listRunIdsFromHttpDirectory()
        .then((ids) => {
          const el = document.getElementById("cmpRunPick");
          if (!el || !ids.length) {
            updateCmpRunStatus();
            refillCompareCohortFilters();
            return;
          }
          const merged = uniq([...ids, ...collectCompareRunOptions(meta, DATA_ALL)]).filter(
            (x) => x !== undefined && x !== null && String(x).trim() !== ""
          ).sort();
          if (merged.length) setOptions(el, merged, false);
          updateCmpRunStatus();
          refillCompareCohortFilters();
        })
        .catch(() => {
          updateCmpRunStatus();
          refillCompareCohortFilters();
        });
    }
    if (!runOptions.length) {
      const o = document.createElement("option");
      o.value = "";
      o.disabled = true;
      o.selected = true;
      o.textContent = "No runs found — open site from project root (see README)";
      cmpRunPick.appendChild(o);
    }
    updateCmpRunStatus();
    refillCompareCohortFilters();
  }

  window.__analyzerRefreshCompareRuns = () => {
    const el = document.getElementById("cmpRunPick");
    if (!el) return;
    const ro = collectCompareRunOptions(meta, DATA_ALL);
    setOptions(el, ro, false);
    if (!ro.length) {
      listRunIdsFromHttpDirectory()
        .then((ids) => {
          const sel = document.getElementById("cmpRunPick");
          if (!sel || !ids.length) {
            updateCmpRunStatus();
            refillCompareCohortFilters();
            return;
          }
          const merged = uniq([...ids, ...collectCompareRunOptions(meta, DATA_ALL)]).filter(
            (x) => x !== undefined && x !== null && String(x).trim() !== ""
          ).sort();
          if (merged.length) setOptions(sel, merged, false);
          updateCmpRunStatus();
          refillCompareCohortFilters();
        })
        .catch(() => {
          updateCmpRunStatus();
          refillCompareCohortFilters();
        });
    } else {
      updateCmpRunStatus();
      refillCompareCohortFilters();
    }
  };

  const cmpRunLabel = document.getElementById("cmpRunLabel");
  const cmpIncludeBaselineRuns = document.getElementById("cmpIncludeBaselineRuns");
  const cmpAddRun = document.getElementById("cmpAddRun");
  const cmpFile = document.getElementById("cmpFile");
  const cmpFileLabel = document.getElementById("cmpFileLabel");
  const cmpAddFile = document.getElementById("cmpAddFile");
  const cmpSourceList = document.getElementById("cmpSourceList");
  const cmpSourceCount = document.getElementById("cmpSourceCount");
  const cmpSourceEmpty = document.getElementById("cmpSourceEmpty");
  const cmpClearSources = document.getElementById("cmpClearSources");

  const cX = document.getElementById("cmpChartXField");
  const cY = document.getElementById("cmpChartYField");
  const cViz = document.getElementById("cmpVizStyle");
  const cPoint = document.getElementById("cmpPointSize");
  const cTheme = document.getElementById("cmpVegaTheme");
  const cSeries = document.getElementById("cmpChartSeriesField");
  const cTitle = document.getElementById("cmpChartTitleInput");
  const cXLab = document.getElementById("cmpChartXLabelInput");
  const cYLab = document.getElementById("cmpChartYLabelInput");
  const cmpChartShell = document.getElementById("cmpChartShell");

  const chartOk = !!(
    cmpChartShell &&
    cX &&
    cY &&
    cSeries &&
    cViz &&
    cPoint &&
    cTheme
  );
  if (!chartOk) {
    const err = document.getElementById("errorBox");
    if (err) {
      err.textContent =
        "Compare chart controls are missing from the page (stale cache?). Hard-refresh with Ctrl+Shift+R.";
    }
    if (cmpChartShell) {
      safeSetHtml(
        cmpChartShell,
        '<p class="named-empty">Chart panel unavailable — refresh the page. You can still use &quot;From loaded runs&quot; if runs appear in the list.</p>'
      );
    }
  }

  const prepareCmpChartMount = () => {
    if (!cmpChartShell) return null;
    cmpChartShell.innerHTML = '<div id="cmpChart"></div>';
    return document.getElementById("cmpChart");
  };

  let cmpVegaApi = null;
  let cmpVegaView = null;

  const finalizeCmpChart = async () => {
    if (cmpVegaApi) {
      try {
        await cmpVegaApi.finalize();
      } catch (_) {
        /* ignore */
      }
      cmpVegaApi = null;
      cmpVegaView = null;
    }
  };

  const onResizeCmp = () => {
    if (!cmpVegaView) return;
    try {
      cmpVegaView.resize().run();
    } catch (_) {
      /* ignore */
    }
  };
  if (chartOk) {
    window.addEventListener("resize", onResizeCmp);
  }

  if (chartOk) {
    setAxisOptions(cX);
    setAxisOptions(cY);
    cX.value = "rank_scalar";
    cY.value = "accuracy";
    cSeries.innerHTML = "";
    SERIES_OPTIONS.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.key;
      o.textContent = s.label;
      cSeries.appendChild(o);
    });
    cSeries.value = "method";
  }

  const persistRuns = () => {
    saveComparePersisted(sources);
  };

  const mergeRowsForPlot = () => {
    const out = [];
    sources.forEach((src) => {
      src.rows.forEach((r) => {
        out.push({ ...r, _cmpSourceLabel: src.label });
      });
    });
    return out;
  };

  const renderSourceList = () => {
    if (cmpSourceCount) cmpSourceCount.textContent = String(sources.length);
    if (cmpSourceEmpty) cmpSourceEmpty.classList.toggle("hidden", sources.length > 0);
    if (!cmpSourceList) return;
    safeSetHtml(
      cmpSourceList,
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
        <button type="button" class="btn-remove" data-remove-src="${escapeHtml(s.id)}">Remove</button>
      </li>`;
        })
        .join("")
    );
    cmpSourceList.querySelectorAll("[data-remove-src]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-remove-src");
        sources = sources.filter((x) => x.id !== id);
        persistRuns();
        renderSourceList();
        renderCmpChart();
      });
    });
  };

  const renderCmpChart = () => {
    if (!chartOk) return;
    renderCmpChartAsync().catch((err) => {
      console.warn("Compare chart:", err);
      safeSetHtml(
        cmpChartShell,
        `<p class="named-empty">Could not draw chart: ${escapeHtml(err && err.message ? err.message : String(err))}</p>`
      );
    });
  };

  async function renderCmpChartAsync() {
    if (!chartOk) return;
    await finalizeCmpChart();

    if (typeof vegaEmbed !== "function") {
      safeSetHtml(
        cmpChartShell,
        '<p class="named-empty">Vega-Lite did not load — check the network or disable script blocking.</p>'
      );
      return;
    }

    const merged = mergeRowsForPlot();
    const multi = sources.length > 1;
    const xKey = cX.value;
    const yKey = cY.value;
    const sKey = cSeries.value;
    const xAxisTitle = (cXLab && cXLab.value.trim()) || fieldLabel(xKey);
    const yAxisTitle = (cYLab && cYLab.value.trim()) || fieldLabel(yKey);
    const chartTitle =
      (cTitle && cTitle.value.trim()) || `${yAxisTitle} vs ${xAxisTitle}${sources.length ? " — compare" : ""}`;

    if (!sources.length) {
      safeSetHtml(
        cmpChartShell,
        '<p class="named-empty">Add at least one run or CSV — every row from each dataset is plotted together.</p>'
      );
      return;
    }

    const baseKeyForCompareSeries = (r) => {
      const lbl = seriesLabel(r, sKey);
      return multi ? `${r._cmpSourceLabel}\t${lbl}` : lbl;
    };

    const rowsForPlot = [];
    merged.forEach((r) => {
      const x = getRowNumeric(r, xKey);
      const y = getRowNumeric(r, yKey);
      if (x === null || y === null) return;
      rowsForPlot.push({ r, x, y });
    });

    const ftSets = new Map();
    rowsForPlot.forEach(({ r }) => {
      const k = baseKeyForCompareSeries(r);
      if (!ftSets.has(k)) ftSets.set(k, new Set());
      ftSets.get(k).add(r.fine_tuning_enabled ? 1 : 0);
    });
    const needsFtSplit = new Set();
    ftSets.forEach((set, k) => {
      if (set.size > 1) needsFtSplit.add(k);
    });

    const methodAndFtCore = (r) => {
      const algo = seriesLabel(r, "method");
      const ft =
        needsFtSplit.has(baseKeyForCompareSeries(r)) ? (r.fine_tuning_enabled ? " - FT" : " - no FT") : "";
      return `${algo}${ft}`;
    };

    const sourcesByCore = new Map();
    rowsForPlot.forEach(({ r }) => {
      const core = methodAndFtCore(r);
      if (!sourcesByCore.has(core)) sourcesByCore.set(core, new Set());
      sourcesByCore.get(core).add(r._cmpSourceLabel || "");
    });
    const coresNeedingDatasetPrefix = new Set();
    sourcesByCore.forEach((srcSet, core) => {
      if (multi && srcSet.size > 1) coresNeedingDatasetPrefix.add(core);
    });

    const values = rowsForPlot
      .map(({ r, x, y }) => {
        const core = methodAndFtCore(r);
        const seriesStr = coresNeedingDatasetPrefix.has(core)
          ? `${r._cmpSourceLabel} — ${core}`
          : core;
        return {
          x,
          y,
          series: seriesStr,
          experiment: r.experiment_name || "",
          method: r.method || "",
          phase: r.phase || "",
        };
      });

    values.sort((a, b) => {
      const cmpSeries = String(a.series).localeCompare(String(b.series));
      if (cmpSeries !== 0) return cmpSeries;
      if (a.x !== b.x) return a.x - b.x;
      return a.y - b.y;
    });

    if (!values.length) {
      safeSetHtml(
        cmpChartShell,
        '<p class="named-empty">No numeric X/Y pairs in the combined data — try other axes or include baseline rows.</p>'
      );
      return;
    }

    const spec = buildCompareVegaLiteSpec({
      values,
      xTitle: xAxisTitle,
      yTitle: yAxisTitle,
      chartTitle,
      vizStyle: cViz.value,
      pointSize: cPoint.value,
      themeKey: cTheme.value,
    });

    const mount = prepareCmpChartMount();
    if (!mount) {
      safeSetHtml(cmpChartShell, '<p class="named-empty">Chart mount missing.</p>');
      return;
    }

    cmpVegaApi = await vegaEmbed(mount, spec, {
      actions: { export: true, compiled: false, source: false, editor: false },
      renderer: "svg",
    });
    cmpVegaView = cmpVegaApi.view;
    cmpVegaView.resize().run();
  }

  if (cmpAddRun && cmpRunPick) {
    cmpRunPick.addEventListener("change", refillCompareCohortFilters);
    cmpAddRun.addEventListener("click", () => {
      const runId = cmpRunPick.value;
      if (!runId) {
        alert("Select a run.");
        return;
      }
      const fm = cmpMethodPick ? cmpMethodPick.value : "__all__";
      const fp = cmpPhasePick ? cmpPhasePick.value : "__all__";
      const fe = cmpExperimentPick ? cmpExperimentPick.value : "__all__";
      const includeBaseline = cmpIncludeBaselineRuns ? cmpIncludeBaselineRuns.checked : true;
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
        alert("This cohort (same run and Method / Phase / Experiment selection) is already in the comparison.");
        return;
      }
      const label = (cmpRunLabel && cmpRunLabel.value.trim()) || runId;
      const rows = hydrateRunCohortRows(runId, includeBaseline, fm, fp, fe);
      if (!rows.length) {
        alert("No rows match that run and filter selection in the loaded dataset.");
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
      persistRuns();
      if (cmpRunLabel) cmpRunLabel.value = "";
      renderSourceList();
      renderCmpChart();
    });
  }

  if (cmpAddFile && cmpFile) {
    cmpAddFile.addEventListener("click", () => {
      if (!cmpFile.files || !cmpFile.files[0]) {
        alert("Choose a results.csv file first.");
        return;
      }
      const file = cmpFile.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        const rawRows = parseCsvText(text);
        if (!rawRows.length) {
          alert("No data rows in that CSV.");
          return;
        }
        if (!rawResultsCsvSampleHasRequiredColumns(rawRows[0])) {
          alert(
            `That CSV must include these columns (from the header row): ${MIN_RESULTS_CSV_COLUMNS.join(", ")}.`
          );
          return;
        }
        const uploadRunId = `upload_${newCompareSourceId().replace(/-/g, "")}`;
        const label =
          (cmpFileLabel && cmpFileLabel.value.trim()) || file.name.replace(/\.csv$/i, "") || uploadRunId;
        const mapped = rawRows.map((row) => datasetRowFromRawCsvRow(row, uploadRunId));
        const rows = withDerived(mapped, mapped);
        sources.push({
          id: newCompareSourceId(),
          kind: "file",
          label,
          rows,
        });
        cmpFile.value = "";
        if (cmpFileLabel) cmpFileLabel.value = "";
        renderSourceList();
        renderCmpChart();
      };
      reader.onerror = () => alert("Could not read the file.");
      reader.readAsText(file);
    });
  }

  if (cmpClearSources) {
    cmpClearSources.addEventListener("click", () => {
      if (!sources.length) return;
      if (!window.confirm("Remove all datasets from the comparison?")) return;
      sources = [];
      persistRuns();
      renderSourceList();
      renderCmpChart();
    });
  }

  if (chartOk) {
    [cX, cY, cViz, cPoint, cTheme, cSeries, cTitle, cXLab, cYLab]
      .filter(Boolean)
      .forEach((el) => {
        el.addEventListener("input", renderCmpChart);
        el.addEventListener("change", renderCmpChart);
      });
  }

  renderSourceList();
  if (chartOk) renderCmpChart();
}
