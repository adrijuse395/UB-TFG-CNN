import { uniq } from "./format-utils.js";
import { parseCsvText } from "./csv.js";
import { datasetRowFromRawCsvRow, rawResultsCsvSampleHasRequiredColumns, MIN_RESULTS_CSV_COLUMNS } from "./rows.js";

/**
 * Names of folders under runs/ that hold results.csv.
 * Official logger uses run_YYYYMMDD_HHMMSS; other dirs (e.g. run_tt_1) are allowed if they match this pattern.
 */
export function isRunDirectoryName(name) {
  const s = String(name || "").trim();
  return /^run_[A-Za-z0-9_.-]+$/.test(s) && s.length > 4;
}

const RUN_DIR_RE = /\brun_[A-Za-z0-9_.-]+\b/g;

/**
 * Parse run_* directory names from directory index HTML (http.server, nginx autoindex, etc.).
 */
export function extractRunIdsFromRunsDirectoryHtml(html) {
  const seen = new Set();
  const ids = [];
  const add = (raw) => {
    const id = String(raw || "")
      .replace(/\/+$/, "")
      .split("/")
      .pop()
      .trim();
    if (!isRunDirectoryName(id) || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  };
  const reHref = /href\s*=\s*["']([^"']*?)(run_[A-Za-z0-9_.-]+)\/?["']/gi;
  let m;
  while ((m = reHref.exec(html)) !== null) add(m[2]);
  const reLoose = /(^|[\s"'>/])(run_[A-Za-z0-9_.-]+)\/?(?=["'\s<]|$)/gm;
  while ((m = reLoose.exec(html)) !== null) add(m[2]);
  let m2;
  RUN_DIR_RE.lastIndex = 0;
  while ((m2 = RUN_DIR_RE.exec(html)) !== null) add(m2[0]);
  return uniq(ids);
}

/**
 * Absolute bases for runs/ (trailing slash). Primary: URL relative to this module —
 * analyzer_web/assets/js/data-loader.js → ../../../runs/ = repo root runs/.
 * Fallbacks: page URL and /runs/ on the same origin.
 */
export function getRunsDirectoryBaseCandidates() {
  const out = [];
  const add = (href) => {
    if (!href) return;
    let s = String(href).trim();
    if (!s) return;
    s = s.replace(/\/?$/, "/");
    if (!out.includes(s)) out.push(s);
  };
  try {
    add(new URL("../../../runs/", import.meta.url).href);
  } catch (_) {
    /* ignore */
  }
  try {
    add(new URL("../runs/", window.location.href).href);
  } catch (_) {
    /* ignore */
  }
  const origin = String(window.location.origin || "").replace(/\/$/, "");
  if (origin) add(`${origin}/runs/`);
  try {
    add(new URL("./runs/", window.location.href).href);
  } catch (_) {
    /* ignore */
  }
  try {
    add(new URL("runs/", new URL(".", window.location.href)).href);
  } catch (_) {
    /* ignore */
  }
  return out;
}

export async function discoverRunsListing() {
  const candidates = getRunsDirectoryBaseCandidates();
  const tried = [];
  for (let listUrl of candidates) {
    listUrl = listUrl.replace(/\/?$/, "/");
    tried.push(listUrl);
    try {
      const res = await fetch(`${listUrl}?cb=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) continue;
      const html = await res.text();
      const ids = extractRunIdsFromRunsDirectoryHtml(html);
      if (!ids.length) continue;
      return { base: listUrl, ids, tried };
    } catch (_) {
      /* try next */
    }
  }
  return { base: null, ids: [], tried };
}

export async function listRunIdsFromHttpDirectory() {
  const { ids } = await discoverRunsListing();
  return ids;
}

/**
 * Run ids for UI pickers: meta + row run_id + directory listing (same bases as live CSV load).
 */
export async function mergeRunIdsForPicker(meta, rows) {
  const fromRows = uniq(
    rows
      .map((r) => r.run_id)
      .filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
      .map((x) => String(x).trim())
  );
  const fromMeta = Array.isArray(meta?.runs)
    ? meta.runs
        .filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
        .map((x) => String(x).trim())
    : [];
  let merged = uniq([...fromMeta, ...fromRows]).sort();
  try {
    const http = await listRunIdsFromHttpDirectory();
    const httpOk = http.map((x) => String(x).trim()).filter((x) => isRunDirectoryName(x));
    merged = uniq([...merged, ...httpOk]).sort();
  } catch (_) {
    /* offline or blocked fetch */
  }
  return merged;
}

export async function loadDataFromRuns() {
  const { base, ids: runIds, tried } = await discoverRunsListing();
  if (!base || !runIds.length) {
    const hint = tried.length ? ` Tried:\n${tried.map((u) => `  • ${u}`).join("\n")}` : "";
    throw new Error(
      "Could not list runs/ or find any run_* result folders (name pattern: run_ plus letters, digits, _, . or -). " +
        "Serve the **repository root** (where both runs/ and analyzer_web/ live), e.g. " +
        "`python -m http.server 8000` from the project root, then open " +
        "http://localhost:8000/analyzer_web/index.html" +
        hint
    );
  }
  try {
    window.__analyzerRunsBaseUrl = base;
  } catch (_) {
    /* ignore */
  }
  const allRows = [];
  const rowCountByRun = {};
  for (const runId of runIds) {
    try {
      const csvUrl = new URL(`${runId}/results.csv?cb=${Date.now()}`, base);
      const res = await fetch(csvUrl.href, { cache: "no-store" });
      if (!res.ok) continue;
      const rawRows = parseCsvText(await res.text());
      if (!rawRows.length) continue;
      if (!rawResultsCsvSampleHasRequiredColumns(rawRows[0])) {
        console.warn(
          `[analyzer] Skipping run ${runId}: results.csv missing required columns (${MIN_RESULTS_CSV_COLUMNS.join(
            ", "
          )}).`
        );
        continue;
      }
      rowCountByRun[runId] = rawRows.length;
      rawRows.forEach((row) => {
        allRows.push(datasetRowFromRawCsvRow(row, runId));
      });
    } catch (err) {
      console.warn(`[analyzer] Skipping run ${runId}:`, err);
    }
  }
  if (!allRows.length && runIds.length) {
    throw new Error(
      "Found run_* folders under runs/ but loaded zero rows. " +
        "Check each run's results.csv (required columns: experiment_name, method) and the browser console for [analyzer] warnings."
    );
  }
  return {
    meta: {
      rows: allRows.length,
      runs: Object.keys(rowCountByRun).length ? Object.keys(rowCountByRun).sort() : [...runIds].sort(),
      runs_fetch_base: base,
      methods: uniq(allRows.map((r) => r.method)),
      phases: uniq(allRows.map((r) => r.phase)),
      latest_run: Object.keys(rowCountByRun).length
        ? [...Object.keys(rowCountByRun)].sort().at(-1)
        : runIds.length
          ? [...runIds].sort().at(-1)
          : "",
      row_count_by_run: rowCountByRun,
      generated_at: new Date().toISOString(),
      source: "live_csv",
    },
    rows: allRows,
  };
}

// Loads live CSV from runs/run_*/results.csv over HTTP (see README). No bundled snapshot fallback.
export async function loadData() {
  return loadDataFromRuns();
}

export function normalizePayload(raw) {
  const rows = Array.isArray(raw?.rows) ? raw.rows : [];
  const meta = raw?.meta && typeof raw.meta === "object" ? { ...raw.meta } : {};
  const methodsU = uniq(rows.map((r) => r.method).filter((x) => x !== undefined && x !== ""));
  const phasesU = uniq(rows.map((r) => r.phase).filter((x) => x !== undefined && x !== ""));
  const runsU = uniq(
    rows.map((r) => r.run_id).filter((x) => x !== undefined && x !== null && String(x).trim() !== "")
  );
  const runsFromMeta = Array.isArray(meta.runs) ? meta.runs.filter((x) => x !== undefined && x !== null && String(x).trim() !== "") : [];
  meta.runs = uniq([...runsFromMeta, ...runsU]).sort();
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

export function pickDefaultRun(meta) {
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
