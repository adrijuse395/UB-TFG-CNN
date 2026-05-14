import { uniq } from "./format-utils.js";
import { parseCsvText } from "./csv.js";
import { datasetRowFromRawCsvRow, rawResultsCsvSampleHasRequiredColumns, MIN_RESULTS_CSV_COLUMNS } from "./rows.js";

/** Parse run_* folder names from typical directory index pages (http.server, nginx, etc.). */
export function extractRunIdsFromRunsDirectoryHtml(html) {
  const seen = new Set();
  const ids = [];
  const add = (raw) => {
    const id = String(raw || "")
      .replace(/\/+$/, "")
      .split("/")
      .pop()
      .trim();
    if (!/^run_\d{8}_\d{6}$/.test(id) || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  };
  const reHref = /href\s*=\s*["']([^"']*?)(run_\d{8}_\d{6})\/?["']/gi;
  let m;
  while ((m = reHref.exec(html)) !== null) add(m[2]);
  const reLoose = /(^|[\s"'>/])(run_\d{8}_\d{6})\/?(?=["'\s<]|$)/gm;
  while ((m = reLoose.exec(html)) !== null) add(m[2]);
  return uniq(ids);
}

export async function discoverRunsListing() {
  const origin = String(window.location.origin || "").replace(/\/$/, "");
  const page = window.location.href;
  const pageDir = new URL(".", page).href;
  const candidates = [
    new URL("../runs/", page).href,
    `${origin}/runs/`,
    new URL("./runs/", page).href,
    new URL("runs/", pageDir).href,
  ];
  const seen = new Set();
  for (let listUrl of candidates) {
    listUrl = listUrl.replace(/\/?$/, "/");
    if (seen.has(listUrl)) continue;
    seen.add(listUrl);
    try {
      const res = await fetch(`${listUrl}?cb=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) continue;
      const html = await res.text();
      const ids = extractRunIdsFromRunsDirectoryHtml(html);
      if (!ids.length) continue;
      return { base: listUrl, ids };
    } catch (_) {
      /* try next */
    }
  }
  return { base: null, ids: [] };
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
    const httpOk = http.map((x) => String(x).trim()).filter((x) => /^run_\d{8}_\d{6}$/.test(x));
    merged = uniq([...merged, ...httpOk]).sort();
  } catch (_) {
    /* offline or blocked fetch */
  }
  return merged;
}

export async function loadDataFromRuns() {
  const { base, ids: runIds } = await discoverRunsListing();
  if (!base || !runIds.length) {
    throw new Error(
      "No runs/ folder found. Start the server in the repo root (folder that contains runs/ and analyzer_web/), then open http://localhost:PORT/analyzer_web/index.html"
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

export async function loadDataFromJsonFallback() {
  const url = `./data/results.json?cb=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Cannot load live CSV or data/results.json (${res.status})`);
  const payload = await res.json();
  payload.meta = payload.meta || {};
  payload.meta.source = payload.meta.source || "results_json";
  return payload;
}

export async function loadData() {
  try {
    return await loadDataFromRuns();
  } catch (_) {
    return loadDataFromJsonFallback();
  }
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
