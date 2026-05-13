export function toFloat(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

export const toBool = (v) => ["true", "1", "yes", "y"].includes(String(v).trim().toLowerCase());

/** When CSV has no fine_tuning_phase, infer compressed vs fine_tuned from experiment_name. */
function inferPhaseFromRow(row) {
  const name = String(row.experiment_name || "").trim();
  const n = name.toLowerCase();
  const m = String(row.method || "").trim();
  if (m === "None" && name === "Baseline") return "baseline";
  if (n.includes("[compressed]")) return "compressed";
  if (n.includes("[fine_tuned]")) return "fine_tuned";
  if (m !== "None" && name && !toBool(row.fine_tuning_enabled)) return "compressed";
  return "";
}

export function rankScalar(raw) {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s || s === "None" || s === "null") return null;
  if (s.includes("|")) {
    const n = Number(s.split("|")[0].trim());
    if (Number.isFinite(n)) return n;
  }
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

export function datasetRowFromRawCsvRow(row, runId) {
  return {
    run_id: runId,
    experiment_name: row.experiment_name || "",
    method: row.method || "",
    phase: row.fine_tuning_phase || inferPhaseFromRow(row) || row.phase || "legacy",
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
    model_memory_mb: toFloat(row.model_memory_mb),
    peak_inference_memory_mb: toFloat(row.peak_inference_memory_mb),
    test_eval_time_s: toFloat(row.test_eval_time_s),
    fine_tuning_time_s: toFloat(row.fine_tuning_time_s),
  };
}

export function isBaselineRow(r) {
  const name = (r.experiment_name || "").trim();
  return (
    r.phase === "baseline" ||
    (r.method === "None" && name === "Baseline") ||
    (r.phase === "legacy" && name === "Baseline")
  );
}

export function baselineMap(rows) {
  const map = new Map();
  rows.forEach((r) => {
    if (isBaselineRow(r)) map.set(r.run_id, r);
  });
  return map;
}

export function withDerived(rows, allRows) {
  const b = baselineMap(allRows);
  return rows.map((r) => {
    const base = b.get(r.run_id);
    const delta =
      base && r.accuracy !== null && base.accuracy !== null ? r.accuracy - base.accuracy : null;
    return { ...r, baseline_accuracy: base ? base.accuracy : null, delta_accuracy_vs_baseline: delta };
  });
}

export function getRowNumeric(r, key) {
  const v = r[key];
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return Number(v);
}

export function seriesLabel(row, key) {
  if (key === "fine_tuning_enabled") return row.fine_tuning_enabled ? "Fine-tuned" : "No FT";
  return String(row[key] ?? "-");
}

export function formatRowOneLine(r) {
  const rk =
    r.rank_raw !== undefined && String(r.rank_raw).trim() !== ""
      ? String(r.rank_raw)
      : String(r.rank_scalar ?? "—");
  const ft = r.fine_tuning_enabled ? "FT" : "no FT";
  return `${r.run_id} · ${r.method} · ${r.experiment_name} · ${r.phase} · r=${rk} · ${ft}`;
}
