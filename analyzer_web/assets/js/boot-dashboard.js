import { LIVE_REFRESH_MS } from "./constants.js";
import { loadData, normalizePayload } from "./data-loader.js";
import { initDashboard } from "./dashboard-main.js";

async function boot() {
  try {
    const payload = normalizePayload(await loadData());
    const signature = JSON.stringify({
      rows: payload.meta.rows,
      latest: payload.meta.latest_run,
      counts: payload.meta.row_count_by_run || {},
      source: payload.meta.source || "",
    });

    initDashboard(payload);

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
        /* keep current view on transient read errors */
      }
    }, LIVE_REFRESH_MS);
  } catch (e) {
    const el = document.getElementById("errorBox");
    if (el) el.textContent = `Error loading data: ${e.message}`;
  }
}

boot();
