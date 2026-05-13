import { loadData, normalizePayload } from "./data-loader.js";
import { initExperimentPage } from "./run-page.js";

async function boot() {
  try {
    const payload = normalizePayload(await loadData());
    initExperimentPage(payload);
  } catch (e) {
    const el = document.getElementById("errorBox");
    if (el) el.textContent = `Error loading data: ${e.message}`;
  }
}

boot();
