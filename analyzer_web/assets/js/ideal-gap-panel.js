import { plotlyPurge } from "./plotly-charts.js";

/**
 * Placeholder for the “ideal frontier vs achieved envelope” method comparison.
 * Cohorts are the same as the trade-off map; ranking logic will land here next.
 */
export function renderIdealGapPanel(chartEl, rows) {
  if (!chartEl) return;
  plotlyPurge(chartEl);
  const n = rows.length;
  if (!n) {
    chartEl.innerHTML =
      '<p class="named-empty">Add one or more cohorts above — the same merged rows will feed this chart once it is implemented.</p>';
    return;
  }
  chartEl.innerHTML = `<div class="ideal-gap-stub" style="padding:14px 16px;border:1px dashed var(--border);border-radius:6px;background:var(--panel);">
    <p><strong>Under development.</strong> This view will draw a reference “ideal” curve in the compression–accuracy plane (for example, holding baseline accuracy across compression, anchored to your data conventions) and compare each method’s upper envelope against it.</p>
    <p class="cmp-hint">Planned metric: integrate the gap between the ideal frontier and each method’s best achievable frontier — smaller area ⇒ closer to the ideal. We will document normalization (axes, baselines, log vs linear compression) so scores are comparable across sweeps.</p>
    <p class="cmp-hint">${n} rows currently merged from your cohorts.</p>
  </div>`;
}
