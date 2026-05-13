import { AXIS_FIELDS } from "./constants.js";

export function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

export function safeSetHtml(el, html) {
  if (el) el.innerHTML = html;
}

export function setOptions(select, values, includeAll = true) {
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
    const s = String(v);
    o.value = s;
    o.textContent = s;
    select.appendChild(o);
  });
}

export function setAxisOptions(select) {
  if (!select) return;
  select.innerHTML = "";
  AXIS_FIELDS.forEach((f) => {
    const o = document.createElement("option");
    o.value = f.key;
    o.textContent = f.label;
    select.appendChild(o);
  });
}

export function updateDashboardFiltersForActiveTab() {
  const bar = document.getElementById("dashboardFiltersBar");
  if (!bar) return;
  const active = document.querySelector(".tab.active");
  const id = active && active.dataset ? active.dataset.tab : "data";
  bar.classList.toggle("hidden", id === "compare" || id === "predefined-graphs");
}
