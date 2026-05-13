import { AXIS_FIELDS } from "./constants.js";

export const fmt = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d);

export const uniq = (arr) =>
  [...new Set(arr.filter((v) => v !== null && v !== undefined && String(v) !== ""))].sort();

export function fieldLabel(key) {
  const f = AXIS_FIELDS.find((x) => x.key === key);
  return f ? f.label : key;
}
