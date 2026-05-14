import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _rank_scalar(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in {"", "None", "null"}:
        return None
    if "|" in s:
        try:
            return float(s.split("|", 1)[0].strip())
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        pass
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and parsed:
            return float(parsed[0])
    except Exception:
        return None
    return None


def collect_rows(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in sorted(runs_dir.glob("run_*/results.csv")):
        run_id = csv_path.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase = row.get("fine_tuning_phase", "") or ""
                if not phase:
                    name = (row.get("experiment_name") or "").strip()
                    nl = name.lower()
                    m = (row.get("method") or "").strip()
                    if m == "None" and name == "Baseline":
                        phase = "baseline"
                    elif "[compressed]" in nl:
                        phase = "compressed"
                    elif "[fine_tuned]" in nl:
                        phase = "fine_tuned"
                    elif m not in {"", "None"} and name and not _to_bool(
                        row.get("fine_tuning_enabled", False)
                    ):
                        phase = "compressed"
                    else:
                        phase = "legacy"
                rec: Dict[str, Any] = {
                    "run_id": run_id,
                    "experiment_name": row.get("experiment_name", ""),
                    "method": row.get("method", ""),
                    "phase": phase,
                    "fine_tuning_enabled": _to_bool(row.get("fine_tuning_enabled", False)),
                    "rank_raw": row.get("rank", ""),
                    "rank_scalar": _rank_scalar(row.get("rank")),
                    "accuracy": _to_float(row.get("accuracy")),
                    "compression_ratio": _to_float(row.get("compression_ratio")),
                    "latency_ms": _to_float(row.get("latency_ms")),
                    "throughput_fps": _to_float(row.get("throughput_fps")),
                    "macs_g": _to_float(row.get("macs_g")),
                    "total_parameters": _to_float(row.get("total_parameters")),
                    "compression_time_s": _to_float(row.get("compression_time_s")),
                    "fine_tuning_time_s": _to_float(row.get("fine_tuning_time_s")),
                }
                rows.append(rec)
    return rows


def _latest_run_id(runs_dir: Path, run_ids: List[str]) -> str:
    """Pick newest run folder by filesystem mtime (robust vs lexical ordering)."""
    if not run_ids:
        return ""
    best = ""
    best_mtime = -1.0
    for rid in run_ids:
        p = runs_dir / rid
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best_mtime:
            best_mtime = m
            best = rid
    return best or sorted(run_ids)[-1]


def build_payload(rows: List[Dict[str, Any]], runs_dir: Path) -> Dict[str, Any]:
    runs = sorted({r["run_id"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    phases = sorted({r["phase"] for r in rows})
    counts = Counter(r["run_id"] for r in rows)
    latest_run = _latest_run_id(runs_dir, runs)
    return {
        "meta": {
            "rows": len(rows),
            "runs": runs,
            "methods": methods,
            "phases": phases,
            "latest_run": latest_run,
            "row_count_by_run": {k: counts[k] for k in sorted(counts.keys())},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON dataset for analyzer_web")
    parser.add_argument("--runs-dir", default="runs", help="Directory with run_*/results.csv")
    parser.add_argument(
        "--output",
        default="runs_analysis_snapshot.json",
        help="Output JSON path (not read by the analyzer; for exports only)",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs_dir)
    rows = collect_rows(runs_path)
    payload = build_payload(rows, runs_path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
