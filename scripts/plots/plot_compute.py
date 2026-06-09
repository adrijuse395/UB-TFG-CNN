import numpy as np
import pandas as pd
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

# Get baseline latency
baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_lat = float(baseline_row["latency_ms"].iloc[0]) if not baseline_row.empty else None

# Filter decomposition rows where fine_tuning_enabled is False
df_m = df[(df["method"].notna()) & (df["method"] != "None") & (df["fine_tuning_enabled"] == False)].copy()

run_id = os.path.basename(os.path.abspath(RUN_DIR))

# We'll gather statistics for each algorithm
ALGORITHMS = ["SVD", "Tucker", "TT", "CP"]

stats = []
# Add baseline first if available
if baseline_lat is not None:
    stats.append({
        "Method": "Baseline",
        "Tensor_order": "N/A",
        "N": 1,
        "Mean": baseline_lat,
        "Std": 0.0,
        "Min": baseline_lat,
        "Max": baseline_lat
    })

for algo in ALGORITHMS:
    sub = df_m[df_m["method"] == algo]
    if sub.empty:
        continue
    latencies = sub["latency_ms"].dropna()
    if len(latencies) == 0:
        continue
    n = len(latencies)
    mean_val = latencies.mean()
    std_val = latencies.std()
    if pd.isna(std_val):
        std_val = 0.0
    min_val = latencies.min()
    max_val = latencies.max()
    stats.append({
        "Method": algo,
        "Tensor_order": "2D",
        "N": n,
        "Mean": mean_val,
        "Std": std_val,
        "Min": min_val,
        "Max": max_val
    })

# Define targets for progression matrix (max 5 columns)
target_crs = [2.5, 5.0, 10.0, 20.0, 40.0]

# Build matrices
matrix_comp_time = {algo: {} for algo in ALGORITHMS}
matrix_latency = {algo: {} for algo in ALGORITHMS}
matrix_actual_cr = {algo: {} for algo in ALGORITHMS}

for algo in ALGORITHMS:
    sub = df_m[df_m["method"] == algo]
    if sub.empty:
        continue
    for target in target_crs:
        # Find closest row by compression_ratio
        idx = (sub["compression_ratio"] - target).abs().idxmin()
        closest_row = sub.loc[idx]
        closest_cr = closest_row["compression_ratio"]
        # Only keep if it is reasonably close (e.g. within target * 0.5)
        if abs(closest_cr - target) <= target * 0.5:
            matrix_comp_time[algo][target] = closest_row["compression_time_s"]
            matrix_latency[algo][target] = closest_row["latency_ms"]
            matrix_actual_cr[algo][target] = closest_cr

# Generate output text
lines = []
lines.append("2D equivalence — Inference latency (ms)")
lines.append(run_id)
lines.append("Aggregated over all compression ratios (2D presets only).")
lines.append("")
lines.append("========================================================================")
lines.append("TABLE — Inference latency (ms)")
lines.append("========================================================================")
lines.append("Method\tTensor_order\tN\tMean\tStd\tMin\tMax")
for s in stats:
    lines.append(f"{s['Method']}\t{s['Tensor_order']}\t{s['N']}\t{s['Mean']:.4f}\t{s['Std']:.4f}\t{s['Min']:.4f}\t{s['Max']:.4f}")

lines.append("")
lines.append("LaTeX rows (Method & Tensor order & Mean & Std & N):")
for s in stats:
    lines.append(f"{s['Method']} & {s['Tensor_order']} & {s['Mean']:.4f} & {s['Std']:.4f} & {s['N']} \\\\")

lines.append("")
lines.append("========================================================================")
lines.append("Mean $\\pm$ Std  [copy-paste friendly]")
lines.append("========================================================================")
decomp_stats = [s for s in stats if s["Method"] != "Baseline"]
baseline_stat = next((s for s in stats if s["Method"] == "Baseline"), None)

for s in decomp_stats:
    lines.append(f"{s['Method']}\t{s['Mean']:.3f} $\\pm$ {s['Std']:.3f}")
if baseline_stat:
    lines.append(f"Baseline\t{baseline_stat['Mean']:.3f}")

# Add matrices sections
lines.append("")
lines.append("========================================================================")
lines.append("MATRIX — Compression Time (s) vs Target Compression Ratio")
lines.append("========================================================================")
header = "Method\t" + "\t".join(f"CR~{t:.1f}" for t in target_crs)
lines.append(header)
for algo in ALGORITHMS:
    row_strs = []
    for t in target_crs:
        val = matrix_comp_time[algo].get(t)
        row_strs.append(f"{val:.3f}" if val is not None else "N/A")
    lines.append(f"{algo}\t" + "\t".join(row_strs))

lines.append("")
lines.append("LaTeX rows (Compression Time):")
latex_header = "Method & " + " & ".join(f"CR~{t:.1f}" for t in target_crs) + " \\\\"
lines.append(latex_header)
for algo in ALGORITHMS:
    row_strs = []
    for t in target_crs:
        val = matrix_comp_time[algo].get(t)
        row_strs.append(f"{val:.3f}" if val is not None else "N/A")
    lines.append(f"{algo} & " + " & ".join(row_strs) + " \\\\")

lines.append("")
lines.append("========================================================================")
lines.append("MATRIX — Inference Latency (ms) vs Target Compression Ratio")
lines.append("========================================================================")
header_lat = "Method\t" + "\t".join(f"CR~{t:.1f}" for t in target_crs)
lines.append(header_lat)
for algo in ALGORITHMS:
    row_strs = []
    for t in target_crs:
        val = matrix_latency[algo].get(t)
        row_strs.append(f"{val:.3f}" if val is not None else "N/A")
    lines.append(f"{algo}\t" + "\t".join(row_strs))
if baseline_lat is not None:
    lines.append(f"Baseline\t" + "\t".join(f"{baseline_lat:.3f}" for _ in target_crs))

lines.append("")
lines.append("LaTeX rows (Inference Latency):")
lines.append(latex_header)
for algo in ALGORITHMS:
    row_strs = []
    for t in target_crs:
        val = matrix_latency[algo].get(t)
        row_strs.append(f"{val:.3f}" if val is not None else "N/A")
    lines.append(f"{algo} & " + " & ".join(row_strs) + " \\\\")
if baseline_lat is not None:
    lines.append(f"Baseline & " + " & ".join(f"{baseline_lat:.3f}" for _ in target_crs) + " \\\\")

# Add detailed mapping for user reference
lines.append("")
lines.append("========================================================================")
lines.append("MAPPING DETAILS — Actual Compression Ratio (x) used for each column")
lines.append("========================================================================")
lines.append(header)
for algo in ALGORITHMS:
    row_strs = []
    for t in target_crs:
        val = matrix_actual_cr[algo].get(t)
        row_strs.append(f"{val:.2f}" if val is not None else "N/A")
    lines.append(f"{algo}\t" + "\t".join(row_strs))

output_text = "\n".join(lines) + "\n"

out_path = os.path.join(PLOT_DIR, "compute_vs_compression.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Saved: {out_path}")
