import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.interpolate import PchipInterpolator
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

# Baseline metrics
baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_lat = float(baseline_row["latency_ms"].iloc[0]) if not baseline_row.empty else None

# Filter to just 'Without FT' since compression time and latency depend only on the rank, not on fine-tuning weights
df_m = df[(df["method"].notna()) & (df["method"] != "None") & (df["fine_tuning_enabled"] == False)].copy()

ALGORITHMS = ["SVD", "Tucker", "TT", "CP"]
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fontsize": 10,
})

def _smooth(x, y, n=250, log_y=False):
    if len(x) < 2: return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if len(x_u) < 2: return x_u, y_u
    x_new = np.linspace(x_u.min(), x_u.max(), n)
    if log_y:
        valid = y_u > 0
        if not np.any(valid): return x_new, np.zeros_like(x_new)
        y_interp = 10 ** PchipInterpolator(x_u[valid], np.log10(y_u[valid]))(x_new)
    else:
        y_interp = PchipInterpolator(x_u, y_u)(x_new)
    return x_new, y_interp

def _fmt_sec(v, _):
    if v <= 0: return ""
    if v >= 10: return f"{int(round(v))}s"
    if v >= 1: return f"{v:.1f}s"
    return f"{v:.2f}s"

fig = plt.figure(figsize=(9, 5.5))
# EXACT SQUARE FOR PANEL A: height = 5.5 * 0.73 = 4.015. width = 9 * 0.4461 = 4.015
ax_a = fig.add_axes([0.08, 0.15, 0.4461, 0.73])
ax_b = fig.add_axes([0.62, 0.15, 0.33, 0.73])

# --- (a) Compression time ---
for algo in ALGORITHMS:
    g = df_m[df_m["method"] == algo].sort_values("compression_ratio")
    if len(g) < 2: continue
    x_s, y_s = _smooth(g["compression_ratio"].values, g["compression_time_s"].values, log_y=True)
    ax_a.plot(x_s, y_s, color=ALGO_COLORS[algo], linewidth=2.5, label=algo)

ax_a.set_yscale("log")
ax_a.set_yticks([1, 5, 20, 100, 500])
ax_a.yaxis.set_major_formatter(FuncFormatter(_fmt_sec))
ax_a.minorticks_off()
ax_a.set_ylabel("Compression time (s)", fontsize=14)
ax_a.set_title("(a)", loc="center", fontsize=14, fontweight="bold", pad=12)
ax_a.grid(True, alpha=0.35, which="both")
ax_a.set_xlabel("Compression Ratio (x)", fontsize=14)
ax_a.set_xlim(0, 50)
ax_a.legend(loc="upper right", framealpha=0.95)

# --- (b) Inference Latency (Bar Chart with Mean and SD) ---
_g = df_m.groupby("method")["latency_ms"]
lat_mean = _g.mean().reindex(ALGORITHMS)
lat_std = _g.std().reindex(ALGORITHMS).fillna(0)

x = np.arange(len(ALGORITHMS))
bar_w = 0.5

ax_b.bar(
    x,
    lat_mean.values,
    bar_w,
    yerr=lat_std.values,
    capsize=5,
    error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "0.15", "zorder": 4},
    color=[ALGO_COLORS[algo] for algo in ALGORITHMS],
    edgecolor="black",
    linewidth=1.2,
    zorder=3,
)

if baseline_lat is not None:
    ax_b.axhline(baseline_lat, color="0.35", linestyle="-.", linewidth=1.5, label="Baseline", zorder=5)
    ax_b.legend(loc="upper right", framealpha=0.95)

ax_b.set_xticks(x)
ax_b.set_xticklabels(ALGORITHMS, fontsize=12, fontweight="bold")
ax_b.set_ylabel("Mean Inference Latency (ms)", fontsize=14)
ax_b.set_title("(b)", loc="center", fontsize=14, fontweight="bold", pad=12)
ax_b.grid(True, alpha=0.35, axis="y", zorder=0)

y_top = (lat_mean.values + lat_std.values).max()
ax_b.set_ylim(0, max(y_top * 1.15, baseline_lat * 1.5 if baseline_lat else 0))

# We don't need subplots_adjust because we used absolute add_axes coordinates

out_path = os.path.join(PLOT_DIR, "compute_vs_compression.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
