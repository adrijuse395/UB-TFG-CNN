import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_params = float(baseline_row["total_parameters"].iloc[0]) / 1e6 if not baseline_row.empty else None
baseline_mem = float(baseline_row["peak_inference_memory_mb"].iloc[0]) if not baseline_row.empty else None
baseline_acc = float(baseline_row["accuracy"].iloc[0]) if not baseline_row.empty else None

df_models = df[df["method"].notna() & (df["method"] != "None")].copy()

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

def _smooth_curve(x: np.ndarray, y: np.ndarray, n: int = 250) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if len(x_unique) < 2:
        return x_unique, y_unique
    x_new = np.linspace(x_unique.min(), x_unique.max(), n)
    y_new = PchipInterpolator(x_unique, y_unique)(x_new)
    return x_new, y_new

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

for algo in ALGORITHMS:
    g_no_ft = df_models[(df_models["method"] == algo) & (df_models["fine_tuning_enabled"] == False)].sort_values("total_parameters", ascending=True)
    if g_no_ft.empty: continue
    
    # Filtro Pareto: Ignorar configuraciones subóptimas (outliers) donde al reducir parámetros se dispara la memoria.
    valid_rows = []
    current_min_mem = float('inf')
    for idx, row in g_no_ft.iloc[::-1].iterrows():
        if row["peak_inference_memory_mb"] < current_min_mem:
            valid_rows.append(row)
            current_min_mem = row["peak_inference_memory_mb"]
            
    g_filtered = pd.DataFrame(valid_rows[::-1])
    if g_filtered.empty: continue
        
    x_vals = g_filtered["peak_inference_memory_mb"].values
    
    # Panel a: Parameters
    y_params = g_filtered["total_parameters"].values / 1e6
    x_sa, y_sa = _smooth_curve(x_vals, y_params)
    ax_a.plot(x_sa, y_sa, color=ALGO_COLORS[algo], linestyle="-", linewidth=2.8, label=algo)
    
    # Panel b: Accuracy
    y_acc = g_filtered["accuracy"].values
    x_sb, y_sb = _smooth_curve(x_vals, y_acc)
    ax_b.plot(x_sb, y_sb, color=ALGO_COLORS[algo], linestyle="-", linewidth=2.8, label=algo)

# Baseline
if baseline_mem is not None:
    if baseline_params is not None:
        ax_a.axhline(y=baseline_params, color="0.35", linestyle="-.", linewidth=1.5, zorder=0)
        ax_a.scatter([baseline_mem], [baseline_params], color="0.35", s=100, zorder=5, edgecolors="black", linewidths=1.2, label="Baseline")
    
    if baseline_acc is not None:
        ax_b.axhline(y=baseline_acc, color="0.35", linestyle="-.", linewidth=1.5, zorder=0)
        ax_b.scatter([baseline_mem], [baseline_acc], color="0.35", s=100, zorder=5, edgecolors="black", linewidths=1.2, label="Baseline")

# Calculate dynamic memory limits
max_mem = max(df_models["peak_inference_memory_mb"].max(), baseline_mem if baseline_mem else 0)
min_mem = min(df_models["peak_inference_memory_mb"].min(), baseline_mem if baseline_mem else float('inf'))
if min_mem == float('inf'): min_mem = 0
mem_range = max_mem - min_mem
min_x = max(0, min_mem - 0.05 * mem_range)
max_x = max_mem + 0.05 * mem_range

# Formatting Panel A
ax_a.set_title("(a)", loc="center", fontsize=14, fontweight="bold", pad=12)
ax_a.set_ylabel("Total Parameters (Millions)", fontsize=14, fontweight="bold")
ax_a.set_xlabel("Peak Inference Memory (MB)", fontsize=14, fontweight="bold")
ax_a.set_xlim(min_x, max_x)
ax_a.set_ylim(bottom=0)
ax_a.grid(True, alpha=0.35)
ax_a.legend(loc="upper left", framealpha=0.95)

# Formatting Panel B
ax_b.set_title("(b)", loc="center", fontsize=14, fontweight="bold", pad=12)
ax_b.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
ax_b.set_xlabel("Peak Inference Memory (MB)", fontsize=14, fontweight="bold")
ax_b.set_xlim(min_x, max_x)
ax_b.set_ylim(0, 100)
ax_b.grid(True, alpha=0.35)
ax_b.legend(loc="lower right", framealpha=0.95)

fig.subplots_adjust(wspace=0.2, bottom=0.15, top=0.88)

out_path = os.path.join(PLOT_DIR, "memory_vs_compression.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
