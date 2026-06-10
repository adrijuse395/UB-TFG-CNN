import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

# Baseline
baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_params = float(baseline_row["total_parameters"].iloc[0]) / 1e6 if not baseline_row.empty else None
baseline_macs_m = float(baseline_row["macs_g"].iloc[0]) * 1000 if not baseline_row.empty else None

# Filter to compressed models (fine_tuning_enabled == False) since parameters/MACs depend only on model structure
df_models = df[df["method"].notna() & (df["method"] != "None") & (df["fine_tuning_enabled"] == False)].copy()
df_models["params_millions"] = df_models["total_parameters"] / 1e6
df_models["macs_millions"] = df_models["macs_g"] * 1000

ALGORITHMS = ["SVD", "Tucker", "TT", "CP"]
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#d4b200",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fontsize": 14,
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

fig, ax = plt.subplots(figsize=(9, 6.0))

# Plot Baseline point (matching TFG style dot)
if baseline_params is not None and baseline_macs_m is not None:
    ax.scatter(
        [baseline_params],
        [baseline_macs_m],
        color="0.35",
        s=64,
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
        label="Baseline",
    )

# Plot each algorithm curve
for algo in ALGORITHMS:
    subset = df_models[df_models["method"] == algo]
    if subset.empty:
        continue
    
    # Sort by parameters to get a clean sequential line
    subset = subset.sort_values("params_millions")
    x_arr = subset["params_millions"].values
    y_arr = subset["macs_millions"].values
    
    # Generate smoothed curve
    x_s, y_s = _smooth_curve(x_arr, y_arr)
    
    # Plot line only (no markers, matching other TFG figures)
    ax.plot(
        x_s, y_s, 
        color=ALGO_COLORS[algo], 
        linewidth=2.5, 
        label=algo,
        zorder=3
    )

ax.set_ylabel("Computation (MMACs)", fontsize=16)
ax.set_xlabel("Total Parameters (Millions)", fontsize=16)

ax.grid(True, alpha=0.35)
ax.legend(loc="lower right", framealpha=0.95)

# Adjust limits and ticks to integer ranges
max_x = max(df_models['params_millions'].max(), baseline_params if baseline_params else 0)
max_y = max(df_models['macs_millions'].max(), baseline_macs_m if baseline_macs_m else 0)

ax.set_xlim(0, max_x * 1.05)
ax.set_ylim(0, max_y * 1.05)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='upper'))

out_path = os.path.join(PLOT_DIR, "macs_vs_parameters.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
