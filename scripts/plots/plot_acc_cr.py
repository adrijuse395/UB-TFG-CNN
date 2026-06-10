import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator
from scipy.interpolate import PchipInterpolator
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

# Baseline
baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_acc = float(baseline_row["accuracy"].iloc[0]) if not baseline_row.empty else None
baseline_cr = float(baseline_row["compression_ratio"].iloc[0]) if not baseline_row.empty else 1.0

df_models = df[df["method"].notna() & (df["method"] != "None")].copy()
df_models["ft_label"] = df_models["fine_tuning_enabled"].apply(
    lambda x: "With FT" if str(x).lower() == "true" else "Without FT"
)
df_models = df_models.sort_values(by=["method", "ft_label", "compression_ratio"])

ALGORITHMS = ["Tucker", "TT", "CP"]
PANEL_LABELS = {"Tucker": "(a)", "TT": "(b)", "CP": "(c)"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.7)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#d4b200",
}
FT_LINESTYLES = {"With FT": "-", "Without FT": "-"}
FT_LINEWIDTHS = {"With FT": 1.2, "Without FT": 2.8}

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

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

# Determine global max for x-axis to align the plots (optional, but good for comparison)
# We will limit it to something reasonable if it spikes too high, or just let it autoscale per plot
MAX_CR_DISPLAY = df_models["compression_ratio"].quantile(0.98) # ignore extreme outliers if any

for ax, algo in zip(axes, ALGORITHMS):
    ax.set_title(PANEL_LABELS[algo], loc="center", fontsize=18, fontweight="bold", pad=8)
    subset = df_models[df_models["method"] == algo]
    
    for ft_lbl, group in subset.groupby("ft_label"):
        group = group.sort_values("compression_ratio")
        x_arr = group["compression_ratio"].values
        y_arr = group["accuracy"].values
        
        x_s, y_s = _smooth_curve(x_arr, y_arr)
        
        ax.plot(
            x_s, y_s, 
            color=ALGO_COLORS[algo], 
            linestyle=FT_LINESTYLES[ft_lbl], 
            linewidth=FT_LINEWIDTHS[ft_lbl], 
            label=ft_lbl
        )
        
    if baseline_acc is not None:
        ax.axhline(
            y=baseline_acc,
            color="0.35",
            linestyle="-.",
            linewidth=1.5,
            label="Baseline",
            zorder=0,
        )
    if baseline_acc is not None:
        ax.scatter(
            [baseline_cr],
            [baseline_acc],
            color="0.35",
            s=48,
            zorder=5,
            edgecolors="black",
            linewidths=0.6,
        )
        
    # Place legend in 'lower left' for (a) and (c), and 'center right' for (b) (Tucker & CP drop slower, TT drops very fast leaving center-right free)
    if algo == "TT":
        ax.legend(loc="center right", framealpha=0.95, fontsize=12.5)
    else:
        ax.legend(loc="lower left", framealpha=0.95, fontsize=12.5)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 100)
    # Limit compression ratio to 0-40 and force exact ticks
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    
    # Adjust alignment of edge labels (0 and 40) to keep them within subplot boundaries
    labels = ax.get_xticklabels()
    # Ensure they are populated before adjusting
    plt.draw()
    labels = ax.get_xticklabels()
    if len(labels) >= 5:
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')

axes[0].set_ylabel("Accuracy (%)", fontsize=18)

fig.subplots_adjust(wspace=0.08, bottom=0.15, top=0.88)
fig.supxlabel("Compression Ratio (x)", fontsize=18)

out_path = os.path.join(PLOT_DIR, "accuracy_vs_compression.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
