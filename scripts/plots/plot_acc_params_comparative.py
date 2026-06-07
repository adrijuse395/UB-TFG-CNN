import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

# Baseline
baseline_row = df[df["method"].isna() | (df["method"] == "None")]
baseline_acc = float(baseline_row["accuracy"].iloc[0]) if not baseline_row.empty else None
baseline_params = float(baseline_row["total_parameters"].iloc[0]) if not baseline_row.empty else None

df_models = df[df["method"].notna() & (df["method"] != "None")].copy()
df_models["ft_label"] = df_models["fine_tuning_enabled"].apply(
    lambda x: "With FT" if str(x).lower() == "true" else "Without FT"
)

ALGORITHMS = ["SVD", "Tucker", "TT", "CP"]
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
}
ALGO_LINESTYLES = {"SVD": "-", "Tucker": "-", "TT": "-", "CP": "-"}

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

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
panels = [
    ("Without FT", "(a)"),
    ("With FT", "(b)")
]

for ax, (ft_cond, title) in zip(axes, panels):
    ax.set_title(title, loc="center", fontsize=14, fontweight="bold", pad=12)
    subset_ft = df_models[df_models["ft_label"] == ft_cond]
    
    for algo in ALGORITHMS:
        group = subset_ft[subset_ft["method"] == algo].sort_values("total_parameters")
        if group.empty:
            continue
            
        x_arr = group["total_parameters"].values / 1e6
        y_arr = group["accuracy"].values
        
        x_s, y_s = _smooth_curve(x_arr, y_arr)
        
        ax.plot(
            x_s, y_s, 
            color=ALGO_COLORS[algo], 
            linestyle=ALGO_LINESTYLES[algo], 
            linewidth=2.5, 
            label=algo
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
    if baseline_acc is not None and baseline_params is not None:
        ax.scatter(
            [baseline_params / 1e6],
            [baseline_acc],
            color="0.35",
            s=64,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 100)
    max_params_m = max(df_models['total_parameters'].max() / 1e6, baseline_params / 1e6 if baseline_params else 0)
    ax.set_xlim(0, max_params_m * 1.05)
    # ax.xaxis.set_major_locator(MultipleLocator(2))

axes[0].set_ylabel("Accuracy (%)", fontsize=14)

fig.subplots_adjust(wspace=0.06, bottom=0.15, top=0.88)
fig.supxlabel("Total parameters (Millions)", fontsize=14)

out_path = os.path.join(PLOT_DIR, "accuracy_vs_parameters_by_ft.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
