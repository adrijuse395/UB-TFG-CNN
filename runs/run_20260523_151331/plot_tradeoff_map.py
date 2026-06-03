import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

def _smooth_curve(x, y, n=250):
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

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(RUN_DIR, "results.csv")

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Drop rows without required metrics
df_valid = df.dropna(subset=["compression_ratio", "accuracy"]).copy()

# Baseline max accuracy
baseline = df_valid[df_valid["method"].isna() | (df_valid["method"] == "None")]
bmax = baseline["accuracy"].max() if not baseline.empty else np.nan

df_models = df_valid[df_valid["method"].notna() & (df_valid["method"] != "None")].copy()

if df_models.empty:
    print("No valid model points to plot.")
    exit(0)

# Configuration for zone separating lines
# Options: "mean", "median", "baseline_pct", or a numeric value for a hardcoded split
VSPLIT_MODE = 'median'       # e.g., "mean", "median", "baseline_pct", or 5.0
HSPLIT_MODE = 88.0                 # e.g., "mean", "median", "baseline_pct", or 90.0

# Calculate vSplit
if isinstance(VSPLIT_MODE, (float, int)):
    vSplit = float(VSPLIT_MODE)
elif VSPLIT_MODE == "median":
    vSplit = df_models["compression_ratio"].median()
elif VSPLIT_MODE == "baseline_pct" and not baseline.empty:
    # 97% of baseline compression ratio (which is usually 1.0)
    vSplit = 0.97 * baseline["compression_ratio"].max()
else:  # "mean" or default
    vSplit = df_models["compression_ratio"].mean()

# Calculate hSplit
if isinstance(HSPLIT_MODE, (float, int)):
    hSplit = float(HSPLIT_MODE)
elif HSPLIT_MODE == "median":
    hSplit = df_models["accuracy"].median()
elif HSPLIT_MODE == "baseline_pct" and pd.notna(bmax) and bmax > 0:
    hSplit = 0.97 * bmax
else:  # "mean" or default
    hSplit = df_models["accuracy"].mean()

x_min, x_max = df_models["compression_ratio"].min(), df_models["compression_ratio"].max()
y_min, y_max = df_models["accuracy"].min(), df_models["accuracy"].max()

# The user wants to limit the graph by compression range 40
x_max = 40.0
x_min = 0.0

x_pad = 0.0
y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.5

x0, x1 = x_min, x_max
y0, y1 = y_min - y_pad, y_max + y_pad

sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(11, 7.5))

# Shaded Quadrants
# Top-Right: Ideal (blue)
ax.add_patch(Rectangle((vSplit, hSplit), x1 - vSplit, y1 - hSplit, color="#bbdefb", alpha=0.42, zorder=0, label="Zone: Ideal"))
# Top-Left: Accuracy-first (green)
ax.add_patch(Rectangle((x0, hSplit), vSplit - x0, y1 - hSplit, color="#c8e6c9", alpha=0.38, zorder=0, label="Zone: Accuracy-first"))
# Bottom-Right: Compression-first (orange)
ax.add_patch(Rectangle((vSplit, y0), x1 - vSplit, hSplit - y0, color="#ffe0b2", alpha=0.45, zorder=0, label="Zone: Compression-first"))
# Bottom-Left: Weak (grey)
ax.add_patch(Rectangle((x0, y0), vSplit - x0, hSplit - y0, color="#e0e0e0", alpha=0.35, zorder=0, label="Zone: Weak"))

# Split lines
ax.axvline(vSplit, color="black", linestyle="-", linewidth=1.5, alpha=0.55, zorder=1)
ax.axhline(hSplit, color="black", linestyle="-", linewidth=1.5, alpha=0.55, zorder=1)

# Annotations moved to legend

# Plot Points
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
}

methods = df_models["method"].unique()
for method in methods:
    group = df_models[df_models["method"] == method]
    color = ALGO_COLORS.get(method, "grey")
    
    # Fine-tuned
    ft_group = group[group["fine_tuning_enabled"] == True]
    if not ft_group.empty:
        x_s, y_s = _smooth_curve(ft_group["compression_ratio"].values, ft_group["accuracy"].values)
        ax.plot(x_s, y_s, color=color, linewidth=2.0, linestyle="--", label=f"{method} (FT)", zorder=3)
        
    # No Fine-tuning
    no_ft_group = group[group["fine_tuning_enabled"] == False]
    if not no_ft_group.empty:
        x_s, y_s = _smooth_curve(no_ft_group["compression_ratio"].values, no_ft_group["accuracy"].values)
        ax.plot(x_s, y_s, color=color, linewidth=2.0, linestyle="-", label=f"{method} (No FT)", zorder=3)



ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_xlabel("Compression Ratio (×)", fontsize=14, fontweight="bold", labelpad=12)
ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold", labelpad=12)

# Split legend into Zones and Methods
handles, labels = ax.get_legend_handles_labels()

zone_h, zone_l = [], []
method_h, method_l = [], []

for h, l in zip(handles, labels):
    if l.startswith("Zone:"):
        zone_h.append(h)
        zone_l.append(l.replace("Zone: ", ""))
    else:
        method_h.append(h)
        method_l.append(l)

# First legend (Zones) at the top
leg_zones = ax.legend(zone_h, zone_l, loc="upper left", bbox_to_anchor=(1.02, 0.95), 
                      frameon=False, title="Performance Zones", fontsize=11)
# Align title and content to the left
leg_zones._legend_box.align = "left"
leg_zones.get_title().set_fontweight("bold")
ax.add_artist(leg_zones)

# Second legend (Methods) below Zones
leg_methods = ax.legend(method_h, method_l, loc="upper left", bbox_to_anchor=(1.02, 0.70), 
                        frameon=False, title="Algorithms", fontsize=11)
# Align title and content to the left
leg_methods._legend_box.align = "left"
leg_methods.get_title().set_fontweight("bold")
ax.add_artist(leg_methods)

# Third legend (Split Thresholds) below Methods
split_h = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, alpha=0.55),
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, alpha=0.55)
]
split_l = [
    f"Compression: {vSplit:.1f}×",
    f"Accuracy: {hSplit:.1f}%"
]
leg_splits = ax.legend(split_h, split_l, loc="upper left", bbox_to_anchor=(1.02, 0.28), 
                       frameon=False, title="Split Thresholds", fontsize=11)
leg_splits._legend_box.align = "left"
leg_splits.get_title().set_fontweight("bold")

out_path = os.path.join(RUN_DIR, "tradeoff_map.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg_zones, leg_methods, leg_splits))
plt.close(fig)

print(f"Saved trade-off map to {out_path}")
