import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib.ticker import MaxNLocator

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
batch_csv_path = os.path.join(RUN_DIR, "batch_memory_results.csv")
res_csv_path = os.path.join(RUN_DIR, "results.csv")

if not os.path.exists(batch_csv_path) or not os.path.exists(res_csv_path):
    print("Error: Missing CSV files.")
    exit(1)

df_batch = pd.read_csv(batch_csv_path)
df_res = pd.read_csv(res_csv_path)

# Reconstruct rank sequence mathematically
svd_ranks = [2, 5, 7, 10, 12, 15, 18, 20, 23, 26, 30, 33, 39, 45, 50, 58, 64, 75, 88, 97, 113, 126, 147, 163, 190, 223, 251, 300, 337, 400]
tucker_ranks = [2, 5, 8, 11, 13, 16, 19, 23, 27, 31, 37, 40, 47, 54, 61, 70, 81, 92, 104, 114, 128, 147, 167, 188, 213, 239, 259, 302, 349, 400]
tt_ranks = [2, 4, 7, 9, 11, 14, 16, 18, 21, 25, 28, 31, 38, 42, 48, 57, 64, 72, 86, 96, 108, 129, 146, 164, 185, 221, 249, 285, 350, 400]
cp_ranks = [2, 5, 8, 11, 14, 17, 20, 25, 28, 34, 40, 49, 58, 70, 84, 100, 120, 143, 171, 204, 243, 290, 326, 389, 465, 576, 732, 931, 1176, 1500]

ranks_seq = []
for _ in range(8): ranks_seq.append(None) # Baseline
for r in svd_ranks:
    for _ in range(8): ranks_seq.append(r)
for r in tucker_ranks:
    for _ in range(8): ranks_seq.append(r)
for r in tt_ranks:
    for _ in range(8): ranks_seq.append(r)
for r in cp_ranks:
    for _ in range(8): ranks_seq.append(r)

if len(ranks_seq) != len(df_batch):
    print(f"Warning: row counts don't match. Expected {len(ranks_seq)} but got {len(df_batch)}.")
    # Fallback padding just in case
    ranks_seq = ranks_seq[:len(df_batch)]
    while len(ranks_seq) < len(df_batch): ranks_seq.append(None)

df_batch["rank"] = ranks_seq

# Extract parameters from results.csv
df_res = df_res[df_res["fine_tuning_enabled"] == False]

# Get Baseline params
try:
    baseline_params = df_res[df_res["method"] == "None"]["total_parameters"].iloc[0]
except:
    baseline_params = 9225482 # fallback VGG11 params

# Get params for compressed
df_params = df_res[["method", "rank", "total_parameters"]].dropna(subset=["method", "rank"]).copy()
# Convert ranks to integer then string for safe merge
df_params["rank"] = df_params["rank"].astype(float).astype(int).astype(str)
df_batch["rank_str"] = df_batch["rank"].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "None")

df_merged = pd.merge(df_batch, df_params, left_on=["method", "rank_str"], right_on=["method", "rank"], how="left")
df_merged.loc[df_merged["method"] == "Baseline", "total_parameters"] = baseline_params

# Clean up
df_merged["total_params_millions"] = df_merged["total_parameters"] / 1e6
df_merged = df_merged.sort_values(["method", "batch_size", "total_params_millions"])

# Plotting Configuration
sns.set_theme(style="whitegrid", context="paper", font_scale=1.65)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
})

ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
}

ALGORITHMS = ["CP", "Tucker", "TT", "SVD"]
BATCH_SIZES_TO_PLOT = [1, 2, 4, 8, 16, 32, 64, 128]

fig, axes = plt.subplots(1, 4, figsize=(12, 5.5), sharex=False, sharey=True)
axes = axes.flatten()

for idx, algo in enumerate(ALGORITHMS):
    ax = axes[idx]
    
    g_algo = df_merged[df_merged["method"] == algo]
    g_base = df_merged[df_merged["method"] == "Baseline"]
    
    # Create a color gradient based on the algorithm's original color
    # Ranging from a light tone to a dark tone (blend with black)
    base_color = ALGO_COLORS[algo]
    colors = sns.blend_palette(["white", base_color, "black"], n_colors=12)[2:10]
    
    for c_idx, bs in enumerate(BATCH_SIZES_TO_PLOT):
        g_bs = g_algo[g_algo["batch_size"] == bs].sort_values("total_params_millions")
        
        # Pareto filter just for plotting clean curves
        valid_rows = []
        current_min_mem = float('inf')
        for _, row in g_bs.iloc[::-1].iterrows():
            if row["peak_inference_memory_mb"] < current_min_mem:
                valid_rows.append(row)
                current_min_mem = row["peak_inference_memory_mb"]
        
        if valid_rows:
            g_clean = pd.DataFrame(valid_rows[::-1])
            ax.plot(
                g_clean["total_params_millions"], 
                g_clean["peak_inference_memory_mb"], 
                color=colors[c_idx], 
                linewidth=2.5, 
                label=str(bs)
            )
            
        # Draw Baseline point (professional style)
        base_mem = g_base[g_base["batch_size"] == bs]["peak_inference_memory_mb"]
        if not base_mem.empty:
            ax.scatter(
                baseline_params / 1e6, 
                base_mem.iloc[0], 
                marker='o', 
                s=80, 
                color=colors[c_idx],
                edgecolor='black',
                zorder=5
            )
            
    # Display index only and lower font size
    ax.set_title(f"{chr(97 + idx)})", fontsize=15, fontweight="bold")
    
    if idx == 0:
        ax.set_ylabel("Peak Inference Memory (MB)", fontsize=16, labelpad=12)
    
    # Force exact limits and ticks
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 30, 60, 90, 120, 150])
    
    # Legend removed from here, moving to global
        
fig.text(0.5, 0.05, "Total Parameters (Millions)", ha='center', fontsize=16)

import matplotlib.lines as mlines
# Create a generic grey gradient legend for Batch Size
grey_palette = sns.blend_palette(["white", "dimgrey", "black"], n_colors=12)[2:10]
legend_elements = [
    mlines.Line2D([0], [0], color=grey_palette[i], lw=3.5, label=str(bs))
    for i, bs in enumerate(BATCH_SIZES_TO_PLOT)
]
legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=9, markeredgecolor='black', label='Baseline'))

# Put the single legend outside the axes, bottom right, 3x3 grid
leg = fig.legend(handles=legend_elements, title="Batch Size", loc="lower right", ncol=3, bbox_to_anchor=(0.98, -0.035), fontsize=13, title_fontsize=14, frameon=False)
leg._legend_box.align = "left"

plt.tight_layout(rect=[0, 0.15, 1, 1])

out_path = os.path.join(PLOT_DIR, "batch_memory_advanced.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved advanced dashboard to {out_path}")
