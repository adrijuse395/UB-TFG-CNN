import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
csv_path = os.path.join(RUN_DIR, "batch_memory_results.csv")

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Run batch_memory_experiment.py first.")
    exit(1)

df = pd.read_csv(csv_path)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fontsize": 11,
})

ALGO_COLORS = {
    "Baseline": "black",
    "CP":       "#1f77b4",
    "Tucker":   "#d62728",
    "TT":       "#2ca02c",
    "SVD":      "#f08c14",
}

fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=df, 
    x="batch_size", 
    y="peak_inference_memory_mb", 
    hue="method", 
    palette=ALGO_COLORS,
    marker="o", 
    linewidth=2.5,
    errorbar="sd",  # Shows standard deviation across different ranks
    ax=ax
)

ax.set_xscale("log", base=2)
ax.set_xticks(sorted(df['batch_size'].unique()))
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_ylabel("Peak Inference Memory (MB)", fontsize=14, fontweight="bold")
ax.set_xlabel("Batch Size", fontsize=14, fontweight="bold")
ax.set_title("Edge vs Server: Peak Memory by Batch Size", fontsize=15, fontweight="bold", pad=12)

ax.grid(True, alpha=0.35)
ax.legend(loc="upper left", framealpha=0.95)

out_path = os.path.join(PLOT_DIR, "batch_memory_scaling.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to {out_path}")
