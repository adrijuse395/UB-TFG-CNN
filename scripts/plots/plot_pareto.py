import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

RUN_DIR = os.environ.get("RUN_DIR", os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

ALGORITHMS = ["CP", "Tucker", "TT", "SVD"]
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#d4b200",
}

# -------------------------------------------------------------------------
# Filter: Solo queremos los modelos con Fine-Tuning para evaluar Accuracy real
# y el Baseline para tener la referencia sin comprimir.
# -------------------------------------------------------------------------
df_ft = df[(df["fine_tuning_enabled"] == True) | (df["method"].isna()) | (df["method"] == "None")].copy()
baseline = df_ft[df_ft["method"].isna() | (df_ft["method"] == "None")]

# =========================================================================
# 1. GRAFICO 3D REAL (x=Memoria, y=Latencia, z=Accuracy)
# =========================================================================
fig_3d = plt.figure(figsize=(11, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Baseline
if not baseline.empty:
    ax_3d.scatter(
        baseline["peak_inference_memory_mb"],
        baseline["latency_ms"],
        baseline["accuracy"],
        color="black",
        marker="*",
        s=300,
        label="Baseline",
        edgecolors="white",
        zorder=10
    )

for algo in ALGORITHMS:
    g = df_ft[df_ft["method"] == algo]
    if len(g) == 0: continue
    ax_3d.scatter(
        g["peak_inference_memory_mb"],
        g["latency_ms"],
        g["accuracy"],
        color=ALGO_COLORS[algo],
        label=algo,
        s=60,
        alpha=0.8,
        edgecolors="w",
        linewidths=0.5
    )

ax_3d.set_xlabel("Peak Memory (MB)", labelpad=12, fontweight="bold")
ax_3d.set_ylabel("Inference Latency (ms)", labelpad=12, fontweight="bold")
ax_3d.set_zlabel("Accuracy (%)", labelpad=12, fontweight="bold")
ax_3d.set_title("3D Space: Accuracy vs Latency vs Memory", pad=25, fontsize=15, fontweight="bold")
ax_3d.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), framealpha=0.95)

# Adjust viewing angle to make the drop clearly visible
ax_3d.view_init(elev=25, azim=-45)

out_3d = os.path.join(PLOT_DIR, "pareto_3d.png")
fig_3d.savefig(out_3d, dpi=300, bbox_inches="tight")
plt.close(fig_3d)


# =========================================================================
# 2. GRAFICO 2D BUBBLE (Pareto Estándar en Papers: x=Latencia, y=Accuracy, Size=Memoria)
# =========================================================================
# This plot is much easier to read on paper/pdf.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
})

fig_2d, ax_2d = plt.subplots(figsize=(10, 7))

# Escala visual para el tamaño de la burbuja (área proporcional a MB)
scale_factor = 25 

if not baseline.empty:
    mem = float(baseline["peak_inference_memory_mb"].iloc[0])
    # Dibujamos la burbuja (semi-transparente)
    ax_2d.scatter(
        baseline["latency_ms"],
        baseline["accuracy"],
        s=mem * scale_factor, 
        color="black",
        alpha=0.2,
        edgecolors="black",
        linewidths=1.5
    )
    # Draw a central point to indicate exact position
    ax_2d.scatter(
        baseline["latency_ms"],
        baseline["accuracy"],
        s=100,
        color="black",
        marker="*",
        label="Baseline"
    )

for algo in ALGORITHMS:
    g = df_ft[df_ft["method"] == algo]
    if len(g) == 0: continue
    
    # Dibujamos burbujas
    ax_2d.scatter(
        g["latency_ms"],
        g["accuracy"],
        s=g["peak_inference_memory_mb"] * scale_factor,
        color=ALGO_COLORS[algo],
        alpha=0.5,
        edgecolors="w",
        linewidths=1
    )
    # Punto central
    ax_2d.scatter(
        g["latency_ms"],
        g["accuracy"],
        s=25,
        color=ALGO_COLORS[algo],
        label=algo
    )

ax_2d.set_xlabel("Inference Latency (ms)  [← Más rápido es mejor]", fontsize=13, fontweight="bold")
ax_2d.set_ylabel("Accuracy (%)  [↑ Más alto es mejor]", fontsize=13, fontweight="bold")
ax_2d.set_title("Pareto Front: Accuracy vs Latency (Bubble Size = Memory)", fontsize=15, fontweight="bold", pad=20)
ax_2d.grid(True, alpha=0.35)

# Force standard size so legend doesn't show huge bubbles
handles, labels = ax_2d.get_legend_handles_labels()
# Los scatter que representan el centro (index impar/pares)
unique_labels = dict(zip(labels, handles)) 
ax_2d.legend(unique_labels.values(), unique_labels.keys(), loc="lower right", framealpha=0.95, scatterpoints=1)

out_2d = os.path.join(PLOT_DIR, "pareto_bubble.png")
fig_2d.savefig(out_2d, dpi=300, bbox_inches="tight")
plt.close(fig_2d)

print(f"Saved: {out_3d}")
print(f"Saved: {out_2d}")
