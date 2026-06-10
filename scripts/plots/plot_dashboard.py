import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

df_ft = df[(df["fine_tuning_enabled"] == True) | (df["method"].isna()) | (df["method"] == "None")].copy()
baseline = df_ft[df_ft["method"].isna() | (df_ft["method"] == "None")]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- PLOT 1: Accuracy vs Memory ---
ax = axes[0]
for algo in ALGORITHMS:
    g = df_ft[df_ft["method"] == algo]
    if len(g) > 0:
        ax.plot(g["peak_inference_memory_mb"], g["accuracy"], marker='o', color=ALGO_COLORS[algo], label=algo, markersize=5, linewidth=1.5, alpha=0.8)
if not baseline.empty:
    ax.scatter(baseline["peak_inference_memory_mb"], baseline["accuracy"], color="black", marker="*", s=250, label="Baseline", zorder=10, edgecolors='w')
ax.set_xlabel("Peak Memory (MB)", fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontweight="bold")
ax.set_title("1. ¿Quién comprime mejor? (Accuracy vs Memoria)", pad=15, fontweight="bold")
# Invertir eje X para que "Mejor" (menos memoria) esté a la derecha
ax.invert_xaxis()

# --- PLOT 2: Accuracy vs Latency ---
ax = axes[1]
for algo in ALGORITHMS:
    g = df_ft[df_ft["method"] == algo]
    if len(g) > 0:
        ax.plot(g["latency_ms"], g["accuracy"], marker='o', color=ALGO_COLORS[algo], label=algo, markersize=5, linewidth=1.5, alpha=0.8)
if not baseline.empty:
    ax.scatter(baseline["latency_ms"], baseline["accuracy"], color="black", marker="*", s=250, label="Baseline", zorder=10, edgecolors='w')
ax.set_xlabel("Inference Latency (ms)", fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontweight="bold")
ax.set_title("2. ¿Quién es más rápido? (Accuracy vs Latencia)", pad=15, fontweight="bold")
# Invertir eje X para que "Mejor" (menos latencia) esté a la derecha
ax.invert_xaxis()

# --- PLOT 3: Latency vs Memory (Hardware Space) ---
ax = axes[2]
for algo in ALGORITHMS:
    g = df_ft[df_ft["method"] == algo]
    if len(g) > 0:
        # Size proportional to accuracy, mapped so differences are visible
        sizes = np.clip((g["accuracy"] - 10) * 3, 10, 300)
        ax.scatter(g["peak_inference_memory_mb"], g["latency_ms"], 
                   c=ALGO_COLORS[algo], s=sizes, alpha=0.6, edgecolors='w', linewidths=0.5)
        # Línea de trayectoria
        ax.plot(g["peak_inference_memory_mb"], g["latency_ms"], color=ALGO_COLORS[algo], label=algo, linewidth=1.5, alpha=0.7)

if not baseline.empty:
    ax.scatter(baseline["peak_inference_memory_mb"], baseline["latency_ms"], color="black", marker="*", s=300, label="Baseline", zorder=10, edgecolors='w')
ax.set_xlabel("Peak Memory (MB)", fontweight="bold")
ax.set_ylabel("Inference Latency (ms)", fontweight="bold")
ax.set_title("3. Espacio Hardware (Memoria vs Latencia)\nTamaño de burbuja = Accuracy", pad=10, fontweight="bold")

# La mejor esquina es abajo-izquierda (menos latencia, menos memoria)
# No invertimos los ejes para mantener la intuición física estándar del plano Hardware

ax.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()
out_file = os.path.join(PLOT_DIR, "pareto_dashboard.png")
fig.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved dashboard to {out_file}")
