import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(RUN_DIR, "results.csv"))

ALGORITHMS = ["CP", "Tucker", "TT", "SVD"]
ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
    "Baseline": "black"
}

# Filter to fine-tuned only + baseline
df_ft = df[(df["fine_tuning_enabled"] == True) | (df["method"].isna()) | (df["method"] == "None")].copy()
df_ft["method"] = df_ft["method"].fillna("Baseline")
df_ft["method"] = df_ft["method"].replace("None", "Baseline")

# Hover info
df_ft["hover_name"] = df_ft.apply(
    lambda row: "Baseline" if row["method"] == "Baseline" 
                else f"{row['method']} (Rank {row['rank']})", 
    axis=1
)

fig = px.scatter_3d(
    df_ft, 
    x="peak_inference_memory_mb", 
    y="latency_ms", 
    z="accuracy",
    color="method",
    color_discrete_map=ALGO_COLORS,
    hover_name="hover_name",
    hover_data={
        "method": False,
        "peak_inference_memory_mb": ':.2f',
        "latency_ms": ':.2f',
        "accuracy": ':.2f',
        "total_parameters": True
    },
    labels={
        "peak_inference_memory_mb": "Peak Memory (MB)",
        "latency_ms": "Latency (ms)",
        "accuracy": "Accuracy (%)"
    },
    title="3D Design Space Exploration: Accuracy vs Latency vs Memory"
)

# Hacer el marker del baseline más grande y con estrella
fig.update_traces(
    marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

for trace in fig.data:
    if trace.name == "Baseline":
        trace.marker.symbol = 'diamond'
        trace.marker.size = 10

fig.update_layout(
    scene=dict(
        xaxis_title="Memory (MB)",
        yaxis_title="Latency (ms)",
        zaxis_title="Accuracy (%)"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

out_file = os.path.join(RUN_DIR, "pareto_interactive.html")
fig.write_html(out_file)
print(f"Saved interactive plot to {out_file}")
