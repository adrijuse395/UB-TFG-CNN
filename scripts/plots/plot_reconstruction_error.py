import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import PchipInterpolator

ALGO_COLORS = {
    "CP":     "#1f77b4",
    "Tucker": "#d62728",
    "TT":     "#2ca02c",
    "SVD":    "#f08c14",
}

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


def plot_reconstruction_error(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)

    # Baseline
    baseline_row = df[df["method"].isna() | (df["method"] == "None")]
    baseline_acc = float(baseline_row["accuracy"].iloc[0]) if not baseline_row.empty else None
    baseline_params = float(baseline_row["total_parameters"].iloc[0]) if not baseline_row.empty else None

    # Filter out Baseline and Fine-tuned rows (we only want compressed models)
    df_models = df[df["method"].notna() & (df["method"] != "None")].copy()
    df_models = df_models[~df_models["experiment_name"].str.endswith("[fine_tuned]")]

    if "reconstruction_error" not in df_models.columns:
        print("Error: 'reconstruction_error' column not found in CSV.")
        return

    # Drop any NaNs
    df_models = df_models.dropna(subset=["reconstruction_error", "accuracy", "total_parameters"])

    if len(df_models) == 0:
        print("No valid rows found after filtering.")
        return

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ALGORITHMS = ["SVD", "Tucker", "TT", "CP"]

    # -------------------------------------------------------------
    # PANEL (a): Reconstruction Error vs Parameters
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_title("(a)", loc="center", fontsize=18, fontweight="bold", pad=8)

    for algo in ALGORITHMS:
        subset = df_models[df_models["method"] == algo]
        if subset.empty: continue
        
        subset = subset.sort_values("reconstruction_error")
        x_arr = subset["reconstruction_error"].values
        # Divide by 1 million for cleaner axis
        y_arr = subset["total_parameters"].values / 1e6
        
        x_s, y_s = _smooth_curve(x_arr, y_arr)
        
        ax1.plot(
            x_s, y_s,
            color=ALGO_COLORS.get(algo, "black"),
            linestyle="-",
            linewidth=2.8,
            label=algo
        )
        
    ax1.set_xlabel("Reconstruction Error", fontsize=18, labelpad=10)
    ax1.set_ylabel("Total Parameters (Millions)", fontsize=18, labelpad=12)
    ax1.grid(True, alpha=0.35)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=12.5)


    # -------------------------------------------------------------
    # PANEL (b): Reconstruction Error vs Accuracy
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_title("(b)", loc="center", fontsize=18, fontweight="bold", pad=8)

    for algo in ALGORITHMS:
        subset = df_models[df_models["method"] == algo]
        if subset.empty: continue
        
        subset = subset.sort_values("reconstruction_error")
        x_arr = subset["reconstruction_error"].values
        y_arr = subset["accuracy"].values
        
        x_s, y_s = _smooth_curve(x_arr, y_arr)
        
        ax2.plot(
            x_s, y_s,
            color=ALGO_COLORS.get(algo, "black"),
            linestyle="-",
            linewidth=2.8,
            label=algo
        )

    if baseline_acc is not None:
        ax2.axhline(
            y=baseline_acc,
            color="0.35",
            linestyle="-.",
            linewidth=1.5,
            label="Baseline",
            zorder=0,
        )
        
    ax2.set_xlabel("Reconstruction Error", fontsize=18, labelpad=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=18, labelpad=12)
    ax2.grid(True, alpha=0.35)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=12.5)

    # Global spacing
    fig.subplots_adjust(wspace=0.25, bottom=0.15, top=0.88)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reconstruction_error_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Reconstruction Error vs Accuracy/Params")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--out", type=str, default=".", help="Output directory for the plot")
    args = parser.parse_args()

    plot_reconstruction_error(args.csv, args.out)
