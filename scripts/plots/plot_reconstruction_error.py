import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ALGO_COLORS = {
    "CP":     "#1f77b4",  # Blue
    "Tucker": "#d62728",  # Red
    "TT":     "#2ca02c",  # Green
    "SVD":    "#d4b200",  # Dark Yellow (better visibility than pure yellow)
}

def plot_reconstruction_error(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)

    # Filter out Baseline and Fine-tuned rows (we only want compressed models)
    df = df[df["method"] != "None"]
    df = df[~df["experiment_name"].str.endswith("[fine_tuned]")]

    if "reconstruction_error" not in df.columns:
        print("Error: 'reconstruction_error' column not found in CSV.")
        return

    # Drop any NaNs
    df = df.dropna(subset=["reconstruction_error", "accuracy"])

    if len(df) == 0:
        print("No valid rows found after filtering.")
        return

    plt.figure(figsize=(8, 6))

    # Scatter plot
    sns.scatterplot(
        data=df,
        x="reconstruction_error",
        y="accuracy",
        hue="method",
        palette=ALGO_COLORS,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5
    )

    # Get baseline accuracy if available
    baseline_df = pd.read_csv(csv_path)
    baseline_df = baseline_df[baseline_df["method"] == "None"]
    if not baseline_df.empty:
        baseline_acc = baseline_df["accuracy"].iloc[0]
        plt.axhline(baseline_acc, color='black', linestyle='--', linewidth=1.5, label=f"Baseline ({baseline_acc:.2f}%)")

    plt.title("Reconstruction Error (Frobenius Norm) vs Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Mean Relative Frobenius Error", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Legend handling
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove 'method' title from legend if seaborn added it
    if labels and labels[0] == "method":
        handles, labels = handles[1:], labels[1:]
    plt.legend(handles, labels, loc='lower left', fontsize=10)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reconstruction_error_vs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Reconstruction Error vs Accuracy")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--out", type=str, default=".", help="Output directory for the plot")
    args = parser.parse_args()

    plot_reconstruction_error(args.csv, args.out)
