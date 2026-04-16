"""
runtime_comparison.py
Generate a runtime comparison bar chart across all four feature sets
(original 28D, PCA 2D, PCA 5D, PCA 10D) using the cluster quality
summary CSV produced by cluster_pca_compare.py.

Usage examples:
    python src/runtime_comparison.py
    python src/runtime_comparison.py --summary Analysis_and_Findings/pca_kmeans_comparison_500k/cluster_quality_summary.csv
    python src/runtime_comparison.py --summary Analysis_and_Findings/pca_kmeans_comparison_1m/cluster_quality_summary.csv --output Analysis_and_Findings/runtime_comparison_1m.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Display labels for each dataset variant
DATASET_LABELS = {
    "original_28d": "Original\n(28D)",
    "pca_2d": "PCA\n(2D)",
    "pca_5d": "PCA\n(5D)",
    "pca_10d": "PCA\n(10D)",
}

# Colors per dataset for visual distinction
DATASET_COLORS = {
    "original_28d": "#2b8cbe",
    "pca_2d":       "#e34a33",
    "pca_5d":       "#fdbb84",
    "pca_10d":      "#756bb1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot k-Means runtime comparison across feature sets."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("Analysis_and_Findings/pca_kmeans_comparison_500k/cluster_quality_summary.csv"),
        help="Path to cluster_quality_summary.csv (default: 500k run)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Analysis_and_Findings/runtime_comparison.png"),
        help="Output path for the runtime comparison plot.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "runtime_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {missing}")
    return df


def build_chart(df: pd.DataFrame, output_path: Path) -> None:
    # Preserve original order from CSV (original_28d first)
    datasets = df["dataset"].tolist()
    runtimes = df["runtime_seconds"].tolist()
    rows_used = int(df["rows_used"].iloc[0]) if "rows_used" in df.columns else None

    labels = [DATASET_LABELS.get(d, d) for d in datasets]
    colors = [DATASET_COLORS.get(d, "#888888") for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, runtimes, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    # Annotate each bar with its runtime value
    for bar, rt in zip(bars, runtimes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(runtimes),
            f"{rt:.3f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    title = "k-Means Runtime Comparison Across Feature Sets"
    if rows_used:
        title += f"\n({rows_used:,} rows, MiniBatchKMeans, k=2)"
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Feature Space", fontsize=11)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_ylim(0, max(runtimes) * 1.18)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved runtime comparison plot to: {output_path}")


def main() -> None:
    args = parse_args()
    df = load_summary(args.summary)
    build_chart(df, args.output)

    # Print a quick summary to console for reference
    print("\nRuntime Summary:")
    for _, row in df.iterrows():
        label = DATASET_LABELS.get(row["dataset"], row["dataset"]).replace("\n", " ")
        print(f"  {label:<20} {row['runtime_seconds']:.4f}s")

    baseline = df.loc[df["dataset"] == "original_28d", "runtime_seconds"]
    if not baseline.empty:
        base_rt = baseline.iloc[0]
        print("\nSpeedup vs original 28D:")
        for _, row in df.iterrows():
            if row["dataset"] == "original_28d":
                continue
            label = DATASET_LABELS.get(row["dataset"], row["dataset"]).replace("\n", " ")
            speedup = base_rt / row["runtime_seconds"]
            print(f"  {label:<20} {speedup:.1f}x faster")


if __name__ == "__main__":
    main()