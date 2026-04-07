"""Compare clustering quality between original and PCA-reduced HIGGS datasets.

This script clusters each dataset variant, reports quality metrics, writes
cluster-labeled CSV outputs, and saves 2D scatter plots for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-Means on original and PCA-reduced datasets and compare quality metrics."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of clusters (default: 2)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "minibatch"],
        default="kmeans",
        help="Clustering algorithm to use (default: kmeans)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Mini-batch size when --algorithm minibatch is selected.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional row limit for each dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=50_000,
        help="Max sample size used for silhouette score computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Analysis_and_Findings/pca_kmeans_comparison"),
        help="Directory for summary CSV and plot outputs.",
    )
    parser.add_argument(
        "--cluster-output-dir",
        type=Path,
        default=Path("data/processed/pca_kmeans_clusters"),
        help="Directory for cluster-labeled output CSV files.",
    )
    parser.add_argument(
        "--cleaned-path",
        type=Path,
        default=Path("data/processed/higgs_cleaned.csv"),
        help="Path to original cleaned HIGGS dataset.",
    )
    parser.add_argument(
        "--pca-2d-path",
        type=Path,
        default=Path("data/processed/higgs_pca_2d.csv"),
        help="Path to PCA 2D dataset.",
    )
    parser.add_argument(
        "--pca-5d-path",
        type=Path,
        default=Path("data/processed/higgs_pca_5d.csv"),
        help="Path to PCA 5D dataset.",
    )
    parser.add_argument(
        "--pca-10d-path",
        type=Path,
        default=Path("data/processed/higgs_pca_10d.csv"),
        help="Path to PCA 10D dataset.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    if args.algorithm == "kmeans":
        return KMeans(n_clusters=args.k, n_init=10, random_state=args.random_state)
    return MiniBatchKMeans(
        n_clusters=args.k,
        n_init=10,
        random_state=args.random_state,
        batch_size=args.batch_size,
    )


def load_features(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if "label" not in df.columns:
        raise ValueError("Input dataset must include a 'label' column.")

    labels = pd.to_numeric(df["label"], errors="coerce")
    canonical = [f"feature_{i}" for i in range(1, 29)]
    if all(col in df.columns for col in canonical):
        numeric = df[canonical].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = df.drop(columns=["label"]).apply(pd.to_numeric, errors="coerce")
    valid = labels.notna() & numeric.notna().all(axis=1)
    return labels.loc[valid].astype(int), numeric.loc[valid].copy()


def compute_compactness_separation(X_scaled: np.ndarray, cluster_labels: np.ndarray, centroids: np.ndarray) -> tuple[float, float, float]:
    """Return compactness and separation metrics.

    compactness: mean Euclidean distance from each point to its assigned centroid
    min_separation: minimum pairwise centroid distance
    mean_separation: mean pairwise centroid distance
    """
    assigned = centroids[cluster_labels]
    compactness = np.linalg.norm(X_scaled - assigned, axis=1).mean()

    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    if not dists:
        return float(compactness), float("nan"), float("nan")
    return float(compactness), float(min(dists)), float(np.mean(dists))


def choose_scatter_projection(X_scaled: np.ndarray, feature_cols: list[str], random_state: int) -> tuple[np.ndarray, str, str]:
    if "pc_1" in feature_cols and "pc_2" in feature_cols:
        i1 = feature_cols.index("pc_1")
        i2 = feature_cols.index("pc_2")
        return X_scaled[:, [i1, i2]], "pc_1", "pc_2"

    if X_scaled.shape[1] == 1:
        return np.column_stack([X_scaled[:, 0], np.zeros(X_scaled.shape[0])]), feature_cols[0], "zero_axis"

    pca2 = PCA(n_components=2, random_state=random_state)
    return pca2.fit_transform(X_scaled), "pca2_1", "pca2_2"


def save_scatter_plot(
    scatter_2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    title_prefix: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(scatter_2d[:, 0], scatter_2d[:, 1], c=cluster_labels, s=6, alpha=0.45, cmap="viridis")
    axes[0].set_title(f"{title_prefix} - KMeans Clusters")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    axes[1].scatter(scatter_2d[:, 0], scatter_2d[:, 1], c=true_labels, s=6, alpha=0.45, cmap="coolwarm")
    axes[1].set_title(f"{title_prefix} - True Labels")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_one_dataset(name: str, path: Path, args: argparse.Namespace) -> dict[str, float | int | str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"\nLoading {name}: {path}")
    df = pd.read_csv(path, low_memory=False, nrows=args.rows)
    true_labels, X = load_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(args)
    start = perf_counter()
    cluster_labels = model.fit_predict(X_scaled)
    elapsed = perf_counter() - start

    # Silhouette is expensive on large data; use sampled subset when needed.
    if len(X_scaled) > args.silhouette_sample:
        rng = np.random.default_rng(args.random_state)
        sample_idx = rng.choice(len(X_scaled), size=args.silhouette_sample, replace=False)
        sil = silhouette_score(X_scaled[sample_idx], cluster_labels[sample_idx])
    else:
        sil = silhouette_score(X_scaled, cluster_labels)

    dbi = davies_bouldin_score(X_scaled, cluster_labels)
    compactness, sep_min, sep_mean = compute_compactness_separation(X_scaled, cluster_labels, model.cluster_centers_)

    # Save cluster-labeled output.
    out_df = pd.DataFrame({"label": true_labels.values})
    out_df[X.columns.tolist()] = X.values
    out_df["cluster"] = cluster_labels.astype(int)
    cluster_path = args.cluster_output_dir / f"{name}_clustered.csv"
    out_df.to_csv(cluster_path, index=False)

    # Save 2D visualization.
    scatter_2d, _, _ = choose_scatter_projection(X_scaled, X.columns.tolist(), args.random_state)
    scatter_path = args.output_dir / f"{name}_scatter.png"
    save_scatter_plot(
        scatter_2d=scatter_2d,
        cluster_labels=cluster_labels,
        true_labels=true_labels.to_numpy(),
        title_prefix=name,
        output_path=scatter_path,
    )

    print(
        f"Done {name}: rows={len(X_scaled):,}, dims={X_scaled.shape[1]}, "
        f"silhouette={sil:.4f}, davies_bouldin={dbi:.4f}, "
        f"compactness={compactness:.4f}, separation_min={sep_min:.4f}, runtime={elapsed:.2f}s"
    )

    return {
        "dataset": name,
        "rows_used": int(len(X_scaled)),
        "dimensions": int(X_scaled.shape[1]),
        "algorithm": args.algorithm,
        "k": int(args.k),
        "silhouette_score": float(sil),
        "davies_bouldin_index": float(dbi),
        "cluster_compactness": float(compactness),
        "cluster_separation_min": float(sep_min),
        "cluster_separation_mean": float(sep_mean),
        "runtime_seconds": float(elapsed),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cluster_output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("original_28d", args.cleaned_path),
        ("pca_2d", args.pca_2d_path),
        ("pca_5d", args.pca_5d_path),
        ("pca_10d", args.pca_10d_path),
    ]

    rows = []
    for name, path in datasets:
        rows.append(run_one_dataset(name=name, path=path, args=args))

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "cluster_quality_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Clustering Quality Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved clustered CSVs to: {args.cluster_output_dir}")
    print(f"Saved scatter plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
