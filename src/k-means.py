"""
k-means.py
Run K-Means clustering on the cleaned HIGGS dataset using all 28 feature dimensions.

Usage examples:
	python src/k-means.py
	python src/k-means.py --algorithm minibatch --rows 500000
	python src/k-means.py --benchmark-sizes 50000,100000,250000,500000
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Cluster the HIGGS dataset with K-Means using all 28 feature columns."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/processed/higgs_cleaned.csv"),
		help="Path to cleaned HIGGS CSV (default: data/processed/higgs_cleaned.csv)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/processed/higgs_clustered.csv"),
		help="Output CSV with assigned cluster labels.",
	)
	parser.add_argument("--k", type=int, default=2, help="Number of clusters (default: 2)")
	parser.add_argument(
		"--algorithm",
		choices=["kmeans", "minibatch"],
		default="kmeans",
		help="Use standard KMeans or MiniBatchKMeans (default: kmeans)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=10_000,
		help="Mini-batch size when --algorithm minibatch is selected.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=None,
		help="Number of rows to use from the input file (default: all rows).",
	)
	parser.add_argument(
		"--benchmark-sizes",
		type=str,
		default=None,
		help="Comma-separated row counts for runtime benchmark, e.g. 50000,100000,250000.",
	)
	parser.add_argument(
		"--plot-output",
		type=Path,
		default=Path("Analysis_and_Findings/runtime_vs_size.png"),
		help="Where to save runtime-vs-size plot in benchmark mode.",
	)
	return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
	missing = [col for col in FEATURE_COLS if col not in df.columns]
	if missing:
		raise ValueError(f"Input file is missing required feature columns: {missing}")


def build_model(args: argparse.Namespace):
	if args.algorithm == "kmeans":
		return KMeans(
			n_clusters=args.k,
			n_init=10,
			random_state=args.random_state,
		)
	return MiniBatchKMeans(
		n_clusters=args.k,
		n_init=10,
		random_state=args.random_state,
		batch_size=args.batch_size,
	)


def parse_benchmark_sizes(raw: str) -> list[int]:
	sizes = []
	for token in raw.split(","):
		value = token.strip()
		if not value:
			continue
		parsed = int(value)
		if parsed <= 0:
			raise ValueError("All benchmark sizes must be positive integers.")
		sizes.append(parsed)
	if not sizes:
		raise ValueError("--benchmark-sizes must contain at least one value.")
	return sorted(set(sizes))


def prepare_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	validate_input(df)
	X = df[FEATURE_COLS].copy().apply(pd.to_numeric, errors="coerce")
	valid_mask = X.notna().all(axis=1)
	dropped = int((~valid_mask).sum())
	if dropped:
		print(f"Dropped {dropped:,} rows with missing/non-numeric feature values.")
	return df.loc[valid_mask].copy(), X.loc[valid_mask]


def run_single_clustering(args: argparse.Namespace, df: pd.DataFrame) -> None:
	df, X = prepare_numeric_features(df)

	print("Standardizing all 28 feature columns...")
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	model = build_model(args)
	print(
		f"Fitting {model.__class__.__name__} with k={args.k} on "
		f"{X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} features..."
	)
	cluster_labels = model.fit_predict(X_scaled)

	df["cluster"] = cluster_labels.astype("int32")
	args.output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output, index=False)

	counts = df["cluster"].value_counts().sort_index()
	print(f"Saved clustered data to: {args.output}")
	print("Cluster counts:")
	for cluster_id, count in counts.items():
		print(f"  cluster {cluster_id}: {count:,}")


def run_runtime_benchmark(args: argparse.Namespace, df: pd.DataFrame) -> None:
	sizes = parse_benchmark_sizes(args.benchmark_sizes)

	df, X = prepare_numeric_features(df)
	available_rows = len(X)
	if available_rows == 0:
		raise ValueError("No valid rows available for benchmarking.")

	usable_sizes = [size for size in sizes if size <= available_rows]
	skipped_sizes = [size for size in sizes if size > available_rows]
	if skipped_sizes:
		print(f"Skipping sizes larger than available rows ({available_rows:,}): {skipped_sizes}")
	if not usable_sizes:
		raise ValueError("No benchmark sizes are <= available valid rows.")

	runtimes = []
	for size in usable_sizes:
		X_subset = X.iloc[:size]
		start = perf_counter()
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X_subset)
		model = build_model(args)
		model.fit(X_scaled)
		elapsed = perf_counter() - start
		runtimes.append(elapsed)
		print(f"Size {size:,}: {elapsed:.3f} seconds")

	args.plot_output.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 5))
	plt.plot(usable_sizes, runtimes, marker="o", linewidth=2)
	plt.title(f"Runtime vs Dataset Size ({args.algorithm}, k={args.k})")
	plt.xlabel("Rows used")
	plt.ylabel("Runtime (seconds)")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(args.plot_output, dpi=150)
	plt.close()

	print(f"Saved runtime plot to: {args.plot_output}")


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	print(f"Loading: {args.input}")
	if args.benchmark_sizes:
		max_rows = max(parse_benchmark_sizes(args.benchmark_sizes))
		if args.rows is not None:
			max_rows = min(max_rows, args.rows)
		df = pd.read_csv(args.input, low_memory=False, nrows=max_rows)
		run_runtime_benchmark(args, df)
		return

	if args.rows is not None:
		if args.rows <= 0:
			raise ValueError("--rows must be a positive integer.")
		df = pd.read_csv(args.input, low_memory=False, nrows=args.rows)
	else:
		df = pd.read_csv(args.input, low_memory=False)

	run_single_clustering(args, df)


if __name__ == "__main__":
	main()
