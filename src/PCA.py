"""
PCA.py
Run PCA on cleaned HIGGS data and export reduced datasets.

Default behavior runs PCA with 2, 5, and 10 components.

Usage examples:
	python src/PCA.py
	python src/PCA.py --rows 500000
	python src/PCA.py --components 2,5,10 --output-dir data/processed
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run PCA on HIGGS features and save reduced datasets."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/processed/higgs_cleaned.csv"),
		help="Path to cleaned HIGGS CSV (default: data/processed/higgs_cleaned.csv)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/processed"),
		help="Directory to save PCA outputs.",
	)
	parser.add_argument(
		"--components",
		type=str,
		default="2,5,10",
		help="Comma-separated PCA component counts (default: 2,5,10)",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=None,
		help="Optional row limit for faster experimentation.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	return parser.parse_args()


def parse_component_counts(raw: str) -> list[int]:
	counts = []
	for token in raw.split(","):
		value = token.strip()
		if not value:
			continue
		parsed = int(value)
		if parsed <= 0:
			raise ValueError("All component counts must be positive integers.")
		counts.append(parsed)

	if not counts:
		raise ValueError("--components must contain at least one integer.")

	return sorted(set(counts))


def validate_input(df: pd.DataFrame) -> None:
	missing = [col for col in FEATURE_COLS if col not in df.columns]
	if missing:
		raise ValueError(f"Input file is missing required feature columns: {missing}")


def prepare_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	validate_input(df)
	X = df[FEATURE_COLS].copy().apply(pd.to_numeric, errors="coerce")
	valid_mask = X.notna().all(axis=1)
	dropped = int((~valid_mask).sum())
	if dropped:
		print(f"Dropped {dropped:,} rows with missing/non-numeric feature values.")
	return df.loc[valid_mask].copy(), X.loc[valid_mask]


def run_pca_for_components(
	df: pd.DataFrame,
	X_scaled,
	n_components: int,
	output_dir: Path,
	random_state: int,
) -> None:
	pca = PCA(n_components=n_components, random_state=random_state)
	X_reduced = pca.fit_transform(X_scaled)

	component_cols = [f"pc_{i}" for i in range(1, n_components + 1)]
	reduced_df = pd.DataFrame(X_reduced, columns=component_cols, index=df.index)

	# Keep label when available for downstream comparisons.
	if "label" in df.columns:
		output_df = pd.concat([df[["label"]], reduced_df], axis=1)
	else:
		output_df = reduced_df

	output_path = output_dir / f"higgs_pca_{n_components}d.csv"
	output_df.to_csv(output_path, index=False)

	explained = pca.explained_variance_ratio_
	explained_total = explained.sum()
	print(
		f"Saved {n_components}-component PCA output to: {output_path} "
		f"(explained variance: {explained_total:.4f})"
	)


def main() -> None:
	args = parse_args()
	component_counts = parse_component_counts(args.components)

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	if args.rows is not None and args.rows <= 0:
		raise ValueError("--rows must be a positive integer.")

	print(f"Loading: {args.input}")
	if args.rows is not None:
		df = pd.read_csv(args.input, low_memory=False, nrows=args.rows)
	else:
		df = pd.read_csv(args.input, low_memory=False)

	df, X = prepare_numeric_features(df)
	if len(X) == 0:
		raise ValueError("No valid rows available after numeric filtering.")

	max_components = X.shape[1]
	invalid = [c for c in component_counts if c > max_components]
	if invalid:
		raise ValueError(
			f"Requested components {invalid} exceed available features ({max_components})."
		)

	print("Standardizing 28 feature columns...")
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	for n_components in component_counts:
		run_pca_for_components(
			df=df,
			X_scaled=X_scaled,
			n_components=n_components,
			output_dir=args.output_dir,
			random_state=args.random_state,
		)


if __name__ == "__main__":
	main()
