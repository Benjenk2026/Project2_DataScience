# Project 2 Data Science - Unsupervised Learning

## Contents 
* [Go to Cleaning](#cleaning-pipeline)
* [Go to k-means](#k-means-clustering-pipeline-k-meanspy)
* [Go to PCA Pipeline](#pca-pipeline-pcapy)
* [Go to EDA Pipeline](#eda-pipeline-edapy)
* [Go to PCA vs Original Cluster Comparison](#pca-vs-original-cluster-comparison-cluster_pca_comparepy)


## Dataset : UCI HIGGS Dataset 
* https://archive.ics.uci.edu/dataset/280/higgs
* Label = 1 → signal 
* Label = 0 → background 
* Features: 28 continuous physics-derived attributes 
* Rows: 11,000,000 
 For more detailed information about each feature see the original paper.


## Cleaning Pipeline (cleaning.py)

The script `cleaning.py` cleans raw data and writes cleaned CSV files to `data/processed/`.

### Main goals
- Standardize column names and string values.
- Handle missing and invalid values safely.
- Enforce numeric feature integrity for HIGGS.
- Flag potential outliers for downstream analysis.
- Keep memory usage manageable for large files (chunked mode).

### What it does to the data
1. Loads source data from CSV or JSON.
2. For HIGGS specifically, assigns explicit column names:
	- `label`
	- `feature_1` through `feature_28`
3. Standardizes all column names to `snake_case`.
4. Applies known alias mapping for location fields (for datasets that include them), such as:
	- `lat` -> `latitude`
	- `long` / `lon` -> `longitude`
	- `zip` / `postal_code` -> `zipcode`
5. Trims whitespace in text/string columns and converts string "nan" values to missing (`NA`).
6. For HIGGS, enforces that `feature_1` through `feature_28` are numeric floats (`float64`).
	- Non-numeric values are coerced to missing (`NaN`) and reported.
7. Drops rows missing critical required fields configured per dataset.
	- For HIGGS, rows missing `label` are dropped.
8. Flags outliers for HIGGS using per-feature IQR fences and adds:
	- `outlier_feature_count` (number of features flagged as outliers in that row)
	- `has_outlier` added column in processed (boolean)
9. Optionally applies dataset-specific normalization (for non-HIGGS files in this script), such as:
	- complaint category normalization
	- Yelp category grouping
	- latitude/longitude range checks and ZIP extraction
10. Deduplicates when a dataset has a configured ID key.
	- If no key is configured/found, dedup is skipped.
	- For HIGGS in this project, dedup is currently skipped (`id_col = None`).
11. Writes cleaned output to `data/processed/higgs_cleaned.csv`.

### Chunked mode for large files
Use chunked mode for HIGGS to avoid loading the full 8+ GB file into memory:

`python cleaning.py --file higgs --chunked`

In chunked mode, the script:
- Reads 100,000 rows at a time.
- Cleans each chunk with the same logic.
- Appends each cleaned chunk to the output CSV.
- Prints progress per chunk and a final row-count summary.

## K-Means Clustering Pipeline (k-means.py)

The script `src/k-means.py` runs clustering on the cleaned HIGGS dataset using all 28 feature columns (`feature_1` to `feature_28`).

### Main goals
- Cluster events using the full 28-dimensional feature space.
- Support both `KMeans` and `MiniBatchKMeans`.
- Allow limiting the number of rows used for faster experiments.
- Benchmark runtime vs dataset size and save a plot.

### Default behavior
Running with no arguments:

`python src/k-means.py`

will:
- Read `data/processed/higgs_cleaned.csv`
- Validate required feature columns
- Convert features to numeric and drop invalid rows
- Standardize all 28 features
- Fit K-Means (`k=2` by default)
- Save output with cluster labels to `data/processed/higgs_clustered.csv`

### Useful options
- `--k`: number of clusters (default `2`)
- `--algorithm`: `kmeans` or `minibatch` (default `kmeans`)
- `--batch-size`: mini-batch size when using `minibatch` (default `10000`)
- `--rows`: limit the number of rows loaded from input
- `--input`: input CSV path (default `data/processed/ higgs_cleaned.csv`)
- `--output`: output CSV path (default `data/processed/higgs_clustered.csv`)
- `--random-state`: random seed (default `42`)

### Example: MiniBatch on a subset

`python src/k-means.py --algorithm minibatch --rows 500000 --k 2`

### Runtime vs size benchmark
Use benchmark mode to measure training runtime for multiple dataset sizes and generate a plot.

Example:

`python src/k-means.py --algorithm minibatch --benchmark-sizes 50000,100000,250000,500000 --plot-output Analysis_and_Findings/runtime_vs_size.png`

Benchmark mode:
- Fits a fresh model for each requested size.
- Reports runtime per size in the terminal.
- Saves a line plot of runtime vs row count.

### Benchmark options
- `--benchmark-sizes`: comma-separated row counts to test
- `--plot-output`: output image path for runtime plot (default `Analysis_and_Findings/runtime_vs_size.png`)

### Subsampling workflow (single entrypoint)
The same script now supports streamlined subsampling and fixed-size justification benchmarking directly.

#### Create default 200k stratified subsample
`python src/k-means.py --create-subsample`

Writes:
- `data/processed/higgs_200k.csv`

#### Create custom-size subsample
`python src/k-means.py --create-subsample --subsample-size 50000 --subsample-output data/processed/higgs_50k.csv`

#### Run fixed subsampling justification benchmark (50k/100k/200k)
`python src/k-means.py --justify-subsampling`

Writes:
- `Analysis_and_Findings/subsample_justification.png`

#### Run benchmark and subsample in one command
`python src/k-means.py --justify-subsampling --create-subsample --subsample-size 200000 --subsample-output data/processed/higgs_200k.csv`

#### New subsampling-specific options
- `--create-subsample`: create and save a stratified sample from input
- `--subsample-size`: output row count for subsampling (default `200000`)
- `--subsample-seed`: random seed for reproducible stratified sampling (default `42`)
- `--subsample-output`: output CSV for subsampled data (default `data/processed/higgs_200k.csv`)
- `--justify-subsampling`: fixed benchmark at `50k/100k/200k` rows
- `--justify-plot-output`: output image path for fixed benchmark plot (default `Analysis_and_Findings/subsample_justification.png`)

## PCA Pipeline (PCA.py)

The script `src/PCA.py` performs Principal Component Analysis on the cleaned HIGGS dataset and writes reduced-dimensional CSV files.

### Main goals
- Reduce 28-dimensional feature space into compact representations.
- Support multiple PCA targets in one run.
- Preserve `label` in outputs for downstream analysis/visualization.

### Default behavior
Running with no arguments:

`python src/PCA.py`

will:
- Read `data/processed/higgs_cleaned.csv`
- Validate `feature_1` through `feature_28`
- Convert features to numeric and drop invalid rows
- Standardize all 28 feature columns
- Run PCA with `2`, `5`, and `10` components
- Save outputs to:
	- `data/processed/higgs_pca_2d.csv`
	- `data/processed/higgs_pca_5d.csv`
	- `data/processed/higgs_pca_10d.csv`
- Print explained variance for each PCA run

### Useful options
- `--components`: comma-separated component counts (default `2,5,10`)
- `--rows`: limit rows loaded for faster experiments
- `--input`: input CSV path (default `data/processed/higgs_cleaned.csv`)
- `--output-dir`: output directory (default `data/processed`)
- `--random-state`: random seed (default `42`)

### Example: run PCA on a subset

`python src/PCA.py --rows 500000 --components 2,5,10`

## EDA Pipeline (eda.py)

The script `src/eda.py` generates exploratory plots for the cleaned HIGGS dataset to inspect class balance, feature distributions, and feature relationships before modeling.

### Main goals
- Visualize label balance between signal (`1`) and background (`0`).
- Inspect per-feature distributions across the cleaned dataset.
- Measure pairwise Pearson correlation across numeric features.
- Compare feature spread by class label using boxplots.

### Default behavior
Running with no arguments:

`cd src && python eda.py`

will:
- Read `data/processed/higgs_cleaned.csv`
- Use the `label` column for class-aware plots
- Automatically include numeric feature columns and exclude non-numeric fields such as boolean flags
- Save plots to `output/eda/`

### Output files
- `output/eda/class_distribution.png`
- `output/eda/feature_histograms.png`
- `output/eda/correlation_matrix.png`
- `output/eda/boxplots_by_label.png`

### Function interface
All plotting functions accept the same core arguments:
- `df`: pandas DataFrame containing `label` and numeric feature columns
- `save`: whether to save the figure to disk (default `True`)
- `output_dir`: directory used when saving plots (default `../output/eda` when running from `src/`)

Available functions:
- `plot_class_distribution(df, save=True, output_dir='../output/eda')` Bar chart of label=0 vs label=1 counts
- `plot_feature_histograms(df, save=True, output_dir='../output/eda')` Histogram grid for all 28 features (4x7 subplot)
- `plot_correlation_matrix(df, save=True, output_dir='../output/eda')` Heatmap of feature-feature Pearson correlations
- `plot_boxplots_by_label(df, save=True, output_dir='../output/eda')` Boxplots per feature, colored by signal/background label

### Example: import and run from Python

```python
import pandas as pd
from src.eda import (
	plot_boxplots_by_label,
	plot_class_distribution,
	plot_correlation_matrix,
	plot_feature_histograms,
)

df = pd.read_csv('data/processed/higgs_cleaned.csv')

plot_class_distribution(df)
plot_feature_histograms(df)
plot_correlation_matrix(df)
plot_boxplots_by_label(df)
```

### Notes
- The cleaned HIGGS file in this project may contain extra engineered columns such as `outlier_feature_count` and `has_outlier`.
- `eda.py` plots all numeric columns except `label`, so if additional numeric engineered features exist, they will also appear in the histogram, correlation, and boxplot outputs.
- Use `save=False` to display plots interactively instead of saving them.

## PCA vs Original Cluster Comparison (cluster_pca_compare.py)

The script `src/cluster_pca_compare.py` runs clustering on:
- original cleaned features (`feature_1` to `feature_28`)
- PCA 2D (`pc_1`, `pc_2`)
- PCA 5D
- PCA 10D

It produces side-by-side quality metrics and scatter visualizations to compare clustering behavior across representations.

### Main goals
- Evaluate clustering quality consistently across original and PCA-reduced datasets.
- Compare metrics using the same `k`, algorithm, and random state.
- Save cluster-labeled CSV outputs for each variant.
- Save 2D scatter plots for visual comparison (cluster assignment vs true labels).

### Metrics reported
- `silhouette_score` (higher is better)
- `davies_bouldin_index` (lower is better)
- `cluster_compactness` (mean distance of points to assigned centroid; lower is tighter)
- `cluster_separation_min` and `cluster_separation_mean` (centroid distances; higher means more separated clusters)

### Default behavior

Running with no arguments:

`python src/cluster_pca_compare.py`

will:
- Read:
	- `data/processed/higgs_cleaned.csv`
	- `data/processed/higgs_pca_2d.csv`
	- `data/processed/higgs_pca_5d.csv`
	- `data/processed/higgs_pca_10d.csv`
- Cluster each dataset with K-Means (`k=2` by default).
- Save summary to `Analysis_and_Findings/pca_kmeans_comparison/cluster_quality_summary.csv`.
- Save per-dataset scatter plots to `Analysis_and_Findings/pca_kmeans_comparison/`.
- Save cluster-labeled CSV files to `data/processed/pca_kmeans_clusters/`.

### Useful options
- `--k`: number of clusters (default `2`)
- `--algorithm`: `kmeans` or `minibatch` (default `kmeans`)
- `--batch-size`: mini-batch size for `minibatch` (default `10000`)
- `--rows`: optional row cap per dataset for faster experiments
- `--silhouette-sample`: max sample size used for silhouette computation on large data (default `50000`)
- `--output-dir`: directory for summary and scatter plots
- `--cluster-output-dir`: directory for cluster-labeled CSVs

### Output files

The script produces two groups of outputs: comparison artifacts and cluster-labeled datasets.

#### 1) Comparison summary CSV
Saved in `--output-dir` as:
- `cluster_quality_summary.csv`

This file contains one row per dataset variant (`original_28d`, `pca_2d`, `pca_5d`, `pca_10d`) and includes:
- `rows_used`: number of rows clustered
- `dimensions`: number of input dimensions used for clustering
- `algorithm`: clustering algorithm used
- `k`: number of clusters
- `silhouette_score`: higher indicates better separation and cohesion
- `davies_bouldin_index`: lower indicates better cluster quality
- `cluster_compactness`: lower indicates tighter clusters around centroids
- `cluster_separation_min`: minimum centroid-to-centroid distance
- `cluster_separation_mean`: mean centroid-to-centroid distance
- `runtime_seconds`: clustering runtime for that dataset

#### 2) Scatter plot images
Saved in `--output-dir` as:
- `original_28d_scatter.png`
- `pca_2d_scatter.png`
- `pca_5d_scatter.png`
- `pca_10d_scatter.png`

Each scatter image contains two side-by-side plots:
- left: points colored by assigned cluster
- right: points colored by true HIGGS label (`0` background, `1` signal)

For PCA datasets, the plot uses `pc_1` and `pc_2` when available. For higher-dimensional inputs, the script projects the scaled data to 2D only for visualization.

#### 3) Cluster-labeled CSV outputs
Saved in `--cluster-output-dir` as:
- `original_28d_clustered.csv`
- `pca_2d_clustered.csv`
- `pca_5d_clustered.csv`
- `pca_10d_clustered.csv`

Each output CSV contains:
- the original `label` column
- the feature columns used for that dataset representation
- a new `cluster` column containing the assigned cluster ID

### Example: read and rank the summary in pandas

```python
import pandas as pd

summary = pd.read_csv(
	"Analysis_and_Findings/pca_kmeans_comparison_200k/cluster_quality_summary.csv"
)

ranked = summary.sort_values("silhouette_score", ascending=False)

print(ranked[[
	"dataset",
	"silhouette_score",
	"davies_bouldin_index",
	"cluster_compactness",
	"cluster_separation_min",
]])
```

This is a quick way to see which representation produced the best clustering quality on a given run.

### Commands used in this project

200k rows:
`python src/cluster_pca_compare.py --algorithm minibatch --rows 200000 --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_200k --cluster-output-dir data/processed/pca_kmeans_clusters_200k`

500k rows:
`python src/cluster_pca_compare.py --algorithm minibatch --rows 500000 --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_500k --cluster-output-dir data/processed/pca_kmeans_clusters_500k`

1M rows:
`python src/cluster_pca_compare.py --algorithm minibatch --rows 1000000 --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_1m --cluster-output-dir data/processed/pca_kmeans_clusters_1m`


### Current 200k summary snapshot (MiniBatch, k=2)

- `original_28d`: silhouette `0.1815`, DBI `2.7934`
- `pca_2d`: silhouette `0.3517`, DBI `1.1369`
- `pca_5d`: silhouette `0.1854`, DBI `2.0516`
- `pca_10d`: silhouette `0.0963`, DBI `3.0828`

In this run, PCA 2D gave the strongest clustering quality by both silhouette and Davies-Bouldin.

### Troubleshooting: full-data run fails (`--rows` omitted)

If this command fails:

`python src/cluster_pca_compare.py --algorithm minibatch --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_full --cluster-output-dir data/processed/pca_kmeans_clusters_full`

the most common cause is memory pressure while loading and scaling all datasets at once (especially when computing metrics on very large arrays).

Recommended safe alternatives:
- Use a bounded row count:
	- `python src/cluster_pca_compare.py --algorithm minibatch --rows 1000000 --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_1m --cluster-output-dir data/processed/pca_kmeans_clusters_1m`
- Reduce silhouette sampling cost:
	- `python src/cluster_pca_compare.py --algorithm minibatch --rows 1000000 --k 2 --silhouette-sample 20000 --output-dir Analysis_and_Findings/pca_kmeans_comparison_1m --cluster-output-dir data/processed/pca_kmeans_clusters_1m`
- Use 500k for fast iteration before attempting larger runs:
	- `python src/cluster_pca_compare.py --algorithm minibatch --rows 500000 --k 2 --output-dir Analysis_and_Findings/pca_kmeans_comparison_500k --cluster-output-dir data/processed/pca_kmeans_clusters_500k`

## Step-by-Step Run Order

Use this order to run the full workflow from raw data to outputs.

### 1) Install dependencies
From the project root:

`python -m pip install -r src/requirements.txt`

### 2) Place raw HIGGS data
Ensure the raw file exists at:

`data/higgs/HIGGS.csv`

### 3) Run cleaning first (required)
For full dataset processing with lower memory pressure:

`python src/cleaning.py --file higgs --chunked`

Expected output:
- `data/processed/higgs_cleaned.csv`

### 4) Run PCA (after cleaning)
Default required PCA outputs (2, 5, 10 components):

`python src/PCA.py`

Expected outputs:
- `data/processed/higgs_pca_2d.csv`
- `data/processed/higgs_pca_5d.csv`
- `data/processed/higgs_pca_10d.csv`

Optional faster trial on subset:

`python src/PCA.py --rows 500000 --components 2,5,10`

### 5) Run EDA (after cleaning, before modeling)
Generate the exploratory plots used to inspect the cleaned dataset:

`cd src && python eda.py`

Expected outputs:
- `output/eda/class_distribution.png`
- `output/eda/feature_histograms.png`
- `output/eda/correlation_matrix.png`
- `output/eda/boxplots_by_label.png`

### 6) Run K-Means (after cleaning, optional PCA-independent branch)
Baseline clustering on full feature set:

`python src/k-means.py --k 2`

Optional MiniBatch for speed:

`python src/k-means.py --algorithm minibatch --rows 500000 --k 2`

Expected output:
- `data/processed/higgs_clustered.csv`

### 7) Compare clustering quality across original vs PCA datasets (optional)
Run side-by-side clustering quality evaluation:

`python src/cluster_pca_compare.py --algorithm minibatch --rows 200000 --k 2`

Expected outputs:
- `Analysis_and_Findings/pca_kmeans_comparison/cluster_quality_summary.csv`
- `Analysis_and_Findings/pca_kmeans_comparison/original_28d_scatter.png`
- `Analysis_and_Findings/pca_kmeans_comparison/pca_2d_scatter.png`
- `Analysis_and_Findings/pca_kmeans_comparison/pca_5d_scatter.png`
- `Analysis_and_Findings/pca_kmeans_comparison/pca_10d_scatter.png`
- `data/processed/pca_kmeans_clusters/original_28d_clustered.csv`
- `data/processed/pca_kmeans_clusters/pca_2d_clustered.csv`
- `data/processed/pca_kmeans_clusters/pca_5d_clustered.csv`
- `data/processed/pca_kmeans_clusters/pca_10d_clustered.csv`

### Recommended minimal command sequence
If you only need the required PCA outputs and baseline clustering:

1. `python -m pip install -r src/requirements.txt`
2. `python src/cleaning.py --file higgs --chunked`
3. `cd src && python eda.py`
4. `python src/PCA.py`
5. `python src/k-means.py --k 2`
6. `python src/cluster_pca_compare.py --algorithm minibatch --rows 200000 --k 2`

