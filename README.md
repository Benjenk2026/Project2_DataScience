# Project 2 Data Science - Unsupervised Learning

## Contents 
* [Go to Cleaning](#cleaning-pipeline)
* [Go to k-means](#k-means-clustering-pipeline-k-meanspy)
* [Go to PCA Pipeline](#pca-pipeline-pcapy)


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
- `--input`: input CSV path (default `data/processed/higgs_cleaned.csv`)
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

### 5) Run K-Means (after cleaning, optional PCA-independent branch)
Baseline clustering on full feature set:

`python src/k-means.py --k 2`

Optional MiniBatch for speed:

`python src/k-means.py --algorithm minibatch --rows 500000 --k 2`

Expected output:
- `data/processed/higgs_clustered.csv`

### Recommended minimal command sequence
If you only need the required PCA outputs and baseline clustering:

1. `python -m pip install -r src/requirements.txt`
2. `python src/cleaning.py --file higgs --chunked`
3. `python src/PCA.py`
4. `python src/k-means.py --k 2`

