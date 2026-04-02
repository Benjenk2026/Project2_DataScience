# Project 2 Data Science - Unsupervised Learning

## Dataset : UCI HIGGS Dataset 
* https://archive.ics.uci.edu/dataset/280/higgs
* Label = 1 → signal 
* Label = 0 → background 
* Features: 28 continuous physics-derived attributes 
* Rows: 11,000,000 

The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level features then 7 high-level features): lepton  pT, lepton  eta, lepton  phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb. For more detailed information about each feature see the original paper.

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

