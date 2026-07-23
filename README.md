# Acute Kidney Injury Prediction

A reproducible binary-classification pipeline for predicting acute kidney injury (AKI) from longitudinal creatinine measurements and patient demographics. The project emphasises recall-aware model selection, leakage-resistant evaluation, and a small, inspectable deployment path.

## What the project does

`model.py`:

- converts variable-length creatinine histories into 12 tabular features;
- handles missing values inside a scikit-learn pipeline;
- trains a class-balanced logistic-regression classifier;
- selects a decision threshold from out-of-fold predictions by maximising F3, which weights recall more heavily than precision;
- evaluates the fixed threshold once on a stratified 10% holdout set;
- applies a minimum holdout F3 quality gate before writing test predictions.

The feature set includes age, encoded sex, test count, baseline and latest creatinine, distribution statistics, and absolute and relative change from baseline.

## Empirical design

The labelled data are divided into:

1. a 90% train/validation partition;
2. five stratified folds within that partition for out-of-fold threshold selection;
3. a 10% holdout used only after the threshold is fixed.

Preprocessing is fitted inside each fold through `sklearn.pipeline.Pipeline`, avoiding leakage from imputation and standardisation. The script reports cross-validation training F3, threshold-selection F3, holdout F3, precision, recall, and the full confusion matrix.

`explore.ipynb` records the exploratory analysis that motivated the features. Its stored outputs describe 7,301 labelled patients, a 79.24%/20.76% non-AKI/AKI class split, and variable-length histories ranging from 1 to 44 creatinine measurements.

No aggregate model score is claimed here because the training data needed to reproduce a run are not included in this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `model.py` | Feature engineering, training, threshold selection, evaluation, and inference |
| `explore.ipynb` | Exploratory analysis of class balance and longitudinal creatinine data |
| `requirements.txt` | Minimal Python dependencies |
| `Dockerfile` | Containerised command-line inference workflow |

## Run locally

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide:

- `training.csv`, containing the labelled training rows;
- a test CSV containing the same input columns, without requiring the `aki` label.

Expected columns include `age`, `sex`, and numbered `creatinine_result_*` fields. The training file must also contain `aki` values encoded as `y` or `n`.

```bash
python model.py --input test.csv --output aki.csv
```

The output is a one-column CSV named `aki`, with `y`/`n` predictions.

## Run with Docker

```bash
docker build -t aki-detection .
docker run --rm \
  -v "$PWD:/data" \
  aki-detection \
  --input=/data/test.csv \
  --output=/data/aki.csv
```

The container expects `/data/training.csv` to be present through the mounted directory.

## Scope

This is a classical, interpretable ML pipeline rather than a large or generative model. Its relevance is in the experimental discipline: explicit split boundaries, out-of-fold operating-point selection, imbalance-aware metrics, deterministic seeds, and a holdout quality gate.
