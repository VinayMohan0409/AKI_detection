# Acute Kidney Injury (AKI) Prediction System

## Overview
This repository implements a machine learning system to predict acute kidney injury (AKI) from patient blood test data, developed in accordance with the coursework brief. The system prioritises recall to minimise false negatives, as missing AKI cases is clinically more harmful than raising false alarms.

The solution consists of:
- explore.ipynb – exploratory data analysis and feature understanding
- model.py – training, validation, and inference pipeline 

## explore.ipynb
The notebook is used to:
- analyse class imbalance (~79% non-AKI),
- motivate the use of derived features (e.g. baseline, recent value, and changes over time).
All design decisions in model.py are motivated by findings in this notebook.

## model.py
- Feature engineering.
- For each patient, the model extracts robust summary features from variable-length creatinine histories, including:
baseline and most recent values, min, max, mean, standard deviation, and range, absolute and relative change from baseline,
number of available tests, along with age and binary-encoded sex. Missing values are handled explicitly.

## Model
A logistic regression classifier is used within a preprocessing pipeline. This model was chosen for its:
- stable probabilistic outputs,
- interpretability,
- suitability for tabular clinical data,

## Recall prioritisation and evaluation
Recall is prioritised in three ways:
- Metric choice: model selection is based on the F3 score, which heavily weights recall.
- Threshold optimisation: using 5-fold stratified cross-validation, out-of-fold predicted probabilities are collected and a decision threshold is chosen to maximise F3 on validation data.
- Locked operating point: the chosen threshold is fixed before evaluation on a held-out set and before generating final predictions.
This ensures recall-biased behaviour without test-set leakage and aligns validation and deployment decisions.

## Summary
This submission delivers a recall-focused AKI prediction system with explicit F3 optimisation, robust feature engineering, and a clean, leakage-free evaluation strategy, fully aligned with the coursework objectives.

### Justification of why the libraries used are reasonable and safe to use in an eventual production system is provided in requirements.txt