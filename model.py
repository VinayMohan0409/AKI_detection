#!/usr/bin/env python3

import argparse

import re
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score


SEED = 36
MAP_SEX = {"M": 1, "F": 0}

def make_features(df: pd.DataFrame, age_col: str = "age", sex_col: str = "sex") -> pd.DataFrame:
    """
    Feature engineering for AKI prediction using ML.
    """
    # Find creatinine result columns (e.g. creatinine_result_0, creatinine_result_1, ...)
    res_cols = [c for c in df.columns if re.search(r"creatinine_result_\d+$", c, re.I)]
    res_cols = sorted(res_cols, key=lambda c: int(re.findall(r"(\d+)$", c)[0])) # For robustness against shuffled columns

    
    if not res_cols:
        raise ValueError(
            "No creatinine_result_* columns found (expected creatinine_result_0, creatinine_result_1, ...)."
        )

    # Saves the creatinine values to V (2D array)
    V = df[res_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(V)

    if np.all(~valid):
        raise ValueError(
            "All creatinine_result_* values are missing or non-numeric after conversion."
        )

    n = len(df)
    baseline = np.full(n, np.nan, dtype=float) # first available creatinine result for all patients
    index = np.full(n, np.nan, dtype=float) # latest creatinine result for all patients

    for i in range(n):
        idx = np.where(valid[i])[0] # index of valid creatinine measurements for patient i
        if idx.size == 0:
            continue
        baseline[i] = V[i, idx.min()] 
        index[i] = V[i, idx.max()] 

    X = pd.DataFrame(index=df.index)

    X["age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col in df.columns else np.nan
    X["sex_binary"] = (
        df[sex_col].astype(str).str.strip().str.upper().map(MAP_SEX) # Map M to 1 and F to 0 to be used by model
        if sex_col in df.columns
        else np.nan
    )

    # Summary stats from creatinine results
    X["n_creatinine_tests"] = valid.sum(axis=1).astype(float)
    X["cr_baseline"] = baseline
    X["cr_index"] = index

    Vm = np.where(valid, V, np.nan)

    X["cr_min"] = np.nanmin(Vm, axis=1)
    X["cr_max"] = np.nanmax(Vm, axis=1)
    X["cr_mean"] = np.nanmean(Vm, axis=1)
    X["cr_std"] = np.nanstd(Vm, axis=1)
    X["cr_range"] = X["cr_max"] - X["cr_min"]

    X["change_from_baseline"] = X["cr_index"] - X["cr_baseline"]
    rel = X["change_from_baseline"] / X["cr_baseline"]
    X["rel_change_from_baseline"] = rel.replace([np.inf, -np.inf], np.nan)

    return X

def parse_labels(df: pd.DataFrame, label_col: str = "aki") -> np.ndarray:
    """
    Map labels y/n to 1/0. Unknown values becomes NaN and will be filtered out.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input CSV.")
    s = df[label_col].astype(str).str.strip().str.lower()
    y = s.map({"y": 1, "n": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), label_col].astype(str).unique()[:5]
        raise ValueError(f"Invalid labels in '{label_col}'. Expected 'y'/'n'. Examples: {bad}")
    return y.to_numpy(dtype=int)

def best_threshold_for_fbeta(y_true: np.ndarray, p_pos: np.ndarray, beta: float = 3.0):
    """
    Selects the decision threshold that maximises the F3 score.

    The classifier outputs probabilities; threshold optimisation is required because
    the default threshold (0.5) does not, in general, maximise F3. Using F3
    prioritises recall over precision, reducing false negatives, as each false negative
    represents a deteriorating patient who may be overlooked
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f = -1.0

    for t in thresholds:
        y_pred = (p_pos >= t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta)
        if f > best_f:
            best_f = f
            best_t = float(t)

    return best_t, best_f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()

    TRAIN_PATH = "/data/training.csv"
    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = "training.csv"

    try:
        df_train = pd.read_csv(TRAIN_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read training file {TRAIN_PATH}: {e}") from e

    X_train = make_features(df_train).to_numpy()
    y_train = parse_labels(df_train)

    # Internal evaluation: 90/10 training/holdout + 5-fold CV on the 90%
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=SEED
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # In case value missing in X_train
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]) # Prevents data leakage during cross-validation.

    oof_p = np.zeros(len(y_trainval)) # Store out of fold probabilities

    train_scores = []

    for tr_idx, val_idx in cv.split(X_trainval, y_trainval):
        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr = y_trainval[tr_idx]

        pipe.fit(X_tr, y_tr)

        y_tr_pred = pipe.predict(X_tr)
        train_scores.append(fbeta_score(y_tr, y_tr_pred, beta=3))

        p_val = pipe.predict_proba(X_val)[:, 1]
        oof_p[val_idx] = p_val

    print(f"Training F3 score (CV mean ± std): {np.mean(train_scores):.3f} ± {np.std(train_scores):.3f}")

    # Threshold tuning using cross validation predictions
    best_t, best_cv_f3 = best_threshold_for_fbeta(y_trainval, oof_p, beta=3.0)
    print(f"Chosen threhsold from CV validation: {best_t:.2f}; F3_score: {best_cv_f3:.3f}")

    # Evaluate held-out set using threshold found
    pipe.fit(X_trainval, y_trainval)
    p_hold = pipe.predict_proba(X_holdout)[:, 1]
    y_hold_pred = (p_hold >= best_t).astype(int)

    hold_f3 = fbeta_score(y_holdout, y_hold_pred, beta=3)
    print(f"Holdout F3 score (10% of training.csv): {hold_f3:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_hold_pred, labels=[0, 1]).ravel()
    print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

    prec = precision_score(y_holdout, y_hold_pred, zero_division=0)
    rec  = recall_score(y_holdout, y_hold_pred, zero_division=0)

    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")

    MIN_EXPECTED_F3 = 0.7 # An F3 score greater than 0.7 would be required for the system to be deployed in practice.

    if hold_f3 < MIN_EXPECTED_F3:
        raise RuntimeError(
            f"Model quality gate failed: holdout F3={hold_f3:.3f} < {MIN_EXPECTED_F3:.2f}"
        )

    # Load test set and predict
    try:
        df_test = pd.read_csv(flags.input)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test file not found: {flags.input}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read test file '{flags.input}': {e}") from e

    X_test = make_features(df_test).to_numpy()

    p_test = pipe.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= best_t).astype(int)


    out = pd.DataFrame({"aki": np.where(y_pred == 1, "y", "n")})
    out.to_csv(flags.output, index=False)


if __name__ == "__main__":
    main()

