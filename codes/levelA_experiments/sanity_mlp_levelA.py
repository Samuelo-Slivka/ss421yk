import argparse
import os
import re
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

LABEL_COL = "study_ptorhc"   # v metadata.csv
LABEL_MAP = {
    "mecfs": 1,
    "control": 0,
}

# -----------------------------
# Helpers: loading + alignment
# -----------------------------

def _read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


def _infer_index_column(df: pd.DataFrame) -> str:
    # Common patterns in this repo: "Unnamed: 0" used as row names (features or sample IDs)
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    # Or first column is an ID-like column
    return df.columns[0]


def load_omics_X(data_dir: str, dataset: str) -> pd.DataFrame:
    """
    Loads one of: immune/kegg/metabolome/specie/quest/omics
    Expected format (repo-style): rows=features, columns=samples (SAMxxxx_tpY)
    Returns: X with rows=samples, cols=features
    """
    path = os.path.join(data_dir, f"{dataset}.csv")
    df = _read_table(path)

    idx_col = _infer_index_column(df)
    df = df.set_index(idx_col)

    # features x samples -> transpose to samples x features
    X = df.T

    # Clean sample ids (strip whitespace)
    X.index = X.index.astype(str).str.strip()
    X.columns = X.columns.astype(str).str.strip()

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    return X


def load_scores_Y(data_dir: str) -> pd.DataFrame:
    """
    Loads score.csv (12 clinical scores). Returns Y with rows=samples.
    """
    path = os.path.join(data_dir, "score.csv")
    df = _read_table(path)

    idx_col = _infer_index_column(df)
    df = df.set_index(idx_col)

    df.index = df.index.astype(str).str.strip()
    # Keep only numeric columns (scores)
    Y = df.apply(pd.to_numeric, errors="coerce")

    # Drop all-null columns just in case
    Y = Y.dropna(axis=1, how="all")
    return Y


def _normalize_label_values(series: pd.Series) -> pd.Series:
    """
    Converts typical metadata labels to 0/1 where:
    0 = control, 1 = patient/ME-CFS
    """
    s = series.copy()

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted([x for x in s.dropna().unique().tolist()])
        if set(uniq).issubset({0, 1}):
            return s.astype(int)

    # String mapping
    s = s.astype(str).str.strip().str.lower()

    def map_val(v: str) -> Optional[int]:
        if v in {"0", "control", "healthy", "hc", "h"}:
            return 0
        if v in {"1", "patient", "case", "mecfs", "me/cfs", "cfs"}:
            return 1
        # heuristic contains
        if "control" in v or "healthy" in v:
            return 0
        if "patient" in v or "mecfs" in v or "me/cfs" in v or "cfs" in v or "case" in v:
            return 1
        return None

    mapped = s.map(map_val)
    if mapped.isna().any():
        # Try numeric strings
        maybe_num = pd.to_numeric(s, errors="coerce")
        if maybe_num.notna().all():
            uniq = set(maybe_num.unique().tolist())
            if uniq.issubset({0, 1}):
                return maybe_num.astype(int)
        # If still ambiguous, raise with diagnostics
        examples = series.dropna().astype(str).unique().tolist()[:10]
        raise ValueError(
            "Could not normalize labels from metadata. "
            f"Examples of raw label values: {examples}\n"
            "Edit _normalize_label_values() mapping to match your metadata column values."
        )
    return mapped.astype(int)


def load_labels_from_metadata(data_dir: str) -> pd.Series:
    path = os.path.join(data_dir, "metadata.csv")
    df = _read_table(path)

    idx_col = _infer_index_column(df)
    df = df.set_index(idx_col)
    df.index = df.index.astype(str).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected label column '{LABEL_COL}' in metadata.csv, but not found. "
            f"Available columns: {list(df.columns)[:80]} ..."
        )

    s = df[LABEL_COL].astype(str).str.strip().str.lower()
    bad = sorted(set(s.unique()) - set(LABEL_MAP.keys()))
    if bad:
        raise ValueError(
            f"Unexpected label values in '{LABEL_COL}': {bad}. "
            f"Allowed: {sorted(LABEL_MAP.keys())}"
        )

    return s.map(LABEL_MAP).astype(int)

def align_XY(X: pd.DataFrame, Y: pd.DataFrame, y_label: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Align on common sample IDs across X, Y, and labels.
    """
    common = X.index.intersection(Y.index).intersection(y_label.index)
    if len(common) < 50:
        raise ValueError(f"Too few aligned samples ({len(common)}). Check sample IDs between files.")

    X2 = X.loc[common].copy()
    Y2 = Y.loc[common].copy()
    lab2 = y_label.loc[common].copy()

    # Drop samples with missing scores
    valid = ~Y2.isna().any(axis=1)
    X2 = X2.loc[valid]
    Y2 = Y2.loc[valid]
    lab2 = lab2.loc[valid]

    # Drop features that are all NaN
    X2 = X2.dropna(axis=1, how="all")
    # Simple impute: fill remaining NaNs with 0 (paper uses various imputations; for sanity check this is fine)
    X2 = X2.fillna(0.0)

    return X2, Y2, lab2


# -----------------------------
# Protocol: heldout + CV + RUS
# -----------------------------

def rus_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random undersampling of majority class to match minority size.
    y is 0/1.
    """
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return np.arange(len(y))

    if len(idx0) > len(idx1):
        keep0 = rng.choice(idx0, size=len(idx1), replace=False)
        keep = np.concatenate([keep0, idx1])
    else:
        keep1 = rng.choice(idx1, size=len(idx0), replace=False)
        keep = np.concatenate([idx0, keep1])

    rng.shuffle(keep)
    return keep


def build_mlp(seed: int) -> Pipeline:
    # Small MLP: fast sanity check, not tuned.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,            # L2
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1
        ))
    ])


def score_layer_auc(y_true_label: np.ndarray, y_pred_scores: np.ndarray, seed: int) -> float:
    """
    Mimics "ScoreLayer": logistic regression on predicted scores -> AUC for disease/control
    """
    clf = LogisticRegression(
        solver="liblinear",
        random_state=seed,
        max_iter=200
    )
    clf.fit(y_pred_scores, y_true_label)
    prob = clf.predict_proba(y_pred_scores)[:, 1]
    return roc_auc_score(y_true_label, prob)


def run_levelA_sanity(data_dir: str, dataset: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    X = load_omics_X(data_dir, dataset)
    Y = load_scores_Y(data_dir)
    labels = load_labels_from_metadata(data_dir)
    X, Y, labels = align_XY(X, Y, labels)

    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Aligned samples: {X.shape[0]}, features: {X.shape[1]}, scores: {Y.shape[1]}")
    print(f"[INFO] Label balance: control={int((labels==0).sum())}, patient={int((labels==1).sum())}")

    # Held-out 10%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, held_idx = next(sss.split(X, labels))

    X_train_all = X.iloc[train_idx].to_numpy(dtype=np.float32)
    Y_train_all = Y.iloc[train_idx].to_numpy(dtype=np.float32)
    lab_train_all = labels.iloc[train_idx].to_numpy(dtype=int)

    X_held = X.iloc[held_idx].to_numpy(dtype=np.float32)
    Y_held = Y.iloc[held_idx].to_numpy(dtype=np.float32)
    lab_held = labels.iloc[held_idx].to_numpy(dtype=int)

    # 10-fold CV on 90%
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    fold_mse = []
    fold_auc = []

    for fold, (tr, va) in enumerate(skf.split(X_train_all, lab_train_all), start=1):
        X_tr, Y_tr, lab_tr = X_train_all[tr], Y_train_all[tr], lab_train_all[tr]
        X_va, Y_va, lab_va = X_train_all[va], Y_train_all[va], lab_train_all[va]

        # RUS on training fold only
        keep = rus_indices(lab_tr, rng)
        X_trb, Y_trb, lab_trb = X_tr[keep], Y_tr[keep], lab_tr[keep]

        model = build_mlp(seed + fold)
        model.fit(X_trb, Y_trb)

        Y_va_pred = model.predict(X_va)
        mse = mean_squared_error(Y_va, Y_va_pred)
        fold_mse.append(mse)

        # ScoreLayer AUC: train logreg on predicted scores of the SAME val set isn't correct.
        # We mimic paper-style: score-layer is an evaluation head.
        # For CV sanity: fit score-layer on TRAIN predictions, evaluate on VAL predictions.
        Y_tr_pred = model.predict(X_tr)
        Y_va_pred = model.predict(X_va)

        clf = LogisticRegression(solver="liblinear", random_state=seed + fold, max_iter=200)
        clf.fit(Y_tr_pred, lab_tr)
        prob_va = clf.predict_proba(Y_va_pred)[:, 1]
        auc = roc_auc_score(lab_va, prob_va)
        fold_auc.append(auc)

        print(f"[CV fold {fold:02d}] MSE={mse:.5f} | AUC={auc:.4f}")

    print("\n[CV summary]")
    print(f"Mean MSE: {np.mean(fold_mse):.5f} ± {np.std(fold_mse):.5f}")
    print(f"Mean AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

    # Final train on full 90% (with one RUS) and evaluate on held-out 10%
    keep_all = rus_indices(lab_train_all, rng)
    X_trb, Y_trb, lab_trb = X_train_all[keep_all], Y_train_all[keep_all], lab_train_all[keep_all]

    final_model = build_mlp(seed)
    final_model.fit(X_trb, Y_trb)

    Y_held_pred = final_model.predict(X_held)
    held_mse = mean_squared_error(Y_held, Y_held_pred)

    # Score-layer: fit on train predictions, evaluate on held predictions
    Y_train_pred = final_model.predict(X_train_all)
    Y_held_pred = final_model.predict(X_held)

    clf = LogisticRegression(solver="liblinear", random_state=seed, max_iter=200)
    clf.fit(Y_train_pred, lab_train_all)
    prob_held = clf.predict_proba(Y_held_pred)[:, 1]
    held_auc = roc_auc_score(lab_held, prob_held)

    print("\n[Held-out 10%]")
    print(f"Held-out MSE: {held_mse:.5f}")
    print(f"Held-out AUC: {held_auc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing *.csv files (codes/AI/input style).")
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["immune", "kegg", "metabolome", "specie", "quest", "omics"],
                    help="Which omics table to use.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_levelA_sanity(args.data_dir, args.dataset, args.seed)


if __name__ == "__main__":
    main()