import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

import lightgbm as lgb


# ----------------------------
# Config
# ----------------------------

LABEL_COL = "study_ptorhc"
LABEL_MAP = {"mecfs": 1, "control": 0}


@dataclass
class TrainConfig:
    seed: int = 0
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 100


# ----------------------------
# IO helpers
# ----------------------------

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _infer_index_col(df: pd.DataFrame) -> str:
    return "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]


def load_omics_X(data_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{dataset}.csv")
    df = _read_csv(path)
    idx = _infer_index_col(df)
    df = df.set_index(idx)

    # repo-style: rows=features, cols=samples -> transpose
    X = df.T
    X.index = X.index.astype(str).str.strip()
    X.columns = X.columns.astype(str).str.strip()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def load_scores_Y(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "score.csv")
    df = _read_csv(path)
    idx = _infer_index_col(df)
    df = df.set_index(idx)

    df.index = df.index.astype(str).str.strip()
    Y = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return Y


def load_labels(data_dir: str) -> pd.Series:
    path = os.path.join(data_dir, "metadata.csv")
    df = _read_csv(path)
    idx = _infer_index_col(df)
    df = df.set_index(idx)
    df.index = df.index.astype(str).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' in metadata.csv. Found: {list(df.columns)[:60]} ...")

    s = df[LABEL_COL].astype(str).str.strip().str.lower()
    bad = sorted(set(s.unique()) - set(LABEL_MAP.keys()))
    if bad:
        raise ValueError(
            f"Unexpected label values in '{LABEL_COL}': {bad}. Allowed: {sorted(LABEL_MAP.keys())}"
        )

    return s.map(LABEL_MAP).astype(int)


def align_XY(X: pd.DataFrame, Y: pd.DataFrame, lab: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    common = X.index.intersection(Y.index).intersection(lab.index)
    if len(common) < 50:
        raise ValueError(f"Too few aligned samples ({len(common)}). Check sample IDs.")

    X2 = X.loc[common].copy()
    Y2 = Y.loc[common].copy()
    lab2 = lab.loc[common].copy()

    # require all 12 scores present
    valid = ~Y2.isna().any(axis=1)
    X2 = X2.loc[valid]
    Y2 = Y2.loc[valid]
    lab2 = lab2.loc[valid]

    # drop all-null features and fill remaining NaNs with 0
    X2 = X2.dropna(axis=1, how="all").fillna(0.0)
    return X2, Y2, lab2


# ----------------------------
# RUS
# ----------------------------

def rus_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


# ----------------------------
# Score-layer AUC
# ----------------------------

def score_layer_auc(y_train_label: np.ndarray, y_train_scores_pred: np.ndarray,
                    y_test_label: np.ndarray, y_test_scores_pred: np.ndarray,
                    seed: int) -> float:
    clf = GradientBoostingClassifier(
        random_state=seed,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9
    )
    clf.fit(y_train_scores_pred, y_train_label)
    prob = clf.predict_proba(y_test_scores_pred)[:, 1]
    return roc_auc_score(y_test_label, prob)


# ----------------------------
# LightGBM multi-output helpers
# ----------------------------

def fit_lightgbm_multi(X_tr: np.ndarray, Y_tr: np.ndarray,
                       X_va: np.ndarray, Y_va: np.ndarray,
                       cfg: TrainConfig) -> List[lgb.LGBMRegressor]:
    models = []

    for j in range(Y_tr.shape[1]):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.seed + j,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(
            X_tr, Y_tr[:, j],
            eval_set=[(X_va, Y_va[:, j])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
        )

        models.append(model)

    return models


def predict_lightgbm_multi(models: List[lgb.LGBMRegressor], X: np.ndarray) -> np.ndarray:
    preds = []
    for model in models:
        p = model.predict(X).reshape(-1, 1)
        preds.append(p)
    return np.hstack(preds)


# ----------------------------
# Main experiment
# ----------------------------

def run_levelA(data_dir: str, dataset: str, cfg: TrainConfig, rus_repeats: int):
    rng = np.random.default_rng(cfg.seed)

    X = load_omics_X(data_dir, dataset)
    Y = load_scores_Y(data_dir)
    lab = load_labels(data_dir)
    X, Y, lab = align_XY(X, Y, lab)

    print(f"[INFO] Model: LightGBM (12x regression)")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Aligned samples: {X.shape[0]}, features: {X.shape[1]}, scores: {Y.shape[1]}")
    print(f"[INFO] Label balance: control={int((lab==0).sum())}, patient={int((lab==1).sum())}")

    X_np = X.to_numpy(dtype=np.float32)
    Y_np = Y.to_numpy(dtype=np.float32)
    lab_np = lab.to_numpy(dtype=int)

    # held-out 10%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.seed)
    train_idx, held_idx = next(sss.split(X_np, lab_np))

    X_train_all, Y_train_all, lab_train_all = X_np[train_idx], Y_np[train_idx], lab_np[train_idx]
    X_held, Y_held, lab_held = X_np[held_idx], Y_np[held_idx], lab_np[held_idx]

    # standardize X on 90% train
    x_scaler = StandardScaler()
    X_train_all = x_scaler.fit_transform(X_train_all)
    X_held = x_scaler.transform(X_held)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)

    fold_mse = []
    fold_auc = []

    for fold, (tr, va) in enumerate(skf.split(X_train_all, lab_train_all), start=1):
        X_tr, Y_tr, lab_tr = X_train_all[tr], Y_train_all[tr], lab_train_all[tr]
        X_va, Y_va, lab_va = X_train_all[va], Y_train_all[va], lab_train_all[va]

        rep_mse = []
        rep_auc = []

        for rep in range(rus_repeats):
            keep = rus_indices(lab_tr, rng)
            X_trb, Y_trb, lab_trb = X_tr[keep], Y_tr[keep], lab_tr[keep]

            # internal validation inside RUS train set
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed + fold + rep)
            tr2, va2 = next(sss2.split(X_trb, lab_trb))
            X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
            X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

            models = fit_lightgbm_multi(X_tr2, Y_tr2, X_va2, Y_va2, cfg)

            # score-layer preds
            Y_tr_pred = predict_lightgbm_multi(models, X_tr)
            Y_va_pred = predict_lightgbm_multi(models, X_va)

            mse = mean_squared_error(Y_va, Y_va_pred)
            auc = score_layer_auc(lab_tr, Y_tr_pred, lab_va, Y_va_pred, seed=cfg.seed + 1000 * fold + rep)

            rep_mse.append(mse)
            rep_auc.append(auc)

        fold_mse.append(float(np.mean(rep_mse)))
        fold_auc.append(float(np.mean(rep_auc)))

        print(f"[CV fold {fold:02d}] MSE={fold_mse[-1]:.5f} | AUC={fold_auc[-1]:.4f}")

    print("\n[CV summary]")
    print(f"Mean MSE: {np.mean(fold_mse):.5f} ± {np.std(fold_mse):.5f}")
    print(f"Mean AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

    # ============================================
    # FINAL TRAINING WITH RUS AVERAGING (paper-like)
    # ============================================

    Y_train_pred_all = []
    Y_held_pred_all = []

    for rep in range(rus_repeats):
        keep_all = rus_indices(lab_train_all, rng)
        X_trb = X_train_all[keep_all]
        Y_trb = Y_train_all[keep_all]
        lab_trb = lab_train_all[keep_all]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed + rep)
        tr2, va2 = next(sss2.split(X_trb, lab_trb))

        X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
        X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

        models = fit_lightgbm_multi(X_tr2, Y_tr2, X_va2, Y_va2, cfg)

        Y_train_pred_rep = predict_lightgbm_multi(models, X_train_all)
        Y_held_pred_rep = predict_lightgbm_multi(models, X_held)

        Y_train_pred_all.append(Y_train_pred_rep)
        Y_held_pred_all.append(Y_held_pred_rep)

    Y_train_pred = np.mean(np.stack(Y_train_pred_all, axis=0), axis=0)
    Y_held_pred = np.mean(np.stack(Y_held_pred_all, axis=0), axis=0)

    held_mse = mean_squared_error(Y_held, Y_held_pred)
    held_auc = score_layer_auc(lab_train_all, Y_train_pred, lab_held, Y_held_pred, seed=cfg.seed)

    print("\n[Held-out 10%]")
    print(f"Held-out MSE: {held_mse:.5f}")
    print(f"Held-out AUC: {held_auc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["immune", "kegg", "metabolome", "specie", "quest", "omics"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rus_repeats", type=int, default=100)

    # LightGBM params
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--min_child_samples", type=int, default=20)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=100)

    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds
    )

    run_levelA(
        data_dir=args.data_dir,
        dataset=args.dataset,
        cfg=cfg,
        rus_repeats=args.rus_repeats
    )


if __name__ == "__main__":
    main()