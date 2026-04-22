import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

import torch
from pytorch_tabnet.tab_model import TabNetRegressor


# ----------------------------
# Config
# ----------------------------
LABEL_COL = "study_ptorhc"
LABEL_MAP = {"mecfs": 1, "control": 0}


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 8e-3
    epochs: int = 500
    patience: int = 100  # TabNet uses early stopping


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
        raise ValueError(f"Unexpected label values in '{LABEL_COL}': {bad}. Allowed: {sorted(LABEL_MAP.keys())}")

    return s.map(LABEL_MAP).astype(int)


def align_XY(X: pd.DataFrame, Y: pd.DataFrame, lab: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    common = X.index.intersection(Y.index).intersection(lab.index)
    if len(common) < 50:
        raise ValueError(f"Too few aligned samples ({len(common)}). Check sample IDs.")

    X2 = X.loc[common].copy()
    Y2 = Y.loc[common].copy()
    lab2 = lab.loc[common].copy()

    valid = ~Y2.isna().any(axis=1)
    X2 = X2.loc[valid]
    Y2 = Y2.loc[valid]
    lab2 = lab2.loc[valid]

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
# TabNet fit/predict
# ----------------------------
def fit_tabnet(X_tr: np.ndarray, Y_tr: np.ndarray,
               X_va: np.ndarray, Y_va: np.ndarray,
               cfg: TrainConfig,
               tabnet_params: dict) -> TabNetRegressor:
    model = TabNetRegressor(**tabnet_params)

    # TabNet expects float32
    X_tr = X_tr.astype(np.float32)
    Y_tr = Y_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    Y_va = Y_va.astype(np.float32)

    model.fit(
        X_train=X_tr, y_train=Y_tr,
        eval_set=[(X_va, Y_va)],
        eval_name=["val"],
        eval_metric=["rmse"],   # regression metric
        max_epochs=cfg.epochs,
        patience=cfg.patience,
        batch_size=cfg.batch_size,
        virtual_batch_size=min(cfg.batch_size, 128),
        num_workers=0,
        drop_last=True
    )
    return model


def predict_tabnet(model: TabNetRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X.astype(np.float32))


# ----------------------------
# Main experiment
# ----------------------------
def run_levelA(data_dir: str, dataset: str, cfg: TrainConfig,
               rus_repeats: int,
               n_d: int, n_a: int, n_steps: int, gamma: float, lambda_sparse: float,
               tabnet_dropout: float):
    rng = np.random.default_rng(cfg.seed)

    X = load_omics_X(data_dir, dataset)
    Y = load_scores_Y(data_dir)
    lab = load_labels(data_dir)
    X, Y, lab = align_XY(X, Y, lab)

    print(f"[INFO] Model: TabNet")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Aligned samples: {X.shape[0]}, features: {X.shape[1]}, scores: {Y.shape[1]}")
    print(f"[INFO] Label balance: control={int((lab==0).sum())}, patient={int((lab==1).sum())}")

    X_np = X.to_numpy(dtype=np.float32)
    Y_np = Y.to_numpy(dtype=np.float32)
    lab_np = lab.to_numpy(dtype=int)

    # heldout 10%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.seed)
    train_idx, held_idx = next(sss.split(X_np, lab_np))

    X_train_all, Y_train_all, lab_train_all = X_np[train_idx], Y_np[train_idx], lab_np[train_idx]
    X_held, Y_held, lab_held = X_np[held_idx], Y_np[held_idx], lab_np[held_idx]

    # standardize X based on 90% train
    x_scaler = StandardScaler()
    X_train_all = x_scaler.fit_transform(X_train_all)
    X_held = x_scaler.transform(X_held)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)

    tabnet_params = dict(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=cfg.lr, weight_decay=cfg.weight_decay),
        mask_type="entmax",
        seed=cfg.seed,
        verbose=0
    )

    fold_mse, fold_auc = [], []

    for fold, (tr, va) in enumerate(skf.split(X_train_all, lab_train_all), start=1):
        X_tr, Y_tr, lab_tr = X_train_all[tr], Y_train_all[tr], lab_train_all[tr]
        X_va, Y_va, lab_va = X_train_all[va], Y_train_all[va], lab_train_all[va]

        rep_mse, rep_auc = [], []
        for rep in range(rus_repeats):
            keep = rus_indices(lab_tr, rng)
            X_trb, Y_trb = X_tr[keep], Y_tr[keep]
            lab_trb = lab_tr[keep]

            # small internal val for early stopping (within the RUS train set)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed + fold + rep)
            tr2, va2 = next(sss2.split(X_trb, lab_trb))
            X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
            X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

            model = fit_tabnet(X_tr2, Y_tr2, X_va2, Y_va2, cfg, tabnet_params)

            # preds for score-layer
            Y_tr_pred = predict_tabnet(model, X_tr)  # full fold train (not undersampled)
            Y_va_pred = predict_tabnet(model, X_va)

            mse = mean_squared_error(Y_va, Y_va_pred)
            auc = score_layer_auc(lab_tr, Y_tr_pred, lab_va, Y_va_pred, seed=cfg.seed + 1000*fold + rep)

            rep_mse.append(mse)
            rep_auc.append(auc)

        fold_mse.append(float(np.mean(rep_mse)))
        fold_auc.append(float(np.mean(rep_auc)))
        print(f"[CV fold {fold:02d}] MSE={fold_mse[-1]:.5f} | AUC={fold_auc[-1]:.4f}")

    print("\n[CV summary]")
    print(f"Mean MSE: {np.mean(fold_mse):.5f} ± {np.std(fold_mse):.5f}")
    print(f"Mean AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

    # --------------------------
    # Final training: average preds over RUS repeats (paper-like)
    # --------------------------
    Y_train_pred_all, Y_held_pred_all = [], []

    for rep in range(rus_repeats):
        keep_all = rus_indices(lab_train_all, rng)
        X_trb, Y_trb, lab_trb = X_train_all[keep_all], Y_train_all[keep_all], lab_train_all[keep_all]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed + rep)
        tr2, va2 = next(sss2.split(X_trb, lab_trb))
        X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
        X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

        model = fit_tabnet(X_tr2, Y_tr2, X_va2, Y_va2, cfg, tabnet_params)

        Y_train_pred_all.append(predict_tabnet(model, X_train_all))
        Y_held_pred_all.append(predict_tabnet(model, X_held))

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

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=8e-3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--rus_repeats", type=int, default=100)

    # TabNet hyperparams
    ap.add_argument("--n_d", type=int, default=32)
    ap.add_argument("--n_a", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--lambda_sparse", type=float, default=1e-4)
    ap.add_argument("--tabnet_dropout", type=float, default=0.0)  # placeholder (TabNet has internal regularization)

    args = ap.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    run_levelA(
        data_dir=args.data_dir,
        dataset=args.dataset,
        cfg=cfg,
        rus_repeats=args.rus_repeats,
        n_d=args.n_d, n_a=args.n_a, n_steps=args.n_steps,
        gamma=args.gamma, lambda_sparse=args.lambda_sparse,
        tabnet_dropout=args.tabnet_dropout
    )


if __name__ == "__main__":
    main()