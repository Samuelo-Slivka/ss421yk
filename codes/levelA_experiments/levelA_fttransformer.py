import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score


# ----------------------------
# Config (clean + explicit)
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
    patience: int = 30          # early stopping patience
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# IO helpers (same as sanity)
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

    # repo-style: rows=features, cols=samples  -> transpose to samples x features
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

    # require all 12 scores present
    valid = ~Y2.isna().any(axis=1)
    X2 = X2.loc[valid]
    Y2 = Y2.loc[valid]
    lab2 = lab2.loc[valid]

    # drop all-null features and fill remaining NaNs with 0
    X2 = X2.dropna(axis=1, how="all").fillna(0.0)
    return X2, Y2, lab2


# ----------------------------
# RUS (paper-style imbalance handling)
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
# Tabular Transformer (FT-Transformer inspired, minimal)
# ----------------------------

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class FTTransformerRegressor(nn.Module):
    """
    Numeric-only FT-Transformer style model with FEATURE IDENTITY.

    Key change vs. previous version:
    - adds a learnable embedding per feature (column identity),
      so the model can distinguish which omics marker it is seeing.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 12,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # value -> token embedding
        self.value_proj = nn.Linear(1, d_model)

        # NEW: feature identity embedding (one vector per column)
        self.feature_emb = nn.Embedding(n_features, d_model)

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_outputs)

        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        nn.init.normal_(self.feature_emb.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (B, F)
        B, F = x.shape
        assert F == self.n_features, "Input feature count mismatch."

        # value tokens
        xv = x.view(B, F, 1)                 # (B, F, 1)
        tok = self.value_proj(xv)            # (B, F, d_model)

        # add feature identity
        feat_ids = torch.arange(F, device=x.device)           # (F,)
        feat_e = self.feature_emb(feat_ids).unsqueeze(0)      # (1, F, d_model)
        tok = tok + feat_e                                    # (B, F, d_model)

        # prepend CLS
        cls = self.cls.expand(B, 1, self.d_model)             # (B, 1, d_model)
        seq = torch.cat([cls, tok], dim=1)                    # (B, 1+F, d_model)

        h = self.encoder(seq)                                 # (B, 1+F, d_model)
        cls_h = self.dropout(h[:, 0, :])                      # (B, d_model)
        out = self.head(cls_h)                                # (B, 12)
        return out

# ----------------------------
# Training + evaluation (CV + heldout + score-layer AUC)
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        outs.append(pred)
    return np.vstack(outs)


def fit_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig) -> nn.Module:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # validate
        model.eval()
        val_losses = []
        for xb, yb in val_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            pred = model(xb)
            val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_layer_auc(y_train_label: np.ndarray, y_train_scores_pred: np.ndarray,
                    y_test_label: np.ndarray, y_test_scores_pred: np.ndarray,
                    seed: int) -> float:
    """
    Score-layer pre AUC: nelineárny klasifikátor na 12 predikovaných score.
    GradientBoosting býva často lepší než logreg, keď je vzťah nelineárny.
    """
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


def run_levelA(data_dir: str, dataset: str, cfg: TrainConfig,
               d_model: int = 128, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.1,
               rus_repeats: int = 1):
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    X = load_omics_X(data_dir, dataset)
    Y = load_scores_Y(data_dir)
    lab = load_labels(data_dir)
    X, Y, lab = align_XY(X, Y, lab)

    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Aligned samples: {X.shape[0]}, features: {X.shape[1]}, scores: {Y.shape[1]}")
    print(f"[INFO] Label balance: control={int((lab==0).sum())}, patient={int((lab==1).sum())}")
    print(f"[INFO] Device: {cfg.device}")

    # numpy arrays
    X_np = X.to_numpy(dtype=np.float32)
    Y_np = Y.to_numpy(dtype=np.float32)
    lab_np = lab.to_numpy(dtype=int)

    # heldout 10%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=cfg.seed)
    train_idx, held_idx = next(sss.split(X_np, lab_np))

    X_train_all, Y_train_all, lab_train_all = X_np[train_idx], Y_np[train_idx], lab_np[train_idx]
    X_held, Y_held, lab_held = X_np[held_idx], Y_np[held_idx], lab_np[held_idx]

    # Standardize X using training split only (analogicky k normalizácii)
    x_scaler = StandardScaler()
    X_train_all = x_scaler.fit_transform(X_train_all)
    X_held = x_scaler.transform(X_held)

    # CV on 90%
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)

    fold_mse = []
    fold_auc = []

    for fold, (tr, va) in enumerate(skf.split(X_train_all, lab_train_all), start=1):
        X_tr, Y_tr, lab_tr = X_train_all[tr], Y_train_all[tr], lab_train_all[tr]
        X_va, Y_va, lab_va = X_train_all[va], Y_train_all[va], lab_train_all[va]

        # repeat RUS (paper sampled majority many times; tu je parameter rus_repeats)
        rep_mse = []
        rep_auc = []

        for rep in range(rus_repeats):
            keep = rus_indices(lab_tr, rng)
            X_trb, Y_trb, lab_trb = X_tr[keep], Y_tr[keep], lab_tr[keep]

            # internal train/val loaders (val is the CV fold validation)
            train_ds = TabDataset(X_trb, Y_trb)
            val_ds = TabDataset(X_va, Y_va)
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

            model = FTTransformerRegressor(
                n_features=X_train_all.shape[1],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                n_outputs=Y_train_all.shape[1],
            )

            model = fit_one(model, train_loader, val_loader, cfg)

            # predictions on train fold (for score-layer training) + val fold (for eval)
            tr_full_loader = DataLoader(TabDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=False)
            va_loader = DataLoader(TabDataset(X_va, Y_va), batch_size=cfg.batch_size, shuffle=False)

            Y_tr_pred = predict(model, tr_full_loader, cfg.device)
            Y_va_pred = predict(model, va_loader, cfg.device)

            mse = mean_squared_error(Y_va, Y_va_pred)
            auc = score_layer_auc(lab_tr, Y_tr_pred, lab_va, Y_va_pred, seed=cfg.seed + fold + rep)

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

    held_mse_list = []
    held_auc_list = []

    Y_train_pred_all = []
    Y_held_pred_all = []

    for rep in range(rus_repeats):
        keep_all = rus_indices(lab_train_all, rng)
        X_trb = X_train_all[keep_all]
        Y_trb = Y_train_all[keep_all]
        lab_trb = lab_train_all[keep_all]

        # internal validation split for early stopping
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed + rep)
        tr2, va2 = next(sss2.split(X_trb, lab_trb))

        X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
        X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

        train_loader = DataLoader(TabDataset(X_tr2, Y_tr2), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(TabDataset(X_va2, Y_va2), batch_size=cfg.batch_size, shuffle=False)

        model = FTTransformerRegressor(
            n_features=X_train_all.shape[1],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_outputs=Y_train_all.shape[1],
        )

        model = fit_one(model, train_loader, val_loader, cfg)

        # predictions
        train_full_loader = DataLoader(TabDataset(X_train_all, Y_train_all), batch_size=cfg.batch_size, shuffle=False)
        held_loader = DataLoader(TabDataset(X_held, Y_held), batch_size=cfg.batch_size, shuffle=False)

        Y_train_pred_rep = predict(model, train_full_loader, cfg.device)
        Y_held_pred_rep = predict(model, held_loader, cfg.device)

        Y_train_pred_all.append(Y_train_pred_rep)
        Y_held_pred_all.append(Y_held_pred_rep)

    # average predictions across repeats
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

    # model hyperparams (tuning-friendly)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training hyperparams
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=8e-3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=100)

    # imbalance handling
    ap.add_argument("--rus_repeats", type=int, default=1,
                    help="How many random undersampling repeats per fold (paper used many; 1-5 is practical).")

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
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        rus_repeats=args.rus_repeats,
    )


if __name__ == "__main__":
    main()