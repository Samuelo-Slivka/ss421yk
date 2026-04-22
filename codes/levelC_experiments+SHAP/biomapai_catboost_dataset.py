import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from collections import Counter


LABEL_COL = "study_ptorhc"
LABEL_MAP = {"mecfs": 1, "control": 0}


@dataclass
class RegressorConfig:
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 100
    loss_function: str = "RMSE"
    verbose: bool = False


@dataclass
class ClassifierConfig:
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 4
    l2_leaf_reg: float = 3.0
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    verbose: bool = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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

    # input tables are features x samples -> transpose to samples x features
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


def fit_catboost_regressors(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    reg_cfg: RegressorConfig,
    seed_base: int,
) -> List[CatBoostRegressor]:
    models = []

    for j in range(Y_train.shape[1]):
        model = CatBoostRegressor(
            loss_function=reg_cfg.loss_function,
            iterations=reg_cfg.iterations,
            learning_rate=reg_cfg.learning_rate,
            depth=reg_cfg.depth,
            l2_leaf_reg=reg_cfg.l2_leaf_reg,
            random_seed=seed_base + j,
            od_type="Iter",
            od_wait=reg_cfg.early_stopping_rounds,
            verbose=reg_cfg.verbose,
            allow_writing_files=False,
        )

        model.fit(
            X_train,
            Y_train[:, j],
            eval_set=(X_val, Y_val[:, j]),
            use_best_model=True,
        )
        models.append(model)

    return models


def predict_regressors(models: List[CatBoostRegressor], X: np.ndarray) -> np.ndarray:
    preds = []
    for m in models:
        preds.append(m.predict(X).reshape(-1, 1))
    return np.hstack(preds)


def fit_score_layer(
    X_scores_train: np.ndarray,
    y_train: np.ndarray,
    clf_cfg: ClassifierConfig,
    seed: int,
) -> CatBoostClassifier:
    clf = CatBoostClassifier(
        loss_function=clf_cfg.loss_function,
        eval_metric=clf_cfg.eval_metric,
        iterations=clf_cfg.iterations,
        learning_rate=clf_cfg.learning_rate,
        depth=clf_cfg.depth,
        l2_leaf_reg=clf_cfg.l2_leaf_reg,
        random_seed=seed,
        verbose=clf_cfg.verbose,
        allow_writing_files=False,
    )
    clf.fit(X_scores_train, y_train)
    return clf


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob),
        "auprc": average_precision_score(y_true, prob),
    }


def compute_shap_df(model: CatBoostRegressor, X_df: pd.DataFrame) -> pd.DataFrame:
    pool = Pool(X_df)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    # CatBoost returns n_samples x (n_features + 1), last col is expected value
    shap_values = shap_values[:, :-1]
    return pd.DataFrame(shap_values, index=X_df.index, columns=X_df.columns)


def find_common_feature(shap_scores: List[pd.DataFrame], best_n: int) -> List[str]:
    feature_lists = []
    for shap_df in shap_scores:
        top_features = (
            shap_df.abs()
            .sum()
            .nlargest(best_n)
            .index
            .to_list()
        )
        feature_lists.append(top_features)

    counter = Counter([f for sub in feature_lists for f in sub])
    common_features = [feature for feature, count in counter.items() if count > 1]
    return common_features


def paper_like_feature_selection(
    shap_scores: List[pd.DataFrame],
    all_feature_names: List[str],
) -> Dict[str, object]:
    # common features from top 20% across scores
    best_n_20 = round(len(all_feature_names) * 0.2)
    feature_common = find_common_feature(shap_scores, best_n_20)

    selected_by_score = {}
    union_selected = set()

    for score_idx, shap_data in enumerate(shap_scores):
        feature_top_n = (
            shap_data.abs()
            .sum()
            .nlargest(best_n_20)
            .index
            .to_list()
        )

        feature_long_tail = [f for f in feature_top_n if f not in feature_common]

        best_n_50 = round(len(all_feature_names) * 0.5)
        feature_top_50 = (
            shap_data.abs()
            .sum()
            .nlargest(best_n_50)
            .index
            .to_list()
        )

        feature_extra = []
        for f in all_feature_names:
            if f in feature_top_50 and f not in feature_common and f not in feature_long_tail:
                feature_extra.append(f)

        score_selected = list(dict.fromkeys(feature_common + feature_long_tail + feature_extra))
        selected_by_score[f"score_{score_idx}"] = score_selected
        union_selected.update(score_selected)

    return {
        "common_features": sorted(feature_common),
        "selected_by_score": selected_by_score,
        "union_selected": sorted(union_selected),
    }


def run_dataset_pipeline(
    data_dir: str,
    dataset: str,
    output_root: str,
    reg_cfg: RegressorConfig,
    clf_cfg: ClassifierConfig,
    seed: int,
    rus_repeats: int,
):
    rng = np.random.default_rng(seed)

    X = load_omics_X(data_dir, dataset)
    Y = load_scores_Y(data_dir)
    lab = load_labels(data_dir)
    X, Y, lab = align_XY(X, Y, lab)

    print(f"[INFO] Model family: CatBoost")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Aligned samples: {X.shape[0]}, features: {X.shape[1]}, scores: {Y.shape[1]}")
    print(f"[INFO] Label balance: control={int((lab==0).sum())}, patient={int((lab==1).sum())}")

    dataset_out = os.path.join(output_root, dataset)
    models_dir = os.path.join(dataset_out, "models")
    metrics_dir = os.path.join(dataset_out, "metrics")
    preds_dir = os.path.join(dataset_out, "predictions")
    shap_dir = os.path.join(dataset_out, "shap")
    sel_dir = os.path.join(dataset_out, "selected_features")

    for d in [dataset_out, models_dir, metrics_dir, preds_dir, shap_dir, sel_dir]:
        ensure_dir(d)

    X_np = X.to_numpy(dtype=np.float32)
    Y_np = Y.to_numpy(dtype=np.float32)
    lab_np = lab.to_numpy(dtype=int)

    # held-out 10%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, held_idx = next(sss.split(X_np, lab_np))

    X_train_all = X_np[train_idx]
    Y_train_all = Y_np[train_idx]
    lab_train_all = lab_np[train_idx]

    X_held = X_np[held_idx]
    Y_held = Y_np[held_idx]
    lab_held = lab_np[held_idx]

    idx_train_all = X.index[train_idx]
    idx_held = X.index[held_idx]

    # --------------------------
    # 10-fold CV
    # --------------------------
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cv_rows = []
    cv_pred_rows = []

    for fold, (tr, va) in enumerate(skf.split(X_train_all, lab_train_all), start=1):
        X_tr, Y_tr, lab_tr = X_train_all[tr], Y_train_all[tr], lab_train_all[tr]
        X_va, Y_va, lab_va = X_train_all[va], Y_train_all[va], lab_train_all[va]
        idx_tr = idx_train_all[tr]
        idx_va = idx_train_all[va]

        rep_score_preds_train = []
        rep_score_preds_val = []

        rep_mse_list = []
        rep_metric_list = []

        for rep in range(rus_repeats):
            keep = rus_indices(lab_tr, rng)
            X_trb, Y_trb, lab_trb = X_tr[keep], Y_tr[keep], lab_tr[keep]

            # inner validation for regressor early stopping
            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed + fold + rep)
            tr2, va2 = next(sss_inner.split(X_trb, lab_trb))

            X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
            X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

            reg_models = fit_catboost_regressors(
                X_tr2, Y_tr2, X_va2, Y_va2, reg_cfg, seed_base=seed + 10000 * fold + 100 * rep
            )

            Y_tr_pred = predict_regressors(reg_models, X_tr)
            Y_va_pred = predict_regressors(reg_models, X_va)

            score_layer = fit_score_layer(
                Y_tr_pred, lab_tr, clf_cfg, seed=seed + 20000 * fold + rep
            )
            val_prob = score_layer.predict_proba(Y_va_pred)[:, 1]

            rep_score_preds_train.append(Y_tr_pred)
            rep_score_preds_val.append(Y_va_pred)
            rep_mse_list.append(mean_squared_error(Y_va, Y_va_pred))
            rep_metric_list.append(binary_metrics(lab_va, val_prob))

        # average over repeats
        Y_va_pred_avg = np.mean(np.stack(rep_score_preds_val, axis=0), axis=0)

        metric_keys = list(rep_metric_list[0].keys())
        mean_metrics = {k: float(np.mean([m[k] for m in rep_metric_list])) for k in metric_keys}
        mean_mse = float(np.mean(rep_mse_list))

        cv_rows.append({
            "fold": fold,
            "mse": mean_mse,
            **mean_metrics
        })

        fold_pred_df = pd.DataFrame(
            Y_va_pred_avg,
            index=idx_va,
            columns=[f"score_pred_{i}" for i in range(Y.shape[1])]
        )
        fold_pred_df["true_label"] = lab_va
        fold_pred_df["fold"] = fold
        cv_pred_rows.append(fold_pred_df)

        print(
            f"[CV fold {fold:02d}] "
            f"MSE={mean_mse:.5f} | "
            f"AUC={mean_metrics['auc']:.4f}"
        )

    cv_metrics_df = pd.DataFrame(cv_rows)
    cv_metrics_df.to_csv(os.path.join(metrics_dir, f"{dataset}_cv_metrics.csv"), index=False)

    cv_pred_df = pd.concat(cv_pred_rows, axis=0)
    cv_pred_df.to_csv(os.path.join(preds_dir, f"{dataset}_cv_score_predictions.csv"))

    cv_summary = {
        "mse_mean": float(cv_metrics_df["mse"].mean()),
        "mse_std": float(cv_metrics_df["mse"].std(ddof=0)),
        "accuracy_mean": float(cv_metrics_df["accuracy"].mean()),
        "accuracy_std": float(cv_metrics_df["accuracy"].std(ddof=0)),
        "precision_mean": float(cv_metrics_df["precision"].mean()),
        "precision_std": float(cv_metrics_df["precision"].std(ddof=0)),
        "recall_mean": float(cv_metrics_df["recall"].mean()),
        "recall_std": float(cv_metrics_df["recall"].std(ddof=0)),
        "f1_mean": float(cv_metrics_df["f1"].mean()),
        "f1_std": float(cv_metrics_df["f1"].std(ddof=0)),
        "auc_mean": float(cv_metrics_df["auc"].mean()),
        "auc_std": float(cv_metrics_df["auc"].std(ddof=0)),
        "auprc_mean": float(cv_metrics_df["auprc"].mean()),
        "auprc_std": float(cv_metrics_df["auprc"].std(ddof=0)),
    }
    with open(os.path.join(metrics_dir, f"{dataset}_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n[CV summary]")
    print(f"Mean MSE: {cv_summary['mse_mean']:.5f} ± {cv_summary['mse_std']:.5f}")
    print(f"Mean AUC: {cv_summary['auc_mean']:.4f} ± {cv_summary['auc_std']:.4f}")

    # --------------------------
    # Final training on 90% train with RUS averaging
    # --------------------------
    final_train_score_preds = []
    final_held_score_preds = []
    held_prob_list = []

    # for SHAP aggregation across repeats
    shap_accumulators = {i: [] for i in range(Y.shape[1])}

    for rep in range(rus_repeats):
        keep_all = rus_indices(lab_train_all, rng)
        X_trb, Y_trb, lab_trb = X_train_all[keep_all], Y_train_all[keep_all], lab_train_all[keep_all]

        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed + rep)
        tr2, va2 = next(sss_inner.split(X_trb, lab_trb))
        X_tr2, Y_tr2 = X_trb[tr2], Y_trb[tr2]
        X_va2, Y_va2 = X_trb[va2], Y_trb[va2]

        reg_models = fit_catboost_regressors(
            X_tr2, Y_tr2, X_va2, Y_va2, reg_cfg, seed_base=seed + 50000 + 100 * rep
        )

        # save models
        rep_dir = os.path.join(models_dir, f"repeat_{rep:03d}")
        ensure_dir(rep_dir)
        for j, m in enumerate(reg_models):
            m.save_model(os.path.join(rep_dir, f"{dataset}_score_{j}.cbm"))

        # predictions
        Y_train_pred_rep = predict_regressors(reg_models, X_train_all)
        Y_held_pred_rep = predict_regressors(reg_models, X_held)

        final_train_score_preds.append(Y_train_pred_rep)
        final_held_score_preds.append(Y_held_pred_rep)

        score_layer = fit_score_layer(
            Y_train_pred_rep, lab_train_all, clf_cfg, seed=seed + 60000 + rep
        )
        score_layer.save_model(os.path.join(rep_dir, f"{dataset}_score_layer.cbm"))
        held_prob = score_layer.predict_proba(Y_held_pred_rep)[:, 1]
        held_prob_list.append(held_prob)

        # SHAP on full aligned dataset X (article-like explanation stage)
        X_full_df = X.copy()
        for j, m in enumerate(reg_models):
            shap_df = compute_shap_df(m, X_full_df)
            shap_accumulators[j].append(shap_df)

    Y_train_pred = np.mean(np.stack(final_train_score_preds, axis=0), axis=0)
    Y_held_pred = np.mean(np.stack(final_held_score_preds, axis=0), axis=0)
    held_prob_avg = np.mean(np.stack(held_prob_list, axis=0), axis=0)

    held_metrics = binary_metrics(lab_held, held_prob_avg)
    held_mse = float(mean_squared_error(Y_held, Y_held_pred))

    held_metrics_row = {
        "mse": held_mse,
        **held_metrics
    }
    pd.DataFrame([held_metrics_row]).to_csv(
        os.path.join(metrics_dir, f"{dataset}_heldout_metrics.csv"),
        index=False
    )

    held_pred_df = pd.DataFrame(
        Y_held_pred,
        index=idx_held,
        columns=[f"score_pred_{i}" for i in range(Y.shape[1])]
    )
    held_pred_df["true_label"] = lab_held
    held_pred_df["prob_mecfs"] = held_prob_avg
    held_pred_df.to_csv(os.path.join(preds_dir, f"{dataset}_heldout_score_predictions.csv"))

    print("\n[Held-out 10%]")
    print(f"Held-out MSE: {held_mse:.5f}")
    print(f"Held-out AUC: {held_metrics['auc']:.4f}")

    # --------------------------
    # SHAP aggregation + save
    # --------------------------
    normalized_rows = []
    std_rows = []
    shap_score_dfs = []

    for score_idx in range(Y.shape[1]):
        shap_avg_df = sum(shap_accumulators[score_idx]) / len(shap_accumulators[score_idx])
        shap_score_dfs.append(shap_avg_df)

        # save raw averaged SHAP df
        joblib.dump(
            shap_avg_df,
            os.path.join(shap_dir, f"{dataset}_score_{score_idx}.pkl")
        )
        shap_avg_df.to_csv(os.path.join(shap_dir, f"{dataset}_score_{score_idx}.csv"))

        abs_sum = shap_avg_df.abs().sum(axis=0)
        norm_abs_sum = abs_sum / abs_sum.sum() if abs_sum.sum() > 0 else abs_sum
        shap_std = shap_avg_df.std(axis=0)

        normalized_rows.append(norm_abs_sum.rename(f"score_{score_idx}"))
        std_rows.append(shap_std.rename(f"score_{score_idx}"))

    normalized_df = pd.DataFrame(normalized_rows)
    std_df = pd.DataFrame(std_rows)

    normalized_df.to_csv(os.path.join(shap_dir, f"Scores_Normalized_SHAP_{dataset}.csv"))
    std_df.to_csv(os.path.join(shap_dir, f"Scores_SHAP_std_{dataset}.csv"))

    # --------------------------
    # Paper-like feature selection
    # --------------------------
    selection = paper_like_feature_selection(shap_score_dfs, list(X.columns))

    pd.DataFrame({"feature": selection["common_features"]}).to_csv(
        os.path.join(sel_dir, f"{dataset}_common_features.csv"), index=False
    )

    union_selected = selection["union_selected"]
    pd.DataFrame({"feature": union_selected}).to_csv(
        os.path.join(sel_dir, f"{dataset}_selected_features.csv"), index=False
    )

    # selected by score
    selected_by_score_rows = []
    for score_name, features in selection["selected_by_score"].items():
        for f in features:
            selected_by_score_rows.append({"score": score_name, "feature": f})
    pd.DataFrame(selected_by_score_rows).to_csv(
        os.path.join(sel_dir, f"{dataset}_selected_by_score.csv"), index=False
    )

    with open(os.path.join(sel_dir, f"{dataset}_selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_input_features": X.shape[1],
                "n_common_features": len(selection["common_features"]),
                "n_union_selected": len(union_selected),
            },
            f,
            indent=2
        )

    print("\n[Feature selection]")
    print(f"Input features: {X.shape[1]}")
    print(f"Common features: {len(selection['common_features'])}")
    print(f"Selected features (union): {len(union_selected)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)   # allow also omics_new
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rus_repeats", type=int, default=100)

    # regressor
    parser.add_argument("--reg_iterations", type=int, default=2000)
    parser.add_argument("--reg_lr", type=float, default=0.03)
    parser.add_argument("--reg_depth", type=int, default=6)
    parser.add_argument("--reg_l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--reg_early_stopping_rounds", type=int, default=100)

    # classifier
    parser.add_argument("--clf_iterations", type=int, default=1000)
    parser.add_argument("--clf_lr", type=float, default=0.03)
    parser.add_argument("--clf_depth", type=int, default=4)
    parser.add_argument("--clf_l2_leaf_reg", type=float, default=3.0)

    args = parser.parse_args()

    reg_cfg = RegressorConfig(
        iterations=args.reg_iterations,
        learning_rate=args.reg_lr,
        depth=args.reg_depth,
        l2_leaf_reg=args.reg_l2_leaf_reg,
        early_stopping_rounds=args.reg_early_stopping_rounds,
    )

    clf_cfg = ClassifierConfig(
        iterations=args.clf_iterations,
        learning_rate=args.clf_lr,
        depth=args.clf_depth,
        l2_leaf_reg=args.clf_l2_leaf_reg,
    )

    run_dataset_pipeline(
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_root=args.output_root,
        reg_cfg=reg_cfg,
        clf_cfg=clf_cfg,
        seed=args.seed,
        rus_repeats=args.rus_repeats,
    )


if __name__ == "__main__":
    main()