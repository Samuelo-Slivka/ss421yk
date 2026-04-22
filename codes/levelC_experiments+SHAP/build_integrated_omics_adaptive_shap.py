import argparse
import os
from typing import List

import numpy as np
import pandas as pd


DATASETS_DEFAULT = ["immune", "specie", "kegg", "metabolome", "quest"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _infer_index_col(df: pd.DataFrame) -> str:
    return "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]


def load_dataset_X(data_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{dataset}.csv")
    df = _read_csv(path)
    idx = _infer_index_col(df)
    df = df.set_index(idx)

    X = df.T
    X.index = X.index.astype(str).str.strip()
    X.columns = X.columns.astype(str).str.strip()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def load_normalized_shap(output_root: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(
        output_root,
        dataset,
        "shap",
        f"Scores_Normalized_SHAP_{dataset}.csv"
    )
    return pd.read_csv(path, index_col=0)


def compute_feature_statistics(
    shap_df: pd.DataFrame,
    top_fraction_per_score: float = 0.2
) -> pd.DataFrame:
    n_features = shap_df.shape[1]
    top_k = max(1, round(n_features * top_fraction_per_score))

    importance_sum = shap_df.sum(axis=0)
    importance_mean = shap_df.mean(axis=0)

    freq_counter = pd.Series(0, index=shap_df.columns, dtype=float)

    for _, row in shap_df.iterrows():
        top_feats = row.sort_values(ascending=False).head(top_k).index
        freq_counter.loc[top_feats] += 1

    weighted_score = importance_sum * np.log1p(freq_counter)

    stats_df = pd.DataFrame({
        "feature": shap_df.columns.astype(str),
        "importance_sum": importance_sum.values,
        "importance_mean": importance_mean.values,
        "freq_top20": freq_counter.values,
        "weighted_score": weighted_score.values,
    })

    stats_df = stats_df.sort_values(
        by=["weighted_score", "importance_sum", "freq_top20"],
        ascending=False
    ).reset_index(drop=True)

    return stats_df


def select_features_by_cumulative_threshold(
    stats_df: pd.DataFrame,
    cumulative_threshold: float = 0.8,
    min_freq_top20: int = 3,
    max_features: int = None
) -> List[str]:
    filtered = stats_df.loc[stats_df["freq_top20"] >= min_freq_top20].copy()

    # ak filter odstráni všetko, vezmi top 1 z pôvodných
    if filtered.empty:
        selected = [stats_df.iloc[0]["feature"]] if len(stats_df) > 0 else []
        if max_features is not None:
            selected = selected[:max_features]
        return selected

    total_weighted = filtered["weighted_score"].sum()
    if total_weighted > 0:
        filtered["weighted_score_fraction"] = filtered["weighted_score"] / total_weighted
        filtered["weighted_score_cumfrac"] = filtered["weighted_score_fraction"].cumsum()
    else:
        filtered["weighted_score_fraction"] = 0.0
        filtered["weighted_score_cumfrac"] = 0.0

    selected_mask = filtered["weighted_score_cumfrac"] <= cumulative_threshold
    selected = filtered.loc[selected_mask, "feature"].tolist()

    # pridaj ešte prvú feature nad threshold
    if len(selected) < len(filtered):
        selected.append(filtered.iloc[len(selected)]["feature"])

    if len(selected) == 0 and len(filtered) > 0:
        selected = [filtered.iloc[0]["feature"]]

    if max_features is not None:
        selected = selected[:max_features]

    return selected


def build_adaptive_integrated_omics(
    data_dir: str,
    output_root: str,
    datasets: List[str],
    out_name: str,
    cumulative_threshold: float,
    top_fraction_per_score: float,
    min_freq_top20: int,
    max_features_per_dataset: int = None,
):
    ensure_dir(output_root)

    selected_feature_summary = {}
    selected_tables = []
    parts = []

    for dataset in datasets:
        print(f"\n[INFO] Processing dataset: {dataset}")

        shap_df = load_normalized_shap(output_root, dataset)
        stats_df = compute_feature_statistics(
            shap_df,
            top_fraction_per_score=top_fraction_per_score
        )

        selected = select_features_by_cumulative_threshold(
            stats_df,
            cumulative_threshold=cumulative_threshold,
            min_freq_top20=min_freq_top20,
            max_features=max_features_per_dataset
        )

        X = load_dataset_X(data_dir, dataset)
        selected_existing = [f for f in selected if f in X.columns]
        X_sel = X.loc[:, selected_existing].copy()

        print(f"[INFO] Input features: {X.shape[1]}")
        print(f"[INFO] Selected features: {len(selected_existing)}")

        selected_feature_summary[dataset] = {
            "input_features": int(X.shape[1]),
            "selected_features": int(len(selected_existing)),
            "cumulative_threshold": float(cumulative_threshold),
            "top_fraction_per_score": float(top_fraction_per_score),
            "min_freq_top20": int(min_freq_top20),
        }

        stats_df["dataset"] = dataset
        stats_df["selected"] = stats_df["feature"].isin(selected_existing)
        selected_tables.append(stats_df)

        parts.append(X_sel)

    common_idx = parts[0].index
    for p in parts[1:]:
        common_idx = common_idx.intersection(p.index)

    parts = [p.loc[common_idx] for p in parts]
    X_integrated = pd.concat(parts, axis=1)
    X_integrated = X_integrated.loc[:, ~X_integrated.columns.duplicated()]

    print("\n[RESULT] Integrated adaptive omics")
    print(f"[RESULT] Samples: {X_integrated.shape[0]}")
    print(f"[RESULT] Features: {X_integrated.shape[1]}")

    out_path = os.path.join(data_dir, out_name)
    X_integrated.T.to_csv(out_path)
    print(f"[SAVED] {out_path}")

    summary_rows = []
    for dataset, info in selected_feature_summary.items():
        row = {"dataset": dataset}
        row.update(info)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_root, "adaptive_omics_selection_summary.csv"),
        index=False
    )

    selected_tables_df = pd.concat(selected_tables, axis=0, ignore_index=True)
    selected_tables_df.to_csv(
        os.path.join(output_root, "adaptive_feature_ranking_all_datasets.csv"),
        index=False
    )

    selected_feature_rows = []
    for dataset in datasets:
        dataset_selected = selected_tables_df[
            (selected_tables_df["dataset"] == dataset) &
            (selected_tables_df["selected"] == True)
        ]["feature"].tolist()
        for feat in dataset_selected:
            selected_feature_rows.append({
                "dataset": dataset,
                "feature": feat
            })

    pd.DataFrame(selected_feature_rows).to_csv(
        os.path.join(output_root, "adaptive_integrated_selected_features.csv"),
        index=False
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="omics_adaptive.csv")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS_DEFAULT,
        help="Datasets to include, default: immune specie kegg metabolome quest"
    )

    parser.add_argument(
        "--cumulative_threshold",
        type=float,
        default=0.80,
        help="Cumulative weighted SHAP threshold"
    )

    parser.add_argument(
        "--top_fraction_per_score",
        type=float,
        default=0.20,
        help="Top fraction used to compute freq_top20"
    )

    parser.add_argument(
        "--min_freq_top20",
        type=int,
        default=3,
        help="Keep only features that appear in top_fraction_per_score in at least this many scores"
    )

    parser.add_argument(
        "--max_features_per_dataset",
        type=int,
        default=None,
        help="Optional cap for selected features per dataset"
    )

    args = parser.parse_args()

    build_adaptive_integrated_omics(
        data_dir=args.data_dir,
        output_root=args.output_root,
        datasets=args.datasets,
        out_name=args.out_name,
        cumulative_threshold=args.cumulative_threshold,
        top_fraction_per_score=args.top_fraction_per_score,
        min_freq_top20=args.min_freq_top20,
        max_features_per_dataset=args.max_features_per_dataset,
    )


if __name__ == "__main__":
    main()