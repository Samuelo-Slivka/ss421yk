import argparse
import os
from typing import Dict, List

import pandas as pd


# počty presne podľa článku
TARGET_COUNTS = {
    "immune": 50,
    "specie": 32,
    "kegg": 30,
    "metabolome": 42,
    # quest sa v paper-omics integrácii nepoužíva
}


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)


def _infer_index_col(df: pd.DataFrame) -> str:
    return "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]


def load_dataset_X(data_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{dataset}.csv")
    df = pd.read_csv(path)
    idx = _infer_index_col(df)
    df = df.set_index(idx)

    # features x samples -> samples x features
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
    df = _read_csv(path)
    return df


def aggregate_feature_importance(normalized_shap_df: pd.DataFrame) -> pd.Series:
    """
    Scores_Normalized_SHAP_*.csv má riadky = score_0 ... score_11
    a stĺpce = feature names.
    Spravíme globálnu dôležitosť feature ako sumu cez všetky score.
    """
    importance = normalized_shap_df.sum(axis=0)
    importance = importance.sort_values(ascending=False)
    return importance


def select_top_features(output_root: str, dataset: str, k: int) -> List[str]:
    shap_df = load_normalized_shap(output_root, dataset)
    importance = aggregate_feature_importance(shap_df)
    top_features = importance.head(k).index.astype(str).tolist()
    return top_features


def build_integrated_omics(
    data_dir: str,
    output_root: str,
    out_name: str = "omics_selected.csv",
) -> None:

    selected_feature_summary = {}
    parts = []

    for dataset, k in TARGET_COUNTS.items():
        X = load_dataset_X(data_dir, dataset)
        selected = select_top_features(output_root, dataset, k)

        selected_existing = [f for f in selected if f in X.columns]
        X_sel = X.loc[:, selected_existing].copy()

        selected_feature_summary[dataset] = {
            "target_k": k,
            "selected_existing": len(selected_existing),
            "features": selected_existing,
        }

        print(f"[INFO] {dataset}: selected {len(selected_existing)} / target {k}")
        parts.append(X_sel)

    # align samples
    common_idx = parts[0].index
    for p in parts[1:]:
        common_idx = common_idx.intersection(p.index)

    parts = [p.loc[common_idx] for p in parts]
    X_integrated = pd.concat(parts, axis=1)

    # remove duplicate column names if any
    X_integrated = X_integrated.loc[:, ~X_integrated.columns.duplicated()]

    print(f"[RESULT] integrated samples: {X_integrated.shape[0]}")
    print(f"[RESULT] integrated features: {X_integrated.shape[1]}")

    # uloženie vo formáte článku: features x samples
    out_path = os.path.join(data_dir, out_name)
    X_integrated.T.to_csv(out_path)
    print(f"[SAVED] {out_path}")

    # summary
    summary_path = os.path.join(output_root, "integrated_omics_summary.json")
    pd.Series({
        "samples": X_integrated.shape[0],
        "features": X_integrated.shape[1],
    }).to_json(summary_path, indent=2)

    # detailný feature summary
    detail_rows = []
    for dataset, info in selected_feature_summary.items():
        for feat in info["features"]:
            detail_rows.append({
                "dataset": dataset,
                "feature": feat
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(
        os.path.join(output_root, "integrated_omics_selected_features.csv"),
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--out_name", type=str, default="omics_selected.csv")
    args = parser.parse_args()

    build_integrated_omics(
        data_dir=args.data_dir,
        output_root=args.output_root,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()