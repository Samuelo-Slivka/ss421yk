import os
import argparse
import pandas as pd
import numpy as np

CONFIG = {
    "immune": {"cum": 0.40, "min_freq": 4},
    "specie": {"cum": 0.60, "min_freq": 3},
    "metabolome": {"cum": 0.30, "min_freq": 5},
    "kegg": {"cum": 0.10, "min_freq": 7},
    "quest": {"cum": 0.50, "min_freq": 4},
}


def load_shap(output_root: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(
        output_root,
        dataset,
        "shap",
        f"Scores_Normalized_SHAP_{dataset}.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP file not found: {path}")
    return pd.read_csv(path, index_col=0)


def load_data(data_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{dataset}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, index_col=0)

    # článkové datasety sú uložené ako features x samples
    # preto ich otočíme na samples x features
    X = df.T
    X.index = X.index.astype(str).str.strip()
    X.columns = X.columns.astype(str).str.strip()
    return X


def select_features(shap_df: pd.DataFrame, dataset_name: str):
    cfg = CONFIG[dataset_name]

    # shap_df: rows = score_0..score_11, cols = features
    importance = shap_df.abs().sum(axis=0).sort_values(ascending=False)

    # cumulative selection
    cum_importance = importance.cumsum() / importance.sum()
    selected_cum = cum_importance[cum_importance <= cfg["cum"]].index.tolist()

    # frequency selection: top 20 % per score
    top_k = max(1, int(len(importance) * 0.20))
    freq_counter = {}

    for _, row in shap_df.iterrows():
        top_features = row.abs().nlargest(top_k).index
        for f in top_features:
            freq_counter[f] = freq_counter.get(f, 0) + 1

    selected_freq = [
        f for f, c in freq_counter.items()
        if c >= cfg["min_freq"]
    ]

    final = sorted(set(selected_cum) & set(selected_freq))

    # fallback: ak by prienik bol prázdny, vezmeme cumulative set
    if len(final) == 0:
        final = selected_cum

    return final, importance, freq_counter


def main(args):
    datasets = ["immune", "specie", "kegg", "metabolome", "quest"]

    selected_feature_rows = []
    selected_dataframes = []

    for ds in datasets:
        print(f"\n[INFO] Processing {ds}")

        shap_df = load_shap(args.output_root, ds)
        X = load_data(args.data_dir, ds)

        selected, importance, freq_counter = select_features(shap_df, ds)

        selected_existing = [f for f in selected if f in X.columns]
        X_sel = X[selected_existing].copy()

        print(f"[INFO] Input features: {X.shape[1]}")
        print(f"[INFO] Selected features: {len(selected_existing)}")

        for feat in selected_existing:
            selected_feature_rows.append({
                "dataset": ds,
                "feature": feat,
                "importance_sum": float(importance.get(feat, 0.0)),
                "freq_top20": int(freq_counter.get(feat, 0)),
            })

        selected_dataframes.append(X_sel)

    # align samples
    common_idx = selected_dataframes[0].index
    for df in selected_dataframes[1:]:
        common_idx = common_idx.intersection(df.index)

    selected_dataframes = [df.loc[common_idx] for df in selected_dataframes]
    merged = pd.concat(selected_dataframes, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    print("\n[RESULT]")
    print(f"Samples: {merged.shape[0]}")
    print(f"Features: {merged.shape[1]}")

    out_path = os.path.join(args.data_dir, args.out_name)

    # uloženie vo formáte features x samples, aby to sedelo s ostatnými datasetmi
    merged.T.to_csv(out_path)
    print(f"[SAVED] {out_path}")

    selected_df = pd.DataFrame(selected_feature_rows)
    selected_df.to_csv(
        os.path.join(args.output_root, "per_dataset_selected_features.csv"),
        index=False
    )
    print(f"[SAVED] {os.path.join(args.output_root, 'per_dataset_selected_features.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--out_name", default="omics_per_dataset.csv")
    args = parser.parse_args()
    main(args)