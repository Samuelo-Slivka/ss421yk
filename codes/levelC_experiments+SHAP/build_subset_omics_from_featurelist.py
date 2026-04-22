import argparse
import os
import pandas as pd


def read_feature_list(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"{path} must contain column 'feature'")
    feats = df["feature"].astype(str).str.strip().tolist()
    # preserve order, remove duplicates
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def read_omics_matrix(path: str) -> pd.DataFrame:
    # paper omics is features x samples
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def build_subset(source_omics: str, feature_list_csv: str, output_csv: str) -> None:
    omics = read_omics_matrix(source_omics)
    wanted = read_feature_list(feature_list_csv)

    matched = [f for f in wanted if f in omics.index]
    missing = [f for f in wanted if f not in omics.index]

    subset = omics.loc[matched].copy()
    subset.to_csv(output_csv)

    print(f"[INFO] Source omics features: {omics.shape[0]}")
    print(f"[INFO] Requested features: {len(wanted)}")
    print(f"[INFO] Matched features: {len(matched)}")
    print(f"[INFO] Missing features: {len(missing)}")
    print(f"[INFO] Output shape (features x samples): {subset.shape}")
    print(f"[SAVED] {output_csv}")

    if missing:
        miss_path = output_csv.replace(".csv", "_missing_features.csv")
        pd.DataFrame({"feature": missing}).to_csv(miss_path, index=False)
        print(f"[SAVED] {miss_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_omics", required=True, help="Path to paper omics.csv")
    ap.add_argument("--feature_list", required=True, help="CSV with column 'feature'")
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    build_subset(args.source_omics, args.feature_list, args.output_csv)


if __name__ == "__main__":
    main()