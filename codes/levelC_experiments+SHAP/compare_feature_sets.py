import pandas as pd
import argparse


def load_features_from_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return set(df.index.astype(str))


def load_features_from_list(path):
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"{path} must contain column 'feature'")
    return set(df["feature"].astype(str))


def main(args):
    print("[INFO] Loading datasets...\n")

    # paper omics
    paper_features = load_features_from_matrix(args.paper)

    # fixed selection (tvoj prvý)
    fixed_features = load_features_from_list(args.fixed)

    # nový per-dataset selection
    new_features = load_features_from_list(args.new)

    print(f"[INFO] Paper features: {len(paper_features)}")
    print(f"[INFO] Fixed features: {len(fixed_features)}")
    print(f"[INFO] New features: {len(new_features)}")

    print("\n==============================")
    print("PAIRWISE OVERLAPS")
    print("==============================")

    pf = paper_features & fixed_features
    pn = paper_features & new_features
    fn = fixed_features & new_features

    print(f"Paper ∩ Fixed: {len(pf)}")
    print(f"Paper ∩ New: {len(pn)}")
    print(f"Fixed ∩ New: {len(fn)}")

    print("\n==============================")
    print("TRIPLE OVERLAP")
    print("==============================")

    triple = paper_features & fixed_features & new_features
    print(f"Paper ∩ Fixed ∩ New: {len(triple)}")

    print("\n==============================")
    print("UNIQUE FEATURES")
    print("==============================")

    only_paper = paper_features - fixed_features - new_features
    only_fixed = fixed_features - paper_features - new_features
    only_new = new_features - paper_features - fixed_features

    print(f"Only Paper: {len(only_paper)}")
    print(f"Only Fixed: {len(only_fixed)}")
    print(f"Only New: {len(only_new)}")

    print("\n==============================")
    print("PERCENT OVERLAPS")
    print("==============================")

    def pct(a, b):
        return 100 * len(a & b) / len(a) if len(a) > 0 else 0

    print(f"Paper vs Fixed: {pct(paper_features, fixed_features):.2f}%")
    print(f"Paper vs New: {pct(paper_features, new_features):.2f}%")
    print(f"Fixed vs New: {pct(fixed_features, new_features):.2f}%")

    print("\n==============================")
    print("SAVING RESULTS")
    print("==============================")

    pd.DataFrame({"feature": list(pf)}).to_csv(args.out + "_paper_fixed.csv", index=False)
    pd.DataFrame({"feature": list(pn)}).to_csv(args.out + "_paper_new.csv", index=False)
    pd.DataFrame({"feature": list(fn)}).to_csv(args.out + "_fixed_new.csv", index=False)
    pd.DataFrame({"feature": list(triple)}).to_csv(args.out + "_triple.csv", index=False)

    pd.DataFrame({"feature": list(only_paper)}).to_csv(args.out + "_only_paper.csv", index=False)
    pd.DataFrame({"feature": list(only_fixed)}).to_csv(args.out + "_only_fixed.csv", index=False)
    pd.DataFrame({"feature": list(only_new)}).to_csv(args.out + "_only_new.csv", index=False)

    print("[DONE] CSV files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--paper", required=True, help="omics.csv from paper")
    parser.add_argument("--fixed", required=True, help="your fixed selection CSV")
    parser.add_argument("--new", required=True, help="per_dataset_selected_features.csv")
    parser.add_argument("--out", default="feature_overlap")

    args = parser.parse_args()
    main(args)