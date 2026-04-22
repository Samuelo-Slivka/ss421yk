import argparse
import importlib.util
import os
import pickle
import random

import numpy as np
import pandas as pd
from joblib import dump


def import_module_with_full_path(file_path: str):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_labels(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    label_map = {
        "control": 0,
        "mecfs": 1,
    }
    bad = sorted(set(s2.unique()) - set(label_map.keys()))
    if bad:
        raise ValueError(f"Unexpected labels: {bad}")
    return s2.map(label_map).astype(int)


def mse_list_from_tables(y_true_table, y_pred_table):
    from sklearn.metrics import mean_squared_error

    y_true = np.asarray(y_true_table)
    y_pred = np.asarray(y_pred_table)

    out = []
    for i in range(y_true.shape[1]):
        out.append(mean_squared_error(y_true[:, i], y_pred[:, i]))
    return out


def rus_balance_indices(y_binary: np.ndarray, rng: np.random.Generator):
    idx0 = np.where(y_binary == 0)[0]
    idx1 = np.where(y_binary == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return np.arange(len(y_binary))

    if len(idx0) > len(idx1):
        keep0 = rng.choice(idx0, size=len(idx1), replace=False)
        keep = np.concatenate([keep0, idx1])
    else:
        keep1 = rng.choice(idx1, size=len(idx0), replace=False)
        keep = np.concatenate([idx0, keep1])

    rng.shuffle(keep)
    return keep


def run_biomapai_subset(
    dnn_py_path: str,
    data_file_path: str,
    metadata_file_path: str,
    scores_data_file_path: str,
    output_dir: str,
    model_name: str = None,
    sample_times: int = 10,
    n_splits: int = 10,
    seed: int = 1015,
):
    import tensorflow as tf
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    import sklearn.metrics as sk_metrics

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    DNN = import_module_with_full_path(dnn_py_path)

    # input omics subset is features x samples -> transpose to samples x features
    data = pd.read_csv(data_file_path, index_col=0).transpose()
    metadata = pd.read_csv(metadata_file_path, index_col=0)
    scores_data = pd.read_csv(scores_data_file_path, index_col=0)

    data.index = data.index.astype(str).str.strip()
    data.columns = data.columns.astype(str).str.strip()
    metadata.index = metadata.index.astype(str).str.strip()
    scores_data.index = scores_data.index.astype(str).str.strip()

    sample_id = data.index.intersection(scores_data.index).intersection(metadata.index)
    data = data.loc[sample_id, :]
    metadata = metadata.loc[sample_id, :]
    scores_data = scores_data.loc[sample_id, :]

    y0_all = normalize_labels(metadata["study_ptorhc"])

    print(f"[INFO] data shape: {data.shape}")
    print(f"[INFO] scores shape: {scores_data.shape}")
    print(f"[INFO] labels shape: {y0_all.shape}")

    # ==========================================
    # Held-out 10% split (analogicky ku CatBoost)
    # ==========================================
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, held_idx = next(sss.split(data, y0_all))

    X_train_all = data.iloc[train_idx, :].copy()
    X_held = data.iloc[held_idx, :].copy()

    y_train_all = scores_data.loc[X_train_all.index, :].copy()
    y_held = scores_data.loc[X_held.index, :].copy()

    y0_train_all = y0_all.loc[X_train_all.index].copy()
    y0_held = y0_all.loc[X_held.index].copy()

    print(f"[INFO] held-out train samples: {X_train_all.shape[0]}")
    print(f"[INFO] held-out test samples: {X_held.shape[0]}")
    print(f"[INFO] held-out train label balance: control={(y0_train_all == 0).sum()}, patient={(y0_train_all == 1).sum()}")
    print(f"[INFO] held-out test label balance: control={(y0_held == 0).sum()}, patient={(y0_held == 1).sum()}")

    # ==========================================
    # CV on 90% train set
    # ==========================================
    rng = np.random.default_rng(seed)

    cv_mse = []
    cv_acc = []

    best_acc = -1.0
    best_omics_score_model = None
    best_omics_score_model_history = None
    best_score_layer = None
    best_scaler = None
    best_fold_meta = None

    step_now = 0

    result_metrics = pd.DataFrame(
        None,
        index=range(sample_times * n_splits),
        columns=["repeat", "fold", "train_acc", "test_acc", "TN", "FN", "FP", "TP"],
    )

    for rep in range(sample_times):
        random_state = int(rng.integers(0, 10_000))
        kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_train_all, y0_train_all), start=1):
            x_train = X_train_all.iloc[tr_idx, :].copy()
            x_val = X_train_all.iloc[va_idx, :].copy()

            y_train = y_train_all.loc[x_train.index, :].copy()
            y_val = y_train_all.loc[x_val.index, :].copy()

            y0_train = y0_train_all.loc[x_train.index].copy()
            y0_val = y0_train_all.loc[x_val.index].copy()

            # RUS len na train fold
            keep = rus_balance_indices(y0_train.to_numpy(), rng)
            x_train_bal = x_train.iloc[keep, :].copy()
            y_train_bal = y_train.loc[x_train_bal.index, :].copy()
            y0_train_bal = y0_train.loc[x_train_bal.index].copy()

            # hlavný multi-output model
            omic_model_runner = DNN.OmicScoreModel(
                epochs=200,
                optimizer=tf.keras.optimizers.Adam(0.0005),
                batch_size=64,
                kernel_regularizer=tf.keras.regularizers.L2(0.008),
                model_name=model_name,
            )

            omics_score_model, omics_score_model_history = omic_model_runner.train(
                x_train_bal, y_train_bal, x_val, y_val
            )

            mse = omic_model_runner.score(omics_score_model, x_val, y_val)
            cv_mse.append(mse)
            print(f"[CV repeat {rep+1:02d} fold {fold:02d}] MSE={mse}")

            # predikované clinical scores
            y_train_pred = omic_model_runner.predict(omics_score_model, x_train_bal)
            y_val_pred = omic_model_runner.predict(omics_score_model, x_val)

            # score layer sa trénuje na predikovaných score
            score_y_model_orig = DNN.ScoreLayer(
                learning_rate=0.01,
                epochs=100
            ).build_model(y_train_pred, y0_train_bal)

            # weight adjustment podľa train reconstruction quality
            weights = 1 - np.array(
                mse_list_from_tables(y_train_bal, y_train_pred)
            )

            score_y_model = DNN.WeightsAdjust(
                omics_score_model, x_train_bal, y_train_bal, score_y_model_orig
            ).adjust_layer_weight(score_y_model_orig, 1, weights)

            # evaluate classification na predikovaných score
            train_eval = score_y_model.evaluate(
                y_train_pred,
                np.asarray(y0_train_bal).astype("float32"),
                verbose=0
            )
            val_eval = score_y_model.evaluate(
                y_val_pred,
                np.asarray(y0_val).astype("float32"),
                verbose=0
            )

            train_accuracy = float(train_eval[1])
            val_accuracy = float(val_eval[1])
            cv_acc.append(val_accuracy)

            val_predict_y = score_y_model.predict(y_val_pred, verbose=0).flatten()
            val_predict_y_bin = (val_predict_y > 0.5).astype(int)

            confusion = sk_metrics.confusion_matrix(y0_val, val_predict_y_bin, labels=[0, 1])
            confusion_fraction = confusion / confusion.sum(axis=1, keepdims=True)

            TN, FP = confusion_fraction[0, 0], confusion_fraction[0, 1]
            FN, TP = confusion_fraction[1, 0], confusion_fraction[1, 1]

            result_metrics.loc[step_now, ["repeat", "fold", "train_acc", "test_acc", "TN", "FN", "FP", "TP"]] = [
                rep + 1, fold, train_accuracy, val_accuracy, TN, FN, FP, TP
            ]

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_omics_score_model = omics_score_model
                best_omics_score_model_history = omics_score_model_history
                best_score_layer = score_y_model
                best_scaler = omic_model_runner.scaler
                best_fold_meta = {"repeat": rep + 1, "fold": fold}

            print(f"[CV repeat {rep+1:02d} fold {fold:02d}] train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}")

            step_now += 1
            tf.keras.backend.clear_session()

    # ==========================================
    # Held-out evaluation using best CV model
    # ==========================================
    held_mse = None
    held_acc = None

    if best_omics_score_model is not None and best_score_layer is not None and best_scaler is not None:
        # použijeme best CV model na held-out
        # potrebujeme runner len kvôli scaler+predict logike
        omic_model_runner = DNN.OmicScoreModel(
            epochs=200,
            optimizer=tf.keras.optimizers.Adam(0.0005),
            batch_size=64,
            kernel_regularizer=tf.keras.regularizers.L2(0.008),
            model_name=model_name,
        )
        omic_model_runner.scaler = best_scaler

        y_held_pred = omic_model_runner.predict(best_omics_score_model, X_held)
        held_mse = mse_list_from_tables(y_held, y_held_pred)

        held_eval = best_score_layer.evaluate(
            y_held_pred,
            np.asarray(y0_held).astype("float32"),
            verbose=0
        )
        held_acc = float(held_eval[1])

        print("\n[Held-out 10%]")
        print(f"Held-out MSE={held_mse}")
        print(f"Held-out acc={held_acc:.4f}")

    # ==========================================
    # Save results under different names
    # ==========================================
    result = {
        "cv_mse_list": cv_mse,
        "cv_acc_list": cv_acc,
        "result_metrics": result_metrics,
        "best_acc": best_acc,
        "best_fold_meta": best_fold_meta,
        "held_mse": held_mse,
        "held_acc": held_acc,
    }

    dump(result, os.path.join(output_dir, "result_cvheldout.joblib"))

    if best_omics_score_model is not None:
        best_omics_score_model.save(os.path.join(output_dir, "best_omics_score_model_cvheldout.keras"))
    if best_score_layer is not None:
        best_score_layer.save(os.path.join(output_dir, "best_adjusted_score_layer_cvheldout.keras"))
    if best_scaler is not None:
        dump(best_scaler, os.path.join(output_dir, "best_scaler_cvheldout.joblib"))
    if best_omics_score_model_history is not None:
        with open(os.path.join(output_dir, "best_history_cvheldout.pkl"), "wb") as f:
            pickle.dump(best_omics_score_model_history.history, f)

    # summary txt
    summary_path = os.path.join(output_dir, "summary_cvheldout.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("[CV summary]\n")
        cv_mse_arr = np.asarray(cv_mse, dtype=float)
        f.write(f"Mean MSE: {cv_mse_arr.mean():.5f} ± {cv_mse_arr.std():.5f}\n")
        f.write(f"Mean val_acc: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}\n")
        f.write(f"Best val_acc: {best_acc:.4f}\n")
        if best_fold_meta is not None:
            f.write(f"Best repeat/fold: {best_fold_meta}\n")
        f.write("\n[Held-out 10%]\n")
        f.write(f"Held-out MSE: {held_mse}\n")
        f.write(f"Held-out acc: {held_acc}\n")

    print("\n[CV summary]")
    cv_mse_arr = np.asarray(cv_mse, dtype=float)
    print(f"Mean MSE: {cv_mse_arr.mean():.5f} ± {cv_mse_arr.std():.5f}")
    print(f"Mean val_acc: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
    print(f"Best val_acc: {best_acc:.4f}")
    if best_fold_meta is not None:
        print(f"Best repeat/fold: {best_fold_meta}")
    print(f"[SAVED] {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnn_py_path", required=True, help="Path to DNN.py")
    ap.add_argument("--data_file_path", required=True)
    ap.add_argument("--metadata_file_path", required=True)
    ap.add_argument("--scores_data_file_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--sample_times", type=int, default=10)
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1015)
    args = ap.parse_args()

    run_biomapai_subset(
        dnn_py_path=args.dnn_py_path,
        data_file_path=args.data_file_path,
        metadata_file_path=args.metadata_file_path,
        scores_data_file_path=args.scores_data_file_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_times=args.sample_times,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()