# Level B – CatBoost Modeling with SHAP-based Feature Selection

This directory contains the **Level B** script used in the second and third stages of the thesis experiments.  
The script extends the basic Level A pipeline by combining:

1. **multi-output regression** of 12 clinical symptom scores from omics data,  
2. **binary classification** of ME/CFS vs Control from the predicted scores, and  
3. **SHAP-based feature selection**, used to identify biologically relevant attributes and construct new integrated datasets.

The script was designed primarily for experiments with the **CatBoost** model family and was used both for:
- evaluating CatBoost on individual omics datasets, and
- generating feature importance profiles and selected feature sets for downstream dataset construction.

---

## Directory Contents

This directory contains the following main script:

- `biomapai_catboost_dataset.py` – CatBoost-based multi-output regression and classification pipeline with SHAP computation and paper-like feature selection.

---

## Modeling Logic

The script follows the same general conceptual framework as the BioMapAI study:

- **Input (X):** omics or multi-omics data  
- **Intermediate output (Y):** 12 clinical symptom scores  
- **Final output (y):** binary label (ME/CFS vs Control)

Unlike the original BioMapAI neural network, this implementation uses:

- **12 independent CatBoost regressors** for the 12 symptom scores
- **1 CatBoost classifier** as the score-layer model for final diagnosis prediction

This makes the pipeline easier to adapt to tabular data and directly compatible with SHAP-based interpretation.

---

## Input Data Requirements

The script expects the following files inside `--data_dir`:

- `<dataset>.csv` – omics or integrated dataset in **features × samples** format  
- `score.csv` – matrix of 12 clinical scores  
- `metadata.csv` – metadata table containing the label column `study_ptorhc`

### Supported labels

The script expects the following values in `metadata.csv`:

- `mecfs` → 1
- `control` → 0

### Typical dataset names

The script can be used with any dataset file name passed to `--dataset`, for example:

- `omics`
- `immune`
- `specie`
- `kegg`
- `metabolome`
- `quest`
- `new_omics`
- `core_omics`
- `papernew_omics`
- other custom datasets prepared in the same format

---

## Main Processing Steps

The script performs the following operations:

### 1. Data loading and alignment
- Loads omics data, score data, and labels
- Transposes the omics matrix from **features × samples** to **samples × features**
- Aligns samples present in all three sources
- Removes samples with missing clinical scores
- Removes feature columns containing only missing values
- Replaces remaining missing feature values with zeros

### 2. Held-out split
- Splits the aligned dataset into:
  - **90% training set**
  - **10% held-out test set**
- Uses stratified splitting to preserve class balance

### 3. Cross-validation
- Performs **10-fold stratified cross-validation** on the 90% training portion
- For each fold:
  - trains 12 CatBoost regressors,
  - predicts the 12 symptom scores,
  - trains a CatBoost classifier on predicted scores,
  - evaluates regression and classification performance

### 4. Class imbalance handling
- Applies **Random Under-Sampling (RUS)** on the training data
- Repeats this balancing procedure multiple times (`--rus_repeats`)
- Aggregates predictions across repeats for more robust results

### 5. Final training on the 90% split
- Repeats the same procedure on the full training portion
- Evaluates the final model on the independent held-out set

### 6. SHAP computation
- Computes SHAP values for each of the 12 CatBoost regression models
- Saves raw and aggregated SHAP outputs
- Produces normalized SHAP summaries across all features

### 7. Feature selection
- Applies a **paper-like feature selection strategy**
- Identifies:
  - **common features** appearing across symptom models
  - **union-selected features** across all scores
  - **score-specific selected features**

These outputs are then used to build new derived datasets for later experiments.

---

## Evaluation Metrics

The script reports both regression and classification metrics.

### Regression
- **MSE** – Mean Squared Error

### Classification
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUC**
- **AUPRC**

Cross-validation metrics are saved fold-wise and also summarized as mean ± standard deviation.

---

## Output Structure

The script creates an output directory inside `--output_root/<dataset>/` with the following subfolders:

- `models/` – saved CatBoost regression models and score-layer classifiers
- `metrics/` – CV metrics, held-out metrics, and summaries
- `predictions/` – predicted clinical scores and classification outputs
- `shap/` – SHAP values and normalized SHAP summaries
- `selected_features/` – selected feature lists and feature-selection summaries

---

## How to Run

### Example 1 – run on the original integrated omics dataset

```bash
python biomapai_catboost_dataset.py \
  --data_dir ./Input \
  --dataset omics \
  --output_root ./output_omics_catboost \
  --seed 0 \
  --rus_repeats 100
```

### Example 2 – run on a custom derived dataset

```bash
python biomapai_catboost_dataset.py \
  --data_dir ./Input \
  --dataset core_omics \
  --output_root ./output_core_catboost \
  --seed 0 \
  --rus_repeats 100
```

---

## Important Hyperparameters

The script allows adjustment of both regressor and classifier settings.

### Regressor parameters
- `--reg_iterations` (default: 2000)
- `--reg_lr` (default: 0.03)
- `--reg_depth` (default: 6)
- `--reg_l2_leaf_reg` (default: 3.0)
- `--reg_early_stopping_rounds` (default: 100)

### Classifier parameters
- `--clf_iterations` (default: 1000)
- `--clf_lr` (default: 0.03)
- `--clf_depth` (default: 4)
- `--clf_l2_leaf_reg` (default: 3.0)

### Experiment parameters
- `--seed`
- `--rus_repeats`

---

## Files Generated by SHAP Analysis

The `shap/` and `selected_features/` folders typically include:

- `Scores_Normalized_SHAP_<dataset>.csv` – normalized SHAP importance per score
- `Scores_SHAP_std_<dataset>.csv` – SHAP variability
- `<dataset>_score_<i>.csv` – SHAP values for each clinical score
- `<dataset>_common_features.csv` – features shared across symptom models
- `<dataset>_selected_features.csv` – union-selected features
- `<dataset>_selected_by_score.csv` – score-specific selected features
- `<dataset>_selection_summary.json` – summary counts

---

## Intended Use in the Thesis

This script was used in the thesis for two main purposes:

### Second experimental phase
To evaluate CatBoost on:
- `immune`
- `specie`
- `kegg`
- `metabolome`
- `quest`

This allowed comparison of the information content of individual omics layers.

### Third experimental phase
To evaluate CatBoost on:
- original integrated datasets from BioMapAI
- newly derived integrated datasets
- SHAP-based feature-selected subsets

This phase focused on the relationship between:
- predictive performance,
- dimensionality reduction,
- and biological interpretability.

---

## Practical Notes

- Input data must be stored in the same CSV orientation as in the thesis repository: **features × samples**
- Sample IDs must match across dataset, score, and metadata files
- The script is designed for **numerical tabular omics data**
- GPU is **not required**
- Larger values of `--rus_repeats` increase runtime but improve robustness

---

## Summary

`biomapai_catboost_dataset.py` is the main CatBoost-based pipeline for advanced thesis experiments.  
It combines:
- multi-output regression,
- downstream classification,
- repeated imbalance correction,
- held-out evaluation,
- and SHAP-based feature selection

into one reproducible workflow suitable for both benchmarking and biological interpretation.
