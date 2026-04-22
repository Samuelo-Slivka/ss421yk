# Final Phase – SHAP-based Dataset Construction and Overlap Analysis

This directory contains the scripts used in the **final phase of the thesis experiments**, focused on:

- SHAP-based feature selection,
- construction of new integrated multi-omics datasets,
- overlap analysis between original and newly derived feature sets,
- and evaluation of selected datasets with both CatBoost and BioMapAI-style modeling.

A special note applies to the deep learning evaluation part:

> `DNN.py` is a **helper script taken directly from the original BioMapAI repository** and was used only as an auxiliary implementation for reproducing BioMapAI-style experiments. It is **not our original script**.

---

## Directory Contents

This directory contains the following scripts:

- `biomapai_catboost_dataset.py` – CatBoost pipeline for regression, classification, SHAP computation, and feature selection
- `build_integrated_omics_from_shap.py` – builds a fixed integrated omics dataset using feature counts analogous to the original article
- `build_integrated_omics_adaptive_shap.py` – builds an adaptive integrated omics dataset from SHAP-based rankings
- `build_integrated_omics_adaptive_shap_updated.py` – updated adaptive SHAP integration script with per-dataset selection rules
- `build_subset_omics_from_featurelist.py` – builds subset omics datasets from a given feature list
- `compare_feature_sets.py` – compares overlaps among original and newly derived feature sets
- `run_biomapai_subset.py` – evaluation script for BioMapAI-style modeling on custom subset datasets
- `DNN.py` – helper neural-network implementation from the BioMapAI authors (external script, not authored in this thesis)

---

## General Purpose of This Directory

These scripts were used to move beyond direct model comparison and address a broader research question:

- whether feature importance profiles derived from CatBoost and SHAP can be used to construct biologically meaningful alternative integrated datasets,
- whether these derived datasets retain predictive value,
- and which features remain stable across different selection strategies.

In practice, this directory supports the creation and evaluation of the following types of datasets:

- **fixed integrated datasets** based on article-like feature counts,
- **adaptive integrated datasets** derived from SHAP selection,
- **subset datasets** created from overlaps or intersections,
- and **comparison outputs** identifying shared and unique biomarkers.

---

## Recommended Workflow

A typical workflow in this directory is:

1. Run CatBoost with SHAP on original or derived datasets
2. Save normalized SHAP outputs
3. Build new integrated datasets from SHAP-based selection
4. Compare overlaps among original, fixed, and adaptive feature sets
5. Extract specific subsets such as `core_omics` or `papernew_omics`
6. Evaluate selected datasets again with:
   - CatBoost
   - BioMapAI-style DNN pipeline

---

## Script Descriptions

## 1. `biomapai_catboost_dataset.py`

This is the main CatBoost-based pipeline for advanced experiments.

### What it does
- loads a dataset, score table, and metadata
- aligns samples across all files
- predicts 12 clinical scores using **12 independent CatBoost regressors**
- trains a **CatBoost classifier** on predicted scores
- performs:
  - 10-fold cross-validation
  - 10% held-out testing
  - repeated random under-sampling (RUS)
- computes SHAP values for each regression model
- saves:
  - metrics
  - predictions
  - SHAP outputs
  - selected features

### Outputs
It creates structured outputs inside:

- `models/`
- `metrics/`
- `predictions/`
- `shap/`
- `selected_features/`

### Example run

```bash
python biomapai_catboost_dataset.py \
  --data_dir ./Input \
  --dataset core_omics \
  --output_root ./output_core_catboost \
  --seed 0 \
  --rus_repeats 100
```

This script is the central source of SHAP-derived feature rankings used by the rest of the scripts.fileciteturn13file0

---

## 2. `build_integrated_omics_from_shap.py`

This script creates a **fixed integrated omics dataset** using SHAP rankings but preserves the same feature counts per omics layer as in the original BioMapAI article.

### Selection logic
It uses predefined target counts:

- immune: 50
- specie: 32
- kegg: 30
- metabolome: 42

The questionnaire dataset is not used in this integration step.

### What it produces
- a merged integrated dataset in article-style format (**features × samples**)
- a summary of selected features
- a CSV listing all selected features

### Example run

```bash
python build_integrated_omics_from_shap.py \
  --data_dir ./Input \
  --output_root ./output \
  --out_name omics_selected.csv
```

This script was used to generate the **fixed_omics**-style dataset.fileciteturn13file3

---

## 3. `build_integrated_omics_adaptive_shap.py`

This script constructs an **adaptive integrated omics dataset** based on global SHAP statistics rather than fixed counts per omics layer.

### Selection logic
For each dataset:
- normalized SHAP values are loaded,
- feature importance statistics are computed,
- features are ranked using:
  - total importance,
  - mean importance,
  - frequency of occurrence among top features,
  - weighted scores

Selection is then performed using:
- cumulative weighted SHAP threshold,
- top-feature frequency filtering,
- optional per-dataset caps.

### Example run

```bash
python build_integrated_omics_adaptive_shap.py \
  --data_dir ./Input \
  --output_root ./output \
  --out_name omics_adaptive.csv \
  --cumulative_threshold 0.80 \
  --top_fraction_per_score 0.20 \
  --min_freq_top20 3
```

This script was intended for constructing a more flexible SHAP-driven integrated dataset.fileciteturn13file1

---

## 4. `build_integrated_omics_adaptive_shap_updated.py`

This is the **updated adaptive SHAP integration script** used in the final version of the experiments.

### Key difference from the previous adaptive script
Instead of using one shared global threshold for all omics layers, it applies **dataset-specific selection rules**:

- immune → cumulative threshold 0.40, minimum frequency 4
- specie → cumulative threshold 0.60, minimum frequency 3
- metabolome → cumulative threshold 0.30, minimum frequency 5
- kegg → cumulative threshold 0.10, minimum frequency 7
- quest → cumulative threshold 0.50, minimum frequency 4

### Selection logic
For each omics layer:
- SHAP importance is aggregated across the 12 score models
- a cumulative selection is performed
- a frequency selection based on top 20% features is applied
- the final selected set is the intersection of both criteria

If that intersection is empty, the cumulative set is used as fallback.

### What it produces
- an integrated omics dataset in **features × samples** format
- a CSV listing the selected features across all omics layers

### Example run

```bash
python build_integrated_omics_adaptive_shap_updated.py \
  --data_dir ./Input \
  --output_root ./output \
  --out_name omics_per_dataset.csv
```

This script was used to generate the final **new_omics / per-dataset adaptive SHAP selection** workflow.fileciteturn13file2

---

## 5. `build_subset_omics_from_featurelist.py`

This script creates a new omics matrix by taking an existing source omics dataset and keeping only features listed in an external CSV file.

### Typical use
It is especially useful for constructing datasets such as:
- `core_omics`
- `papernew_omics`
- other overlap-based subsets

### Input requirements
- source omics matrix in **features × samples** format
- feature list CSV containing a `feature` column

### Output
- subset omics CSV
- optional CSV of missing features

### Example run

```bash
python build_subset_omics_from_featurelist.py \
  --source_omics ./Input/omics.csv \
  --feature_list ./overlap_results_triple.csv \
  --output_csv ./Input/core_omics.csv
```

This script was used to create concrete subset datasets from overlap analyses.fileciteturn13file4

---

## 6. `compare_feature_sets.py`

This script compares three feature sets:

- the original article feature set,
- the fixed SHAP-based feature set,
- the adaptive SHAP-based feature set.

### What it computes
- pairwise overlaps
- triple overlap
- unique features in each set
- overlap percentages
- CSV exports of all overlap groups

### Typical outputs
It can identify:
- stable shared features across all methods
- features unique to the article
- features unique to the adaptive SHAP selection

### Example run

```bash
python compare_feature_sets.py \
  --paper ./Input/omics.csv \
  --fixed ./output/integrated_omics_selected_features.csv \
  --new ./output/per_dataset_selected_features.csv \
  --out overlap_results
```

This script was essential for identifying:
- `core` biomarkers,
- overlap-based subsets,
- and potentially novel SHAP-derived candidate biomarkers.fileciteturn13file5

---

## 7. `run_biomapai_subset.py`

This script evaluates a selected custom dataset using a **BioMapAI-style deep learning pipeline**.

### Important note
This script is our evaluation wrapper, but it depends on the helper file `DNN.py`, which comes from the original BioMapAI authors and was reused here only to reproduce their style of modeling.

### What it does
- loads a custom omics subset dataset
- loads metadata and score tables
- performs:
  - held-out 10% split
  - repeated stratified cross-validation
  - random under-sampling on training folds
- trains the BioMapAI-style multi-output model
- trains a score-layer classification model
- applies weight adjustment
- saves:
  - CV metrics
  - held-out metrics
  - best model
  - best score layer
  - scaler
  - training history

### Example run

```bash
python run_biomapai_subset.py \
  --dnn_py_path ./DNN.py \
  --data_file_path ./Input/core_omics.csv \
  --metadata_file_path ./Input/metadata.csv \
  --scores_data_file_path ./Input/score.csv \
  --output_dir ./output_core_biomapai \
  --model_name core_omics \
  --sample_times 10 \
  --n_splits 10
```

This script was used only for **comparative evaluation**, not for developing a new neural-network architecture.fileciteturn13file7

---

## 8. `DNN.py`

This file contains the BioMapAI-style neural-network components used by `run_biomapai_subset.py`.

### Important authorship note
This script is **not our original implementation**. It was taken from the BioMapAI project and used only as a supporting tool for reproducing BioMapAI-like experiments on our newly created datasets.

### Main components
It defines helper classes such as:
- `OmicScoreModel`
- `ScoreLayer`
- `ScoreYModel`
- `WeightsAdjust`

These implement the two-stage modeling logic:
- multi-output prediction of symptom scores
- classification from predicted scores

It was included in the repository only because it is required for reproducible execution of the subset evaluation pipeline.fileciteturn13file6

---

## Data Relationships and Provenance

This directory works with both:

### 1. Original BioMapAI-derived data
These include:
- the original omics dataset,
- original individual omics layers,
- original metadata and score files.

### 2. Newly derived thesis datasets
These were created by us using SHAP outputs and overlap analysis, for example:
- `fixed_omics`
- `new_omics`
- `core_omics`
- `papernew_omics`

Their construction depends on the scripts listed above, especially:
- `build_integrated_omics_from_shap.py`
- `build_integrated_omics_adaptive_shap_updated.py`
- `compare_feature_sets.py`
- `build_subset_omics_from_featurelist.py`

---

## Practical Summary

This directory contains the scripts necessary to:

- derive new integrated datasets from SHAP feature importance,
- compare fixed and adaptive selection strategies,
- identify overlap-based biomarker candidates,
- build subset datasets,
- and evaluate them using both CatBoost and BioMapAI-style modeling.

It therefore represents the main technical implementation of the **third experimental phase** of the thesis.
