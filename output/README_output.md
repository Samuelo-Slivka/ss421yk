# Output Directory

This directory contains all experimental outputs generated during **Level B (Phase 2)** modeling.  
The outputs are organized by dataset type (e.g., immune, kegg, metabolome, etc.) and further divided into structured subdirectories.

---

## Structure Overview

### 🔹 Dataset-specific experiment folders

Each dataset has its own folder:

- `immune/`
- `kegg/`
- `metabolome/`
- `specie/`
- `quest/`

Each of these folders contains results from experiments performed on the corresponding omics dataset.

---

## Common Subdirectory Structure

Each dataset folder contains the following subdirectories:

### 📁 `metrics/`
Contains evaluation metrics:

- `<dataset>_cv_metrics.xlsx` – cross-validation metrics
- `<dataset>_cv_summary.json` – summarized CV results
- `<dataset>_heldout_metrics.xlsx` – performance on held-out dataset

---

### 📁 `models/`
Contains trained models.

Structure:
- `repeat_000/` … `repeat_099/` (100 repetitions)

Inside each repeat:
- `<dataset>_score_0.cbm` … `<dataset>_score_11.cbm` → 12 regression models  
- `<dataset>_score_layer.cbm` → classification layer model  

---

### 📁 `predictions/`
Contains model predictions:

- `<dataset>_cv_score_predictions.xlsx` – predictions from CV
- `<dataset>_heldout_score_predictions.xlsx` – predictions from held-out test

---

### 📁 `selected_features/`
Contains selected feature sets:

- `<dataset>_selected_features.xlsx`
- `<dataset>_selected_by_score.xlsx`
- `<dataset>_common_features.xlsx`
- `<dataset>_selection_summary.json`

---

### 📁 `shap/`
Contains SHAP-based interpretability outputs:

For each regression model:
- `<dataset>_score_0.xlsx` … `<dataset>_score_11.xlsx`
- `<dataset>_score_0.pkl` … `<dataset>_score_11.pkl`

Additional files:
- `Scores_Normalized_SHAP_<dataset>.xlsx`
- `Scores_SHAP_std_<dataset>.xlsx`

---

## Additional Output Files

At root level:

- `adaptive_feature_ranking_all_datasets.xlsx`
- `adaptive_integrated_selected_features.xlsx`
- `adaptive_omics_selection_summary.xlsx`
- `integrated_omics_selected_features.xlsx`
- `integrated_omics_summary.json`
- `per_dataset_selected_features.xlsx`

---

## Output Folders for Integrated Experiments

- `output_core_biomapai/`
- `output_core_catboost/`
- `output_papernew_biomapai/`
- `output_papernew_catboost/`
- `output_perdataset_biomapai/`
- `output_perdataset_catboost/`

These correspond to different dataset construction strategies.

---

## Logs

- `levelB_experiments_terminal_outputs.txt`
- `levelC_experiments_terminal_outputs.txt`

---

## Summary

This directory contains:

- trained models (100 repetitions)
- predictions and evaluation metrics
- SHAP interpretability outputs
- selected feature sets

All outputs follow a consistent structure across datasets, enabling reproducibility and comparison.
