# Input Data Directory

This directory contains all datasets used throughout the thesis experiments.  
It includes both:

- original datasets provided by the BioMapAI framework (**marked as adopted from BioMapAI authors**),
- and newly constructed datasets created during this thesis.

---

## Structure Overview

The directory contains:

### 🔹 Original BioMapAI datasets (ADOPTED)

These datasets were directly taken from the BioMapAI project:

- `immune.csv` *(adopted from BioMapAI authors)* – immune profiling data  
- `specie.csv` *(adopted from BioMapAI authors)* – microbiome species abundance  
- `kegg.csv` *(adopted from BioMapAI authors)* – KEGG gene data  
- `metabolome.csv` *(adopted from BioMapAI authors)* – metabolomics data  
- `quest.csv` *(adopted from BioMapAI authors)* – clinical measurements  
- `omics.csv` *(adopted from BioMapAI authors)* – integrated multi-omics dataset  
- `metadata.csv` *(adopted from BioMapAI authors)* – sample labels  
- `score.csv` *(adopted from BioMapAI authors)* – clinical symptom scores  

---

### 🔹 Feature metadata

- `feature_meta/` – auxiliary feature descriptions *(adopted from BioMapAI authors)*

---

### 🔹 Newly created datasets (THIS THESIS)

#### SHAP-based datasets

- `omics_selected.csv` – fixed SHAP selection  
- `omics_adaptive.csv` – adaptive SHAP selection  
- `omics_per_dataset.csv` – per-dataset SHAP selection  

#### Subset datasets

- `core_omics.csv` – intersection of multiple selections  
- `papernew_omics.csv` – merged original + new features  
- `per_dataset_omics.csv` – dataset from per-dataset selection  
- `per_dataset_omics_missing_features.csv` – missing features log  

---

## Usage

Datasets were used across three phases:

1. Original omics dataset  
2. Individual omics datasets  
3. SHAP-based constructed datasets  

---

## Summary

This directory combines:

- adopted BioMapAI datasets  
- newly engineered datasets  

enabling comparison between original and newly constructed representations.
