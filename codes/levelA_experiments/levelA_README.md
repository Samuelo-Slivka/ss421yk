# Level A – Multi-output Modeling (Omics → Clinical Scores → Classification)

## Level A – Multi-output Modeling (Omics → Clinical Scores → Classification)

This directory contains scripts implementing **Level A modeling**, which follows the same conceptual pipeline as the BioMapAI framework:

- Step 1: Predict **clinical symptom scores (Y)** from omics data (X) using regression models  
- Step 2: Use predicted scores (Ŷ) as input for a **classification layer** (ME/CFS vs Control)

This two-stage approach mimics the architecture described in the BioMapAI study and allows modeling disease heterogeneity via intermediate clinical representations.

---

## Contents

This directory includes the following scripts:

- `levelA_catboost.py` – CatBoost-based multi-output regression  
- `levelA_lightgbm.py` – LightGBM-based multi-output regression  
- `levelA_fttransformer.py` – Transformer-based deep learning model (FT-Transformer style)  
- `levelA_tabnet.py` – TabNet deep learning model for tabular data  
- `sanity_mlp_levelA.py` – baseline MLP sanity check model  

---

## Modeling Concept

All scripts follow the same pipeline:

1. **Input (X)**: omics dataset (e.g., metabolome, immune, kegg, etc.)
2. **Intermediate prediction (Y)**: 12 clinical symptom scores (multi-output regression)
3. **Final prediction (y)**: binary classification (ME/CFS vs Control)

The classification step is implemented using a **score-layer model** (Gradient Boosting or Logistic Regression), trained on predicted scores.

---

## Input Data Requirements

Each script expects the following files in `--data_dir`:

- `<dataset>.csv` → omics data (features × samples)
- `score.csv` → clinical scores (Y)
- `metadata.csv` → labels (`study_ptorhc` column)

### Supported datasets:
- immune, kegg, metabolome, specie, quest, omics

### Label format:
- mecfs → 1
- control → 0

---

## General Workflow

All scripts:

- Align samples across X, Y, and labels  
- Remove samples with missing scores  
- Standardize features (StandardScaler)  
- Handle class imbalance using **Random Under-Sampling (RUS)**  
- Use:
  - 10-fold cross-validation (on 90% data)
  - 10% held-out test set  

Evaluation metrics:

- Regression: **MSE**
- Classification: **AUC (ROC)**

---

## How to Run

### Example (CatBoost):

```
python levelA_catboost.py \
  --data_dir path/to/data \
  --dataset metabolome \
  --seed 42
```

---

## Script Descriptions

### levelA_catboost.py

- Implements **12 independent CatBoost regressors** (one per clinical score)  
- Uses:
  - RMSE loss
  - early stopping
- Predictions are aggregated into matrix Ŷ  
- Final classification via Gradient Boosting

---

### levelA_lightgbm.py

- Similar to CatBoost but uses **LightGBM regressors**  
- Efficient and scalable boosting method  
- Multi-output handled via separate models per score  

---

### levelA_fttransformer.py

- Deep learning model inspired by **FT-Transformer architecture**  
- Key components:
  - Feature tokenization (numeric → embeddings)
  - Feature identity embeddings
  - Multi-head self-attention
  - CLS token for final prediction    

---

### levelA_tabnet.py

- Uses **TabNetRegressor** for multi-output prediction  
- Attention-based feature selection  
- Built-in interpretability  

---

### sanity_mlp_levelA.py

- Simple **MLP baseline model**  
- Used for sanity checks and debugging  
- Includes:
  - StandardScaler
  - Logistic regression score-layer  

---

## Class Imbalance Handling

All scripts use:

- **Random Under-Sampling (RUS)**  
- Optionally repeated multiple times (`--rus_repeats`)  

This follows the methodology used in the reference study.

---

## Output

Each script prints:

- Cross-validation results (mean ± std)
- Held-out test performance:
  - MSE
  - AUC

---

## Notes

- These scripts replicate the BioMapAI pipeline logic, but:
  - use alternative models (CatBoost, LightGBM, DL)
  - allow easier experimentation and comparison  

- Multi-output regression is implemented as:
  - separate models (boosting)
  - shared model (DL)

---

## Purpose

This directory serves as:

- Experimental validation of the BioMapAI modeling paradigm  
- Comparison of classical ML vs deep learning approaches  
- Basis for feature selection (used later in SHAP analysis)  
