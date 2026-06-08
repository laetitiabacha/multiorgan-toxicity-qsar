# Multi-Organ Toxicity Prediction Using QSAR

**MEng Individual Project — Laetitia Bacha**  
Supervisor: Dr Pedro Ballester  
Imperial College London, June 2026

---

## Overview

This repository contains all code and data for a descriptor-based QSAR study predicting toxicity across eight organ systems. Five machine learning classifiers (Random Forest, XGBoost, SVM-RBF, Logistic Regression, CART) were trained on UniTox (2,196 FDA-approved small molecules) and evaluated under three data-splitting strategies of increasing structural stringency (random, scaffold, clustering). External validation used 153 clinically withdrawn drugs from Withdrawn 2.0.

---

## Repository Structure

### Preprocessing — run in order before analysis scripts

| Script | Input | Output |
|--------|-------|--------|
| `001_unitox_smiles_curation.py` | `UniTox.csv` | `UniTox_with_recovered_typos_v3.csv` |
| `002_withdrawn_data_preparation.py` | `withdrawn_drugs.csv` | `withdrawn_external_validation.csv` |
| `003_mordred_feature_computation.py` | both cleaned CSVs | `mordred_features_cached.csv`, `mordred_withdrawn_cached.csv` |

### Analysis scripts

Scripts 01–18 are self-contained and load their own data. They can be run in any order, though running them in numerical order follows the logical pipeline from internal benchmarking through to external validation.

### Data files included

| File | Description |
|------|-------------|
| `UniTox_with_recovered_typos_v3.csv` | Curated UniTox dataset with resolved canonical SMILES (2,196 small molecules, 8 binary endpoints) |
| `withdrawn_external_validation.csv` | 153 Withdrawn 2.0 drugs with mapped toxicity endpoints |

> `mordred_withdrawn_cached.csv` and `mordred_features_cached.csv` are not included. Run `003_mordred_feature_computation.py` to generate them.

---

## Data Sources

- **UniTox**: Silberg et al. *UniTox: Leveraging LLMs to Curate a Unified Dataset of Drug-Induced Toxicity from FDA Labels.* NeurIPS 2024. Download the raw dataset from the paper before running `001_unitox_smiles_curation.py`.
- **Withdrawn 2.0**: Gallo et al. *Withdrawn 2.0 — update on withdrawn drugs with pharmacovigilance data.* Nucleic Acids Research, 2023.

---

## Dependencies

```bash
pip install pandas numpy rdkit mordred scikit-learn xgboost matplotlib \
            pubchempy chembl-webresource-client scipy tabulate
```

Python 3.10+ recommended.

---

## Reproducing the Results

```bash
# Step 1 — Data preparation (requires raw UniTox.csv and withdrawn_drugs.csv)
python 001_unitox_smiles_curation.py
python 002_withdrawn_data_preparation.py
python 003_mordred_feature_computation.py

# Step 2 — Run analysis scripts
python 01_...py
# ... through
python 18_...py
```

All figures and result CSVs are saved to the working directory.
