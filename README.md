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
| `01_mordred_feature_computation.py` | `UniTox_with_recovered_typos_v3.csv`, `withdrawn_external_validation.csv` | `mordred_features_cached.csv`, `mordred_withdrawn_cached.csv` |

> `002_data_exploration.py` combines data exploration with Withdrawn 2.0 preparation (toxtype normalisation, endpoint mapping, overlap removal). The cleaned file `withdrawn_external_validation.csv` is provided directly and does not need to be regenerated.

### Analysis scripts

Scripts 02–18 are self-contained and load their own data. They can be run in any order, though running them in numerical order follows the logical pipeline from internal benchmarking through to external validation.

| Script | Description |
|--------|-------------|
| `02_data_exploration.py` | Dataset statistics and class distribution plots |
| `03_splits.py` | Random, scaffold, and clustering split construction |
| `04_train_all_models.py` | Trains all five classifiers across all splits and endpoints |
| `05_cluster_size_analysis.py` | Cluster-count validation (silhouette, Calinski-Harabasz, dendrogram) |
| `06_svm_kernel_comparison.py` | SVM kernel selection (linear, RBF, polynomial) |
| `07_multimodel_boxplot.py` | Cross-architecture MCC and ROC-AUC box plots |
| `08_split_degradation.py` | Split degradation plot: random → scaffold → clustering per model |
| `09_mcc_heatmaps.py` | Per-endpoint MCC heatmaps across classifiers and splits |
| `10_pca_chemical_space.py` | PCA chemical space overlap (UniTox vs Withdrawn 2.0) |
| `11_generate_all_metrics.py` | Generates full metrics tables for all model/split combinations |
| `12_rf_performance_vs_similarity.py` | RF MCC vs Tanimoto similarity scatter plot |
| `13_svm_rfe_feature_selection.py` | SVM-RFE feature selection pipeline |
| `14_external_validation.py` | External validation of RF and XGBoost on Withdrawn 2.0 |
| `15_per_drug_mcc_venn.py` | Per-drug MCC analysis and Venn diagram case studies |
| `16_external_recovery_plot.py` | Clinical recovery rates and sensitivity plots |
| `17_mcc_heatmap_grid.py` | MCC/sensitivity/specificity heatmap grid for external validation |
| `18_generalisation_gap.py` | Internal vs external MCC generalisation gap plot |

### Data files included

| File | Description |
|------|-------------|
| `UniTox_with_recovered_typos_v3.csv` | Curated UniTox dataset with resolved canonical SMILES (2,196 small molecules, 8 binary endpoints) |
| `withdrawn_external_validation.csv` | 153 Withdrawn 2.0 drugs with mapped toxicity endpoints |

> `mordred_features_cached.csv` and `mordred_withdrawn_cached.csv` are not included due to file size. Run `01_mordred_feature_computation.py` to generate them (~10–15 min on a standard CPU).

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
# Step 1 — Data preparation (requires raw UniTox.csv)
python 001_unitox_smiles_curation.py
python 01_mordred_feature_computation.py

# Step 2 — Run analysis scripts (withdrawn_external_validation.csv is provided)
python 02_data_exploration.py
python 03_splits.py
# ... through
python 18_generalisation_gap.py
```

All figures and result CSVs are saved to the working directory.
