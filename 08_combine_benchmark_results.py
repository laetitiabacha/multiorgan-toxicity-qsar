"""
08_combine_benchmark_results.py
================================
Reads all individual Table2_model_metrics_<model>_<split>.csv files
produced by 04_train_all_models.py and merges them into one master CSV:

    Multi_Model_Benchmark_Results.csv
    columns: Split | Model | Toxicity | ROC-AUC | AUPRC | MCC | Accuracy

REQUIRES: Table2_model_metrics_<model>_<split>.csv  (from 04_train_all_models.py)
PRODUCES: Multi_Model_Benchmark_Results.csv
"""

import os
import pandas as pd

MODEL_KEYS = ["rf", "xgb", "logreg", "svm", "cart"]
SPLIT_KEYS = ["random", "scaffold", "clustering"]

MODEL_LABELS = {
    "rf":     "Random Forest",
    "xgb":    "XGBoost",
    "logreg": "Logistic Regression",
    "svm":    "SVM (RBF)",
    "cart":   "CART",
}
SPLIT_LABELS = {
    "random":     "Random Split",
    "scaffold":   "Scaffold Split",
    "clustering": "Clustering Split",
}

all_frames = []
for split_key in SPLIT_KEYS:
    for model_key in MODEL_KEYS:
        fname = f"Table2_model_metrics_{model_key}_{split_key}.csv"
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        df = pd.read_csv(fname)
        df.insert(0, "Split", SPLIT_LABELS[split_key])
        df.insert(1, "Model", MODEL_LABELS[model_key])
        all_frames.append(df)
        print(f"  Loaded {fname}  ({len(df)} rows)")

if not all_frames:
    raise FileNotFoundError(
        "No CSV files found. Run 04_train_all_models.py first.")

combined = pd.concat(all_frames, ignore_index=True)

# ── Summary pivot ─────────────────────────────────────────────────────────────
print("\n── Mean MCC per model × split ───────────────────────────────────────────")
pivot = (combined.groupby(["Model", "Split"])["MCC"]
         .mean()
         .unstack("Split")
         .reindex(columns=[SPLIT_LABELS[s] for s in SPLIT_KEYS])
         .round(3))
print(pivot.to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
out = "Multi_Model_Benchmark_Results.csv"
combined.to_csv(out, index=False)
print(f"\nSaved → {out}  ({len(combined)} rows, {combined['Model'].nunique()} models × "
      f"{combined['Split'].nunique()} splits × {combined['Toxicity'].nunique()} endpoints)")
