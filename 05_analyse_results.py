"""
05_analyse_results.py
=====================
Loads the per-model-per-split CSV files from 04_train_all_models.py
and produces summary tables and figures.

Figures:
  Figure_split_degradation.png     — MCC drop from random → scaffold → clustering
  Figure_mcc_heatmap_<model>.png   — MCC heatmap per endpoint × split
  Figure_internal_vs_external.png  — Internal vs external MCC gap

REQUIRES: Table2_model_metrics_<model>_<split>.csv  (from 04_train_all_models.py)
          External validation results if available
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_KEYS   = ["rf", "xgb", "logreg", "svm", "cart"]
MODEL_LABELS = {
    "rf":     "Random Forest",
    "xgb":    "XGBoost",
    "logreg": "Logistic Regression",
    "svm":    "SVM (RBF)",
    "cart":   "CART",
}
SPLIT_KEYS   = ["random", "scaffold", "clustering"]
SPLIT_LABELS = {
    "random":     "Random Split",
    "scaffold":   "Scaffold Split",
    "clustering": "Clustering Split",
}
SPLIT_COLOURS = {
    "random":     "#4C8BE0",
    "scaffold":   "#E05050",
    "clustering": "#5BBF5B",
}

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all CSVs ─────────────────────────────────────────────────────────────
frames = []
for split in SPLIT_KEYS:
    for model in MODEL_KEYS:
        fname = f"Table2_model_metrics_{model}_{split}.csv"
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        df = pd.read_csv(fname)
        df["Split"] = SPLIT_LABELS[split]
        df["Model"] = MODEL_LABELS[model]
        df["split_key"] = split
        df["model_key"] = model
        frames.append(df)

if not frames:
    raise FileNotFoundError(
        "No metric CSVs found. Run 04_train_all_models.py first.")

all_data = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(all_data)} rows across {all_data['Model'].nunique()} models × "
      f"{all_data['Split'].nunique()} splits")

# ── Table 1: Mean MCC per model × split ──────────────────────────────────────
pivot = (all_data.groupby(["Model", "Split"])["MCC"]
         .mean()
         .unstack("Split")
         .reindex(columns=[SPLIT_LABELS[s] for s in SPLIT_KEYS])
         .round(3))
print("\n── Mean MCC per model × split ──")
print(pivot.to_string())
pivot.to_csv("results_summary_mcc.csv")
print("Saved → results_summary_mcc.csv")

# ── Figure 1: Split degradation (RF) ─────────────────────────────────────────
rf_data = all_data[all_data["model_key"] == "rf"]
endpoints = rf_data["Toxicity"].unique()

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

x = np.arange(len(endpoints))
bar_w = 0.25
for i, split in enumerate(SPLIT_KEYS):
    sub = rf_data[rf_data["split_key"] == split]
    mccs = [sub[sub["Toxicity"] == ep]["MCC"].values[0]
            if len(sub[sub["Toxicity"] == ep]) else np.nan
            for ep in endpoints]
    ax.bar(x + (i - 1) * bar_w, mccs, bar_w,
           color=SPLIT_COLOURS[split], alpha=0.85, zorder=3,
           label=SPLIT_LABELS[split])

ax.set_xticks(x)
ax.set_xticklabels([ep.replace(" Toxicity", "").replace(" toxicity", "")
                    for ep in endpoints], rotation=30, ha="right", fontsize=10)
ax.set_ylabel("MCC")
ax.set_title("Random Forest: MCC per Endpoint by Split Strategy",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
out = f"{OUT_DIR}/Figure_split_degradation.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")

# ── Figure 2: MCC heatmap per model ──────────────────────────────────────────
for model_key, model_label in MODEL_LABELS.items():
    sub = all_data[all_data["model_key"] == model_key]
    if sub.empty:
        continue
    matrix = sub.pivot_table(index="Toxicity", columns="Split",
                              values="MCC", aggfunc="mean")
    matrix = matrix.reindex(columns=[SPLIT_LABELS[s] for s in SPLIT_KEYS])

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=-0.1, vmax=0.6,
                   aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="MCC")
    ax.set_title(f"{model_label}: MCC Heatmap", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/Figure_mcc_heatmap_{model_key}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")

print("\nDone — all analysis figures saved.")
