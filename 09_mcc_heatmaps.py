"""
09_mcc_heatmaps.py
==================
Loads per-model per-split metric CSVs and produces three MCC heatmaps
(one per split strategy), showing MCC per model × endpoint.

Figures:
  figures/figure9_heatmap_random.png
  figures/figure9_heatmap_scaffold.png
  figures/figure9_heatmap_clustering.png

REQUIRES: Table2_model_metrics_<model>_<split>.csv  (from 04_train_all_models.py)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = ["rf", "logreg", "svm", "cart", "xgb"]
SPLITS = ["random", "scaffold", "clustering"]
PRETTY = {
    "rf":     "Random Forest",
    "logreg": "Logistic Reg.",
    "svm":    "SVM (RBF)",
    "cart":   "CART",
    "xgb":    "XGBoost",
}
OUT_DIR  = "figures"
ANNOT_FS = 15
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load CSVs ─────────────────────────────────────────────────────────────────
DATA = {}
EP_LABELS, EP_DISPLAY = [], []

for split in SPLITS:
    for model in MODELS:
        fname = f"Table2_model_metrics_{model}_{split}.csv"
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        df = pd.read_csv(fname)
        df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
        DATA[(model, split)] = df
        # Detect endpoint labels from first loaded file
        if not EP_LABELS and "Toxicity" in df.columns:
            EP_LABELS = df["Toxicity"].dropna().unique().tolist()
            EP_DISPLAY = [ep.replace("__binary", "").replace("_", " ").title()
                          for ep in EP_LABELS]
        print(f"  Loaded {fname}")

if not DATA:
    raise FileNotFoundError("No metric CSVs found. Run 04_train_all_models.py first.")

# ── Plot one heatmap per split ─────────────────────────────────────────────────
for split in SPLITS:
    matrix, row_labels = [], []

    for model in MODELS:
        df = DATA.get((model, split))
        if df is None:
            continue
        row_labels.append(PRETTY[model])
        ep_mcc = {r["Toxicity"]: r["MCC"] if pd.notna(r["MCC"]) else np.nan
                  for _, r in df.iterrows()}
        matrix.append([ep_mcc.get(ep, np.nan) for ep in EP_LABELS])

    if not matrix:
        print(f"  No data for {split} split — skipping")
        continue

    mat  = np.array(matrix, dtype=float)
    vmax = max(np.nanmax(np.abs(mat)), 0.1)

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("MCC", fontsize=ANNOT_FS, labelpad=10)
    cbar.ax.tick_params(labelsize=ANNOT_FS - 1)

    ax.set_xticks(range(len(EP_LABELS)))
    ax.set_xticklabels(EP_DISPLAY, rotation=40, ha="right", fontsize=ANNOT_FS)
    ax.tick_params(axis="x", which="both", length=6, width=1.5, pad=6)
    ax.set_xlabel("Toxicity Endpoint", fontsize=ANNOT_FS, labelpad=12)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=ANNOT_FS - 1)
    ax.tick_params(axis="y", which="both", length=6, width=1.5, pad=6)
    ax.set_ylabel("Model", fontsize=ANNOT_FS, labelpad=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#333333")

    for i in range(len(row_labels)):
        for j in range(len(EP_LABELS)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=12,
                        color="black" if abs(val) < 0.6 * vmax else "white")

    ax.set_title(f"{split.capitalize()} Split", fontsize=17, pad=16)
    plt.tight_layout()
    out = f"{OUT_DIR}/figure9_heatmap_{split}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")