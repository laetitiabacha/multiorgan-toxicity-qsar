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
"""
09_multimodel_boxplot.py
========================
Loads Multi_Model_Benchmark_Results.csv and produces a grouped boxplot
comparing MCC across 5 models × 3 splits, with significance brackets.

Figure:
  Figure4_Multi_Model_Benchmark_Boxplot.png

REQUIRES: Multi_Model_Benchmark_Results.csv  (from 08_combine_benchmark_results.py)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV_PATH = "Multi_Model_Benchmark_Results.csv"
OUT      = "Figure4_Multi_Model_Benchmark_Boxplot.png"
ANNOT_FS = 13

MODEL_ORDER   = ["RF", "SVM (RBF)", "LogReg", "XGBoost", "CART"]
SPLIT_ORDER   = ["random", "scaffold", "clustering"]
SPLIT_LABELS  = {"random": "Random Split", "scaffold": "Scaffold Split",
                 "clustering": "Clustering Split"}
SPLIT_COLOURS = {"random": "#4C8BE0", "scaffold": "#E05050", "clustering": "#5BBF5B"}

SPLIT_NORM = {
    "random": "random", "random split": "random",
    "scaffold": "scaffold", "scaffold split": "scaffold",
    "clustering": "clustering", "clustering split": "clustering",
}
MODEL_NORM = {
    "rf": "RF", "random forest": "RF",
    "xgb": "XGBoost", "xgboost": "XGBoost",
    "logreg": "LogReg", "logistic regression": "LogReg",
    "svm_rbf": "SVM (RBF)", "svm (rbf)": "SVM (RBF)",
    "cart": "CART",
}


def wilcoxon_p(a, b):
    a = np.array([v for v in a if not np.isnan(v)])
    b = np.array([v for v in b if not np.isnan(v)])
    n = min(len(a), len(b))
    if n < 5:
        return 1.0
    try:
        _, p = wilcoxon(a[:n], b[:n])
        return p
    except Exception:
        return 1.0


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Split"] = df["Split"].str.lower().str.strip().map(SPLIT_NORM)
    df["Model"] = df["Model"].str.lower().str.strip().map(MODEL_NORM)
    before = len(df)
    df = df.dropna(subset=["Split", "Model"])
    if len(df) < before:
        print(f"  WARNING: dropped {before - len(df)} unmapped rows")
    return df


def plot(df, out=OUT):
    splits = [s for s in SPLIT_ORDER if s in df["Split"].unique()]
    models = [m for m in MODEL_ORDER if m in df["Model"].unique()]

    x       = np.arange(len(models))
    n       = len(splits)
    width   = 0.22
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.spines["left"].set_color("black");   ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("black"); ax.spines["bottom"].set_linewidth(1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linestyle="--", linewidth=0.8, zorder=0)

    vals_dict      = {}
    legend_handles = []

    for i, split in enumerate(splits):
        colour    = SPLIT_COLOURS[split]
        positions = x + offsets[i]

        for j, model in enumerate(models):
            vals = df[(df["Model"] == model) &
                      (df["Split"] == split)]["MCC"].dropna().values
            vals_dict[(model, split)] = vals
            if len(vals) == 0:
                continue

            bp = ax.boxplot(
                vals,
                positions=[positions[j]],
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                zorder=3,
                boxprops=dict(facecolor=colour, color="black",
                              linewidth=1.0, alpha=1),
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(color="black", linewidth=1.0),
                capprops=dict(color="black", linewidth=1.0),
                flierprops=dict(marker="", markersize=0),
            )

            np.random.seed(42)
            jitter = np.random.normal(0, width * 0.08, size=len(vals))
            ax.scatter(positions[j] + jitter, vals,
                       s=60, facecolors=colour, edgecolors="black",
                       linewidths=1.2, alpha=0.9, zorder=2)

            median_val = np.median(vals)
            ax.text(positions[j],
                    bp["caps"][1].get_ydata()[0] + 0.012,
                    f"{median_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=14, fontweight="semibold",
                    color="black", zorder=5)

        legend_handles.append(
            mpatches.Patch(facecolor=colour, edgecolor="black",
                           label=SPLIT_LABELS[split], alpha=0.85)
        )

    leg = ax.legend(handles=legend_handles, fontsize=ANNOT_FS + 2,
                    frameon=True, facecolor="white", edgecolor="black",
                    loc="upper right")
    ax.add_artist(leg)

    ax.set_ylim(-0.35, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=ANNOT_FS + 1)
    ax.set_ylabel("MCC (across 8 endpoints)", fontsize=ANNOT_FS + 3, labelpad=10)
    ax.tick_params(axis="both", length=6, width=1.5,
                   labelsize=ANNOT_FS + 3, color="black", labelcolor="black")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.3)
    ax.set_title("Multi-Model MCC Comparison by Split Strategy\n"
                 "(distributions across 8 toxicity endpoints)",
                 fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    df = load(CSV_PATH)
    plot(df, out=OUT)
