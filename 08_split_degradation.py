"""
08_split_degradation.py
==================
Run this AFTER train_all_models.py has produced all CSV files.
Produces : Split degradation plot: random → scaffold → clustering per model

"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# ── Config ────────────────────────────────────────────────────────────────────
MODELS = ["rf", "logreg", "svm", "cart", "xgb"]
SPLITS = ["random", "scaffold", "clustering"]
PRETTY = {"rf": "Random Forest", "logreg": "Logistic Reg.",
          "svm": "SVM (RBF)", "cart": "CART", "xgb": "XGBoost"}
SPLIT_COLORS = {"random": "#4A90D9", "scaffold": "#E8A838", "clustering": "#E85C5C"}
UNITOX_PATH    = "UniTox_with_recovered_typos_v3.csv"
MORDRED_PATH   = "mordred_features_cached.csv"
WITHDRAWN_FILE = "withdrawn_external_validation.csv"
OUT_DIR = "analysis_figures"
os.makedirs(OUT_DIR, exist_ok=True)
# ── Font size config ──────────────────────────────────────────────────────────
ANNOT_FS     = 15   # axis labels and tick labels
LEGEND_FS    = 12   # legend font size
BAR_LABEL_FS = 11   # numbers printed above bars
# ── File path helpers (unified naming + legacy RF fallback) ───────────────────
def metrics_path(model, split):
    primary = f"Table2_model_metrics_{model}_{split}.csv"
    if os.path.exists(primary):
        return primary
    legacy = {"random": "Table2_model_metrics.csv",
              "scaffold": "Table2_model_metrics_scaffold.csv",
              "clustering": "Table2_model_metrics_clustering.csv"}.get(split, "")
    return legacy if legacy and os.path.exists(legacy) else primary
def cm_path(model, split):
    primary = f"cm_table_{model}_{split}.csv"
    if os.path.exists(primary):
        return primary
    legacy = {"random": "cm_table.csv",
              "scaffold": "cm_table_scaffold.csv",
              "clustering": "cm_table_clustering.csv"}.get(split, "")
    return legacy if legacy and os.path.exists(legacy) else primary
# ── Auto-detect endpoint labels from actual CSV data ─────────────────────────
def detect_ep_labels():
    """Read Toxicity column from first available cm_table CSV.
    Returns (ep_raw_list, ep_display_list) — no hardcoding."""
    for model in MODELS:
        for split in SPLITS:
            cp = cm_path(model, split)
            if os.path.exists(cp):
                df = pd.read_csv(cp)
                if "Toxicity" in df.columns:
                    eps = df["Toxicity"].dropna().unique().tolist()
                    display = [ep.replace("__binary", "").replace("_", " ").title()
                               for ep in eps]
                    return eps, display
    return [], []
# ── Load all results ──────────────────────────────────────────────────────────
def load_all():
    data = {}
    for model in MODELS:
        for split in SPLITS:
            mp = metrics_path(model, split)
            cp = cm_path(model, split)
            if not os.path.exists(mp) or os.path.getsize(mp) == 0:
                continue
            if not os.path.exists(cp) or os.path.getsize(cp) == 0:
                continue
            try:
                m  = pd.read_csv(mp)
                cm = pd.read_csv(cp)
            except pd.errors.EmptyDataError:
                print(f"  Skipping empty file: {mp} or {cp}")
                continue
            for col in ["ROC-AUC", "AUPRC", "MCC", "Accuracy"]:
                if col in m.columns:
                    m[col] = pd.to_numeric(m[col], errors="coerce")
            for col in ["TP", "TN", "FP", "FN"]:
                if col in cm.columns:
                    cm[col] = pd.to_numeric(cm[col], errors="coerce").fillna(0)
                else:
                    cm[col] = 0.0
            cm["Sensitivity"] = np.where(
                (cm["TP"] + cm["FN"]) > 0,
                cm["TP"] / (cm["TP"] + cm["FN"]),
                np.nan)
            cm["Specificity"] = np.where(
                (cm["TN"] + cm["FP"]) > 0,
                cm["TN"] / (cm["TN"] + cm["FP"]),
                np.nan)
            merged = m.merge(
                cm[["Toxicity", "TP", "TN", "FP", "FN",
                    "Sensitivity", "Specificity"]],
                on="Toxicity", how="left")
            data[(model, split)] = merged
    return data
print("Loading results...")
DATA = load_all()
print(f"  Loaded {len(DATA)} model/split combinations.")
EP_LABELS, EP_DISPLAY = detect_ep_labels()
print(f"  Endpoints from CSV: {EP_LABELS}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SPLIT DEGRADATION PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2 — SPLIT DEGRADATION PLOT")
print("="*80)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric in zip(axes, ["MCC", "ROC-AUC"]):
    for model in MODELS:
        means = []
        for split in SPLITS:
            df = DATA.get((model, split))
            means.append(df[metric].mean()
                         if df is not None and metric in df.columns else np.nan)
        ax.plot(SPLITS, means, marker="o", linewidth=2.5,
                label=PRETTY[model], alpha=0.9)
    ax.set_title(f"Mean {metric} across splits", fontsize=13, fontweight="bold")
    ax.set_xlabel("Split strategy", fontsize=ANNOT_FS)
    ax.set_ylabel(f"Mean {metric}", fontsize=ANNOT_FS)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Random", "Scaffold", "Clustering"], fontsize=ANNOT_FS - 1)
    ax.tick_params(axis="both", labelsize=ANNOT_FS - 1)
    ax.legend(fontsize=LEGEND_FS,
              frameon=True,
              facecolor="white",
              edgecolor="black",
              loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#F8F8F8")
fig.suptitle("Performance Degradation: Random → Scaffold → Clustering\n"
             "(Drop indicates loss of generalization due to structural novelty)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = f"{OUT_DIR}/step2_split_degradation.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
