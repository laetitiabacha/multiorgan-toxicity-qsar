"""
07_svm_kernel_comparison.py
============================
Trains SVM with three kernels (Linear, RBF, Polynomial) across all three
data splits and all 8 endpoints. Produces a grouped bar chart comparing
ROC-AUC, AUPRC, MCC, and Accuracy.

Figure:
  Figure_SVM_comparison_errorbars.png

REQUIRES: UniTox_with_recovered_typos_v3.csv, mordred_features_cached.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

UNITOX_FILE  = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE = "mordred_features_cached.csv"
RANDOM_STATE = 42
N_CLUSTERS   = 150

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = (pd.read_csv(UNITOX_FILE)
        .dropna(subset=["SMILES_filled"])
        .reset_index(drop=True))
X = np.nan_to_num(pd.read_csv(MORDRED_FILE).values.astype(float))
smiles    = df["SMILES_filled"].values
ENDPOINTS = [c for c in df.columns if c.endswith("__binary")]
EP_LABELS = [c.replace("__binary", "").replace("_", " ").title() for c in ENDPOINTS]
print(f"  {len(df)} molecules × {X.shape[1]} descriptors")


# ── Split functions ───────────────────────────────────────────────────────────
def split_random(n, test_frac=0.2):
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_frac))
    return idx[:split], idx[split:]


def split_scaffold(smiles, test_frac=0.2):
    scaffold_map = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sc = "__INVALID__"
        elif not any(atom.IsInRing() for atom in mol.GetAtoms()):
            sc = "__ACYCLIC__"
        else:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_map[sc].append(i)

    n_test = int(len(smiles) * test_frac)
    te, tr = [], []
    for g in sorted(scaffold_map.values(), key=len, reverse=True):
        (te if len(te) < n_test else tr).extend(g)
    return np.array(tr), np.array(te)


def split_clustering(X, n_clusters=N_CLUSTERS, test_frac=0.2):
    Xs     = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Xs)
    cl2idx = defaultdict(list)
    for i, c in enumerate(labels):
        cl2idx[c].append(i)
    groups = sorted(cl2idx.values(), key=len, reverse=True)
    n_test = int(len(X) * test_frac)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)
    return np.array(tr), np.array(te)


splits = {
    "random":     split_random(len(df)),
    "scaffold":   split_scaffold(smiles),
    "clustering": split_clustering(X),
}

# ── Train all SVM kernels ─────────────────────────────────────────────────────
svm_configs = {
    "svm_linear": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
    "svm_rbf":    SVC(kernel="rbf",    class_weight="balanced", probability=True, random_state=42),
    "svm_poly":   SVC(kernel="poly",   class_weight="balanced", probability=True, random_state=42),
}

records = []
for split_name, (tr_idx, te_idx) in splits.items():
    X_tr, X_te = X[tr_idx], X[te_idx]
    for model_key, base_clf in svm_configs.items():
        print(f"  {split_name} / {model_key} ...")
        clf = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])
        for ep_col, ep_label in zip(ENDPOINTS, EP_LABELS):
            y_tr = df[ep_col].values[tr_idx]
            y_te = df[ep_col].values[te_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            preds = (probs >= y_tr.mean()).astype(int)
            records.append({
                "split":    split_name,
                "model":    model_key,
                "Toxicity": ep_label,
                "ROC-AUC":  roc_auc_score(y_te, probs),
                "AUPRC":    average_precision_score(y_te, probs),
                "MCC":      matthews_corrcoef(y_te, preds),
                "Accuracy": accuracy_score(y_te, preds),
            })

svm_df = pd.DataFrame(records)
print(svm_df.groupby(["split", "model"])["MCC"].mean().round(3).to_string())

# ── Plot ──────────────────────────────────────────────────────────────────────
svm_df["Model"] = svm_df["model"].map(
    {"svm_linear": "Linear", "svm_rbf": "RBF", "svm_poly": "Poly"})
summary = (svm_df
           .groupby(["split", "Model"])[["ROC-AUC", "AUPRC", "MCC", "Accuracy"]]
           .agg(["mean", "std"])
           .reset_index())

SPLITS       = ["random", "scaffold", "clustering"]
MODELS       = ["Linear", "RBF", "Poly"]
METRICS      = ["ROC-AUC", "AUPRC", "MCC", "Accuracy"]
COLORS       = {"Linear": "#4C8BE0", "RBF": "#E05050", "Poly": "#F5A623"}
SPLIT_LABELS = {"random": "Random", "scaffold": "Scaffold", "clustering": "Clustering"}

bar_w = 0.25
x     = np.arange(len(SPLITS))

fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 5), facecolor="white")
fig.suptitle("SVM Kernel Comparison Across Splits\n(mean ± std over endpoints)",
             fontsize=14, fontweight="bold")

for i, metric in enumerate(METRICS):
    ax = axes[i]
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for j, model in enumerate(MODELS):
        means, stds = [], []
        for split in SPLITS:
            row = summary[(summary["split"] == split) & (summary["Model"] == model)]
            if len(row):
                means.append(row[(metric, "mean")].values[0])
                stds.append(row[(metric, "std")].values[0])
            else:
                means.append(np.nan); stds.append(np.nan)
        pos = x + (j - 1) * bar_w
        ax.bar(pos, means, width=bar_w, color=COLORS[model], alpha=0.85, zorder=3)
        ax.errorbar(pos, means, yerr=stds, fmt="none", color="black",
                    capsize=4, linewidth=1.2, zorder=4)

    ax.set_title(metric, fontsize=18, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 1.15)

axes[0].set_ylabel("Score", fontsize=18)
handles = [mpatches.Patch(facecolor=COLORS[m], alpha=0.85, label=m) for m in MODELS]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14,
           frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.14),
           title="Kernel:", title_fontsize=14)

plt.tight_layout()
plt.savefig("Figure_SVM_comparison_errorbars.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("Saved → Figure_SVM_comparison_errorbars.png")
