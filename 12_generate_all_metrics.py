"""
12_generate_all_metrics.py
===========================
Runs repeated cross-validation for Random Forest across all three split
strategies, collecting ROC-AUC, AUPRC, MCC, and Accuracy per fold.
Produces grouped boxplots (across splits) and per-endpoint boxplots (Figure 6).

Figures:
  Figure_grouped_boxplot.png          — 4 metrics × 3 splits, median annotated
  Figure6_random.png                  — horizontal boxplots per endpoint, random
  Figure6_scaffold.png                — horizontal boxplots per endpoint, scaffold
  Figure6_clustering.png              — horizontal boxplots per endpoint, clustering

REQUIRES: UniTox_with_recovered_typos_v3.csv, mordred_features_cached.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GroupKFold
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                             average_precision_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# ── Config ────────────────────────────────────────────────────────────────────
N_FOLDS      = 5
N_REPEATS    = 2
N_CLUSTERS   = 150
N_TOP_FEAT   = 300
RANDOM_STATE = 42

COLOURS     = {"random": "#4C8BE0", "scaffold": "#E05050", "clustering": "#5BBF5B"}
LABELS      = {"random": "Random Split", "scaffold": "Scaffold Split",
               "clustering": "Clustering Split"}
SPLIT_ORDER  = ["random", "scaffold", "clustering"]
METRIC_ORDER = ["ROC-AUC", "AUPRC", "MCC", "Accuracy"]
FIG6_METRICS = ["ROC-AUC", "AUPRC", "MCC"]

EP_SHORT = {
    "cardiotoxicity__binary":          "Cardiotoxicity",
    "dermatological_toxicity__binary": "Dermatological toxicity",
    "hematological__binary":           "Hematological toxicity",
    "infertility__binary":             "Infertility",
    "liver_toxicity__binary":          "Liver toxicity",
    "ototoxicity__binary":             "Ototoxicity",
    "pulmonary_toxicity__binary":      "Pulmonary toxicity",
    "renal_toxicity__binary":          "Renal toxicity",
}
RF_PARAMS = dict(n_estimators=100, max_features="sqrt",
                 class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = (pd.read_csv("UniTox_with_recovered_typos_v3.csv")
            .dropna(subset=["SMILES_filled"])
            .reset_index(drop=True))
    X  = np.nan_to_num(pd.read_csv("mordred_features_cached.csv")
                          .values.astype(np.float64))
    smiles    = df["SMILES_filled"].tolist()
    endpoints = sorted([c for c in df.columns if c.endswith("__binary")])
    # Pre-filter by variance
    top_idx = np.argsort(X.var(axis=0))[-N_TOP_FEAT:]
    X       = X[:, top_idx]
    print(f"Loaded {len(df)} molecules × {X.shape[1]} top-variance descriptors")
    return df, X, smiles, endpoints


def preprocess(X_tr, X_te):
    keep = X_tr.var(axis=0) > 0
    sc   = StandardScaler()
    return sc.fit_transform(X_tr[:, keep]), sc.transform(X_te[:, keep])


def get_scaffold_groups(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if not sc:
                sc = str(smi)
        except Exception:
            sc = str(smi)
        groups.append(sc)
    unique = {s: i for i, s in enumerate(dict.fromkeys(groups))}
    return np.array([unique[g] for g in groups])


def get_cluster_groups(X):
    sc = StandardScaler()
    return AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric="euclidean", linkage="ward"
    ).fit_predict(sc.fit_transform(X))


def evaluate_fold(X_tr, X_te, y_tr, y_te):
    nan_row = {m: np.nan for m in METRIC_ORDER}
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return nan_row
    clf   = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= np.mean(y_tr)).astype(int)
    return {
        "ROC-AUC":  roc_auc_score(y_te, proba),
        "AUPRC":    average_precision_score(y_te, proba),
        "MCC":      matthews_corrcoef(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
    }


def run_random(df, X, endpoints):
    rows = []
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
    idx = np.arange(len(df))
    for ep in endpoints:
        y = df[ep].values
        for tr_idx, te_idx in rskf.split(idx, y):
            X_tr, X_te = preprocess(X[tr_idx], X[te_idx])
            m = evaluate_fold(X_tr, X_te, y[tr_idx], y[te_idx])
            for met, val in m.items():
                rows.append({"split": "random", "endpoint": ep, "metric": met, "value": val})
        print(f"  [random]      {ep}: mean MCC = {np.nanmean([r['value'] for r in rows if r['endpoint']==ep and r['metric']=='MCC']):.3f}", flush=True)
    return rows


def run_group(df, X, endpoints, groups, label):
    rows = []
    gkf  = GroupKFold(n_splits=N_FOLDS)
    idx  = np.arange(len(df))
    for ep in endpoints:
        y = df[ep].values
        for tr_idx, te_idx in gkf.split(idx, y, groups):
            X_tr, X_te = preprocess(X[tr_idx], X[te_idx])
            m = evaluate_fold(X_tr, X_te, y[tr_idx], y[te_idx])
            for met, val in m.items():
                rows.append({"split": label, "endpoint": ep, "metric": met, "value": val})
        ep_mcc = [r["value"] for r in rows if r["endpoint"] == ep and r["metric"] == "MCC" and r["split"] == label]
        print(f"  [{label:10s}] {ep}: mean MCC = {np.nanmean(ep_mcc):.3f}", flush=True)
    return rows


# ── Plot helpers ──────────────────────────────────────────────────────────────
def style_ax(ax, orient="vertical"):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(1.3)
    ax.tick_params(colors="black", width=1.2, length=5, labelsize=12)
    if orient == "vertical":
        ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    else:
        ax.xaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_bp_vertical(ax, data, pos, colour, width=0.22):
    clean = np.array([v for v in data if not np.isnan(v)])
    if len(clean) == 0: clean = np.array([np.nan])
    bp = ax.boxplot([clean], positions=[pos], widths=width, vert=True,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2.2),
                    whiskerprops=dict(color="black", linewidth=1.2),
                    capprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3.5,
                                   markerfacecolor="gray",
                                   markeredgecolor="none", alpha=0.5),
                    zorder=3)
    bp["boxes"][0].set_facecolor(colour); bp["boxes"][0].set_alpha(0.85)
    return float(np.nanmedian(clean)), bp["whiskers"][1].get_ydata()[1]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, X, smiles, endpoints = load_data()

    print("\nBuilding scaffold and clustering groups ...")
    scaf_groups = get_scaffold_groups(smiles)
    clus_groups = get_cluster_groups(X)

    print("\nRunning random CV ...")
    all_rows = run_random(df, X, endpoints)
    print("\nRunning scaffold CV ...")
    all_rows += run_group(df, X, endpoints, scaf_groups, "scaffold")
    print("\nRunning clustering CV ...")
    all_rows += run_group(df, X, endpoints, clus_groups, "clustering")

    results = pd.DataFrame(all_rows)

    # ── Figure: grouped boxplot ───────────────────────────────────────────────
    splits  = SPLIT_ORDER
    metrics = METRIC_ORDER
    x       = np.arange(len(splits))
    width   = 0.22

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), facecolor="white")
    fig.suptitle("Random Forest: Repeated CV Performance by Split Strategy",
                 fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        style_ax(ax)
        offsets = np.linspace(-(len(splits)-1)*width/2,
                               (len(splits)-1)*width/2, len(splits))
        for i, split in enumerate(splits):
            data = results[(results["split"] == split) &
                           (results["metric"] == metric)]["value"].dropna().values
            med, top = draw_bp_vertical(ax, data, x[i] + offsets[i], COLOURS[split])
            ax.text(x[i] + offsets[i], top + 0.015, f"{med:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in splits], fontsize=9, rotation=15)
        ax.set_ylim(-0.3, 1.1)

    handles = [mpatches.Patch(facecolor=COLOURS[s], alpha=0.85, label=LABELS[s])
               for s in splits]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.savefig("Figure_grouped_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved → Figure_grouped_boxplot.png")

    # ── Figure 6: per-endpoint horizontal boxplots ────────────────────────────
    ep_labels_short = [EP_SHORT.get(ep, ep) for ep in endpoints]

    for split in SPLIT_ORDER:
        fig, axes = plt.subplots(1, len(FIG6_METRICS), figsize=(16, 6), facecolor="white")
        fig.suptitle(f"Random Forest per Endpoint — {LABELS[split]}",
                     fontsize=13, fontweight="bold")
        colour = COLOURS[split]

        for ax, metric in zip(axes, FIG6_METRICS):
            style_ax(ax, orient="horizontal")
            y_positions = np.arange(len(endpoints))
            for j, ep in enumerate(endpoints):
                data = results[(results["split"] == split) &
                               (results["endpoint"] == ep) &
                               (results["metric"] == metric)]["value"].dropna().values
                if len(data) == 0: continue
                bp = ax.boxplot([data], positions=[y_positions[j]], widths=0.5,
                                vert=False, patch_artist=True,
                                medianprops=dict(color="black", linewidth=2),
                                whiskerprops=dict(color="black", linewidth=1),
                                capprops=dict(color="black", linewidth=1),
                                flierprops=dict(marker="o", markersize=3, alpha=0.4),
                                zorder=3)
                bp["boxes"][0].set_facecolor(colour); bp["boxes"][0].set_alpha(0.8)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(ep_labels_short, fontsize=10)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"Figure6_{split}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved → Figure6_{split}.png")

    print("\n✓ Done.")
