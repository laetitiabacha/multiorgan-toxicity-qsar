"""
12_rf_performance_vs_similarity.py
====================================
Trains Random Forest for each split strategy and plots MCC as a function
of average Tanimoto similarity between test and training sets.
Shows that splits with lower train-test similarity yield lower MCC.

Figure:
  Figure_HitRate_vs_Similarity.png

REQUIRES: UniTox_with_recovered_typos_v3.csv, mordred_features_cached.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

UNITOX_FILE  = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE = "mordred_features_cached.csv"
SMILES_COL   = "SMILES_filled"
RANDOM_STATE = 42
N_CLUSTERS   = 150
OUT_FIG      = "Figure_HitRate_vs_Similarity.png"
RF_PARAMS    = dict(n_estimators=300, max_features="sqrt",
                    class_weight="balanced", random_state=42, n_jobs=-1)

SPLIT_COLOURS = {"random": "#4C8BE0", "scaffold": "#E05050", "clustering": "#5BBF5B"}
SPLIT_LABELS  = {"random": "Random", "scaffold": "Scaffold", "clustering": "Clustering"}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(UNITOX_FILE).dropna(subset=[SMILES_COL]).reset_index(drop=True)
X  = np.nan_to_num(pd.read_csv(MORDRED_FILE).values.astype(np.float64))
assert len(df) == len(X)
smiles    = df[SMILES_COL].tolist()
ENDPOINTS = [c for c in df.columns if c.endswith("__binary")]
EP_LABELS = [
    "Cardiotoxicity", "Dermatological", "Hematological",
    "Infertility", "Liver Toxicity", "Ototoxicity",
    "Pulmonary Toxicity", "Renal Toxicity",
]
print(f"  {len(df)} molecules | {len(ENDPOINTS)} endpoints")

# ── Split functions ───────────────────────────────────────────────────────────
def split_random(n):
    tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)
    return list(tr), list(te)


def split_scaffold(smiles):
    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            if not sc: sc = smi
        except Exception:
            sc = smi
        scaf2idx[sc].append(i)
    n_test = int(len(smiles) * 0.2)
    te, tr = [], []
    for s in sorted(scaf2idx, key=lambda s: -len(scaf2idx[s])):
        (te if len(te) < n_test else tr).extend(scaf2idx[s])
    return tr, te


def split_clustering(X, n_clusters=N_CLUSTERS):
    Xs     = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(Xs)
    rng    = np.random.default_rng(RANDOM_STATE)
    cl2idx = defaultdict(list)
    for i, c in enumerate(labels):
        cl2idx[c].append(i)
    groups = list(cl2idx.values())
    rng.shuffle(groups)
    n_test = int(len(X) * 0.2)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)
    return tr, te


splits = {
    "random":     split_random(len(df)),
    "scaffold":   split_scaffold(smiles),
    "clustering": split_clustering(X),
}

# ── Morgan fingerprints ───────────────────────────────────────────────────────
fps = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)

# ── Compute similarity + MCC per split × endpoint ────────────────────────────
results = []
for split_name, (tr_idx, te_idx) in splits.items():
    tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
    X_tr, X_te = X[tr_idx], X[te_idx]

    tr_fps  = [fps[i] for i in tr_idx if fps[i] is not None]
    sims    = [max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps))
               for i in te_idx if fps[i] is not None]
    avg_sim = np.mean(sims)

    for ep_col, ep_label in zip(ENDPOINTS, EP_LABELS):
        y_tr = df[ep_col].values[tr_idx]
        y_te = df[ep_col].values[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        mcc   = matthews_corrcoef(y_te, preds)
        results.append({"split": split_name, "endpoint": ep_label,
                        "avg_sim": avg_sim, "MCC": mcc})
    print(f"  {split_name:<12} avg_sim={avg_sim:.3f}")

res_df = pd.DataFrame(results)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
ax.set_facecolor("white")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e5e5e5", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

split_order = sorted(splits.keys(),
                     key=lambda s: res_df[res_df["split"] == s]["avg_sim"].mean())

for pos, split_name in enumerate(split_order):
    sub      = res_df[res_df["split"] == split_name]
    avg_sim  = sub["avg_sim"].mean()
    mcc_vals = sub["MCC"].values
    colour   = SPLIT_COLOURS[split_name]

    ax.boxplot(mcc_vals, positions=[pos], widths=0.4,
               patch_artist=True, manage_ticks=False, zorder=3,
               boxprops=dict(facecolor=colour, color="black", linewidth=1.2),
               medianprops=dict(color="black", linewidth=2.0),
               whiskerprops=dict(color="black", linewidth=1.2),
               capprops=dict(color="black", linewidth=1.2),
               flierprops=dict(marker="", markersize=0))

    np.random.seed(42)
    jitter = np.random.normal(0, 0.05, size=len(mcc_vals))
    ax.scatter(pos + jitter, mcc_vals, color=colour,
               edgecolors="black", linewidths=0.8, s=60, alpha=0.85, zorder=2)
    print(f"  {split_name:<12} median MCC = {np.median(mcc_vals):.3f}")

ax.set_xticks(range(len(split_order)))
ax.set_xticklabels(
    [f"{res_df[res_df['split']==s]['avg_sim'].mean():.3f}" for s in split_order],
    fontsize=12)
ax.set_xlabel("Avg. Max Tanimoto Similarity to Training Set", fontsize=12)
ax.set_ylabel("MCC (RF, per endpoint)", fontsize=12)
ax.set_title("RF Performance vs. Training Set Similarity\n"
             "(each dot = one toxicity endpoint)",
             fontsize=13, fontweight="bold")
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.3)

handles = [mpatches.Patch(facecolor=SPLIT_COLOURS[s], alpha=0.85,
                           label=SPLIT_LABELS[s]) for s in split_order]
ax.legend(handles=handles, fontsize=11, frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_FIG}")
