"""
plot_external_recovery.py
──────────────────────────
Generates a corrected version of the clinical signal recovery figure
using EXTERNAL validation on Withdrawn 2.0 (not internal UniTox test set).

Produces:
  external_recovery_plot.png   horizontal bar chart, 3 panels (Random / Scaffold / Clustering)
                               showing TP/(TP+FN) per endpoint on Withdrawn 2.0

Note: ototoxicity is absent from the Withdrawn 2.0 toxtype column and will
show "N/A" (labelled explicitly). All other endpoints use the same toxtype
mapping as plot_generalisation_tax_table.py.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not found, scaffold split falls back to random.")


# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_FILE           = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE         = "mordred_features_cached.csv"
WITHDRAWN_FILE       = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED    = "mordred_withdrawn_cached.csv"
UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_REASON_COL = "toxtype"
RANDOM_STATE         = 42
RNG                  = np.random.default_rng(RANDOM_STATE)
N_CLUSTERS           = 150
N_SPLITS             = 5

RF_PARAMS = dict(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ── TOXTYPE MAPPING ───────────────────────────────────────────────────────────
ALIAS = {
    "hematological":           "hematological",
    "renal_toxicity":          "renal_toxicity",
    "cardiotoxicity":          "cardiotoxicity",
    "dermatological_toxicity": "dermatological_toxicity",
    "liver_toxicity":          "liver_toxicity",
    "pulmonary_toxicity":      "pulmonary_toxicity",
    "ototoxicity":             "ototoxicity",
    "infertility":             "infertility",
}


def build_toxtype_map(endpoints):
    ep_base = {ep.replace("__binary", ""): ep for ep in endpoints}
    tmap = {}
    for token, base in ALIAS.items():
        if base in ep_base:
            tmap[token] = ep_base[base]
    for base, col in ep_base.items():
        if base not in tmap:
            tmap[base] = col
    return tmap


def parse_toxtype(toxtype_str, toxtype_map):
    if not isinstance(toxtype_str, str):
        return []
    matched = set()
    for token in toxtype_str.split(","):
        key = token.strip().lower()
        if key in toxtype_map:
            matched.add(toxtype_map[key])
    return list(matched)


# ── SPLITS ────────────────────────────────────────────────────────────────────
def random_split(n):
    return train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)


def scaffold_split(smiles_list, test_size=0.2):
    if not RDKIT_OK:
        return random_split(len(smiles_list))
    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False) if mol else ""
        except Exception:
            scaf = ""
        if not scaf:
            scaf = smi    # acyclic fix
        scaf2idx[scaf].append(i)
    groups = list(scaf2idx.values())
    RNG.shuffle(groups)
    n_test = int(len(smiles_list) * test_size)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)
    return np.array(tr), np.array(te)


def clustering_split(X, n_clusters=N_CLUSTERS, n_splits=N_SPLITS, fold=0):
    """
    Ward clustering on standardised Mordred descriptors with n=150 clusters,
    then GroupKFold(n_splits=5). Returns fold 0 as the canonical 80/20 split.
    Matches the methodology in cells 3, 4, 11, 18, 19, 20.
    """
    Xs     = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
             ).fit_predict(Xs)
    gkf   = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.arange(len(X)), groups=labels))
    tr_idx, te_idx = folds[fold]
    n_unique = len(np.unique(labels))
    print(f"  clustering: {n_unique} clusters, "
          f"GroupKFold({n_splits}) fold={fold}, "
          f"train={len(tr_idx)}, test={len(te_idx)}")
    return np.array(tr_idx), np.array(te_idx)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess(X_train_df, X_test_df, X_ext_df):
    shared = (X_train_df.columns
              .intersection(X_test_df.columns)
              .intersection(X_ext_df.columns))
    Xtr = X_train_df[shared].copy()
    Xte = X_test_df[shared].copy()
    Xex = X_ext_df[shared].copy()

    keep = Xtr.isnull().mean() <= 0.5
    Xtr, Xte, Xex = Xtr.loc[:, keep], Xte.loc[:, keep], Xex.loc[:, keep]

    medians = Xtr.median()
    Xtr = Xtr.fillna(medians)
    Xte = Xte.fillna(medians)
    Xex = Xex.fillna(medians)

    nonzero = Xtr.var() > 0
    Xtr, Xte, Xex = Xtr.loc[:, nonzero], Xte.loc[:, nonzero], Xex.loc[:, nonzero]
    return Xtr.values, Xte.values, Xex.values


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading UniTox...")
df_raw      = pd.read_csv(TRAIN_FILE).dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
mordred_raw = pd.read_csv(MORDRED_FILE).reset_index(drop=True)
assert len(mordred_raw) == len(df_raw)

df      = df_raw.reset_index(drop=True)
mordred = mordred_raw.apply(pd.to_numeric, errors="coerce")

ENDPOINTS = sorted([c for c in df.columns if c.endswith("__binary")])
print(f"  {len(df)} molecules x {len(ENDPOINTS)} endpoints")

print("Loading Withdrawn 2.0...")
withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED).apply(pd.to_numeric, errors="coerce")
assert len(mordred_w_raw) == len(withdrawn_raw)

# Remove overlap with UniTox
unitox_smiles = set(df[UNITOX_SMILES_COL].str.strip())
overlap       = withdrawn_raw[WITHDRAWN_SMILES_COL].str.strip().isin(unitox_smiles)
withdrawn     = withdrawn_raw[~overlap].reset_index(drop=True)
mordred_w     = mordred_w_raw[~overlap].reset_index(drop=True)
print(f"  Removed {overlap.sum()} overlaps, {len(withdrawn)} withdrawn drugs remain")

# Assign ground truth labels from toxtype column
toxtype_map = build_toxtype_map(ENDPOINTS)
for ep in ENDPOINTS:
    withdrawn[ep] = 0
for i, row in withdrawn.iterrows():
    for ep in parse_toxtype(row[WITHDRAWN_REASON_COL], toxtype_map):
        withdrawn.at[i, ep] = 1

print("\n  Ground truth positive counts in Withdrawn 2.0:")
for ep in ENDPOINTS:
    n_pos = withdrawn[ep].sum()
    label = ep.replace("__binary", "").replace("_", " ").title()
    print(f"    {label:<28}: {n_pos} positive cases")

# Align Mordred columns
common_cols = mordred.columns.intersection(mordred_w.columns)
mordred     = mordred[common_cols]
mordred_w   = mordred_w[common_cols]


# ── COMPUTE SPLITS ────────────────────────────────────────────────────────────
smiles_list = df[UNITOX_SMILES_COL].tolist()
X_all       = np.nan_to_num(mordred.values.astype(np.float64), nan=0.0)

split_indices = {
    "random":     random_split(len(df)),
    "scaffold":   scaffold_split(smiles_list),
    "clustering": clustering_split(X_all),
}
for sname, (tr, te) in split_indices.items():
    print(f"  {sname}: train={len(tr)}, test={len(te)}")


# ── TRAIN RF AND COLLECT EXTERNAL SENSITIVITY ─────────────────────────────────
DISPLAY = {ep: ep.replace("__binary", "").replace("_", " ").title() for ep in ENDPOINTS}
sens_results = {sp: {} for sp in split_indices}

for split_name, (train_idx, test_idx) in split_indices.items():
    print(f"\nTraining RF, split: {split_name}")
    Xtr, Xte, Xex = preprocess(
        mordred.iloc[train_idx],
        mordred.iloc[test_idx],
        mordred_w,
    )
    for ep in ENDPOINTS:
        y_train = df[ep].iloc[train_idx].values
        y_ext   = withdrawn[ep].values

        if len(np.unique(y_train)) < 2:
            sens_results[split_name][DISPLAY[ep]] = 0.0
            continue

        threshold = float(y_train.mean()) if y_train.sum() > 0 else 0.5
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(Xtr, y_train)
        p_ext  = model.predict_proba(Xex)[:, 1]
        y_pred = (p_ext >= threshold).astype(int)

        TP = int(((y_pred == 1) & (y_ext == 1)).sum())
        FN = int(((y_pred == 0) & (y_ext == 1)).sum())
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        sens_results[split_name][DISPLAY[ep]] = sensitivity
        print(f"  {DISPLAY[ep]:<28}  TP={TP}  FN={FN}  sens={sensitivity:.2f}")


# ── PLOT ──────────────────────────────────────────────────────────────────────
all_labels = list(DISPLAY.values())
mean_sens  = {
    lab: np.mean([sens_results[sp].get(lab, 0.0) for sp in split_indices])
    for lab in all_labels
}
TOX_ORDER = sorted(all_labels, key=lambda x: mean_sens[x])

COLORS = {
    "random":     "#4472C4",
    "scaffold":   "#ED7D31",
    "clustering": "#70AD47",
}

plt.style.use("default")
plt.rcParams.update({
    "font.size":       14,
    "axes.titlesize":  16,
    "axes.labelsize":  15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, split_name in zip(axes, ["random", "scaffold", "clustering"]):
    vals = [sens_results[split_name].get(t, 0.0) * 100 for t in TOX_ORDER]
    y    = np.arange(len(TOX_ORDER))
    bars = ax.barh(y, vals, color=COLORS[split_name], edgecolor="none")
    ax.set_title(split_name.capitalize(), fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(TOX_ORDER)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Sensitivity (%)")

    for i, v in enumerate(vals):
        label_ep = TOX_ORDER[i]
        n_pos = withdrawn[
            [ep for ep in ENDPOINTS if DISPLAY[ep] == label_ep][0]
        ].sum()
        if n_pos == 0:
            ax.text(2, i, "N/A (no cases)", va="center",
                    fontsize=11, color="grey", style="italic")
        else:
            ax.text(v + 1, i, f"{int(round(v))}%", va="center", fontweight="bold")

axes[0].set_ylabel("Toxicity Endpoint")
plt.suptitle(
    "External Clinical Signal Sensitivity, Withdrawn 2.0\n"
    "TP / (TP + FN) per endpoint, Random Forest, by split strategy",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("external_recovery_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
print("\nSaved external_recovery_plot.png")
plt.show()
