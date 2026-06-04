"""
plot_mcc_heatmap_grid.py  fixed layout, larger fonts, no overlaps
Scaffold split fix: acyclic / invalid molecules fall back to their own
SMILES instead of "" so they are not all lumped into one scaffold group.
Clustering split fix: Ward (n=150) + GroupKFold(n_splits=5), fold 0
(consistent with cells 3, 4, 11, 18, 19, 20, 21).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import sqrt
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not found, scaffold split falls back to random.")


# ── FILE PATHS ────────────────────────────────────────────────────────────────
TRAIN_FILE        = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE      = "mordred_features_cached.csv"
WITHDRAWN_FILE    = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED = "mordred_withdrawn_cached.csv"
OUT_FILE          = "figure_mcc_heatmap_grid.png"

UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_REASON_COL = "toxtype"

SPLITS       = ["random", "scaffold", "clustering"]
SPLIT_LABELS = ["Random", "Scaffold", "Clustering"]
RF_PARAMS    = dict(n_estimators=300, max_features="sqrt",
                    class_weight="balanced", random_state=42, n_jobs=-1)
RANDOM_STATE = 42
RNG          = np.random.default_rng(RANDOM_STATE)
N_CLUSTERS   = 150
N_SPLITS     = 5

TITLE_FS     = 22
COL_HEAD_FS  = 19
ROW_LABEL_FS = 18
MCC_FS       = 20
SENS_FS      = 16

ENDPOINTS        = None
ENDPOINT_DISPLAY = None


def make_display_label(col):
    return col.replace("__binary", "").replace("_", " ").title()


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


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_mcc(TP, TN, FP, FN):
    d = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return (TP * TN - FP * FN) / d if d else 0.0


def compute_sens(TP, FN):
    return TP / (TP + FN) if (TP + FN) else 0.0


def compute_spec(TN, FP):
    return TN / (TN + FP) if (TN + FP) else 0.0


# ── SCAFFOLD HELPER (acyclic fix) ─────────────────────────────────────────────
def compute_scaffold(smiles):
    """
    Murcko scaffold of a molecule. Acyclic or invalid molecules fall back
    to their original SMILES (singleton group), preventing them from being
    lumped under a single empty-string scaffold key.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaf if scaf else smiles
    except Exception:
        return smiles


# ── ENDPOINT / TOXTYPE UTILITIES ──────────────────────────────────────────────
def build_toxtype_map(endpoints):
    ep_base = {ep.replace("__binary", ""): ep for ep in endpoints}
    tmap = {}
    for token, base in ALIAS.items():
        if base in ep_base:
            tmap[token] = ep_base[base]
    for base, col in ep_base.items():
        if base not in tmap.values():
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


# ── PRE-PROCESSING ────────────────────────────────────────────────────────────
def preprocess(X_train, X_test, X_ext):
    shared  = X_train.columns.intersection(X_test.columns).intersection(X_ext.columns)
    X_train = X_train[shared].copy()
    X_test  = X_test[shared].copy()
    X_ext   = X_ext[shared].copy()

    keep    = X_train.isnull().mean() <= 0.5
    X_train = X_train.loc[:, keep]
    X_test  = X_test.loc[:, keep]
    X_ext   = X_ext.loc[:, keep]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)
    X_ext   = X_ext.fillna(medians)

    nonzero = X_train.var() > 0
    X_train = X_train.loc[:, nonzero]
    X_test  = X_test.loc[:, nonzero]
    X_ext   = X_ext.loc[:, nonzero]
    return X_train.values, X_test.values, X_ext.values


# ── SPLIT STRATEGIES ──────────────────────────────────────────────────────────
def random_split(n):
    return train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)


def scaffold_split(smiles_list, test_size=0.2, rng=RNG):
    """Murcko scaffold split with acyclic fix; no size bias."""
    if not RDKIT_OK:
        return random_split(len(smiles_list))
    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = compute_scaffold(smi)
        scaf2idx[scaf].append(i)
    scaffold_groups = list(scaf2idx.values())
    rng.shuffle(scaffold_groups)
    n_test = int(len(smiles_list) * test_size)
    te, tr = [], []
    for group in scaffold_groups:
        (te if len(te) < n_test else tr).extend(group)
    return np.array(tr), np.array(te)


def mordred_clustering_split(X, n_clusters=N_CLUSTERS, n_splits=N_SPLITS, fold=0):
    """
    Ward clustering on standardised Mordred descriptors with n=150 clusters,
    then GroupKFold(n_splits=5). Returns fold 0 as the canonical 80/20 split.
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


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    global ENDPOINTS, ENDPOINT_DISPLAY
    print("Loading UniTox...")
    df_raw      = pd.read_csv(TRAIN_FILE).dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
    mordred_raw = pd.read_csv(MORDRED_FILE)
    assert len(mordred_raw) == len(df_raw)

    if RDKIT_OK:
        keep_mask = [
            compute_scaffold(smi) != ""
            for smi in df_raw[UNITOX_SMILES_COL]
        ]
        df      = df_raw[keep_mask].reset_index(drop=True)
        mordred = mordred_raw[keep_mask].reset_index(drop=True)
    else:
        df, mordred = df_raw, mordred_raw

    ENDPOINTS        = sorted([c for c in df.columns if c.endswith("__binary")])
    ENDPOINT_DISPLAY = [make_display_label(ep) for ep in ENDPOINTS]
    mordred = mordred.apply(pd.to_numeric, errors="coerce")

    withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
    mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED)
    assert len(mordred_w_raw) == len(withdrawn_raw)

    valid_mask = withdrawn_raw[WITHDRAWN_SMILES_COL].notna()
    withdrawn  = withdrawn_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w.apply(pd.to_numeric, errors="coerce")

    common_cols = mordred.columns.intersection(mordred_w.columns)
    mordred     = mordred[common_cols].reset_index(drop=True)
    mordred_w   = mordred_w[common_cols].reset_index(drop=True)

    toxtype_map = build_toxtype_map(ENDPOINTS)
    for ep in ENDPOINTS:
        withdrawn[ep] = 0
    for i, row in withdrawn.iterrows():
        for ep in parse_toxtype(row[WITHDRAWN_REASON_COL], toxtype_map):
            withdrawn.at[i, ep] = 1

    unitox_smiles = set(df[UNITOX_SMILES_COL].str.strip())
    overlap   = withdrawn[WITHDRAWN_SMILES_COL].str.strip().isin(unitox_smiles)
    withdrawn = withdrawn[~overlap].reset_index(drop=True)
    mordred_w = mordred_w[~overlap].reset_index(drop=True)

    print(f"  Final external set: {len(withdrawn)} drugs")
    return df, mordred, withdrawn, mordred_w


# ── MODEL TRAINING / EVALUATION ───────────────────────────────────────────────
def run_split(split_name, df, mordred, withdrawn, mordred_w):
    print(f"\n--- Split: {split_name} ---")
    smiles = df[UNITOX_SMILES_COL].tolist()
    X_all  = np.nan_to_num(mordred.values.astype(np.float64), nan=0.0)

    if   split_name == "random":     train_idx, test_idx = random_split(len(df))
    elif split_name == "scaffold":   train_idx, test_idx = scaffold_split(smiles, rng=RNG)
    else:                            train_idx, test_idx = mordred_clustering_split(X_all)

    X_train, _, X_ext = preprocess(
        mordred.iloc[train_idx], mordred.iloc[test_idx], mordred_w
    )

    ep_results = {}
    for ep in ENDPOINTS:
        y_train   = df[ep].iloc[train_idx].values
        y_ext     = withdrawn[ep].values
        threshold = float(y_train.sum()) / len(y_train) if y_train.sum() > 0 else 0.5

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        y_pred = (rf.predict_proba(X_ext)[:, 1] >= threshold).astype(int)

        TP = int(np.sum((y_pred == 1) & (y_ext == 1)))
        TN = int(np.sum((y_pred == 0) & (y_ext == 0)))
        FP = int(np.sum((y_pred == 1) & (y_ext == 0)))
        FN = int(np.sum((y_pred == 0) & (y_ext == 1)))

        ep_results[ep] = dict(
            mcc=compute_mcc(TP, TN, FP, FN),
            sens=compute_sens(TP, FN),
            spec=compute_spec(TN, FP),
            TP=TP, TN=TN, FP=FP, FN=FN,
        )
        print(f"  {ep:<38} MCC={ep_results[ep]['mcc']:+.3f}")
    return ep_results


# ── FIGURE ────────────────────────────────────────────────────────────────────
def plot_mcc_heatmap_grid(results_by_split):
    n_ep = len(ENDPOINTS)
    n_sp = len(SPLITS)

    mcc_mat  = np.zeros((n_ep, n_sp))
    sens_mat = np.zeros((n_ep, n_sp))
    spec_mat = np.zeros((n_ep, n_sp))

    for ci, split in enumerate(SPLITS):
        for ri, ep in enumerate(ENDPOINTS):
            res = results_by_split[split][ep]
            mcc_mat[ri, ci]  = res["mcc"]
            sens_mat[ri, ci] = res["sens"]
            spec_mat[ri, ci] = res["spec"]

    vmax = max(np.abs(mcc_mat).max(), 0.05)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cell_w = 3.4
    cell_h = 1.6
    lpad   = 2.6
    rpad   = 1.4
    tpad   = 1.6
    bpad   = 0.4

    fig_w = lpad + n_sp * cell_w + rpad
    fig_h = tpad + n_ep * cell_h + bpad

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    l = lpad / fig_w
    b = bpad / fig_h
    w = (n_sp * cell_w) / fig_w
    h = (n_ep * cell_h) / fig_h
    ax = fig.add_axes([l, b, w, h])

    im = ax.imshow(mcc_mat, cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest")

    for x in range(1, n_sp):
        ax.axvline(x - 0.5, color="white", linewidth=2.5)
    for y in range(1, n_ep):
        ax.axhline(y - 0.5, color="white", linewidth=2.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("#444444")
    ax.set_xticks([])
    ax.set_yticks([])

    for ri in range(n_ep):
        for ci in range(n_sp):
            mcc  = mcc_mat[ri, ci]
            sens = sens_mat[ri, ci]
            spec = spec_mat[ri, ci]
            sign = "+" if mcc >= 0 else ""
            rgba = cmap(norm(mcc))
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_col     = "white" if lum < 0.45 else "black"
            txt_col_sub = "white" if lum < 0.45 else "#333333"
            ax.text(ci, ri - 0.13,
                    f"MCC {sign}{mcc:.2f}",
                    ha="center", va="center",
                    fontsize=MCC_FS, fontweight="bold", color=txt_col)
            ax.text(ci, ri + 0.22,
                    f"Sens {sens:.2f}   Spec {spec:.2f}",
                    ha="center", va="center",
                    fontsize=SENS_FS, fontweight="normal", color=txt_col_sub)

    col_y_fig = b + h + (0.38 / fig_h)
    for ci, label in enumerate(SPLIT_LABELS):
        col_x_fig = l + w * (ci + 0.5) / n_sp
        fig.text(col_x_fig, col_y_fig, label,
                 ha="center", va="bottom",
                 fontsize=COL_HEAD_FS, fontweight="bold")

    for ri, disp in enumerate(ENDPOINT_DISPLAY):
        row_y_fig = b + h * (1 - (ri + 0.5) / n_ep)
        row_x_fig = l - (0.12 / fig_w)
        fig.text(row_x_fig, row_y_fig, disp,
                 ha="right", va="center",
                 fontsize=ROW_LABEL_FS)

    title_y_fig = b + h + (0.95 / fig_h)
    fig.text(0.5, title_y_fig,
             "External Validation: Per-Endpoint MCC by Split Strategy\nRF, Withdrawn 2.0",
             ha="center", va="bottom",
             fontsize=TITLE_FS, fontweight="bold")

    cbar_l  = l + w + 0.025
    cbar_b  = b + 0.05 * h
    cbar_w  = 0.022
    cbar_h  = 0.90 * h
    cbar_ax = fig.add_axes([cbar_l, cbar_b, cbar_w, cbar_h])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MCC", fontsize=ROW_LABEL_FS, labelpad=8)
    cbar.ax.tick_params(labelsize=ROW_LABEL_FS - 1)
    cbar.ax.axhline(y=norm(0), color="black", linewidth=1.2,
                    linestyle="--", alpha=0.6)

    fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {OUT_FILE}")
    plt.show()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    df, mordred, withdrawn, mordred_w = load_data()
    results_by_split = {}
    for split in SPLITS:
        results_by_split[split] = run_split(split, df, mordred, withdrawn, mordred_w)
    plot_mcc_heatmap_grid(results_by_split)
    print("Done.")


if __name__ == "__main__":
    main()
