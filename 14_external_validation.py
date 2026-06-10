"""
plot_external_validation.py
============================
End-to-end external validation: loads data, trains RF, computes all metrics
from actual predictions, and produces four figures.
  figure1_mcc_sensitivity.png         — MCC + Sensitivity per endpoint × split
  figure2_confusion_bars_<split>.png  — TP / FP / FN per endpoint, one panel per split
  figure3_drugs_vs_mcc.png            — Grouped bar chart: drugs per MCC bin × split
  figure4_tp_fn_stacked_<split>.png   — Stacked TP / FN per endpoint, one figure per split
Pipeline (consistent with cells 3/4/11/18):
  - Acyclic molecules retained via fix: each gets its own unique scaffold key (smi)
  - Clustering split: AgglomerativeClustering(n_clusters=150, ward) on standardised
                      Mordred descriptors, partitioned via GroupKFold(n_splits=5).
  - Scaffold split:   random-shuffled scaffold groups (no size bias)
  - Single global RNG for reproducibility
REQUIRED FILES (same directory as this script):
  UniTox_with_recovered_typos_v3.csv
  mordred_features_cached.csv
  withdrawn_external_validation.csv    columns: drugname, smiles, toxtype, predicted_labels
  mordred_withdrawn_cached.csv
"""
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    print("WARNING: RDKit not found — scaffold split falls back to random.")
# ── FILE PATHS ────────────────────────────────────────────────────────────────
TRAIN_FILE        = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE      = "mordred_features_cached.csv"
WITHDRAWN_FILE    = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED = "mordred_withdrawn_cached.csv"
# ── COLUMN NAMES ──────────────────────────────────────────────────────────────
UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_REASON_COL = "toxtype"
# ── FONT SIZE CONFIG ──────────────────────────────────────────────────────────
SUPTITLE_FS  = 18
TITLE_FS     = 14
AXLABEL_FS   = 17
TICK_FS      = 17
LEGEND_FS    = 15
ANNOT_FS     = 13
BAR_LABEL_FS = 12
# ── SPLIT / MODEL CONFIG ──────────────────────────────────────────────────────
SPLITS        = ["random", "scaffold", "clustering"]
SPLIT_LABELS  = ["Random Split", "Scaffold Split", "Clustering Split"]
SPLIT_COLOURS = ["#4472C4", "#ED7D31", "#70AD47"]
RF_PARAMS     = dict(n_estimators=300, max_features="sqrt",
                     class_weight="balanced", random_state=42, n_jobs=-1)
RANDOM_STATE  = 42
RNG           = np.random.default_rng(RANDOM_STATE)
N_CLUSTERS    = 150
N_SPLITS      = 5
# ── ENDPOINT DETECTION ────────────────────────────────────────────────────────
ENDPOINTS        = None
ENDPOINT_DISPLAY = None
def make_display_label(col: str) -> str:
    return col.replace("__binary", "").replace("_", " ").title()
# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_mcc(TP, TN, FP, FN):
    d = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP*TN - FP*FN) / d if d else 0.0
def compute_sens(TP, FN):
    return TP / (TP + FN) if (TP + FN) else 0.0
def compute_spec(TN, FP):
    return TN / (TN + FP) if (TN + FP) else 0.0
def drug_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return compute_mcc(TP, TN, FP, FN)
# ── TOXTYPE PARSING ───────────────────────────────────────────────────────────
def canon_label(s: str) -> str:
    """Normalise a label to lowercase, underscore-delimited canonical form."""
    s = str(s).lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    return s

def build_toxtype_map(endpoints: list) -> dict:
    """
    Derive the toxtype → endpoint column map directly from endpoint column names.
    No hardcoding required: strips '__binary', canonicalises, maps to full col name.
    """
    return {re.sub(r"__binary$", "", ep.lower()): ep for ep in endpoints}

def parse_toxtype(toxtype_str: str, toxtype_map: dict) -> list:
    """
    Split Withdrawn 2.0 toxtype field on semicolons, commas, and forward slashes,
    canonicalise each token, and return all matching endpoint column names.
    """
    if not isinstance(toxtype_str, str):
        return []
    matched = set()
    for token in re.split(r"[;,/]\s*", toxtype_str):
        key = canon_label(token)
        if key in toxtype_map:
            matched.add(toxtype_map[key])
    return list(matched)
# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, X_ext: pd.DataFrame):
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
# ── SPLITTING ─────────────────────────────────────────────────────────────────
def random_split(n):
    return train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)
def scaffold_split(smiles_list, test_size=0.2, rng=RNG):
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
        if not scaf:          # acyclic fix — each acyclic gets its own unique key
            scaf = smi
        scaf2idx[scaf].append(i)
    scaffold_groups = list(scaf2idx.values())
    rng.shuffle(scaffold_groups)
    n_test = int(len(smiles_list) * test_size)
    te, tr = [], []
    for group in scaffold_groups:
        (te if len(te) < n_test else tr).extend(group)
    print(f"  scaffold → unique scaffolds: {len(scaf2idx)}, "
          f"train={len(tr)}, test={len(te)}")
    return np.array(tr), np.array(te)
def mordred_clustering_split(X, n_clusters=N_CLUSTERS, n_splits=N_SPLITS, fold=0):
    """
    Ward clustering (n=150) on standardised Mordred descriptors,
    then GroupKFold(n_splits=5) — entire clusters kept in train OR test.
    Returns (train_idx, test_idx) for the requested fold (default = 0 → 80/20).
    """
    Xs     = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
             ).fit_predict(Xs)
    gkf   = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.arange(len(X)), groups=labels))
    tr_idx, te_idx = folds[fold]
    n_unique = len(np.unique(labels))
    print(f"  clustering → {n_unique} clusters, "
          f"GroupKFold({n_splits}) fold={fold}, "
          f"train={len(tr_idx)}, test={len(te_idx)}")
    return np.array(tr_idx), np.array(te_idx)
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    global ENDPOINTS, ENDPOINT_DISPLAY
    print("Loading UniTox...")
    df = pd.read_csv(TRAIN_FILE).dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
    print(f"  Raw: {len(df)} compounds")
    print("Loading Mordred (UniTox)...")
    mordred = pd.read_csv(MORDRED_FILE)
    assert len(mordred) == len(df), \
        f"Row mismatch UniTox={len(df)} vs Mordred={len(mordred)}"
    # Acyclic scaffold fix
    if RDKIT_OK:
        acyclic_count = 0
        for smi in df[UNITOX_SMILES_COL]:
            mol = Chem.MolFromSmiles(smi)
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False) if mol else ""
                if not scaf:
                    acyclic_count += 1
            except Exception:
                pass
        print(f"  Acyclic fix applied: {acyclic_count} acyclic molecules retained "
              f"(each treated as its own scaffold). Total: {len(df)} molecules.")
    else:
        print("  RDKit unavailable — acyclic fix skipped")
    ENDPOINTS        = sorted([c for c in df.columns if c.endswith("__binary")])
    ENDPOINT_DISPLAY = [make_display_label(ep) for ep in ENDPOINTS]
    print(f"  Endpoints detected: {ENDPOINTS}")
    mordred = mordred.apply(pd.to_numeric, errors="coerce")
    print("Loading Withdrawn 2.0...")
    withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
    print(f"  {len(withdrawn_raw)} drugs (raw), columns: {list(withdrawn_raw.columns)}")
    print("Loading Mordred (Withdrawn)...")
    mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED)
    assert len(mordred_w_raw) == len(withdrawn_raw), \
        f"Row mismatch Withdrawn={len(withdrawn_raw)} vs Mordred={len(mordred_w_raw)}"
    valid_mask = withdrawn_raw[WITHDRAWN_SMILES_COL].notna()
    withdrawn  = withdrawn_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w_raw[valid_mask].reset_index(drop=True)
    print(f"  After SMILES dropna: {len(withdrawn)} drugs remain")
    mordred_w = mordred_w.apply(pd.to_numeric, errors="coerce")
    common_cols = mordred.columns.intersection(mordred_w.columns)
    mordred     = mordred[common_cols].reset_index(drop=True)
    mordred_w   = mordred_w[common_cols].reset_index(drop=True)
    print(f"  Common Mordred features: {len(common_cols)}")
    toxtype_map = build_toxtype_map(ENDPOINTS)
    print(f"  Toxtype map: {toxtype_map}")
    for ep in ENDPOINTS:
        withdrawn[ep] = 0
    unmatched = []
    for i, row in withdrawn.iterrows():
        hits = parse_toxtype(row[WITHDRAWN_REASON_COL], toxtype_map)
        if not hits:
            unmatched.append(row.get("drugname", i))
        for ep in hits:
            withdrawn.at[i, ep] = 1
    if unmatched:
        print(f"  WARNING: {len(unmatched)} drugs had no matching endpoint:")
        for d in unmatched[:10]:
            print(f"    '{d}'")
    unitox_smiles = set(df[UNITOX_SMILES_COL].str.strip())
    overlap = withdrawn[WITHDRAWN_SMILES_COL].str.strip().isin(unitox_smiles)
    print(f"  Removing {overlap.sum()} SMILES overlaps with UniTox")
    withdrawn = withdrawn[~overlap].reset_index(drop=True)
    mordred_w = mordred_w[~overlap].reset_index(drop=True)
    print(f"  Final external set: {len(withdrawn)} drugs")
    return df, mordred, withdrawn, mordred_w
# ── RUN ONE SPLIT ─────────────────────────────────────────────────────────────
def run_split(split_name, df, mordred, withdrawn, mordred_w):
    print(f"\n--- Split: {split_name} ---")
    smiles = df[UNITOX_SMILES_COL].tolist()
    X_all  = mordred.values.astype(np.float64)
    X_all  = np.nan_to_num(X_all, nan=0.0)
    if   split_name == "random":     train_idx, test_idx = random_split(len(df))
    elif split_name == "scaffold":   train_idx, test_idx = scaffold_split(smiles, rng=RNG)
    else:                            train_idx, test_idx = mordred_clustering_split(X_all)
    print(f"  Train={len(train_idx)}  Test={len(test_idx)}")
    X_train, _, X_ext = preprocess(
        mordred.iloc[train_idx], mordred.iloc[test_idx], mordred_w)
    print(f"  Features after preprocessing: {X_train.shape[1]}")
    n_ext       = len(withdrawn)
    pred_matrix = np.zeros((n_ext, len(ENDPOINTS)), dtype=int)
    true_matrix = np.zeros((n_ext, len(ENDPOINTS)), dtype=int)
    ep_results  = {}
    for idx, ep in enumerate(ENDPOINTS):
        y_train   = df[ep].iloc[train_idx].values
        y_ext     = withdrawn[ep].values
        threshold = float(y_train.sum()) / len(y_train) if y_train.sum() > 0 else 0.5
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        proba  = rf.predict_proba(X_ext)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        TP = int(np.sum((y_pred == 1) & (y_ext == 1)))
        TN = int(np.sum((y_pred == 0) & (y_ext == 0)))
        FP = int(np.sum((y_pred == 1) & (y_ext == 0)))
        FN = int(np.sum((y_pred == 0) & (y_ext == 1)))
        ep_results[ep] = dict(
            mcc  = compute_mcc(TP, TN, FP, FN),
            sens = compute_sens(TP, FN),
            spec = compute_spec(TN, FP),
            TP=TP, TN=TN, FP=FP, FN=FN
        )
        pred_matrix[:, idx] = y_pred
        true_matrix[:, idx] = y_ext
        print(f"  {ep:<38} MCC={ep_results[ep]['mcc']:+.3f} "
              f"Sens={ep_results[ep]['sens']:.3f}  "
              f"TP={TP} FP={FP} FN={FN}")
    per_drug_mccs = np.array([
        drug_mcc(true_matrix[i], pred_matrix[i]) for i in range(n_ext)
    ])
    print(f"  Per-drug MCC: mean={per_drug_mccs.mean():.3f}  "
          f"median={np.median(per_drug_mccs):.3f}  std={per_drug_mccs.std():.3f}")
    return ep_results, per_drug_mccs
# ── FIGURE 1 — MCC and Sensitivity per endpoint ───────────────────────────────
def plot_mcc_sensitivity(results_by_split):
    labels  = ENDPOINT_DISPLAY
    n_ep    = len(ENDPOINTS)
    n_sp    = len(SPLITS)
    x       = np.arange(n_ep)
    width   = 0.22
    offsets = np.linspace(-(n_sp-1)*width/2, (n_sp-1)*width/2, n_sp)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)
    for ax, metric, ylabel in zip(axes, ["mcc", "sens"], ["MCC", "Sensitivity"]):
        for i, (split, slabel, col) in enumerate(zip(SPLITS, SPLIT_LABELS, SPLIT_COLOURS)):
            vals = [results_by_split[split][ep][metric] for ep in ENDPOINTS]
            ax.bar(x + offsets[i], vals, width=width,
                   color=col, label=slabel, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS - 4)
        ax.set_ylabel(ylabel, fontsize=AXLABEL_FS - 2)
        ax.set_title(f"{ylabel} per Endpoint by Split", fontsize=TITLE_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS - 3)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.82)
    fig.legend(handles, labels_leg,
               loc="upper center", ncol=len(SPLITS),
               fontsize=LEGEND_FS, frameon=True, framealpha=0.9,
               edgecolor="#aaaaaa", bbox_to_anchor=(0.5, 0.98))
    plt.savefig("figure1_mcc_sensitivity.png", dpi=180, bbox_inches="tight")
    print("Saved → figure1_mcc_sensitivity.png")
    plt.show()
# ── FIGURE 2 — TP / FP / FN per endpoint, one figure per split ───────────────
def plot_confusion_bars(results_by_split):
    labels     = ENDPOINT_DISPLAY
    n_ep       = len(ENDPOINTS)
    x          = np.arange(n_ep)
    width      = 0.26
    cm_colours = {"TP": "#2ECC71", "FP": "#F39C12", "FN": "#E74C3C"}
    cm_labels  = {"TP": "TP", "FP": "FP", "FN": "FN"}
    for split, slabel in zip(SPLITS, SPLIT_LABELS):
        fig, ax = plt.subplots(figsize=(13, 5))
        tp_vals = [results_by_split[split][ep]["TP"] for ep in ENDPOINTS]
        fp_vals = [results_by_split[split][ep]["FP"] for ep in ENDPOINTS]
        fn_vals = [results_by_split[split][ep]["FN"] for ep in ENDPOINTS]
        ax.bar(x - width, tp_vals, width=width, color=cm_colours["TP"],
               label=cm_labels["TP"], edgecolor="white")
        ax.bar(x,         fp_vals, width=width, color=cm_colours["FP"],
               label=cm_labels["FP"], edgecolor="white")
        ax.bar(x + width, fn_vals, width=width, color=cm_colours["FN"],
               label=cm_labels["FN"], edgecolor="white")
        ax.set_title(f"{slabel} Strategy", fontsize=TITLE_FS, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS - 4)
        ax.set_ylabel("Number of Drugs", fontsize=AXLABEL_FS - 2)
        ax.tick_params(axis="y", labelsize=TICK_FS - 3)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=LEGEND_FS, framealpha=0.85, edgecolor="#cccccc")
        plt.tight_layout()
        fname = f"figure2_confusion_bars_{split}.png"
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"Saved → {fname}")
        plt.show()
# ── FIGURE 3 — Grouped bar chart: drugs per MCC bin × split ──────────────────
def plot_drugs_vs_mcc(drug_mccs_by_split):
    bin_edges  = np.arange(-1.0, 1.01, 0.2)
    bin_labels = [f"{bin_edges[i]:.1f}–\n{bin_edges[i+1]:.1f}"
                  for i in range(len(bin_edges)-1)]
    n_bins  = len(bin_labels)
    n_sp    = len(SPLITS)
    x       = np.arange(n_bins)
    width   = 0.22
    offsets = np.linspace(-(n_sp-1)*width/2, (n_sp-1)*width/2, n_sp)
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Per-Drug MCC Distribution — Withdrawn 2.0\nacross Splitting Strategies",
                 fontsize=SUPTITLE_FS, fontweight="bold")
    for i, (split, slabel, col) in enumerate(zip(SPLITS, SPLIT_LABELS, SPLIT_COLOURS)):
        vals   = np.clip(drug_mccs_by_split[split], -1.0, 0.9999)
        counts = np.histogram(vals, bins=bin_edges)[0]
        bars   = ax.bar(x + offsets[i], counts, width=width,
                        color=col, label=slabel, edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        str(count), ha="center", va="bottom",
                        fontsize=ANNOT_FS, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=TICK_FS)
    ax.set_xlabel("Per-Drug MCC Range", fontsize=AXLABEL_FS, labelpad=8)
    ax.set_ylabel("Number of Withdrawn Drugs", fontsize=AXLABEL_FS, labelpad=8)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.85, edgecolor="#cccccc")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("figure3_drugs_vs_mcc.png", dpi=180, bbox_inches="tight")
    print("Saved → figure3_drugs_vs_mcc.png")
    plt.show()
# ── FIGURE 4 — Stacked TP / FN per endpoint, one figure per split ────────────
def plot_tp_fn_stacked(results_by_split, n_withdrawn):
    for split, slabel in zip(SPLITS, SPLIT_LABELS):
        res     = results_by_split[split]
        tp_vals = [res[ep]["TP"] for ep in ENDPOINTS]
        fn_vals = [res[ep]["FN"] for ep in ENDPOINTS]
        totals  = [tp + fn for tp, fn in zip(tp_vals, fn_vals)]
        x       = np.arange(len(ENDPOINTS))
        width   = 0.55
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x, tp_vals, width=width, color="#2ECC71",
               label="Matched (TP)", edgecolor="white")
        ax.bar(x, fn_vals, width=width, bottom=tp_vals,
               color="#E74C3C", label="Not Matched (FN)", edgecolor="white")
        for i, (tp, fn, tot) in enumerate(zip(tp_vals, fn_vals, totals)):
            if tp > 0:
                ax.text(x[i], tp/2, str(tp),
                        ha="center", va="center", fontsize=BAR_LABEL_FS,
                        fontweight="bold", color="white")
            if fn > 0:
                ax.text(x[i], tp + fn/2, str(fn),
                        ha="center", va="center", fontsize=BAR_LABEL_FS,
                        fontweight="bold", color="white")
            if tot > 0:
                ax.text(x[i], tot + 0.5, f"Total: {tot}",
                        ha="center", va="bottom", fontsize=ANNOT_FS,
                        fontweight="bold", color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(ENDPOINT_DISPLAY, rotation=15, ha="right", fontsize=TICK_FS)
        ax.set_xlabel("Toxicity Endpoint", fontsize=AXLABEL_FS, labelpad=8)
        ax.set_ylabel("Number of Drugs (Count)", fontsize=AXLABEL_FS, labelpad=8)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ymax = max(totals) if max(totals) > 0 else 1
        ax.set_ylim(0, ymax * 1.20)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=LEGEND_FS, framealpha=0.85, loc="upper right")
        ax.set_title(
            f"Rigorous Success Quantification: {slabel.upper()} Split "
            f"(N={n_withdrawn} Drugs)",
            fontsize=TITLE_FS, fontweight="bold", pad=12)
        fname = f"figure4_tp_fn_stacked_{split}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"Saved → {fname}")
        plt.show()
# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    df, mordred, withdrawn, mordred_w = load_data()
    n_withdrawn = len(withdrawn)
    results_by_split   = {}
    drug_mccs_by_split = {}
    for split in SPLITS:
        ep_res, drug_mccs = run_split(split, df, mordred, withdrawn, mordred_w)
        results_by_split[split]   = ep_res
        drug_mccs_by_split[split] = drug_mccs
    print("\nGenerating figures...")
    plot_mcc_sensitivity(results_by_split)
    plot_confusion_bars(results_by_split)
    plot_drugs_vs_mcc(drug_mccs_by_split)
    plot_tp_fn_stacked(results_by_split, n_withdrawn)
    print("\nAll done.")
if __name__ == "__main__":
    main()try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not found — scaffold split falls back to random.")


# ── FILE PATHS ────────────────────────────────────────────────────────────────
TRAIN_FILE        = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE      = "mordred_features_cached.csv"
WITHDRAWN_FILE    = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED = "mordred_withdrawn_cached.csv"

# ── COLUMN NAMES ──────────────────────────────────────────────────────────────
UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_REASON_COL = "toxtype"

# ── FONT SIZE CONFIG ──────────────────────────────────────────────────────────
SUPTITLE_FS  = 18
TITLE_FS     = 14
AXLABEL_FS   = 17
TICK_FS      = 17
LEGEND_FS    = 15
ANNOT_FS     = 13
BAR_LABEL_FS = 12

# ── SPLIT / MODEL CONFIG ──────────────────────────────────────────────────────
SPLITS        = ["random", "scaffold", "clustering"]
SPLIT_LABELS  = ["Random Split", "Scaffold Split", "Clustering Split"]
SPLIT_COLOURS = ["#4472C4", "#ED7D31", "#70AD47"]
RF_PARAMS     = dict(n_estimators=300, max_features="sqrt",
                     class_weight="balanced", random_state=42, n_jobs=-1)
RANDOM_STATE  = 42
RNG           = np.random.default_rng(RANDOM_STATE)
N_CLUSTERS    = 150
N_SPLITS      = 5

# ── ENDPOINT DETECTION ────────────────────────────────────────────────────────
ENDPOINTS        = None
ENDPOINT_DISPLAY = None


def make_display_label(col: str) -> str:
    return col.replace("__binary", "").replace("_", " ").title()


# ── TOXTYPE MAP ───────────────────────────────────────────────────────────────
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
    d = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP*TN - FP*FN) / d if d else 0.0


def compute_sens(TP, FN):
    return TP / (TP + FN) if (TP + FN) else 0.0


def compute_spec(TN, FP):
    return TN / (TN + FP) if (TN + FP) else 0.0


def drug_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return compute_mcc(TP, TN, FP, FN)


# ── TOXTYPE PARSING ───────────────────────────────────────────────────────────
def build_toxtype_map(endpoints: list) -> dict:
    ep_base = {ep.replace("__binary", ""): ep for ep in endpoints}
    tmap = {}
    for token, base in ALIAS.items():
        if base in ep_base:
            tmap[token] = ep_base[base]
    for base, col in ep_base.items():
        if base not in tmap.values():
            tmap[base] = col
    return tmap


def parse_toxtype(toxtype_str: str, toxtype_map: dict) -> list:
    if not isinstance(toxtype_str, str):
        return []
    matched = set()
    for token in toxtype_str.split(","):
        key = token.strip().lower()
        if key in toxtype_map:
            matched.add(toxtype_map[key])
    return list(matched)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, X_ext: pd.DataFrame):
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


# ── SPLITTING ─────────────────────────────────────────────────────────────────
def random_split(n):
    return train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE)


def scaffold_split(smiles_list, test_size=0.2, rng=RNG):
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
        if not scaf:          # acyclic fix — each acyclic gets its own unique key
            scaf = smi
        scaf2idx[scaf].append(i)
    scaffold_groups = list(scaf2idx.values())
    rng.shuffle(scaffold_groups)
    n_test = int(len(smiles_list) * test_size)
    te, tr = [], []
    for group in scaffold_groups:
        (te if len(te) < n_test else tr).extend(group)
    print(f"  scaffold → unique scaffolds: {len(scaf2idx)}, "
          f"train={len(tr)}, test={len(te)}")
    return np.array(tr), np.array(te)


def mordred_clustering_split(X, n_clusters=N_CLUSTERS, n_splits=N_SPLITS, fold=0):
    """
    Ward clustering (n=150) on standardised Mordred descriptors,
    then GroupKFold(n_splits=5) — entire clusters kept in train OR test.
    Returns (train_idx, test_idx) for the requested fold (default = 0 → 80/20).
    """
    Xs     = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
             ).fit_predict(Xs)

    gkf   = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.arange(len(X)), groups=labels))
    tr_idx, te_idx = folds[fold]

    n_unique = len(np.unique(labels))
    print(f"  clustering → {n_unique} clusters, "
          f"GroupKFold({n_splits}) fold={fold}, "
          f"train={len(tr_idx)}, test={len(te_idx)}")
    return np.array(tr_idx), np.array(te_idx)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    global ENDPOINTS, ENDPOINT_DISPLAY
    print("Loading UniTox...")
    df = pd.read_csv(TRAIN_FILE).dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
    print(f"  Raw: {len(df)} compounds")

    print("Loading Mordred (UniTox)...")
    mordred = pd.read_csv(MORDRED_FILE)
    assert len(mordred) == len(df), \
        f"Row mismatch UniTox={len(df)} vs Mordred={len(mordred)}"

    # Acyclic scaffold fix
    if RDKIT_OK:
        acyclic_count = 0
        for smi in df[UNITOX_SMILES_COL]:
            mol = Chem.MolFromSmiles(smi)
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False) if mol else ""
                if not scaf:
                    acyclic_count += 1
            except Exception:
                pass
        print(f"  Acyclic fix applied: {acyclic_count} acyclic molecules retained "
              f"(each treated as its own scaffold). Total: {len(df)} molecules.")
    else:
        print("  RDKit unavailable — acyclic fix skipped")

    ENDPOINTS        = sorted([c for c in df.columns if c.endswith("__binary")])
    ENDPOINT_DISPLAY = [make_display_label(ep) for ep in ENDPOINTS]
    print(f"  Endpoints detected: {ENDPOINTS}")

    mordred = mordred.apply(pd.to_numeric, errors="coerce")

    print("Loading Withdrawn 2.0...")
    withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
    print(f"  {len(withdrawn_raw)} drugs (raw), columns: {list(withdrawn_raw.columns)}")

    print("Loading Mordred (Withdrawn)...")
    mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED)
    assert len(mordred_w_raw) == len(withdrawn_raw), \
        f"Row mismatch Withdrawn={len(withdrawn_raw)} vs Mordred={len(mordred_w_raw)}"

    valid_mask = withdrawn_raw[WITHDRAWN_SMILES_COL].notna()
    withdrawn  = withdrawn_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w_raw[valid_mask].reset_index(drop=True)
    print(f"  After SMILES dropna: {len(withdrawn)} drugs remain")

    mordred_w = mordred_w.apply(pd.to_numeric, errors="coerce")

    common_cols = mordred.columns.intersection(mordred_w.columns)
    mordred     = mordred[common_cols].reset_index(drop=True)
    mordred_w   = mordred_w[common_cols].reset_index(drop=True)
    print(f"  Common Mordred features: {len(common_cols)}")

    toxtype_map = build_toxtype_map(ENDPOINTS)
    print(f"  Toxtype map: {toxtype_map}")

    for ep in ENDPOINTS:
        withdrawn[ep] = 0
    unmatched = []
    for i, row in withdrawn.iterrows():
        hits = parse_toxtype(row[WITHDRAWN_REASON_COL], toxtype_map)
        if not hits:
            unmatched.append(row.get("drugname", i))
        for ep in hits:
            withdrawn.at[i, ep] = 1

    if unmatched:
        print(f"  WARNING: {len(unmatched)} drugs had no matching endpoint:")
        for d in unmatched[:10]:
            print(f"    '{d}'")

    unitox_smiles = set(df[UNITOX_SMILES_COL].str.strip())
    overlap = withdrawn[WITHDRAWN_SMILES_COL].str.strip().isin(unitox_smiles)
    print(f"  Removing {overlap.sum()} SMILES overlaps with UniTox")
    withdrawn = withdrawn[~overlap].reset_index(drop=True)
    mordred_w = mordred_w[~overlap].reset_index(drop=True)
    print(f"  Final external set: {len(withdrawn)} drugs")

    return df, mordred, withdrawn, mordred_w


# ── RUN ONE SPLIT ─────────────────────────────────────────────────────────────
def run_split(split_name, df, mordred, withdrawn, mordred_w):
    print(f"\n--- Split: {split_name} ---")
    smiles = df[UNITOX_SMILES_COL].tolist()
    X_all  = mordred.values.astype(np.float64)
    X_all  = np.nan_to_num(X_all, nan=0.0)

    if   split_name == "random":     train_idx, test_idx = random_split(len(df))
    elif split_name == "scaffold":   train_idx, test_idx = scaffold_split(smiles, rng=RNG)
    else:                            train_idx, test_idx = mordred_clustering_split(X_all)

    print(f"  Train={len(train_idx)}  Test={len(test_idx)}")

    X_train, _, X_ext = preprocess(
        mordred.iloc[train_idx], mordred.iloc[test_idx], mordred_w)
    print(f"  Features after preprocessing: {X_train.shape[1]}")

    n_ext       = len(withdrawn)
    pred_matrix = np.zeros((n_ext, len(ENDPOINTS)), dtype=int)
    true_matrix = np.zeros((n_ext, len(ENDPOINTS)), dtype=int)
    ep_results  = {}

    for idx, ep in enumerate(ENDPOINTS):
        y_train   = df[ep].iloc[train_idx].values
        y_ext     = withdrawn[ep].values
        threshold = float(y_train.sum()) / len(y_train) if y_train.sum() > 0 else 0.5

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        proba  = rf.predict_proba(X_ext)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        TP = int(np.sum((y_pred == 1) & (y_ext == 1)))
        TN = int(np.sum((y_pred == 0) & (y_ext == 0)))
        FP = int(np.sum((y_pred == 1) & (y_ext == 0)))
        FN = int(np.sum((y_pred == 0) & (y_ext == 1)))

        ep_results[ep] = dict(
            mcc  = compute_mcc(TP, TN, FP, FN),
            sens = compute_sens(TP, FN),
            spec = compute_spec(TN, FP),
            TP=TP, TN=TN, FP=FP, FN=FN
        )
        pred_matrix[:, idx] = y_pred
        true_matrix[:, idx] = y_ext

        print(f"  {ep:<38} MCC={ep_results[ep]['mcc']:+.3f} "
              f"Sens={ep_results[ep]['sens']:.3f}  "
              f"TP={TP} FP={FP} FN={FN}")

    per_drug_mccs = np.array([
        drug_mcc(true_matrix[i], pred_matrix[i]) for i in range(n_ext)
    ])
    print(f"  Per-drug MCC: mean={per_drug_mccs.mean():.3f}  "
          f"median={np.median(per_drug_mccs):.3f}  std={per_drug_mccs.std():.3f}")

    return ep_results, per_drug_mccs


# ── FIGURE 1 — MCC and Sensitivity per endpoint ───────────────────────────────
def plot_mcc_sensitivity(results_by_split):
    labels  = ENDPOINT_DISPLAY
    n_ep    = len(ENDPOINTS)
    n_sp    = len(SPLITS)
    x       = np.arange(n_ep)
    width   = 0.22
    offsets = np.linspace(-(n_sp-1)*width/2, (n_sp-1)*width/2, n_sp)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

    for ax, metric, ylabel in zip(axes, ["mcc", "sens"], ["MCC", "Sensitivity"]):
        for i, (split, slabel, col) in enumerate(zip(SPLITS, SPLIT_LABELS, SPLIT_COLOURS)):
            vals = [results_by_split[split][ep][metric] for ep in ENDPOINTS]
            ax.bar(x + offsets[i], vals, width=width,
                   color=col, label=slabel, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS - 4)
        ax.set_ylabel(ylabel, fontsize=AXLABEL_FS - 2)
        ax.set_title(f"{ylabel} per Endpoint by Split", fontsize=TITLE_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS - 3)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.82)
    fig.legend(handles, labels_leg,
               loc="upper center", ncol=len(SPLITS),
               fontsize=LEGEND_FS, frameon=True, framealpha=0.9,
               edgecolor="#aaaaaa", bbox_to_anchor=(0.5, 0.98))

    plt.savefig("figure1_mcc_sensitivity.png", dpi=180, bbox_inches="tight")
    print("Saved → figure1_mcc_sensitivity.png")
    plt.show()


# ── FIGURE 2 — TP / FP / FN per endpoint, one figure per split ───────────────
def plot_confusion_bars(results_by_split):
    labels     = ENDPOINT_DISPLAY
    n_ep       = len(ENDPOINTS)
    x          = np.arange(n_ep)
    width      = 0.26
    cm_colours = {"TP": "#2ECC71", "FP": "#F39C12", "FN": "#E74C3C"}
    cm_labels  = {"TP": "TP", "FP": "FP", "FN": "FN"}

    for split, slabel in zip(SPLITS, SPLIT_LABELS):
        fig, ax = plt.subplots(figsize=(13, 5))
        tp_vals = [results_by_split[split][ep]["TP"] for ep in ENDPOINTS]
        fp_vals = [results_by_split[split][ep]["FP"] for ep in ENDPOINTS]
        fn_vals = [results_by_split[split][ep]["FN"] for ep in ENDPOINTS]

        ax.bar(x - width, tp_vals, width=width, color=cm_colours["TP"],
               label=cm_labels["TP"], edgecolor="white")
        ax.bar(x,         fp_vals, width=width, color=cm_colours["FP"],
               label=cm_labels["FP"], edgecolor="white")
        ax.bar(x + width, fn_vals, width=width, color=cm_colours["FN"],
               label=cm_labels["FN"], edgecolor="white")

        ax.set_title(f"{slabel} Strategy", fontsize=TITLE_FS, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS - 4)
        ax.set_ylabel("Number of Drugs", fontsize=AXLABEL_FS - 2)
        ax.tick_params(axis="y", labelsize=TICK_FS - 3)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        fname = f"figure2_confusion_bars_{split}.png"
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"Saved → {fname}")
        plt.show()


# ── FIGURE 3 — Grouped bar chart: drugs per MCC bin × split ──────────────────
def plot_drugs_vs_mcc(drug_mccs_by_split):
    bin_edges  = np.arange(-1.0, 1.01, 0.2)
    bin_labels = [f"{bin_edges[i]:.1f}–\n{bin_edges[i+1]:.1f}"
                  for i in range(len(bin_edges)-1)]
    n_bins  = len(bin_labels)
    n_sp    = len(SPLITS)
    x       = np.arange(n_bins)
    width   = 0.22
    offsets = np.linspace(-(n_sp-1)*width/2, (n_sp-1)*width/2, n_sp)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Per-Drug MCC Distribution — Withdrawn 2.0\nacross Splitting Strategies",
                 fontsize=SUPTITLE_FS, fontweight="bold")

    for i, (split, slabel, col) in enumerate(zip(SPLITS, SPLIT_LABELS, SPLIT_COLOURS)):
        vals   = np.clip(drug_mccs_by_split[split], -1.0, 0.9999)
        counts = np.histogram(vals, bins=bin_edges)[0]
        bars   = ax.bar(x + offsets[i], counts, width=width,
                        color=col, label=slabel, edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        str(count), ha="center", va="bottom",
                        fontsize=ANNOT_FS, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=TICK_FS)
    ax.set_xlabel("Per-Drug MCC Range", fontsize=AXLABEL_FS, labelpad=8)
    ax.set_ylabel("Number of Withdrawn Drugs", fontsize=AXLABEL_FS, labelpad=8)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.85, edgecolor="#cccccc")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("figure3_drugs_vs_mcc.png", dpi=180, bbox_inches="tight")
    print("Saved → figure3_drugs_vs_mcc.png")
    plt.show()


# ── FIGURE 4 — Stacked TP / FN per endpoint, one figure per split ────────────
def plot_tp_fn_stacked(results_by_split, n_withdrawn):
    for split, slabel in zip(SPLITS, SPLIT_LABELS):
        res     = results_by_split[split]
        tp_vals = [res[ep]["TP"] for ep in ENDPOINTS]
        fn_vals = [res[ep]["FN"] for ep in ENDPOINTS]
        totals  = [tp + fn for tp, fn in zip(tp_vals, fn_vals)]
        x       = np.arange(len(ENDPOINTS))
        width   = 0.55

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x, tp_vals, width=width, color="#2ECC71",
               label="Matched (TP)", edgecolor="white")
        ax.bar(x, fn_vals, width=width, bottom=tp_vals,
               color="#E74C3C", label="Not Matched (FN)", edgecolor="white")

        for i, (tp, fn, tot) in enumerate(zip(tp_vals, fn_vals, totals)):
            if tp > 0:
                ax.text(x[i], tp/2, str(tp),
                        ha="center", va="center", fontsize=BAR_LABEL_FS,
                        fontweight="bold", color="white")
            if fn > 0:
                ax.text(x[i], tp + fn/2, str(fn),
                        ha="center", va="center", fontsize=BAR_LABEL_FS,
                        fontweight="bold", color="white")
            if tot > 0:
                ax.text(x[i], tot + 0.5, f"Total: {tot}",
                        ha="center", va="bottom", fontsize=ANNOT_FS,
                        fontweight="bold", color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(ENDPOINT_DISPLAY, rotation=15, ha="right", fontsize=TICK_FS)
        ax.set_xlabel("Toxicity Endpoint", fontsize=AXLABEL_FS, labelpad=8)
        ax.set_ylabel("Number of Drugs (Count)", fontsize=AXLABEL_FS, labelpad=8)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ymax = max(totals) if max(totals) > 0 else 1
        ax.set_ylim(0, ymax * 1.20)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=LEGEND_FS, framealpha=0.85, loc="upper right")
        ax.set_title(
            f"Rigorous Success Quantification: {slabel.upper()} Split "
            f"(N={n_withdrawn} Drugs)",
            fontsize=TITLE_FS, fontweight="bold", pad=12)

        fname = f"figure4_tp_fn_stacked_{split}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"Saved → {fname}")
        plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    df, mordred, withdrawn, mordred_w = load_data()
    n_withdrawn = len(withdrawn)

    results_by_split   = {}
    drug_mccs_by_split = {}

    for split in SPLITS:
        ep_res, drug_mccs = run_split(split, df, mordred, withdrawn, mordred_w)
        results_by_split[split]   = ep_res
        drug_mccs_by_split[split] = drug_mccs

    print("\nGenerating figures...")
    plot_mcc_sensitivity(results_by_split)
    plot_confusion_bars(results_by_split)
    plot_drugs_vs_mcc(drug_mccs_by_split)
    plot_tp_fn_stacked(results_by_split, n_withdrawn)

    print("\nAll done.")


if __name__ == "__main__":
    main()
