"""
per_drug_mcc_peon.py
====================
Trains per-endpoint Random Forest models on UniTox + Mordred descriptors,
predicts toxicity profiles for withdrawn drugs, computes per-drug MCC,
and illustrates per-drug MCC with three Venn-diagram case studies
(good / partial / near-zero prediction) — all auto-selected from results.

Split strategy is controlled by SPLIT_TYPE:
  "random"   — standard 80/20 random split
  "scaffold" — Bemis-Murcko scaffold split with acyclic molecule fix
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not found — scaffold split falls back to random.")

# ─────────────────────────────────────────────────────────────────────────────
# FILES
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FILE        = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE      = "mordred_features_cached.csv"
WITHDRAWN_FILE    = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED = "mordred_withdrawn_cached.csv"

UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_REASON_COL = "toxtype"
DRUGNAME_COL         = "drugname"

TEST_SIZE    = 0.2
RANDOM_STATE = 42
SPLIT_TYPE   = "scaffold"   # "random" or "scaffold"

RNG = np.random.default_rng(RANDOM_STATE)

RF_PARAMS = dict(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

ENDPOINTS = None   # filled by load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SPLITS
# ─────────────────────────────────────────────────────────────────────────────
def random_split(n):
    return train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def scaffold_split(smiles_list, test_size=TEST_SIZE):
    """
    Bemis-Murcko scaffold split with acyclic molecule fix.
    Acyclic molecules (no ring system → empty scaffold) are assigned
    their full SMILES as a fallback identifier, preventing them from
    being artificially grouped into a single large scaffold.
    """
    if not RDKIT_OK:
        print("RDKit unavailable — falling back to random split.")
        return random_split(len(smiles_list))

    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            ) if mol else ""
        except Exception:
            scaf = ""
        # Acyclic fix: use full SMILES so each acyclic molecule forms its own group
        if not scaf:
            scaf = smi
        scaf2idx[scaf].append(i)

    groups = list(scaf2idx.values())
    RNG.shuffle(groups)

    n_test = int(len(smiles_list) * test_size)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)

    return np.array(tr), np.array(te)


# ─────────────────────────────────────────────────────────────────────────────
# CASE STUDY SELECTION  —  fully automatic, no hardcoding
# ─────────────────────────────────────────────────────────────────────────────
def select_case_studies(mccs, drug_names, true_matrix):
    """
    Auto-selects three representative case studies from computed results.

      1. Good prediction     — highest MCC < 1.0 among drugs with ≥1 observed endpoint (excludes perfect/trivial cases)
      2. Partial prediction  — MCC closest to the median (excluding good pick)
      3. Near-zero prediction— |MCC| closest to 0 (excluding good and partial),
                               preferring negative values to illustrate failure

    Returns (case_studies, case_labels).
    """
    mccs        = np.asarray(mccs)
    true_matrix = np.asarray(true_matrix)

    # Only consider drugs with at least one observed toxic endpoint
    has_obs  = true_matrix.sum(axis=1) > 0
    eligible = [i for i in range(len(drug_names)) if has_obs[i]]
    if len(eligible) < 3:
        eligible = list(range(len(drug_names)))  # fallback

    # 1. Good — highest MCC that is strictly less than 1.0 (avoid perfect/trivial case)
    non_perfect = [i for i in eligible if mccs[i] < 1.0]
    if not non_perfect:
        non_perfect = eligible  # fallback if everything is 1.0
    good_idx = non_perfect[int(np.argmax(mccs[non_perfect]))]

    # 2. Partial — closest to median (excluding good)
    remaining = [i for i in eligible if i != good_idx]
    med       = np.median(mccs[remaining])
    mid_idx   = remaining[int(np.argmin(np.abs(mccs[remaining] - med)))]

    # 3. Near-zero — |MCC| closest to 0, preferring negative values
    final_pool = [i for i in remaining if i != mid_idx]
    neg  = [i for i in final_pool if mccs[i] < 0]
    pool = neg if neg else final_pool
    near_zero_idx = pool[int(np.argmin(np.abs(mccs[pool])))]

    studies = [drug_names[good_idx], drug_names[mid_idx], drug_names[near_zero_idx]]
    tags    = ["Good prediction", "Partial prediction", "Near-zero prediction"]
    return studies, tags


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    fn_fill   = "#B5D4F4",
    tp_fill   = "#AFA9EC",
    fp_fill   = "#F5C4B3",
    tn_fill   = "#EAF3DE",
    obs_edge  = "#185FA5",
    pred_edge = "#993C1D",
    tn_edge   = "#639922",
    tp_edge   = "#7F77DD",
    fn_txt    = "#0C447C",
    tp_txt    = "#3C3489",
    fp_txt    = "#712B13",
    tn_txt    = "#3B6D11",
    good_mcc  = "#2e7d32",
    mid_mcc   = "#c07a00",
    bad_mcc   = "#b71c1c",
)
MCC_COLOR_THRESHOLDS = [(0.30, C["good_mcc"]), (0.10, C["mid_mcc"])]

def mcc_color(val):
    for thr, col in MCC_COLOR_THRESHOLDS:
        if val >= thr:
            return col
    return C["bad_mcc"]


# ─────────────────────────────────────────────────────────────────────────────
# LABEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pretty_label(ep: str) -> str:
    return ep.replace("__binary", "").replace("_", " ").title()

SHORT_MAP = {
    "Cardiotoxicity":          "Cardiotox.",
    "Dermatological Toxicity": "Dermatol.",
    "Hematological":           "Hemotox.",
    "Infertility":             "Infertility",
    "Liver Toxicity":          "Hepatotox.",
    "Ototoxicity":             "Ototox.",
    "Pulmonary Toxicity":      "Pulm. tox.",
    "Renal Toxicity":          "Nephrotox.",
}

def short(name: str) -> str:
    return SHORT_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# MCC  (per-drug: across endpoints for a single molecule)
# ─────────────────────────────────────────────────────────────────────────────
def peon_mcc(y_true, y_pred):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return float(TP * TN - FP * FN) / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(X_train, X_test, X_ext):
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test.select_dtypes(include=[np.number])
    X_ext   = X_ext.select_dtypes(include=[np.number])
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_ext   = X_ext.reindex(columns=X_train.columns, fill_value=0)
    med     = X_train.median()
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)
    X_ext   = X_ext.fillna(med)
    keep    = X_train.var() > 0
    X_train = X_train.loc[:, keep]
    X_test  = X_test.loc[:, keep]
    X_ext   = X_ext.loc[:, keep]
    if X_train.shape[1] == 0:
        raise RuntimeError("No features left after preprocessing")
    return X_train.values, X_test.values, X_ext.values


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    global ENDPOINTS
    df        = pd.read_csv(TRAIN_FILE)
    df        = df.dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
    ENDPOINTS = sorted([c for c in df.columns if c.endswith("__binary")])
    mordred   = pd.read_csv(MORDRED_FILE)
    withdrawn = pd.read_csv(WITHDRAWN_FILE)
    withdrawn = withdrawn.dropna(subset=[WITHDRAWN_SMILES_COL]).reset_index(drop=True)
    mordred_w = pd.read_csv(WITHDRAWN_MORDRED)

    for ep in ENDPOINTS:
        withdrawn[ep] = 0
    for i, row in withdrawn.iterrows():
        if isinstance(row[WITHDRAWN_REASON_COL], str):
            toks = [t.strip().lower() for t in row[WITHDRAWN_REASON_COL].split(",")]
            for ep in ENDPOINTS:
                base = ep.replace("__binary", "")
                if base in toks:
                    withdrawn.at[i, ep] = 1

    return df, mordred, withdrawn, mordred_w


# ─────────────────────────────────────────────────────────────────────────────
# RUN SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def run_split(df, mordred, withdrawn, mordred_w):
    smiles_list = df[UNITOX_SMILES_COL].tolist()

    if SPLIT_TYPE == "scaffold":
        print(f"Using scaffold split (acyclic fix enabled, RDKit={'OK' if RDKIT_OK else 'MISSING'})...")
        train_idx, _ = scaffold_split(smiles_list)
    else:
        print("Using random split...")
        train_idx, _ = random_split(len(df))

    X_tr       = mordred.iloc[train_idx].reset_index(drop=True)
    y_tr       = df.iloc[train_idx][ENDPOINTS].reset_index(drop=True)
    X_ext      = mordred_w.reset_index(drop=True)
    y_ext      = withdrawn[ENDPOINTS].reset_index(drop=True)
    drug_names = withdrawn[DRUGNAME_COL].fillna("Unknown").tolist()

    X_tr_np, _, X_ext_np = preprocess(X_tr, X_tr, X_ext)

    models, thresholds = [], []
    for ep in ENDPOINTS:
        y = y_tr[ep].values
        if len(np.unique(y)) < 2:
            models.append(None)
            thresholds.append(0.5)
            continue
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr_np, y)
        models.append(clf)
        thresholds.append(np.mean(y))

    thresholds  = np.array(thresholds)
    n_drugs     = len(X_ext_np)
    n_ep        = len(ENDPOINTS)
    pred_matrix = np.zeros((n_drugs, n_ep))
    true_matrix = y_ext.values

    for j, model in enumerate(models):
        if model is None:
            continue
        probs = model.predict_proba(X_ext_np)[:, 1]
        pred_matrix[:, j] = (probs >= thresholds[j]).astype(int)

    mccs = np.array([
        peon_mcc(true_matrix[i], pred_matrix[i])
        for i in range(n_drugs)
    ])
    return mccs, drug_names, pred_matrix, true_matrix


# ─────────────────────────────────────────────────────────────────────────────
# VENN DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def _draw_venn_ax(ax, fn_set, tp_set, fp_set, tn_count):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    r  = 2.6
    cy = 5.2
    has_overlap = len(tp_set) > 0
    cx_obs, cx_pred = (4.05, 5.95) if has_overlap else (2.0, 8.0)

    ax.add_patch(FancyBboxPatch(
        (0.2, 0.6), 9.6, 8.8,
        boxstyle="round,pad=0.15",
        linewidth=1.2, linestyle=(0, (5, 3)),
        edgecolor=C["tn_edge"], facecolor=C["tn_fill"],
        alpha=0.55, zorder=1
    ))
    ax.text(0.58, 8.9, f"TN = {tn_count}",
            fontsize=13, fontweight="bold", color=C["tn_txt"], zorder=6)

    ax.add_patch(Circle(
        (cx_obs, cy), r,
        facecolor=C["fn_fill"], edgecolor=C["obs_edge"],
        linewidth=1.5, alpha=0.82, zorder=2
    ))
    ax.add_patch(Circle(
        (cx_pred, cy), r,
        facecolor=C["fp_fill"], edgecolor=C["pred_edge"],
        linewidth=1.5, alpha=0.82, zorder=2
    ))

    if has_overlap:
        tp_patch = Circle(
            (cx_obs, cy), r,
            facecolor=C["tp_fill"], edgecolor="none",
            alpha=0.90, zorder=3
        )
        ax.add_patch(tp_patch)
        tp_patch.set_clip_path(
            Circle((cx_pred, cy), r, transform=ax.transData)
        )

    nkw = dict(ha="center", va="center", fontsize=36, fontweight="bold", zorder=7)
    if has_overlap:
        ax.text(cx_obs  - 1.55, cy, str(len(fn_set)),  color=C["fn_txt"], **nkw)
        ax.text((cx_obs + cx_pred) / 2, cy, str(len(tp_set)), color=C["tp_txt"], **nkw)
        ax.text(cx_pred + 1.55, cy, str(len(fp_set)),  color=C["fp_txt"], **nkw)
    else:
        ax.text(cx_obs,  cy, str(len(fn_set)), color=C["fn_txt"],  **nkw)
        ax.text(cx_pred, cy, str(len(fp_set)), color=C["fp_txt"],  **nkw)

    ax.text(cx_obs,  0.95, "Observed",  ha="center",
            fontsize=15, color=C["obs_edge"],  zorder=6)
    ax.text(cx_pred, 0.95, "Predicted", ha="center",
            fontsize=15, color=C["pred_edge"], zorder=6)


def _draw_legend_ax(ax, fn_set, tp_set, fp_set, tn_count):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    LABEL_FS = 19
    ITEM_FS  = 19
    LINE_H   = 0.24
    X_LBL    = 0.02
    X_ITEMS  = 0.18
    PER_LINE = 3

    rows = [
        ("TP :", C["tp_txt"], sorted(tp_set) if tp_set else None),
        ("FP :", C["fp_txt"], sorted(fp_set) if fp_set else None),
        ("FN :", C["fn_txt"], sorted(fn_set) if fn_set else None),
    ]
    y = 0.98
    for label, color, items in rows:
        if items is None:
            continue
        chunks = []
        for i in range(0, len(items), PER_LINE):
            chunks.append(",  ".join(
                short(x) if x != str(tn_count) else x
                for x in items[i:i + PER_LINE]
            ))
        ax.text(X_LBL, y, label,
                fontsize=LABEL_FS, fontweight="bold", color=color,
                ha="left", va="top", transform=ax.transAxes)
        ax.text(X_ITEMS, y, chunks[0],
                fontsize=ITEM_FS, color=color,
                ha="left", va="top", transform=ax.transAxes)
        for chunk in chunks[1:]:
            y -= LINE_H * 0.65
            ax.text(X_ITEMS, y, chunk,
                    fontsize=ITEM_FS, color=color,
                    ha="left", va="top", transform=ax.transAxes)
        y -= LINE_H


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_case_study_figure(
    mccs, drug_names, true_matrix, pred_matrix,
    endpoints, case_studies, case_labels,
    out_path="venn_confusion_matrix.png",
):
    mccs        = np.asarray(mccs)
    true_matrix = np.asarray(true_matrix)
    pred_matrix = np.asarray(pred_matrix)
    labels      = [pretty_label(ep) for ep in endpoints]
    name_to_idx = {n: i for i, n in enumerate(drug_names)}
    n_cols      = len(case_studies)
    n_ep        = len(endpoints)

    fig = plt.figure(figsize=(7 * n_cols, 10), facecolor="white")
    gs  = GridSpec(
        3, n_cols, figure=fig,
        top=0.93, bottom=0.22,
        left=0.03, right=0.97,
        hspace=0.0, wspace=0.06,
        height_ratios=[0.13, 2.5, 1.1],
    )
    fig.text(0.5, 0.97, "Per-Drug Confusion Matrix — Venn Diagrams",
             ha="center", fontsize=15, fontweight="bold", color="#1a1a18")

    for col, (drug, tag) in enumerate(zip(case_studies, case_labels)):
        idx   = name_to_idx[drug]
        i_mcc = float(mccs[idx])
        obs      = {labels[j] for j in range(n_ep) if true_matrix[idx, j] == 1}
        pred_set = {labels[j] for j in range(n_ep) if pred_matrix[idx, j] == 1}
        tp_set   = obs & pred_set
        fn_set   = obs - pred_set
        fp_set   = pred_set - obs
        tn_count = n_ep - len(tp_set) - len(fn_set) - len(fp_set)

        ax_t = fig.add_subplot(gs[0, col])
        ax_t.axis("off")
        ax_t.set_facecolor("white")
        ax_t.text(
            0.5, 0.5,
            f"{drug}\n{tag} — MCC={i_mcc:+.2f}",
            transform=ax_t.transAxes,
            fontsize=17, fontweight="bold",
            color=mcc_color(i_mcc),
            ha="center", va="center", linespacing=1.5,
        )

        ax_v = fig.add_subplot(gs[1, col])
        ax_v.set_facecolor("white")
        _draw_venn_ax(ax_v, fn_set, tp_set, fp_set, tn_count)

        ax_l = fig.add_subplot(gs[2, col])
        ax_l.set_facecolor("white")
        _draw_legend_ax(ax_l, fn_set, tp_set, fp_set, tn_count)

    handles = [
        mpatches.Patch(facecolor=C["fn_fill"], edgecolor=C["obs_edge"],
                       linewidth=0.9, label="Observed only (FN)"),
        mpatches.Patch(facecolor=C["tp_fill"], edgecolor=C["tp_edge"],
                       linewidth=0.9, label="Observed & Predicted (TP)"),
        mpatches.Patch(facecolor=C["fp_fill"], edgecolor=C["pred_edge"],
                       linewidth=0.9, label="Predicted only (FP)"),
        mpatches.Patch(facecolor=C["tn_fill"], edgecolor=C["tn_edge"],
                       linewidth=0.9, label="Neither (TN)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4,
        fontsize=18, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", facecolor="#ffffff",
        bbox_to_anchor=(0.5, 0.18),
        handlelength=1.6, handleheight=1.0, borderpad=0.7,
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df, mordred, withdrawn, mordred_w = load_data()
    print(f"Training: {len(df)} drugs | Withdrawn: {len(withdrawn)} drugs "
          f"| Endpoints: {len(ENDPOINTS)}")

    mccs, drug_names, pred_matrix, true_matrix = run_split(
        df, mordred, withdrawn, mordred_w
    )

    print(f"\nPer-drug MCC:  mean = {np.mean(mccs):.3f},  "
          f"median = {np.median(mccs):.3f}")
    print(f"  MCC = 1.0 : {np.sum(mccs == 1.0)}")
    print(f"  MCC > 0.5 : {np.sum(mccs > 0.5)}")
    print(f"  MCC > 0   : {np.sum(mccs > 0)}")
    print(f"  MCC <= 0  : {np.sum(mccs <= 0)}")

    case_studies, case_labels = select_case_studies(mccs, drug_names, true_matrix)

    ep_labels   = [pretty_label(ep) for ep in ENDPOINTS]
    name_to_idx = {n: i for i, n in enumerate(drug_names)}

    print()
    for drug, tag in zip(case_studies, case_labels):
        i    = name_to_idx[drug]
        obs  = sorted(ep_labels[j] for j in range(len(ep_labels)) if true_matrix[i, j] == 1)
        pred = sorted(ep_labels[j] for j in range(len(ep_labels)) if pred_matrix[i, j] == 1)
        print(f"  [{tag}]  {drug}  (MCC = {mccs[i]:+.3f})")
        print(f"    Observed:  {obs}")
        print(f"    Predicted: {pred}")
        print()

    make_case_study_figure(
        mccs, drug_names, true_matrix, pred_matrix,
        ENDPOINTS, case_studies, case_labels,
        out_path="venn_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
