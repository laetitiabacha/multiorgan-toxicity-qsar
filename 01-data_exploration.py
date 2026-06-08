#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
01_data_exploration
======================
Produces all data characterisation figures for the report.
Figures produced:
  figures/figure2_smiles_recovery.png          -- SMILES resolution coverage per pipeline step
  figures/figure3_unitox_class_distribution.png -- UniTox stacked bar + toxic prevalence line
  figures/figure4_unitox_endpoint_histogram.png -- Distribution of endpoints per molecule (UniTox)
  figures/figure5_split_characteristics.png    -- Train/test size, class balance, scaffold overlap
  figures/figure6_withdrawn_histograms.png     -- Withdrawn 2.0 full + retained 153 drugs distribution
REQUIRES:
  UniTox_with_recovered_typos_v3.csv
  withdrawn_external_validation.csv
  mordred_features_cached.csv    (for Figure 5 clustering split)
  dataset.csv                    (raw Withdrawn 2.0 from https://withdrawn.charite.de)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── CONFIG ────────────────────────────────────────────────────────────────────
UNITOX_FILE         = "UniTox_with_recovered_typos_v3.csv"
WITHDRAWN_FILE      = "withdrawn_external_validation.csv"
WITHDRAWN_FULL_FILE = "dataset.csv"       # raw Withdrawn 2.0 from https://withdrawn.charite.de
MORDRED_FILE        = "mordred_features_cached.csv"
OUT_DIR             = "figures"
RANDOM_STATE        = 42
N_CLUSTERS          = 150
os.makedirs(OUT_DIR, exist_ok=True)

ENDPOINTS = [
    "dermatological_toxicity__binary",
    "hematological__binary",
    "cardiotoxicity__binary",
    "liver_toxicity__binary",
    "pulmonary_toxicity__binary",
    "renal_toxicity__binary",
    "infertility__binary",
    "ototoxicity__binary",
]
EP_SHORT  = ["Derm", "Hema", "Cardio", "Liver", "Pulm", "Renal", "Infert", "Oto"]
EP_LABELS = ["Dermatological", "Haematological", "Cardiotoxicity", "Liver Toxicity",
             "Pulmonary Toxicity", "Renal Toxicity", "Infertility", "Ototoxicity"]

SPLIT_COLOURS = {
    "random":     "#4C8BE0",
    "scaffold":   "#E05050",
    "clustering": "#5BBF5B",
}
SPLIT_LABELS = {
    "random":     "Random Split",
    "scaffold":   "Scaffold Split",
    "clustering": "Clustering Split",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "font.size":         13,
    "axes.labelsize":    14,
    "axes.titlesize":    14,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e5e5",
    "grid.linewidth":    0.8,
    "axes.axisbelow":    True,
})


# In[10]:


# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(UNITOX_FILE).dropna(subset=["SMILES_filled"]).reset_index(drop=True)
X  = pd.read_csv(MORDRED_FILE).values.astype(np.float64)
X  = np.nan_to_num(X, nan=0.0)
assert len(df) == len(X), f"Row mismatch: {len(df)} vs {len(X)}"
print(f"UniTox: {len(df)} molecules x {X.shape[1]} Mordred descriptors")


# In[69]:


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- SMILES Resolution Pipeline Coverage
# ══════════════════════════════════════════════════════════════════════════════
SMILES_STEP_FILES = [
    None,
    None,
    None,
    None,
    "UniTox_with_recovered_typos_v3.csv",
]
SMILES_COL = "SMILES_filled"
BAD_SMILES = {"", "nan", "none", "null", "na", "n/a", "not found",
              "missing", "unknown", "invalid smiles", "standardization error"}

def _coverage(csv_path):
    df = pd.read_csv(csv_path)
    total = len(df)
    n_missing = df[SMILES_COL].apply(
        lambda x: pd.isna(x) or str(x).strip().lower() in BAD_SMILES
    ).sum()
    return round(100 * (1 - n_missing / total), 1), int(n_missing)

def plot_smiles_recovery():
    steps = ["Initial\nmapping", "Step 1:\nID cleanup", "Step 2:\nSalt/form",
             "Step 3:\nPubChem",  "Step 4:\nSpelling"]
    # Fallback coverage values are manually recorded rather than loaded from saved CSVs.
    # The SMILES curation pipeline was iterative: intermediate outputs were not persisted to disk, so missing counts were noted at each step during the original run.
    # Only the final output (Step 4) is available as a CSV; all earlier steps use these recorded values. Step 4 is recomputed from the file when present.
    _pct_fallback     = [83.1, 86.6, 87.0, 90.5, 90.9]
    _missing_fallback = [408,  324,  314,  230,  221]
    pct, missing = [], []
    for i, fpath in enumerate(SMILES_STEP_FILES):
        if fpath and os.path.exists(fpath):
            p, m = _coverage(fpath)
        else:
            p, m = _pct_fallback[i], _missing_fallback[i]
        pct.append(p)
        missing.append(m)
    colors = ["#7bafd4", "#4a90d9", "#2a6aad", "#1b4f8a", "#0d3060"]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.xaxis.grid(False)
    bars = ax.bar(steps, pct, color=colors, width=0.55, zorder=3)
    for bar, p, m in zip(bars, pct, missing):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{p}%\n({m} missing)",
                ha="center", va="bottom", fontsize=12, color="#222222")
    ax.set_ylim(80, 95)
    ax.set_ylabel("SMILES Coverage (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["left"].set_color("#aaaaaa")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    plt.tight_layout()
    out = f"{OUT_DIR}/figure2_smiles_recovery.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")
plot_smiles_recovery()


# In[16]:


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- UniTox Class Distribution (stacked bar + prevalence line)
# ══════════════════════════════════════════════════════════════════════════════
def plot_unitox_class_distribution(df):
    rows = []
    for col, label in zip(ENDPOINTS, EP_LABELS):
        s = df[col].dropna()
        n_toxic    = int((s == 1).sum())
        n_nontoxic = int((s == 0).sum())
        prevalence = n_toxic / len(s) * 100
        rows.append({"label": label, "toxic": n_toxic,
                     "nontoxic": n_nontoxic, "prevalence": prevalence})
    df_plot = pd.DataFrame(rows).sort_values("prevalence", ascending=False)
    fig, ax1 = plt.subplots(figsize=(11, 6), facecolor="white")
    ax2 = ax1.twinx()
    ax1.set_facecolor("white")
    x = np.arange(len(df_plot))
    ax1.bar(x, df_plot["nontoxic"], color="#4C9BE8", label="Non-toxic (0)", zorder=3)
    ax1.bar(x, df_plot["toxic"],    color="#E05050", label="Toxic (1)",
            bottom=df_plot["nontoxic"], zorder=3)
    ax2.plot(x, df_plot["prevalence"], color="black",
             marker="o", linewidth=2, zorder=4, label="Toxic prevalence (%)")
    ax2.set_ylabel("Toxic Prevalence (%)", fontsize=14)
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot["label"], rotation=30, ha="right", fontsize=13)
    ax1.set_ylabel("Number of Compounds", fontsize=14)
    ax1.set_xlabel("Organ-Toxicity Endpoint", fontsize=14)
    ax1.tick_params(axis="y", labelsize=13)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(True)
    ax1.spines["left"].set_color("#aaaaaa")
    ax1.spines["bottom"].set_color("#aaaaaa")
    ax1.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax1.set_axisbelow(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper left", frameon=True,
               framealpha=0.7, edgecolor="#cccccc", fontsize=12)
    ax2.legend(h2, l2, loc="upper right", frameon=True,
               framealpha=0.7, edgecolor="#cccccc", fontsize=12)
    plt.tight_layout()
    out = f"{OUT_DIR}/figure3_unitox_class_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")
plot_unitox_class_distribution(df)


# In[73]:


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 -- UniTox endpoint histogram (endpoints per molecule)
# ══════════════════════════════════════════════════════════════════════════════
def plot_unitox_histogram(df):
    df_full = pd.read_csv(UNITOX_FILE)
    df_full["n_toxic"] = (df_full[ENDPOINTS] == 1).sum(axis=1)
    coverage = df_full["n_toxic"].value_counts().sort_index()
    n_unique = len(df_full)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    bars = ax.bar(coverage.index, coverage.values,
                  color="#4F81BD", edgecolor="white", linewidth=0.8, width=0.65)
    ax.set_xlabel("Number of toxic endpoints per molecule (k)", fontsize=14)
    ax.set_ylabel("Number of molecules", fontsize=14)
    ax.set_xticks(range(int(coverage.index.max()) + 1))
    ax.tick_params(axis="both", labelsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    offset = max(coverage.values) * 0.015
    for bar, val in zip(bars, coverage.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:,}", ha="center", va="bottom", fontsize=12, color="#333333")
    legend_patch = mpatches.Patch(
        facecolor="#4F81BD", edgecolor="#444444",
        label=f"UniTox (n = {n_unique:,}, 8 endpoints)"
    )
    ax.legend(handles=[legend_patch], loc="upper right",
              bbox_to_anchor=(1, 1.09),
              frameon=True, framealpha=1, edgecolor="#cccccc", fontsize=12)
    plt.tight_layout()
    out = f"{OUT_DIR}/figure4_unitox_endpoint_histogram.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")
plot_unitox_histogram(df)


# In[75]:


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 -- Split Characteristics (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else smi
    except Exception:
        return smi

def build_splits(df, X):
    smiles = df["SMILES_filled"].tolist()
    idx = np.arange(len(df))
    tr_r, te_r = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    scaf2idx = defaultdict(list)
    scaffolds = [get_scaffold(s) for s in smiles]
    for i, sc in enumerate(scaffolds):
        scaf2idx[sc].append(i)
    rng = np.random.default_rng(RANDOM_STATE)
    groups = list(scaf2idx.values())
    rng.shuffle(groups)
    n_test = int(len(smiles) * 0.2)
    te_sc, tr_sc = [], []
    for g in groups:
        (te_sc if len(te_sc) < n_test else tr_sc).extend(g)
    Xs = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(Xs)
    cl2idx = defaultdict(list)
    for i, lab in enumerate(labels):
        cl2idx[lab].append(i)
    groups_c = list(cl2idx.values())
    rng.shuffle(groups_c)
    te_cl, tr_cl = [], []
    for g in groups_c:
        (te_cl if len(te_cl) < n_test else tr_cl).extend(g)
    return {
        "random":     (np.array(tr_r), np.array(te_r)),
        "scaffold":   (np.array(tr_sc), np.array(te_sc)),
        "clustering": (np.array(tr_cl), np.array(te_cl)),
    }, np.array(scaffolds)

def plot_split_characteristics(df, X):
    splits, scaffold_keys = build_splits(df, X)
    split_names = list(splits.keys())
    colours = [SPLIT_COLOURS[s] for s in split_names]
    labels  = [SPLIT_LABELS[s]  for s in split_names]
    x = np.arange(len(split_names))
    stats = {}
    for name, (tr_idx, te_idx) in splits.items():
        tr_pos   = [df[ep].values[tr_idx].mean() for ep in ENDPOINTS]
        tr_scafs = set(scaffold_keys[tr_idx])
        te_scafs = set(scaffold_keys[te_idx])
        overlap  = len(tr_scafs & te_scafs) / len(te_scafs) * 100
        tr_div   = len(tr_scafs) / len(tr_idx) * 100
        te_div   = len(te_scafs) / len(te_idx) * 100
        stats[name] = {
            "n_train":       len(tr_idx),
            "n_test":        len(te_idx),
            "tr_pos_per_ep": tr_pos,
            "overlap":       overlap,
            "tr_div":        tr_div,
            "te_div":        te_div,
        }
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    # ── Panel (a): Dataset Size ───────────────────────────────────────────────
    bar_w = 0.3
    ax = axes[0]
    for i, name in enumerate(split_names):
        c = colours[i]
        ax.bar(i - bar_w/2, stats[name]["n_train"], width=bar_w, color=c, alpha=0.9, zorder=3)
        ax.bar(i + bar_w/2, stats[name]["n_test"],  width=bar_w, color=c, alpha=0.4,
               hatch="//", edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(i - bar_w/2, stats[name]["n_train"] + 20,
                str(stats[name]["n_train"]), ha="center", va="bottom",
                fontsize=15, fontweight="bold")
        ax.text(i + bar_w/2, stats[name]["n_test"] + 20,
                str(stats[name]["n_test"]), ha="center", va="bottom",
                fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17, rotation=10)
    ax.set_ylabel("Number of molecules", fontsize=20)
    ax.set_title("Dataset Size", fontsize=20, fontweight="bold")
    ax.set_ylim(0, max(s["n_train"] for s in stats.values()) * 1.25)
    ax.tick_params(axis="y", labelsize=17)
    train_p = mpatches.Patch(facecolor="grey", alpha=0.9, label="Train")
    test_p  = mpatches.Patch(facecolor="grey", alpha=0.4, hatch="//",
                              edgecolor="black", label="Test")
    ax.legend(handles=[train_p, test_p], fontsize=14, loc="upper right")
    # ── Panel (b): Class Balance ──────────────────────────────────────────────
    ep_x   = np.arange(len(ENDPOINTS))
    bar_w2 = 0.12
    ax = axes[1]
    for i, name in enumerate(split_names):
        offset = (i - 1) * bar_w2
        ax.bar(ep_x + offset, stats[name]["tr_pos_per_ep"],
               width=bar_w2, color=colours[i], alpha=0.9, zorder=3)
    ax.set_xticks(ep_x)
    ax.set_xticklabels(EP_SHORT, fontsize=17, rotation=25, ha="right")
    ax.set_ylabel("Positive rate", fontsize=20)
    ax.set_title("Class Balance per Endpoint (Train)", fontsize=20, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=17)
    handles = [mpatches.Patch(facecolor=colours[i], alpha=0.9, label=labels[i])
               for i in range(len(split_names))]
    ax.legend(handles=handles, fontsize=14, loc="upper right")
    # ── Panel (c): Scaffold Overlap & Diversity ───────────────────────────────
    bar_w3 = 0.25
    ax = axes[2]
    ov_vals = [stats[n]["overlap"] for n in split_names]
    tr_div  = [stats[n]["tr_div"]  for n in split_names]
    te_div  = [stats[n]["te_div"]  for n in split_names]
    ax.bar(x - bar_w3, ov_vals, width=bar_w3, color=colours, alpha=0.9, zorder=3)
    ax.bar(x,          tr_div,  width=bar_w3, color=colours, alpha=0.5,
           hatch="//", edgecolor="black", linewidth=0.5, zorder=3)
    ax.bar(x + bar_w3, te_div,  width=bar_w3, color=colours, alpha=0.3,
           hatch="xx", edgecolor="black", linewidth=0.5, zorder=3)
    for i in range(len(split_names)):
        ax.text(x[i] - bar_w3, max(ov_vals[i], 0) + max(ov_vals) * 0.02 + 0.5,
                f"{ov_vals[i]:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.text(x[i],           tr_div[i] + 0.5,
                f"{tr_div[i]:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.text(x[i] + bar_w3, te_div[i] + 0.5,
                f"{te_div[i]:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    # Star for scaffold 0% overlap — placed just above the % label
    sc_idx = split_names.index("scaffold")
    star_y = max(ov_vals[sc_idx], 0) + max(ov_vals) * 0.02 + 0.5 + 7
    ax.plot(x[sc_idx] - bar_w3, star_y, marker="*", color="black",
            markersize=14, zorder=5, linestyle="None")
    import matplotlib.lines as mlines
    ov_p  = mpatches.Patch(facecolor="grey", alpha=0.9, label="Scaffold overlap")
    trd_p = mpatches.Patch(facecolor="grey", alpha=0.5, hatch="//",
                            edgecolor="black", label="Train diversity")
    ted_p = mpatches.Patch(facecolor="grey", alpha=0.3, hatch="xx",
                            edgecolor="black", label="Test diversity")
    star  = mlines.Line2D([], [], color="black", marker="*", linestyle="None",
                          markersize=13, label="0% = by design (acyclic fix)")
    ax.legend(handles=[ov_p, trd_p, ted_p, star], fontsize=15,
              frameon=True, facecolor="white", edgecolor="black", loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17, rotation=13)
    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.set_title("Scaffold Overlap & Diversity", fontsize=20, fontweight="bold")
    ax.set_ylim(0, 130)
    ax.tick_params(axis="y", labelsize=17)
    plt.tight_layout()
    out = f"{OUT_DIR}/figure5_split_characteristics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")
plot_split_characteristics(df, X)


# In[77]:


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 -- Withdrawn 2.0 endpoint distributions (two panels)
# Panel (a): full Withdrawn 2.0 from website (dataset.csv) — uses Charite vocabulary
# Panel (b): retained 153 drugs after harmonisation and overlap removal
# ══════════════════════════════════════════════════════════════════════════════
_TOXTYPE_MAP_FULL = {
    "cardiovascular": "cardiotoxicity__binary",
    "dermatological": "dermatological_toxicity__binary",
    "hematological":  "hematological__binary",
    "reproductive":   "infertility__binary",
    "hepatic":        "liver_toxicity__binary",
    "ototoxicity":    "ototoxicity__binary",
    "respiratory":    "pulmonary_toxicity__binary",
    "renal":          "renal_toxicity__binary",
}
_TOXTYPE_MAP_RETAINED = {
    "cardiotoxicity": "cardiotoxicity__binary",
    "dermatological": "dermatological_toxicity__binary",
    "hematological":  "hematological__binary",
    "infertility":    "infertility__binary",
    "liver":          "liver_toxicity__binary",
    "ototoxicity":    "ototoxicity__binary",
    "pulmonary":      "pulmonary_toxicity__binary",
    "renal":          "renal_toxicity__binary",
}

def _count_endpoints(toxtype_str, toxtype_map):
    if not isinstance(toxtype_str, str):
        return 0
    toks = [t.strip().lower() for t in toxtype_str.split(",")]
    return sum(1 for key in toxtype_map if any(key in tok for tok in toks))

def plot_withdrawn_histograms():
    BAR_COLOR = "#4472C4"
    df_full = pd.read_csv(WITHDRAWN_FULL_FILE)
    df_full["n_endpoints"] = df_full["toxtype"].apply(
        lambda x: _count_endpoints(x, _TOXTYPE_MAP_FULL)
    )
    counts_full = df_full["n_endpoints"].value_counts().sort_index().to_dict()
    n_full = len(df_full)

    df_w = pd.read_csv(WITHDRAWN_FILE)
    df_w["n_endpoints"] = df_w["toxtype"].apply(
        lambda x: _count_endpoints(x, _TOXTYPE_MAP_RETAINED)
    )
    counts_retained = df_w["n_endpoints"].value_counts().sort_index().to_dict()
    n_retained = len(df_w)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    for ax, counts, n, label, panel in [
        (axes[0], counts_full,     n_full,
         f"Withdrawn 2.0 (n = {n_full:,}, 8 endpoints)",
         "(a) Full dataset prior to harmonisation"),
        (axes[1], counts_retained, n_retained,
         f"Withdrawn 2.0 retained (n = {n_retained:,})",
         "(b) Retained after harmonisation & overlap removal"),
    ]:
        ax.set_facecolor("white")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.8, linestyle="-")
        ax.xaxis.grid(False)
        x_vals = list(counts.keys())
        y_vals = list(counts.values())
        bars = ax.bar(x_vals, y_vals, color=BAR_COLOR, edgecolor="none",
                      width=0.65, zorder=3)
        for bar, val in zip(bars, y_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(y_vals) * 0.012,
                    str(val), ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#222222")
        if 0 in counts:
            zero_bar = bars[x_vals.index(0)]
            ax.text(zero_bar.get_x() + zero_bar.get_width() / 2,
                    zero_bar.get_height() + max(y_vals) * 0.012 + max(y_vals) * 0.06,
                    "non-UniTox\nwithdrawal",
                    ha="center", va="bottom", fontsize=11, color="#666666", style="italic")
        ax.set_xlabel("Number of toxic endpoints per molecule (k)", fontsize=14, labelpad=8)
        ax.set_ylabel("Number of molecules", fontsize=14, labelpad=8)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(k) for k in x_vals], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, max(y_vals) * 1.22)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(
            50 if max(y_vals) > 100 else 20
        ))
        ax.spines["bottom"].set_color("#999999")
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_title(panel, fontsize=14, fontweight="bold")
        patch = mpatches.Patch(facecolor=BAR_COLOR, edgecolor="none", label=label)
        ax.legend(handles=[patch], loc="upper right",
                  frameon=True, framealpha=1, edgecolor="#cccccc", fontsize=12)

    plt.tight_layout()
    out = f"{OUT_DIR}/figure6_withdrawn_histograms.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")
plot_withdrawn_histograms()


# In[ ]:




