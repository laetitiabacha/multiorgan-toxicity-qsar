"""
plot_generalisation_tax_table.py
─────────────────────────────────
Derives and exports the "Generalisation Tax" summary table entirely from
computed results, no hardcoded values.
Produces:
  generalisation_gap.png          two-model (RF + XGBoost) grouped bar chart
  generalisation_tax_table.csv    per-endpoint summary table

Clustering split: Ward (n=150) + GroupKFold(n_splits=5), fold 0
(consistent with cells 3, 4, 11, 18, 19, 20, 21, 22).
"""
import os
import re
import csv
import warnings
import numpy as np
import pandas as pd
from math import sqrt
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# ── COLUMN NAMES ──────────────────────────────────────────────────────────────
COLUMNS = [
    "Toxicity Endpoint",
    "Resilience to Novelty",
    "Generalisation Gap (Delta MCC)",
    "Clinical Recovery Rate",
    "Scientific Interpretation",
]


def make_display_label(col: str) -> str:
    return col.replace("__binary", "").replace("_", " ").title()


def resilience_label(mean_ext_mcc: float, min_ext_mcc: float) -> str:
    if mean_ext_mcc >= 0.15:
        return "High"
    elif mean_ext_mcc >= 0.05:
        return "Moderate"
    elif mean_ext_mcc < 0.0:
        return "Anomalous"
    else:
        return "Low"


def gap_description(mean_gap: float, gap_values: list) -> str:
    if all(g < 0 for g in gap_values):
        return "Negative (Inversion)"
    if mean_gap >= 0.30:
        return f"High (Delta ~{mean_gap:.2f})"
    elif mean_gap >= 0.10:
        return f"Moderate (Delta ~{mean_gap:.2f})"
    elif mean_gap <= 0.0:
        return f"Negative (Delta ~{mean_gap:.2f})"
    else:
        return f"Low (Delta ~{mean_gap:.2f})"


def recovery_rate_string(sens_by_split: dict) -> str:
    pct = {k: round(v * 100) for k, v in sens_by_split.items()}
    values = list(pct.values())
    if all(v == 0 for v in values):
        return "0% (All splits)"
    low_splits = {k: v for k, v in pct.items() if v < 30}
    if low_splits and len(low_splits) < len(pct):
        worst_split = min(low_splits, key=low_splits.get)
        label = worst_split.replace("_", " ").title() + " Split"
        return f"{low_splits[worst_split]}% ({label})"
    mn, mx = min(values), max(values)
    if mn == mx:
        return f"{mn}%"
    return f"{mn}%\u2013{mx}%"


def mean_gap_from_desc(gap_desc: str) -> float:
    m = re.search(r"~?(\d+\.\d+)", gap_desc)
    return float(m.group(1)) if m else 0.0


def auto_interpretation(
    ep, mean_ext_mcc, mean_sens, resilience, gap_desc, prevalence_train, sens_range
) -> str:
    if mean_sens == 0.0:
        imbalance_note = ""
        if prevalence_train < 0.12:
            imbalance_note = (
                f" Extreme class imbalance ({prevalence_train*100:.1f}% prevalence)"
                " prevents learning a transferable signal."
            )
        return "Model fails to recover any positive cases in the external set." + imbalance_note
    if resilience == "Anomalous":
        return (
            "External MCC is negative, indicating prediction inversion on novel structures. "
            "Scaffold-dependent training may cause memorisation of ring-system-specific patterns "
            "rather than transferable physicochemical signals."
        )
    if resilience == "High" and mean_gap_from_desc(gap_desc) >= 0.30:
        return (
            f"High clinical recovery ({mean_sens*100:.0f}% mean sensitivity) "
            "suggests a strong physicochemical signal, but the large generalisation gap "
            "indicates partial scaffold dependency."
        )
    if resilience == "High":
        return (
            f"Robust external recovery ({mean_sens*100:.0f}% mean sensitivity). "
            "Performance is driven by physicochemical properties that transfer well "
            "to structurally novel compounds."
        )
    if resilience == "Low" and sens_range >= 0.30:
        return (
            f"Performance is highly split-dependent (sensitivity range: {sens_range*100:.0f}%). "
            "Recovery depends on scaffold similarity between training and external sets; "
            "fails when core ring systems are novel."
        )
    if resilience == "Low":
        return (
            "Limited generalisation to novel structures. "
            "The model captures some endpoint-relevant signal but struggles with "
            "structurally dissimilar external compounds."
        )
    return (
        f"Moderate external recovery ({mean_sens*100:.0f}% mean sensitivity). "
        "The physicochemical signal partially transfers to the external set."
    )


def build_generalisation_tax_table(
    results_by_split: dict,
    internal_by_split: dict,
    withdrawn_df: pd.DataFrame,
) -> pd.DataFrame:
    splits    = list(results_by_split.keys())
    endpoints = sorted({ep for sp in results_by_split.values() for ep in sp})
    rows = []
    for ep in endpoints:
        display = make_display_label(ep)
        ext_mccs, ext_sens, int_mccs = {}, {}, {}
        for sp in splits:
            if ep in results_by_split[sp]:
                ext_mccs[sp] = results_by_split[sp][ep]["mcc"]
                ext_sens[sp] = results_by_split[sp][ep]["sens"]
            if internal_by_split and ep in internal_by_split.get(sp, {}):
                int_mccs[sp] = internal_by_split[sp][ep]["mcc"]
        if not ext_mccs:
            continue
        mean_ext_mcc = float(np.mean(list(ext_mccs.values())))
        min_ext_mcc  = float(np.min(list(ext_mccs.values())))
        mean_sens    = float(np.mean(list(ext_sens.values())))
        sens_range   = float(np.max(list(ext_sens.values())) -
                             np.min(list(ext_sens.values())))
        if int_mccs:
            mean_int_mcc = float(np.mean(list(int_mccs.values())))
            gap_values   = [
                int_mccs.get(sp, mean_int_mcc) - ext_mccs.get(sp, mean_ext_mcc)
                for sp in splits
            ]
            mean_gap = mean_int_mcc - mean_ext_mcc
        else:
            gap_values = [0.0]
            mean_gap   = 0.0
        prevalence_train = float(withdrawn_df[ep].mean()) if ep in withdrawn_df.columns else 0.0
        resil        = "None" if mean_sens == 0.0 else resilience_label(mean_ext_mcc, min_ext_mcc)
        gap_desc     = gap_description(mean_gap, gap_values)
        recovery_str = recovery_rate_string(ext_sens)
        interp       = auto_interpretation(
            ep, mean_ext_mcc, mean_sens, resil, gap_desc, prevalence_train, sens_range
        )
        rows.append((display, resil, gap_desc, recovery_str, interp))
    return pd.DataFrame(rows, columns=COLUMNS)


def export_generalisation_tax_csv(
    results_by_split: dict,
    internal_by_split: dict,
    withdrawn_df: pd.DataFrame,
    out_path: str = "generalisation_tax_table.csv",
) -> pd.DataFrame:
    df = build_generalisation_tax_table(results_by_split, internal_by_split, withdrawn_df)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))
    return df


def plot_generalisation_gap(
    results_by_split: dict,
    internal_by_split: dict,
    out_file: str = "generalisation_gap.png",
    results_by_split_2: dict = None,
    internal_by_split_2: dict = None,
    model_1_label: str = "RF",
    model_2_label: str = "XGBoost",
) -> None:
    splits   = ["random", "scaffold", "clustering"]
    xlabels  = ["Random", "Scaffold", "Clustering"]
    COLOR_1  = "#4C72B0"
    COLOR_2  = "#C44E52"
    single_model = results_by_split_2 is None

    def split_means(res_ext, res_int):
        int_means, ext_means = [], []
        for sp in splits:
            ext_vals = [res_ext[sp][ep]["mcc"] for ep in res_ext.get(sp, {})]
            int_vals = [res_int[sp][ep]["mcc"]
                        for ep in res_int.get(sp, {})
                        if ep in res_ext.get(sp, {})]
            ext_means.append(np.mean(ext_vals) if ext_vals else 0.0)
            int_means.append(np.mean(int_vals) if int_vals else 0.0)
        return int_means, ext_means

    int1, ext1 = split_means(results_by_split, internal_by_split)
    if not single_model:
        int2, ext2 = split_means(results_by_split_2, internal_by_split_2)

    n     = len(splits)
    x     = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor("#F8F8F8")

    if single_model:
        ax.bar(x - width / 2, int1, width,
               color=COLOR_1, edgecolor="black", linewidth=0.8, zorder=3)
        ax.bar(x + width / 2, ext1, width,
               color=COLOR_1, alpha=0.35,
               hatch="//", edgecolor=COLOR_1, linewidth=0.8, zorder=3)
        for i in range(n):
            gap = int1[i] - ext1[i]
            y   = max(int1[i], ext1[i]) + 0.012
            ax.text(x[i], y, f"D={gap:+.3f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=COLOR_1)
    else:
        off_ri = -1.5 * width - 0.04
        off_re = -0.5 * width - 0.04
        off_xi =  0.5 * width + 0.04
        off_xe =  1.5 * width + 0.04
        bar_data = [
            (int1, COLOR_1, 1.0,  None, off_ri),
            (ext1, COLOR_1, 0.40, "//", off_re),
            (int2, COLOR_2, 1.0,  None, off_xi),
            (ext2, COLOR_2, 0.40, "//", off_xe),
        ]
        for vals, col, alp, hat, off in bar_data:
            ax.bar(x + off, vals, width,
                   color=col, alpha=alp,
                   hatch=hat, edgecolor=col, linewidth=0.8, zorder=3)
        for i in range(n):
            for int_v, ext_v, col, off_i, off_e in [
                (int1[i], ext1[i], COLOR_1, off_ri, off_re),
                (int2[i], ext2[i], COLOR_2, off_xi, off_xe),
            ]:
                gap   = int_v - ext_v
                mid_x = x[i] + (off_i + off_e) / 2
                y     = max(int_v, ext_v) + 0.014
                ax.text(mid_x, y, f"D={gap:+.3f}",
                        ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color=col)

    ax.set_xticks(x, labels=xlabels, fontsize=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Mean Matthews Correlation Coefficient (MCC)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    if single_model:
        legend_handles = [
            mpatches.Patch(facecolor="white", edgecolor=COLOR_1, linewidth=1.5,
                           label="Internal (UniTox Test Set)"),
            mpatches.Patch(facecolor="white", edgecolor=COLOR_1, linewidth=1.5,
                           hatch="//", label="External (Withdrawn 2.0)"),
        ]
        ncol = 2
    else:
        legend_handles = [
            mpatches.Patch(facecolor="white", edgecolor=COLOR_1, linewidth=1.5,
                           label=f"{model_1_label}: Internal (UniTox Test Set)"),
            mpatches.Patch(facecolor="white", edgecolor=COLOR_1, linewidth=1.5,
                           hatch="//", label=f"{model_1_label}: External (Withdrawn 2.0)"),
            mpatches.Patch(facecolor="white", edgecolor=COLOR_2, linewidth=1.5,
                           label=f"{model_2_label}: Internal (UniTox Test Set)"),
            mpatches.Patch(facecolor="white", edgecolor=COLOR_2, linewidth=1.5,
                           hatch="//", label=f"{model_2_label}: External (Withdrawn 2.0)"),
        ]
        ncol = 2

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=ncol,
        frameon=True,
        fontsize=15,
        borderaxespad=0,
        edgecolor="#cccccc",
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {out_file}")
    plt.show()


# ── STANDALONE ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from xgboost import XGBClassifier

    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDKIT_OK = True
    except ImportError:
        RDKIT_OK = False
        print("WARNING: RDKit not found, scaffold split falls back to random.")

    # ── CONFIG ────────────────────────────────────────────────────────────────
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

    RF_PARAMS  = dict(n_estimators=300, max_features="sqrt",
                      class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    XGB_PARAMS = dict(n_estimators=300, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

    # ── METRICS ───────────────────────────────────────────────────────────────
    def compute_mcc(TP, TN, FP, FN):
        d = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        return (TP*TN - FP*FN) / d if d else 0.0

    def compute_sens(TP, FN):
        return TP / (TP + FN) if (TP + FN) else 0.0

    def compute_spec(TN, FP):
        return TN / (TN + FP) if (TN + FP) else 0.0

    def eval_predictions(y_true, y_pred):
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        return dict(mcc=compute_mcc(TP,TN,FP,FN), sens=compute_sens(TP,FN),
                    spec=compute_spec(TN,FP), TP=TP, TN=TN, FP=FP, FN=FN)

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    def preprocess(X_train, X_test, X_ext):
        shared  = X_train.columns.intersection(X_test.columns).intersection(X_ext.columns)
        X_train = X_train[shared].copy()
        X_test  = X_test[shared].copy()
        X_ext   = X_ext[shared].copy()
        keep    = X_train.isnull().mean() <= 0.5
        X_train = X_train.loc[:, keep]; X_test = X_test.loc[:, keep]; X_ext = X_ext.loc[:, keep]
        medians = X_train.median()
        X_train = X_train.fillna(medians); X_test = X_test.fillna(medians); X_ext = X_ext.fillna(medians)
        nonzero = X_train.var() > 0
        X_train = X_train.loc[:, nonzero]; X_test = X_test.loc[:, nonzero]; X_ext = X_ext.loc[:, nonzero]
        return X_train.values, X_test.values, X_ext.values

    # ── SPLITS ────────────────────────────────────────────────────────────────
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
            if not scaf:
                scaf = smi
            scaf2idx[scaf].append(i)
        groups = list(scaf2idx.values())
        rng.shuffle(groups)
        n_test = int(len(smiles_list) * test_size)
        te, tr = [], []
        for g in groups:
            (te if len(te) < n_test else tr).extend(g)
        print(f"  scaffold: unique scaffolds: {len(scaf2idx)}, train={len(tr)}, test={len(te)}")
        return np.array(tr), np.array(te)

    def mordred_clustering_split(X, n_clusters=N_CLUSTERS, n_splits=N_SPLITS, fold=0):
        """
        Ward clustering on standardised Mordred descriptors with n=150 clusters,
        then GroupKFold(n_splits=5). Returns fold 0 as the canonical 80/20 split.
        Matches the methodology in cells 3, 4, 11, 18, 19, 20, 21, 22.
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

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    print("Loading UniTox...")
    df_raw      = pd.read_csv(TRAIN_FILE).dropna(subset=[UNITOX_SMILES_COL]).reset_index(drop=True)
    mordred_raw = pd.read_csv(MORDRED_FILE)
    assert len(mordred_raw) == len(df_raw), \
        f"Row mismatch UniTox={len(df_raw)} vs Mordred={len(mordred_raw)}"

    df      = df_raw.reset_index(drop=True)
    mordred = mordred_raw.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    print(f"  {len(df)} molecules retained (acyclics included)")

    ENDPOINTS = sorted([c for c in df.columns if c.endswith("__binary")])
    print(f"  Endpoints: {ENDPOINTS}")

    # ── WITHDRAWN ─────────────────────────────────────────────────────────────
    print("Loading Withdrawn 2.0...")
    withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
    mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED)
    assert len(mordred_w_raw) == len(withdrawn_raw)

    valid_mask = withdrawn_raw[WITHDRAWN_SMILES_COL].notna()
    withdrawn  = withdrawn_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w_raw[valid_mask].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")

    common_cols = mordred.columns.intersection(mordred_w.columns)
    mordred     = mordred[common_cols].reset_index(drop=True)
    mordred_w   = mordred_w[common_cols].reset_index(drop=True)
    print(f"  Common Mordred features: {len(common_cols)}")

    ALIAS = {
        "hematological": "hematological", "renal_toxicity": "renal_toxicity",
        "cardiotoxicity": "cardiotoxicity",
        "dermatological_toxicity": "dermatological_toxicity",
        "liver_toxicity": "liver_toxicity", "pulmonary_toxicity": "pulmonary_toxicity",
        "ototoxicity": "ototoxicity", "infertility": "infertility",
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
    print(f"  Removed {overlap.sum()} overlaps, {len(withdrawn)} Withdrawn drugs remain")

    # ── SPLIT INDICES (computed once, shared by both models) ──────────────────
    smiles = df[UNITOX_SMILES_COL].tolist()
    X_all  = np.nan_to_num(mordred.values.astype(np.float64), nan=0.0)
    split_indices = {
        "random":     random_split(len(df)),
        "scaffold":   scaffold_split(smiles, rng=RNG),
        "clustering": mordred_clustering_split(X_all),
    }
    for sname, (tr, te) in split_indices.items():
        print(f"  {sname}: train={len(tr)}, test={len(te)}")

    # ── TRAINING LOOP (runs for each model) ───────────────────────────────────
    def run_model(model_factory, model_label):
        results_ext = {}
        results_int = {}
        for split_name, (train_idx, test_idx) in split_indices.items():
            print(f"\n  [{model_label}] Split: {split_name}")
            X_train, X_test, X_ext_arr = preprocess(
                mordred.iloc[train_idx], mordred.iloc[test_idx], mordred_w
            )
            ep_ext, ep_int = {}, {}
            for ep in ENDPOINTS:
                y_train = df[ep].iloc[train_idx].values
                y_test  = df[ep].iloc[test_idx].values
                y_ext   = withdrawn[ep].values
                if len(np.unique(y_train)) < 2:
                    continue
                threshold = float(y_train.sum()) / len(y_train) if y_train.sum() > 0 else 0.5
                spw   = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                model = model_factory(spw)
                model.fit(X_train, y_train)
                p_int = model.predict_proba(X_test)[:, 1]
                ep_int[ep] = eval_predictions(y_test, (p_int >= threshold).astype(int))
                p_ext = model.predict_proba(X_ext_arr)[:, 1]
                ep_ext[ep] = eval_predictions(y_ext, (p_ext >= threshold).astype(int))
                print(f"    {ep:<38} int={ep_int[ep]['mcc']:+.3f}  ext={ep_ext[ep]['mcc']:+.3f}")
            results_ext[split_name] = ep_ext
            results_int[split_name] = ep_int
        return results_ext, results_int

    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    rf_ext, rf_int = run_model(
        lambda spw: RandomForestClassifier(**RF_PARAMS),
        "RF"
    )

    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    xgb_ext, xgb_int = run_model(
        lambda spw: XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS),
        "XGBoost"
    )

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    export_generalisation_tax_csv(
        rf_ext, rf_int, withdrawn,
        out_path="generalisation_tax_table.csv",
    )

    plot_generalisation_gap(
        results_by_split=rf_ext,
        internal_by_split=rf_int,
        results_by_split_2=xgb_ext,
        internal_by_split_2=xgb_int,
        model_1_label="RF",
        model_2_label="XGBoost",
        out_file="generalisation_gap.png",
    )
