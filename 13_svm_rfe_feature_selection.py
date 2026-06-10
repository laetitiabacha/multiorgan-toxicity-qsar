"""
13_svm_rfe_feature_selection.py
=================================
Feature selection via SVM-RFE per toxicity endpoint.
Strategy:
  1. RF importance pre-screen: top 300 features
  2. SVM-RFE on those 300 → N features (tested at N = 50, 100, 150, 200, 250, 300)
  3. Retrain RF on selected features, compare MCC before/after
  4. Best N selected by highest mean test MCC
  5. Union of selected descriptors across endpoints saved for reproducibility
REQUIRES:
  UniTox_with_recovered_typos_v3.csv
  mordred_features_cached.csv
  withdrawn_external_validation.csv
  mordred_withdrawn_cached.csv
PRODUCES:
  rfe_results.csv
  selected_descriptors_rfe.csv
  rfe_comparison_no_overlap.png
"""
import sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")

TRAIN_FILE        = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE      = "mordred_features_cached.csv"
WITHDRAWN_FILE    = "withdrawn_external_validation.csv"
WITHDRAWN_MORDRED = "mordred_withdrawn_cached.csv"

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
PRESCREEN_TOP = 300
RFE_TARGETS   = [50, 100, 150, 200, 250, 300]

RF_PARAMS = dict(
    n_estimators=100, max_features="sqrt", class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=1,
)

# ── TOXTYPE MAPPING (consistent with other scripts) ───────────────────────────
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

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_and_split():
    df = pd.read_csv(TRAIN_FILE)
    df = df.dropna(subset=["SMILES_filled"]).reset_index(drop=True)
    endpoints = sorted([c for c in df.columns if c.endswith("__binary")])
    mordred   = pd.read_csv(MORDRED_FILE)

    withdrawn_raw = pd.read_csv(WITHDRAWN_FILE)
    mordred_w_raw = pd.read_csv(WITHDRAWN_MORDRED)

    # Filter valid SMILES
    valid_mask = withdrawn_raw["smiles"].notna()
    withdrawn  = withdrawn_raw[valid_mask].reset_index(drop=True)
    mordred_w  = mordred_w_raw[valid_mask].reset_index(drop=True)

    # Remove overlap with UniTox training set
    unitox_smiles = set(df["SMILES_filled"].str.strip())
    overlap   = withdrawn["smiles"].str.strip().isin(unitox_smiles)
    withdrawn = withdrawn[~overlap].reset_index(drop=True)
    mordred_w = mordred_w[~overlap].reset_index(drop=True)
    print(f"  Removed {overlap.sum()} overlaps, {len(withdrawn)} withdrawn drugs remain",
          flush=True)

    # Assign ground truth labels using ALIAS-based toxtype mapping
    toxtype_map = build_toxtype_map(endpoints)
    for ep in endpoints:
        withdrawn[ep] = 0
    for i, row in withdrawn.iterrows():
        for ep in parse_toxtype(row["toxtype"], toxtype_map):
            withdrawn.at[i, ep] = 1

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_tr = mordred.iloc[train_idx].reset_index(drop=True).select_dtypes(include=[np.number])
    keep = X_tr.var() > 0
    X_tr = X_tr.loc[:, keep]
    cols = X_tr.columns.tolist()

    X_te  = mordred.iloc[test_idx].reset_index(drop=True).select_dtypes(
                include=[np.number]).reindex(columns=cols, fill_value=0)
    X_ext = mordred_w.reset_index(drop=True).select_dtypes(
                include=[np.number]).reindex(columns=cols, fill_value=0)

    med   = X_tr.median()
    X_tr  = X_tr.fillna(med).values
    X_te  = X_te.fillna(med).values
    X_ext = X_ext.fillna(med).values

    y_tr  = df.iloc[train_idx][endpoints].reset_index(drop=True)
    y_te  = df.iloc[test_idx][endpoints].reset_index(drop=True)
    y_ext = withdrawn[endpoints].reset_index(drop=True)

    return X_tr, X_te, X_ext, y_tr, y_te, y_ext, endpoints, cols


def eval_rf(X_tr, X_te, X_ext, y_tr_ep, y_te_ep, y_ext_ep):
    y = y_tr_ep.values
    if len(np.unique(y)) < 2:
        return 0.0, 0.0
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y)
    thresh = np.mean(y)
    p_te  = (clf.predict_proba(X_te)[:, 1]  >= thresh).astype(int)
    p_ext = (clf.predict_proba(X_ext)[:, 1] >= thresh).astype(int)
    return (matthews_corrcoef(y_te_ep.values, p_te),
            matthews_corrcoef(y_ext_ep.values, p_ext))


def main():
    print("Loading data...", flush=True)
    X_tr, X_te, X_ext, y_tr, y_te, y_ext, endpoints, cols = load_and_split()
    n_feat_total = X_tr.shape[1]
    ep_short = [ep.replace("__binary", "").replace("_", " ").title() for ep in endpoints]

    print(f"Train: {X_tr.shape} | Test: {X_te.shape} | Ext: {X_ext.shape}", flush=True)
    print(f"Features after variance filter: {n_feat_total}", flush=True)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n=== Baseline (all features) ===", flush=True)
    base_test, base_ext = {}, {}
    for ep, name in zip(endpoints, ep_short):
        mt, me = eval_rf(X_tr, X_te, X_ext, y_tr[ep], y_te[ep], y_ext[ep])
        base_test[ep], base_ext[ep] = mt, me
        print(f"  {name:<25} test={mt:+.3f}  ext={me:+.3f}", flush=True)

    # ── Per-endpoint: RF pre-screen → SVM-RFE ─────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)

    results  = {n: {} for n in RFE_TARGETS}
    sel_idxs = {n: {} for n in RFE_TARGETS}

    for ep, name in zip(endpoints, ep_short):
        y = y_tr[ep].values
        if len(np.unique(y)) < 2:
            for n in RFE_TARGETS:
                results[n][ep]  = (0.0, 0.0)
                sel_idxs[n][ep] = []
            continue

        print(f"\n  {name}: RF pre-screening to top {PRESCREEN_TOP}...",
              end="", flush=True)
        t0     = time.time()
        rf_pre = RandomForestClassifier(**RF_PARAMS)
        rf_pre.fit(X_tr, y)
        top_idx = np.argsort(rf_pre.feature_importances_)[::-1][:PRESCREEN_TOP]
        print(f" done ({time.time()-t0:.0f}s)", flush=True)

        X_tr_sub = X_tr_sc[:, top_idx]

        for n_feat in RFE_TARGETS:
            if n_feat >= PRESCREEN_TOP:
                sel_idx = top_idx
            else:
                print(f"    RFE → {n_feat}...", end="", flush=True)
                t1  = time.time()
                svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                                dual="auto", random_state=RANDOM_STATE)
                rfe = RFE(estimator=svm, n_features_to_select=n_feat, step=20)
                rfe.fit(X_tr_sub, y)
                sel_idx = top_idx[rfe.support_]
                print(f" done ({time.time()-t1:.0f}s)", flush=True)

            mt, me = eval_rf(X_tr[:, sel_idx], X_te[:, sel_idx], X_ext[:, sel_idx],
                             y_tr[ep], y_te[ep], y_ext[ep])
            results[n_feat][ep]  = (mt, me)
            sel_idxs[n_feat][ep] = sel_idx.tolist()
            print(f"      N={n_feat}: test={mt:+.3f}  ext={me:+.3f}", flush=True)

    # ── Find best N ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90, flush=True)
    mean_per_n = {}
    for n in RFE_TARGETS:
        mean_per_n[n] = np.mean([results[n][ep][0] for ep in endpoints])
        print(f"  N={n:>3}: mean test MCC = {mean_per_n[n]:+.3f}", flush=True)
    baseline_mean = np.mean(list(base_test.values()))
    print(f"  ALL : mean test MCC = {baseline_mean:+.3f} (baseline)", flush=True)
    best_n = max(mean_per_n, key=mean_per_n.get)
    print(f"\nBest N = {best_n} (mean MCC = {mean_per_n[best_n]:+.3f})", flush=True)

    # ── Save selected descriptor names ────────────────────────────────────────
    all_selected = sorted(set(
        cols[i]
        for ep in endpoints
        for i in sel_idxs[best_n].get(ep, [])
    ))
    pd.DataFrame({"descriptor": all_selected}).to_csv(
        "selected_descriptors_rfe.csv", index=False)
    print(f"\nSaved: selected_descriptors_rfe.csv "
          f"({len(all_selected)} unique descriptors across endpoints)", flush=True)

    # ── CSV output ────────────────────────────────────────────────────────────
    N = best_n
    rows = []
    for ep, name in zip(endpoints, ep_short):
        tf, tr = base_test[ep], results[N][ep][0]
        ef, er = base_ext[ep], results[N][ep][1]
        rows.append({
            "Endpoint":            name,
            "N_features_baseline": n_feat_total,
            "N_features_rfe":      N,
            "MCC_test_all":        round(tf, 3),
            "MCC_test_rfe":        round(tr, 3),
            "Delta_MCC_test":      round(tr - tf, 3),
            "MCC_ext_all":         round(ef, 3),
            "MCC_ext_rfe":         round(er, 3),
            "Delta_MCC_ext":       round(er - ef, 3),
        })
    mef = np.mean(list(base_ext.values()))
    mer = np.mean([results[N][ep][1] for ep in endpoints])
    rows.append({
        "Endpoint":            "MEAN",
        "N_features_baseline": n_feat_total,
        "N_features_rfe":      N,
        "MCC_test_all":        round(baseline_mean, 3),
        "MCC_test_rfe":        round(mean_per_n[N], 3),
        "Delta_MCC_test":      round(mean_per_n[N] - baseline_mean, 3),
        "MCC_ext_all":         round(mef, 3),
        "MCC_ext_rfe":         round(mer, 3),
        "Delta_MCC_ext":       round(mer - mef, 3),
    })
    results_df = pd.DataFrame(rows)
    results_df.to_csv("rfe_results.csv", index=False)
    print(f"\nSaved: rfe_results.csv", flush=True)
    print(results_df.to_string(index=False), flush=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    ep_rows   = [r for r in rows if r["Endpoint"] != "MEAN"]
    test_all  = np.array([r["MCC_test_all"] for r in ep_rows])
    test_rfe  = np.array([r["MCC_test_rfe"] for r in ep_rows])
    ep_labels = [r["Endpoint"] for r in ep_rows]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelweight": "bold",
    })

    COLOR_ALL = "#4472C4"
    COLOR_RFE = "#ED7D31"
    y = np.arange(len(ep_labels))
    h = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True,
                                   gridspec_kw={"width_ratios": [1.6, 1]},
                                   constrained_layout=True)

    ax1.barh(y + 0.19, test_all, h,
             label=f"Baseline (All Descriptors, N={n_feat_total})",
             color=COLOR_ALL, edgecolor="white")
    ax1.barh(y - 0.19, test_rfe, h,
             label=f"SVM-RFE (Selected N={N})",
             color=COLOR_RFE, edgecolor="white")

    for i, v in enumerate(test_all):
        ax1.text(v + 0.01, y[i] + 0.19, f"{v:.3f}", va="center", ha="left",
                 fontsize=15, fontweight="bold", color="black")
    for i, v in enumerate(test_rfe):
        ax1.text(v + 0.01, y[i] - 0.19, f"{v:.3f}", va="center", ha="left",
                 fontsize=15, fontweight="bold", color="black")

    ax1.set_yticks(y)
    ax1.set_yticklabels(ep_labels, fontsize=16)
    ax1.set_xlabel("Matthews Correlation Coefficient (MCC)", fontsize=16,
                   fontweight="normal", labelpad=15)
    ax1.set_ylabel("Toxicity Endpoints", fontsize=16, fontweight="normal", labelpad=8)
    ax1.set_title("Performance Comparison", fontsize=16, pad=15)
    ax1.set_xlim(0, 0.9)
    ax1.invert_yaxis()
    ax1.legend(loc="upper right", frameon=True, fontsize=13, edgecolor="#cccccc")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)

    deltas = test_rfe - test_all
    colors = [("#27ae60" if d >= 0 else "#c0392b") for d in deltas]
    ax2.barh(y, deltas, 0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for i, d in enumerate(deltas):
        ha     = "left"  if d >= 0 else "right"
        offset = 0.002   if d >= 0 else -0.002
        ax2.text(d + offset, y[i], f"{d:+.3f}", va="center", ha=ha,
                 fontsize=15, fontweight="bold", color="black")

    ax2.set_xlabel("Change in MCC (ΔMCC)", fontsize=17, fontweight="normal", labelpad=15)
    ax2.set_title("Impact of Feature Reduction", fontsize=16, fontweight="normal", pad=15)
    ax2.axvline(0, color="black", linewidth=2, zorder=5)
    ax2.set_xlim(-0.07, 0.07)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5)

    for ax in [ax1, ax2]:
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig("rfe_comparison_no_overlap.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    print("Saved: rfe_comparison_no_overlap.png", flush=True)
    plt.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
