"""
04_train_all_models.py
======================
Trains RF, XGBoost, Logistic Regression, SVM (RBF), and CART across all three
data splits (random, scaffold, clustering) and all 8 binary toxicity endpoints.

Outputs per model × split:
  Table2_model_metrics_<model>_<split>.csv    — ROC-AUC, AUPRC, MCC, Accuracy
  cm_table_<model>_<split>.csv                — TP, TN, FP, FN

REQUIRES: UniTox_with_recovered_typos_v3.csv, mordred_features_cached.csv
"""

import warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, confusion_matrix,
                              matthews_corrcoef)
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
UNITOX_FILE  = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE = "mordred_features_cached.csv"
RANDOM_STATE = 42
N_CLUSTERS   = 150

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = (pd.read_csv(UNITOX_FILE)
        .dropna(subset=["SMILES_filled"])
        .reset_index(drop=True))
X = np.nan_to_num(pd.read_csv(MORDRED_FILE).values.astype(np.float64))
assert len(df) == len(X), f"Row mismatch: df={len(df)}, X={len(X)}"
smiles = df["SMILES_filled"].tolist()
print(f"  {len(df)} molecules × {X.shape[1]} descriptors")

ENDPOINTS = [c for c in df.columns if c.endswith("__binary")]
EP_LABELS  = [c.replace("__binary", "").replace("_", " ").title() for c in ENDPOINTS]
print(f"  {len(ENDPOINTS)} endpoints: {EP_LABELS}")


# ── Split functions ───────────────────────────────────────────────────────────
def split_random(n):
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    return list(tr), list(te)


def split_scaffold(smiles):
    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            if not sc:
                sc = smi
        except Exception:
            sc = smi
        scaf2idx[sc].append(i)
    n_test = int(len(smiles) * 0.2)
    te, tr = [], []
    for s in sorted(scaf2idx, key=lambda s: -len(scaf2idx[s])):
        (te if len(te) < n_test else tr).extend(scaf2idx[s])
    return tr, te


def split_clustering(X, n_clusters=N_CLUSTERS, test_frac=0.2):
    rng = np.random.default_rng(RANDOM_STATE)
    Xs  = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Xs)
    cl2idx = defaultdict(list)
    for i, c in enumerate(labels):
        cl2idx[c].append(i)
    groups = list(cl2idx.values())
    rng.shuffle(groups)
    n_test = int(len(X) * test_frac)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)
    sizes = [len(v) for v in cl2idx.values()]
    print(f"    n_clusters={n_clusters}  min_size={min(sizes)}  max_size={max(sizes)}")
    return tr, te


# ── Build splits ──────────────────────────────────────────────────────────────
splits = {
    "random":     split_random(len(df)),
    "scaffold":   split_scaffold(smiles),
    "clustering": split_clustering(X),
}
for name, (tr, te) in splits.items():
    print(f"  {name:12s}  train={len(tr)}  test={len(te)}")


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model(key, spw=1.0):
    clf_map = {
        "rf":     RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1),
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000,
                                     random_state=RANDOM_STATE),
        "svm":    SVC(kernel="rbf", class_weight="balanced",
                      probability=True, random_state=RANDOM_STATE),
        "cart":   DecisionTreeClassifier(class_weight="balanced",
                                         random_state=RANDOM_STATE),
        "xgb":    XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                                random_state=42, verbosity=0),
    }
    clf = clf_map[key]
    if key in {"logreg", "svm"}:
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return clf


# ── Training loop ─────────────────────────────────────────────────────────────
MODEL_KEYS = ["rf", "xgb", "logreg", "svm", "cart"]

for split_name, (tr_idx, te_idx) in splits.items():
    print(f"\n{'='*60}")
    print(f"SPLIT: {split_name.upper()}  (train={len(tr_idx)}, test={len(te_idx)})")
    print(f"{'='*60}")
    X_tr = X[tr_idx]
    X_te = X[te_idx]

    for model_key in MODEL_KEYS:
        print(f"\n  ── {model_key.upper()} ──")
        metric_rows, cm_rows = [], []

        for ep_col, ep_label in zip(ENDPOINTS, EP_LABELS):
            y_tr = df[ep_col].values[tr_idx]
            y_te = df[ep_col].values[te_idx]

            if len(np.unique(y_tr)) < 2:
                print(f"    Skipped {ep_label}: only one class in train")
                continue
            if len(np.unique(y_te)) < 2:
                print(f"    Skipped {ep_label}: only one class in test")
                continue

            num_pos = (y_tr == 1).sum()
            num_neg = (y_tr == 0).sum()
            current_spw = num_neg / num_pos if num_pos > 0 else 1.0

            model = make_model(model_key, spw=current_spw)
            model.fit(X_tr, y_tr)

            probs     = model.predict_proba(X_te)[:, 1]
            threshold = y_tr.sum() / len(y_tr)
            preds     = (probs >= threshold).astype(int)

            mcc   = matthews_corrcoef(y_te, preds)
            auc   = roc_auc_score(y_te, probs)
            auprc = average_precision_score(y_te, probs)
            acc   = accuracy_score(y_te, preds)

            metric_rows.append({
                "Toxicity": ep_label,
                "ROC-AUC":  round(auc,   4),
                "AUPRC":    round(auprc, 4),
                "MCC":      round(mcc,   4),
                "Accuracy": round(acc,   4),
            })

            tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
            cm_rows.append({
                "Toxicity": ep_label,
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            })
            print(f"    {ep_label:<22} MCC={mcc:.3f}  AUC={auc:.3f}  SPW={current_spw:.2f}")

        pd.DataFrame(metric_rows).to_csv(
            f"Table2_model_metrics_{model_key}_{split_name}.csv", index=False)
        pd.DataFrame(cm_rows).to_csv(
            f"cm_table_{model_key}_{split_name}.csv", index=False)
        print(f"    Saved → Table2_model_metrics_{model_key}_{split_name}.csv")

print("\n✓ ALL MODELS × ALL SPLITS COMPLETE.")
print(f"  Clustering split used n_clusters={N_CLUSTERS} (AgglomerativeClustering, Ward linkage)")
