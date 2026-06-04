"""
03_splits.py
============
Defines and prints the three data-partitioning strategies used throughout
this project:
  - Random split (80/20)
  - Scaffold split (Bemis-Murcko, acyclic-fix)
  - Clustering split (AgglomerativeClustering, n=150, Ward linkage)

Run this to verify split sizes before training.

REQUIRES: UniTox_with_recovered_typos_v3.csv, mordred_features_cached.csv
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

UNITOX_FILE  = "UniTox_with_recovered_typos_v3.csv"
MORDRED_FILE = "mordred_features_cached.csv"
SMILES_COL   = "SMILES_filled"
RANDOM_STATE = 42
N_CLUSTERS   = 150

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
df     = pd.read_csv(UNITOX_FILE).dropna(subset=[SMILES_COL]).reset_index(drop=True)
smiles = df[SMILES_COL].tolist()
X      = np.nan_to_num(pd.read_csv(MORDRED_FILE).values.astype(float))
assert len(df) == len(X), f"Row mismatch: df={len(df)}, X={len(X)}"
print(f"  {len(df)} molecules × {X.shape[1]} descriptors")


# ── Split functions ───────────────────────────────────────────────────────────
def split_random(n, test_frac=0.2):
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_frac, random_state=RANDOM_STATE)
    return list(tr), list(te)


def split_scaffold(smiles, test_frac=0.2):
    """Bemis-Murcko scaffold split.
    Acyclic molecules each get their own unique key (SMILES) so they are not
    all lumped into one enormous group that would dominate the test set."""
    scaf2idx = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            if not sc:          # acyclic — unique key per molecule
                sc = smi
        except Exception:
            sc = smi
        scaf2idx[sc].append(i)

    n_test = int(len(smiles) * test_frac)
    te, tr = [], []
    for sc in sorted(scaf2idx, key=lambda s: -len(scaf2idx[s])):
        (te if len(te) < n_test else tr).extend(scaf2idx[sc])
    return tr, te


def split_clustering(X, n_clusters=N_CLUSTERS, test_frac=0.2):
    """AgglomerativeClustering (Ward, n=150) on standardised Mordred descriptors.
    Clusters are sorted by size (largest first) and greedily assigned to test
    until ~20% is reached; remaining go to train."""
    rng = np.random.default_rng(RANDOM_STATE)
    Xs  = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward"
    ).fit_predict(Xs)

    cl2idx = defaultdict(list)
    for i, c in enumerate(labels):
        cl2idx[c].append(i)

    groups = list(cl2idx.values())
    rng.shuffle(groups)          # avoid systematic size bias
    n_test = int(len(X) * test_frac)
    te, tr = [], []
    for g in groups:
        (te if len(te) < n_test else tr).extend(g)

    sizes = [len(v) for v in cl2idx.values()]
    print(f"  n_clusters={n_clusters}  min_size={min(sizes)}  max_size={max(sizes)}")
    return tr, te


# ── Build & report ────────────────────────────────────────────────────────────
splits = {
    "random":     split_random(len(df)),
    "scaffold":   split_scaffold(smiles),
    "clustering": split_clustering(X),
}

print("\nSplit sizes:")
for name, (tr, te) in splits.items():
    print(f"  {name:12s}  train={len(tr)}  test={len(te)}  "
          f"({len(te)/(len(tr)+len(te))*100:.1f}% test)")

print("\nDone.")
