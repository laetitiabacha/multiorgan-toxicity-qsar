"""
02_mordred_features.py
======================
Computes Mordred 2D descriptors for all molecules in UniTox and caches
to mordred_features_cached.csv.

Run once (~10 min). Subsequent scripts load the cache.

REQUIRES: UniTox_with_recovered_typos_v3.csv
PRODUCES: mordred_features_cached.csv
"""

import time
import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors as mordred_desc

UNITOX_FILE  = "UniTox_with_recovered_typos_v3.csv"
SMILES_COL   = "SMILES_filled"
OUT_FILE     = "mordred_features_cached.csv"

print("Loading UniTox ...")
df = (pd.read_csv(UNITOX_FILE)
        .dropna(subset=[SMILES_COL])
        .reset_index(drop=True))
print(f"  {len(df)} molecules")

print("Parsing SMILES ...")
mols = []
for smi in df[SMILES_COL]:
    mol = Chem.MolFromSmiles(str(smi))
    mols.append(mol)

print(f"  {sum(m is not None for m in mols)} valid molecules")

print("Computing Mordred 2D descriptors (ignore_3D=True) ...")
t0   = time.time()
calc = Calculator(mordred_desc, ignore_3D=True)
desc = calc.pandas(mols)
desc = desc.apply(pd.to_numeric, errors="coerce")
print(f"  Done in {time.time()-t0:.1f}s  |  shape: {desc.shape}")

desc.to_csv(OUT_FILE, index=False)
print(f"Saved → {OUT_FILE}")
