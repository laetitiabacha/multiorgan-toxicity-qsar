"""
10_pca_chemical_space.py
=========================
Chemical space analysis:
  1. PCA on Mordred descriptors (fit on UniTox, project Withdrawn 2.0)
  2. Scree plot
  3. Tanimoto similarity distribution (Morgan FP, each Withdrawn drug vs nearest UniTox)

REQUIRES:
  UniTox_with_recovered_typos_v3.csv
  mordred_features_cached.csv
  withdrawn_external_validation.csv
  mordred_withdrawn_cached.csv  (computed by 02_mordred_features.py if missing)

PRODUCES:
  pca_chemical_space_mordred.png
  pca_scree_mordred.png
  tanimoto_similarity_mordred.png
  chemical_space_summary.csv
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
UNITOX_CSV           = "UniTox_with_recovered_typos_v3.csv"
UNITOX_MORDRED       = "mordred_features_cached.csv"
WITHDRAWN_CSV        = "withdrawn_external_validation.csv"
WITHDRAWN_SMILES_COL = "smiles"
WITHDRAWN_MORDRED    = "mordred_withdrawn_cached.csv"
N_COMPONENTS         = 20
RANDOM_STATE         = 42
OOD_THRESHOLD        = 0.4

# ── STEP 1: Load UniTox ───────────────────────────────────────────────────────
print("Loading UniTox data and Mordred descriptors ...")
df_unitox    = (pd.read_csv(UNITOX_CSV)
                  .dropna(subset=["SMILES_filled"])
                  .reset_index(drop=True))
X_unitox_raw = pd.read_csv(UNITOX_MORDRED)
feat_names   = X_unitox_raw.columns.tolist()
X_unitox_raw = np.nan_to_num(X_unitox_raw.values.astype(np.float64))
assert len(df_unitox) == len(X_unitox_raw)
print(f"  {len(df_unitox)} molecules × {X_unitox_raw.shape[1]} descriptors")

# Acyclic count (for reporting)
acyclic_count = sum(
    1 for smi in df_unitox["SMILES_filled"]
    if (mol := Chem.MolFromSmiles(smi)) and
    not any(atom.IsInRing() for atom in mol.GetAtoms()))
print(f"  {acyclic_count} acyclic molecules (each treated as its own scaffold).")

# ── STEP 2: Load or compute Mordred for Withdrawn 2.0 ────────────────────────
if os.path.exists(WITHDRAWN_MORDRED):
    print(f"\nLoading Withdrawn Mordred cache: {WITHDRAWN_MORDRED}")
    X_withdrawn_raw = np.nan_to_num(
        pd.read_csv(WITHDRAWN_MORDRED).values.astype(np.float64))
    df_withdrawn = pd.read_csv(WITHDRAWN_CSV)
else:
    print("\nComputing Mordred descriptors for Withdrawn 2.0 ...")
    from mordred import Calculator, descriptors as mordred_desc
    import time
    df_withdrawn = pd.read_csv(WITHDRAWN_CSV)
    for col in [WITHDRAWN_SMILES_COL, "SMILES", "smiles", "Smiles"]:
        if col in df_withdrawn.columns:
            WITHDRAWN_SMILES_COL = col
            break
    raw_smiles = df_withdrawn[WITHDRAWN_SMILES_COL].dropna().tolist()
    mols, valid_idx = [], []
    for i, smi in enumerate(raw_smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            mols.append(mol); valid_idx.append(i)
    print(f"  {len(mols)}/{len(raw_smiles)} valid molecules")
    t0      = time.time()
    calc    = Calculator(mordred_desc, ignore_3D=True)
    desc_df = calc.pandas(mols).apply(pd.to_numeric, errors="coerce")
    for col in feat_names:
        if col not in desc_df.columns:
            desc_df[col] = np.nan
    desc_df = desc_df[feat_names]
    print(f"  Done in {time.time()-t0:.1f}s")
    X_w = np.full((len(raw_smiles), len(feat_names)), np.nan)
    for li, gi in enumerate(valid_idx):
        X_w[gi] = desc_df.values[li]
    X_w = np.nan_to_num(X_w)
    pd.DataFrame(X_w, columns=feat_names).to_csv(WITHDRAWN_MORDRED, index=False)
    print(f"  Saved cache: {WITHDRAWN_MORDRED}")
    X_withdrawn_raw = X_w
print(f"  Withdrawn: {X_withdrawn_raw.shape[0]} × {X_withdrawn_raw.shape[1]}")

# ── STEP 3: Remove overlap ────────────────────────────────────────────────────
print("\nRemoving overlapping compounds ...")
df_withdrawn_full = pd.read_csv(WITHDRAWN_CSV)
for col in ["smiles", "SMILES", "Smiles"]:
    if col in df_withdrawn_full.columns:
        WITHDRAWN_SMILES_COL = col; break

unitox_smiles_set = set(df_unitox["SMILES_filled"].str.strip().str.upper())
withdrawn_smiles  = df_withdrawn_full[WITHDRAWN_SMILES_COL].fillna("").str.strip().str.upper().tolist()
keep_mask         = np.array([s not in unitox_smiles_set for s in withdrawn_smiles])
X_withdrawn_raw   = X_withdrawn_raw[keep_mask]
print(f"  Removed {(~keep_mask).sum()} overlapping compounds  "
      f"| {X_withdrawn_raw.shape[0]} Withdrawn molecules remain")

# ── STEP 4: Standardise (fit on UniTox) ──────────────────────────────────────
nonconstant = X_unitox_raw.std(axis=0) > 1e-8
X_uni = X_unitox_raw[:, nonconstant].astype(np.float32)
X_wit = X_withdrawn_raw[:, nonconstant].astype(np.float32)
scaler    = StandardScaler()
X_uni_s   = scaler.fit_transform(X_uni)
X_wit_s   = scaler.transform(X_wit)
print(f"\n  {X_uni_s.shape[1]} non-constant descriptors retained")

# ── STEP 5: PCA ───────────────────────────────────────────────────────────────
pca   = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
Z_uni = pca.fit_transform(X_uni_s)
Z_wit = pca.transform(X_wit_s)
ev    = pca.explained_variance_ratio_
print(f"  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%  "
      f"Top {N_COMPONENTS}: {ev.sum()*100:.1f}% total")

# ── FIGURE 1: PCA scatter ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Z_uni[:, 0], Z_uni[:, 1], s=12, alpha=0.35, color="#4A90D9",
           label=f"UniTox (n={len(Z_uni)})", zorder=2)
ax.scatter(Z_wit[:, 0], Z_wit[:, 1], s=20, alpha=0.70, color="#E85C5C",
           marker="^", label=f"Withdrawn 2.0 (n={len(Z_wit)})", zorder=3)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance explained)", fontsize=19)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance explained)", fontsize=19)
ax.set_title("Chemical Space — Mordred Descriptor PCA\n"
             "PCA fitted on UniTox; Withdrawn projected onto same axes",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=12, framealpha=0.9)
ax.set_facecolor("#F8F8F8")
ax.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
fig.savefig("pca_chemical_space_mordred.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: pca_chemical_space_mordred.png")

# ── FIGURE 2: Scree plot ──────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
pcs = range(1, N_COMPONENTS + 1)
ax1.bar(pcs, ev * 100, color="#4A90D9", alpha=0.7, label="Individual")
ax2.plot(pcs, np.cumsum(ev) * 100, color="#E85C5C",
         marker="o", linewidth=2, markersize=5, label="Cumulative")
ax1.set_xlabel("Principal Component", fontsize=19)
ax1.set_ylabel("Explained Variance (%)", fontsize=19, color="#4A90D9")
ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=18, color="#E85C5C")
ax1.set_xticks(list(pcs))
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, fontsize=14, loc="center right",
           bbox_to_anchor=(0.98, 0.5), frameon=True, facecolor="white")
plt.tight_layout()
fig.savefig("pca_scree_mordred.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: pca_scree_mordred.png")

# ── FIGURE 3: Tanimoto similarity ─────────────────────────────────────────────
print("\nComputing Tanimoto similarities (Morgan FP, radius=2, 2048 bits) ...")

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None

uni_fps = [fp for fp in [smiles_to_fp(s) for s in df_unitox["SMILES_filled"]]
           if fp is not None]
wit_smiles  = [s for s, k in zip(withdrawn_smiles, keep_mask) if k]
wit_fps     = [fp for fp in [smiles_to_fp(s) for s in wit_smiles] if fp is not None]

max_sims, mean_top3 = [], []
for fp in wit_fps:
    sims = sorted(DataStructs.BulkTanimotoSimilarity(fp, uni_fps), reverse=True)
    max_sims.append(sims[0])
    mean_top3.append(np.mean(sims[:3]))

max_sims = np.array(max_sims)
mean_top3 = np.array(mean_top3)
pct_ood   = (max_sims < OOD_THRESHOLD).mean() * 100
print(f"  Mean max Tanimoto: {max_sims.mean():.3f}")
print(f"  Median:            {np.median(max_sims):.3f}")
print(f"  % OOD (<{OOD_THRESHOLD}):   {pct_ood:.1f}%")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(max_sims, bins=30, color="#4A90D9", alpha=0.75, edgecolor="white",
        label="Max Tanimoto (nearest neighbour)")
ax.hist(mean_top3, bins=30, color="#E8A838", alpha=0.60, edgecolor="white",
        label="Mean top-3 Tanimoto")
ax.axvline(OOD_THRESHOLD, color="#E85C5C", linestyle="--", linewidth=1.5,
           label=f"OOD threshold (< {OOD_THRESHOLD})")
ax.text(OOD_THRESHOLD + 0.01, ax.get_ylim()[1] * 0.85,
        f"{pct_ood:.0f}% of Withdrawn drugs\nhave max Tanimoto < {OOD_THRESHOLD}",
        fontsize=9, color="#E85C5C")
ax.set_xlabel("Tanimoto Similarity to nearest UniTox compound\n"
              "(Morgan fingerprints, radius=2, 2048 bits)", fontsize=11)
ax.set_ylabel("Number of Withdrawn drugs", fontsize=11)
ax.set_title("Chemical Similarity: Withdrawn 2.0 vs UniTox",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("#F8F8F8")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
fig.savefig("tanimoto_similarity_mordred.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: tanimoto_similarity_mordred.png")

pd.DataFrame({
    "metric": ["mean_max_tanimoto", "median_max_tanimoto",
               f"pct_ood_below_{OOD_THRESHOLD}", "mean_top3_tanimoto"],
    "value":  [round(max_sims.mean(), 3), round(np.median(max_sims), 3),
               round(pct_ood, 1),         round(mean_top3.mean(), 3)]
}).to_csv("chemical_space_summary.csv", index=False)
print("Saved: chemical_space_summary.csv")
print("\n✓ PCA analysis complete.")
