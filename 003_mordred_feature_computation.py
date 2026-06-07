"""
02_mordred_feature_computation.py
==================================
Computes Mordred 2D physicochemical descriptors for UniTox and Withdrawn 2.0.

Mordred computes 1,613 2D descriptors per molecule covering physicochemical
properties (molecular weight, logP, solubility), topological indices
(connectivity, shape), and electronic descriptors (polar surface area,
hydrogen-bond counts). After removing zero-variance descriptors, 1,404
are retained for modelling.

Outputs:
  mordred_features_cached.csv      — descriptor matrix for UniTox (2,196 rows)
  mordred_withdrawn_cached.csv     — descriptor matrix for Withdrawn 2.0 (153 rows)

Row order matches the corresponding CSV files exactly:
  mordred_features_cached.csv      → UniTox_with_recovered_typos_v3.csv
                                     (rows where SMILES_filled is not null)
  mordred_withdrawn_cached.csv     → withdrawn_external_validation.csv

Dependencies:
    pip install mordred rdkit pandas numpy
    (mordred requires rdkit >= 2020)

Runtime: approximately 10-15 minutes for UniTox on a standard CPU.
"""

import time
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors as mordred_desc

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── FILE PATHS ────────────────────────────────────────────────────────────────
UNITOX_CSV           = "UniTox_with_recovered_typos_v3.csv"
WITHDRAWN_CSV        = "withdrawn_external_validation.csv"
UNITOX_OUT           = "mordred_features_cached.csv"
WITHDRAWN_OUT        = "mordred_withdrawn_cached.csv"

UNITOX_SMILES_COL    = "SMILES_filled"
WITHDRAWN_SMILES_COL = "smiles"


# ── MORDRED CALCULATOR ────────────────────────────────────────────────────────
def compute_mordred(smiles_list: list[str], label: str = "") -> pd.DataFrame:
    """
    Compute Mordred 2D descriptors for a list of SMILES strings.

    Invalid SMILES are represented as rows of NaN. Row order matches
    the input list exactly, preserving alignment with the source CSV.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to featurise.
    label : str
        Label for progress reporting.

    Returns
    -------
    pd.DataFrame
        Shape (n_molecules, n_descriptors). Numeric descriptors only;
        inf values replaced with NaN.
    """
    calc = Calculator(mordred_desc, ignore_3D=True)

    mols = []
    invalid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        mols.append(mol)
        if mol is None:
            invalid_idx.append(i)

    valid_mols = [m for m in mols if m is not None]
    valid_idx  = [i for i, m in enumerate(mols) if m is not None]

    print(f"  {label}: {len(valid_mols)} valid / {len(mols)} total molecules")
    if invalid_idx:
        print(f"  Invalid SMILES at indices: {invalid_idx[:10]}{'...' if len(invalid_idx) > 10 else ''}")

    # Compute descriptors for valid molecules
    t0 = time.time()
    desc_df = calc.pandas(valid_mols)
    print(f"  Computation time: {time.time() - t0:.0f}s")

    # Keep numeric columns only; replace inf with NaN
    desc_df = desc_df.select_dtypes(include=[np.number])
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.reset_index(drop=True)

    # Reconstruct full matrix with NaN rows for invalid molecules
    n_total = len(smiles_list)
    n_feat  = desc_df.shape[1]
    full_matrix = np.full((n_total, n_feat), np.nan)
    for row_pos, orig_idx in enumerate(valid_idx):
        full_matrix[orig_idx] = desc_df.iloc[row_pos].values

    result = pd.DataFrame(full_matrix, columns=desc_df.columns)
    print(f"  Output shape: {result.shape}")
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── UniTox ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("UniTox Mordred descriptors")
    print("=" * 60)

    unitox = (
        pd.read_csv(UNITOX_CSV)
        .dropna(subset=[UNITOX_SMILES_COL])
        .reset_index(drop=True)
    )
    print(f"Loaded {len(unitox)} UniTox molecules with valid SMILES")

    unitox_desc = compute_mordred(
        unitox[UNITOX_SMILES_COL].tolist(),
        label="UniTox"
    )

    unitox_desc.to_csv(UNITOX_OUT, index=False)
    print(f"Saved -> {UNITOX_OUT}  ({unitox_desc.shape[0]} rows x {unitox_desc.shape[1]} descriptors)")

    # ── Withdrawn 2.0 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Withdrawn 2.0 Mordred descriptors")
    print("=" * 60)

    withdrawn = pd.read_csv(WITHDRAWN_CSV)
    print(f"Loaded {len(withdrawn)} Withdrawn 2.0 molecules")

    withdrawn_desc = compute_mordred(
        withdrawn[WITHDRAWN_SMILES_COL].tolist(),
        label="Withdrawn 2.0"
    )

    withdrawn_desc.to_csv(WITHDRAWN_OUT, index=False)
    print(f"Saved -> {WITHDRAWN_OUT}  ({withdrawn_desc.shape[0]} rows x {withdrawn_desc.shape[1]} descriptors)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"UniTox:      {unitox_desc.shape[0]} molecules x {unitox_desc.shape[1]} descriptors")
    print(f"Withdrawn:   {withdrawn_desc.shape[0]} molecules x {withdrawn_desc.shape[1]} descriptors")

    # Check for zero-variance columns (informational only — filtering is done
    # inside the modelling scripts on the training partition to prevent leakage)
    zero_var = (unitox_desc.var(axis=0) == 0).sum()
    print(f"Zero-variance descriptors (UniTox): {zero_var}")
    print(f"Retained after variance filter:     {unitox_desc.shape[1] - zero_var}")
    print("\nDone.")
