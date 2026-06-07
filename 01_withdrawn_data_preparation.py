"""
01_withdrawn_data_preparation.py
=================================
Pre-processing pipeline for the Withdrawn 2.0 external validation dataset.

Produces withdrawn_external_validation.csv from the raw Withdrawn 2.0 export.

Steps:
  1. Load raw Withdrawn 2.0 data (withdrawn_drugs.csv)
  2. Map toxtype field to UniTox endpoint names using a rule-based dictionary
     - Unrecognised categories (neurological, carcinogenicity, muscular,
       ophthalmic, etc.) are excluded
     - Multi-reason entries are split on commas and each term mapped
       independently; all matched endpoints are retained
     - Drugs with no mappable category are excluded entirely
  3. Remove overlap with UniTox training set via canonical SMILES matching
     (generated with RDKit)
  4. Save final external validation set

Input:  withdrawn_drugs.csv       (raw Withdrawn 2.0 export)
        UniTox_with_recovered_typos_v3.csv  (for overlap removal)
Output: withdrawn_external_validation.csv  (153 unique drugs)

Dependencies:
    pip install pandas rdkit
"""

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ── FILE PATHS ────────────────────────────────────────────────────────────────
WITHDRAWN_RAW   = "withdrawn_drugs.csv"
UNITOX_FILE     = "UniTox_with_recovered_typos_v3.csv"
OUTPUT_FILE     = "withdrawn_external_validation.csv"

WITHDRAWN_SMILES_COL = "smiles"
UNITOX_SMILES_COL    = "SMILES_filled"

# ── TOXTYPE MAPPING ───────────────────────────────────────────────────────────
# Maps raw Withdrawn 2.0 toxtype tokens to UniTox binary endpoint names.
# Withdrawal reasons not present in this dictionary (e.g. neurological,
# carcinogenicity, muscular, ophthalmic) are excluded from the validation set.
MAPPER = {
    "hepatic":        "liver_toxicity",
    "cardiovascular": "cardiotoxicity",
    "dermatological": "dermatological_toxicity",
    "hematological":  "hematological",
    "renal":          "renal_toxicity",
    "respiratory":    "pulmonary_toxicity",
    "reproductive":   "infertility",
    "ototoxicity":    "ototoxicity",
}


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
def load_withdrawn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[["drugname", WITHDRAWN_SMILES_COL, "toxtype"]].copy()
    df = df.dropna(subset=[WITHDRAWN_SMILES_COL, "toxtype"]).reset_index(drop=True)
    print(f"Raw Withdrawn 2.0: {len(df)} rows")
    return df


# ── STEP 2: TOXTYPE MAPPING ───────────────────────────────────────────────────
def map_toxtype(raw: str) -> str | None:
    """
    Split a raw toxtype string on commas, map each token through MAPPER,
    deduplicate, and return as a comma-separated string.
    Returns None if no token matches any of the 8 endpoints.
    """
    if not isinstance(raw, str):
        return None
    tokens = [t.strip().lower() for t in raw.split(",")]
    mapped = []
    seen = set()
    for t in tokens:
        endpoint = MAPPER.get(t)
        if endpoint and endpoint not in seen:
            mapped.append(endpoint)
            seen.add(endpoint)
    return ", ".join(mapped) if mapped else None


def apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["toxtype"] = df["toxtype"].apply(map_toxtype)
    n_before = len(df)
    df = df.dropna(subset=["toxtype"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} drugs with no mappable endpoint")
    print(f"Remaining: {len(df)} drugs")

    # Deduplicate on (smiles, toxtype)
    df = df.drop_duplicates(subset=[WITHDRAWN_SMILES_COL, "toxtype"]).reset_index(drop=True)
    print(f"After deduplication: {len(df)} drugs")

    # Verify ototoxicity count
    oto = df["toxtype"].str.contains("ototoxicity", na=False).sum()
    print(f"Drugs with ototoxicity: {oto}  (expected: 0)")
    return df


# ── STEP 3: OVERLAP REMOVAL ───────────────────────────────────────────────────
def canonical(smi: str) -> str | None:
    """Return RDKit canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def remove_overlap(df: pd.DataFrame, unitox_path: str) -> pd.DataFrame:
    """Remove any withdrawn drug whose canonical SMILES matches a UniTox molecule."""
    unitox = pd.read_csv(unitox_path)
    unitox_canon = set(
        unitox[UNITOX_SMILES_COL]
        .dropna()
        .apply(canonical)
        .dropna()
    )
    print(f"\nUniTox canonical SMILES: {len(unitox_canon)}")

    df = df.copy()
    df["canon_smiles"] = df[WITHDRAWN_SMILES_COL].apply(canonical)
    df = df.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    n_before = len(df)
    df = df[~df["canon_smiles"].isin(unitox_canon)].reset_index(drop=True)
    print(f"Removed {n_before - len(df)} drugs overlapping with UniTox")
    print(f"Remaining after overlap removal: {len(df)} drugs")
    return df


# ── STEP 4: SAVE ─────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: str) -> None:
    out = df[["drugname", WITHDRAWN_SMILES_COL, "toxtype"]].copy()
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
    print(f"Final dataset: {len(out)} drugs")
    print("\nEndpoint distribution:")
    dist = (
        out["toxtype"]
        .str.split(", ")
        .explode()
        .value_counts()
    )
    for ep, count in dist.items():
        print(f"  {ep:<35} {count}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_withdrawn(WITHDRAWN_RAW)

    print("\n--- Step 2: Toxtype mapping ---")
    df = apply_mapping(df)

    print("\n--- Step 3: Overlap removal ---")
    df = remove_overlap(df, UNITOX_FILE)

    print("\n--- Step 4: Save ---")
    save(df, OUTPUT_FILE)
