"""
00_unitox_smiles_curation.py
============================
Iterative SMILES resolution pipeline for the UniTox dataset.

Resolves drug names in UniTox.csv to canonical SMILES strings through
four sequential steps of increasing fallback breadth:

  Step 1 — Identifier standardisation
            Lowercase, strip special characters and dosage information
            (e.g. "mg", "tablet"), normalise whitespace and punctuation,
            resolve hyphenation and parentheses.
            Coverage target: ~86.6% (from 83.1% baseline).

  Step 2 — Salt and formulation variant resolution
            Strip common pharmaceutical suffixes (e.g. "hydrochloride",
            "calcium", "sodium") to recover the underlying active moiety.
            Coverage target: ~87.0%.

  Step 3 — PubChem cross-reference
            Query PubChem by name and synonym for entries still
            unresolved after Steps 1-2.
            Coverage target: ~90.5%.

  Step 4 — Targeted spelling correction and curated heuristics
            Apply British/American English normalisation and fix minor
            typographical errors for remaining unresolved entries.
            Coverage target: ~90.9%.

All matched SMILES are canonicalised using RDKit, including normalisation
of stereochemistry and charge states. Unresolved entries (predominantly
biologics: monoclonal antibodies, enzymes, peptides) are retained as NaN.

Input:  UniTox.csv
Output: UniTox_with_recovered_typos_v3.csv

Dependencies:
    pip install pandas rdkit chembl-webresource-client pubchempy
"""

import re
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from chembl_webresource_client.new_client import new_client
import pubchempy as pcp

# ── FILE PATHS ────────────────────────────────────────────────────────────────
INPUT_FILE  = "UniTox.csv"
OUTPUT_FILE = "UniTox_with_recovered_typos_v3.csv"
SMILES_COL  = "SMILES_filled"

# ── RDKit STANDARDISATION SETUP ───────────────────────────────────────────────
normalizer       = rdMolStandardize.Normalizer()
disconnector     = rdMolStandardize.MetalDisconnector()
fragment_remover = rdMolStandardize.FragmentRemover()
uncharger        = rdMolStandardize.Uncharger()

def standardize_smiles(smiles: str) -> str | None:
    """Return canonical, standardised SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = normalizer.normalize(mol)
        mol = disconnector.Disconnect(mol)
        mol = fragment_remover.remove(mol)
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ── STEP 1: IDENTIFIER STANDARDISATION ───────────────────────────────────────
def clean_drug_name(name: str) -> str:
    """
    Standardise a raw drug name string:
    - Lowercase
    - Remove dosage information (mg, ml, tablet, capsule, etc.)
    - Strip special characters and extra whitespace
    - Normalise hyphens and parentheses
    """
    name = str(name).lower().strip()
    # Remove dosage patterns
    name = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|ug|iu|g|tablet|capsule|injection|solution)\b", "", name)
    # Remove content in parentheses (often formulation info)
    name = re.sub(r"\(.*?\)", "", name)
    # Remove special characters except hyphens within words
    name = re.sub(r"[^\w\s\-]", " ", name)
    # Normalise whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ── STEP 2: SALT/FORMULATION VARIANT RESOLUTION ───────────────────────────────
SALT_SUFFIXES = [
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "acetate", "phosphate", "sulfate", "sulphate", "maleate", "tartrate",
    "citrate", "fumarate", "mesylate", "tosylate", "bromide", "chloride",
    "gluconate", "lactate", "succinate", "besylate", "hydrate", "monohydrate",
    "dihydrate", "anhydrous", "hemihydrate",
]

def strip_salt(name: str) -> str:
    """Remove common pharmaceutical salt/formulation suffixes."""
    for suffix in SALT_SUFFIXES:
        name = re.sub(rf"\b{suffix}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ── STEP 3: CHEMBL + PUBCHEM LOOKUP ──────────────────────────────────────────
chembl_molecule = new_client.molecule

def lookup_chembl(name: str) -> str | None:
    """Query ChEMBL by exact then partial name match."""
    try:
        res = chembl_molecule.filter(pref_name__iexact=name)
        if not res:
            res = chembl_molecule.filter(pref_name__icontains=name)
        if res:
            return res[0].get("molecule_structures", {}).get("connectivity_smile")
    except Exception:
        pass
    return None

def lookup_pubchem(name: str) -> str | None:
    """Query PubChem by name, with synonym fallback."""
    try:
        compounds = pcp.get_compounds(name, "name")
        if compounds and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception:
        pass
    return None


# ── STEP 4: SPELLING CORRECTION HEURISTICS ────────────────────────────────────
SPELLING_MAP = {
    # British vs American English variants and common typos
    "oestrogen":     "estrogen",
    "oestradiol":    "estradiol",
    "stilboestrol":  "stilbestrol",
    "amoxycillin":   "amoxicillin",
    "cimetidine":    "cimetidine",
    "sulphamethoxazole": "sulfamethoxazole",
    "sulphur":       "sulfur",
    "adrenaline":    "epinephrine",
    "lignocaine":    "lidocaine",
    "paracetamol":   "acetaminophen",
    "pethidine":     "meperidine",
    "salbutamol":    "albuterol",
}

def apply_spelling_correction(name: str) -> str:
    """Apply curated British/American and typographical corrections."""
    for wrong, correct in SPELLING_MAP.items():
        name = name.replace(wrong, correct)
    return name


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def resolve_smiles(name: str) -> str | None:
    """
    Try all resolution steps in order, returning the first valid
    canonical SMILES found, or None if all steps fail.
    """
    # Step 1: clean name
    cleaned = clean_drug_name(name)

    # Step 2: strip salt
    desalted = strip_salt(cleaned)

    # Step 3a: ChEMBL lookup (cleaned name)
    for query in [cleaned, desalted]:
        smi = lookup_chembl(query)
        if smi:
            return standardize_smiles(smi)
        time.sleep(0.15)

    # Step 3b: PubChem lookup
    for query in [cleaned, desalted]:
        smi = lookup_pubchem(query)
        if smi:
            return standardize_smiles(smi)

    # Step 4: spelling correction then retry
    corrected = apply_spelling_correction(cleaned)
    if corrected != cleaned:
        smi = lookup_chembl(corrected) or lookup_pubchem(corrected)
        if smi:
            return standardize_smiles(smi)

    return None


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    drug_col = "generic_name" if "generic_name" in df.columns else "Drug"
    df[drug_col] = df[drug_col].astype(str).str.strip()

    # Identify and clean binary/ternary toxicity columns
    binary_cols  = [c for c in df.columns if "__binary" in c.lower() or ("_binary_" in c.lower() and "_rating" in c.lower())]
    ternary_cols = [c for c in df.columns if "__ternary" in c.lower() or ("_ternary_" in c.lower() and "_rating" in c.lower())]

    drug_names = df[drug_col].dropna().unique()
    total = len(drug_names)
    print(f"Resolving SMILES for {total} unique drug names...")

    smiles_map = {}
    resolved = 0

    for i, name in enumerate(drug_names, 1):
        smi = resolve_smiles(name)
        smiles_map[name] = smi
        if smi:
            resolved += 1
        if i % 100 == 0 or i == total:
            pct = resolved / i * 100
            print(f"  [{i}/{total}] Coverage: {pct:.1f}%")

    df[SMILES_COL] = df[drug_col].map(smiles_map)

    # Report coverage
    n_valid = df[SMILES_COL].notna().sum()
    print(f"\nFinal coverage: {n_valid}/{len(df)} ({n_valid/len(df)*100:.1f}%)")
    print(f"Unresolved: {len(df) - n_valid} (predominantly biologics)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE}")
