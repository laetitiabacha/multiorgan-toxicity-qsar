"""
01_data_exploration.py
======================
Produces endpoint distribution histograms for UniTox and Withdrawn 2.0.
Figures:
  - figure_unitox_endpoint_histogram.png
  - figure_withdrawn_endpoint_histogram.png

REQUIRES: UniTox_with_recovered_typos_v3.csv, withdrawn_external_validation.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os

# ── Config ────────────────────────────────────────────────────────────────────
UNITOX_FILE    = "UniTox_with_recovered_typos_v3.csv"
WITHDRAWN_FILE = "withdrawn_external_validation.csv"
OUT_DIR        = "figures"
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
})

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — UniTox endpoint distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Loading UniTox ...")
df = pd.read_csv(UNITOX_FILE)
df["n_toxic"] = (df[ENDPOINTS] == 1).sum(axis=1)
coverage = df["n_toxic"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7.2, 4.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

bars = ax.bar(coverage.index, coverage.values,
              color="#4F81BD", edgecolor="white", linewidth=0.8, width=0.65)

ax.set_xlabel("Number of toxic endpoints per molecule")
ax.set_ylabel("Number of molecules")
ax.set_title("UniTox: Distribution of toxicity endpoint annotations per molecule",
             fontsize=11, fontweight="bold")
ax.set_xticks(range(int(coverage.index.max()) + 1))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#aaaaaa")
ax.spines["bottom"].set_color("#aaaaaa")

offset = max(coverage.values) * 0.015
for bar, val in zip(bars, coverage.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:,}", ha="center", va="bottom", fontsize=9, color="#333333")

plt.tight_layout()
out = f"{OUT_DIR}/figure_unitox_endpoint_histogram.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Withdrawn 2.0 endpoint distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Withdrawn 2.0 ...")
df_w = pd.read_csv(WITHDRAWN_FILE)

# Count how many of the 8 endpoints each drug is annotated with
TOXTYPE_MAP = {
    "cardiotoxicity":           "cardiotoxicity__binary",
    "dermatological":           "dermatological_toxicity__binary",
    "hematological":            "hematological__binary",
    "infertility":              "infertility__binary",
    "liver":                    "liver_toxicity__binary",
    "ototoxicity":              "ototoxicity__binary",
    "pulmonary":                "pulmonary_toxicity__binary",
    "renal":                    "renal_toxicity__binary",
}

def count_endpoints(toxtype_str):
    if not isinstance(toxtype_str, str):
        return 0
    toks = [t.strip().lower() for t in toxtype_str.split(",")]
    return sum(1 for key in TOXTYPE_MAP if any(key in tok for tok in toks))

df_w["n_toxic"] = df_w["toxtype"].apply(count_endpoints)
coverage_w = df_w["n_toxic"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7.2, 4.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

bars = ax.bar(coverage_w.index, coverage_w.values,
              color="#4472C4", edgecolor="white", linewidth=0.8, width=0.65)

ax.set_xlabel("Number of toxic endpoints per drug")
ax.set_ylabel("Number of drugs")
ax.set_title(f"Withdrawn 2.0 (n={len(df_w)}): Distribution of toxicity endpoint annotations",
             fontsize=11, fontweight="bold")
ax.set_xticks(range(int(coverage_w.index.max()) + 1))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#aaaaaa")
ax.spines["bottom"].set_color("#aaaaaa")

if len(coverage_w) > 0:
    offset = max(coverage_w.values) * 0.015
    for bar, val in zip(bars, coverage_w.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:,}", ha="center", va="bottom", fontsize=9, color="#333333")

plt.tight_layout()
out = f"{OUT_DIR}/figure_withdrawn_endpoint_histogram.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
print("\nDone — both histograms saved.")
